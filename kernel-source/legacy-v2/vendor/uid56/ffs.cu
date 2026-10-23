// FFS: native sm_120 version of fla_fused._ps_seq_x_kernel (e4c ps mode, PEEL, BV=32 / 4 warps in Triton), output
// bit-identical to the Triton kernel.
//
// Triton kernel arithmetic (PTX/SASS of the e4c build, dump/ps_seq_x.*), per chunk of 64 rows, per (value head, V):
//   hs   = cvt.rn.bf16(h)                                           (h fp32 state, [K=128, V])
//   vn   = float(u) - mma(w, hs)          K=128 ascending (w[:, :64] then w[:, 64:]), acc from +0      (chunk 0: vn = u)
//   oa   = mma(q, hs)                     same order, acc from +0                                   (chunk 0: not run)
//   d2   = mma(Aq, bf16(vn))              K=64 ascending, acc from +0
//   o    = bf16( fma.rn(scale, d2, scale * (oa * ex2(g_t))) )       (chunk 0: scale * (ex2(g_t) * 0))
//   vd   = bf16( vn * ex2(g_last - g_t) )
//   h    = mma(k^T, vd, h * ex2(g_last))  K=64 ascending (chunk 0: acc = ex2(g_last) * 0; last chunk: no update)
// ex2 = ex2.approx.ftz.f32, every mul / add / fma a single RN op (no further contraction in the SASS), mma =
// mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32.
//
// Here every product is evaluated TRANSPOSED (v as the M dimension): hT, vnT, oT are m16n8 accumulators whose C layout
// is directly the A-operand layout of the next mma (no shared-memory round trip for h / v_new / vd), each warp owns 16
// value columns for all K and all 64 chunk rows, and the chunk operands (w, q, k, Aq: shared by all value columns of a
// head) are loaded once per CTA into swizzled smem and read with ldmatrix (k with .trans). Each output element gets the
// same products in the same k16 order through the same instruction, so the bits are the Triton kernel's (verified).
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

namespace {
constexpr int H = 16, HV = 32, KD = 128, VD = 128, BT = 64;

DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
DEVI void cp16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}
DEVI void cp4(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(src) : "memory");
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
DEVI void ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr) : "memory");
}
DEVI void ldsm4t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr) : "memory");
}
DEVI void stsm4t(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1,%2,%3,%4};\n" ::"r"(addr), "r"(r0), "r"(r1),
                 "r"(r2), "r"(r3) : "memory");
}
DEVI void mma(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
DEVI float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}
// two floats -> bf16x2 (lo in the low half), RN (cvt.rn.bf16x2.f32, as Triton)
DEVI uint32_t pk(float lo, float hi) {
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
DEVI float blo_f(uint32_t x) { return __uint_as_float(x << 16); }
DEVI float bhi_f(uint32_t x) { return __uint_as_float(x & 0xffff0000u); }
DEVI float fmul(float a, float b) { return __fmul_rn(a, b); }
DEVI float fsub(float a, float b) { return __fsub_rn(a, b); }
DEVI float ffma(float a, float b, float c) { return __fmaf_rn(a, b, c); }

// swizzled smem tiles (byte offsets): 64 rows x 256 B (w, q, k), 64 x 128 B (Aq), 64 x 32 B (per-warp u / o)
DEVI uint32_t o256(int r, int ch) { return r * 256 + ((ch ^ (r & 7)) << 4); }
DEVI uint32_t o128(int r, int ch) { return r * 128 + ((ch ^ (r & 7)) << 4); }
DEVI uint32_t o32(int r, int ch) { return r * 32 + ((ch ^ ((r >> 2) & 1)) << 4); }

constexpr int S_W = 0, S_Q = 16384, S_K = 32768, S_A = 49152, S_G = 57344, S_U = 57856;

template <int WPC>
struct SeqCfg {
    static constexpr int NTH = WPC * 32;
    static constexpr int S_O = S_U + WPC * 2048;
    static constexpr int SMEM = S_O + WPC * 2048;
};

template <int WPC, bool SKIP>
DEVI void seq_body(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                   const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ u,
                   const __nv_bfloat16* __restrict__ Aq, const float* __restrict__ g, __nv_bfloat16* __restrict__ o,
                   const int* __restrict__ slen, int T, float scale) {
    using C = SeqCfg<WPC>;
    constexpr int NTH = C::NTH;
    extern __shared__ __align__(128) uint8_t smem[];
    const int inh = blockIdx.y;
    const int n = inh / HV, h = inh % HV;
    if (slen[n] == 0) return;
    const int hk = h / (HV / H);
    const int NT = T / BT;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gq = lane >> 2, cq = lane & 3;
    const int v0 = blockIdx.x * 16 * WPC + warp * 16;
    const uint32_t sb = smem_u32(smem);
    const uint32_t sU = sb + S_U + warp * 2048, sO = sb + C::S_O + warp * 2048;
    const size_t row0 = static_cast<size_t>(n) * T;

    auto issue_a = [&](int i) {   // w, q (not for chunk 0), this warp's u columns, g
        const size_t r0 = row0 + static_cast<size_t>(i) * BT;
        if (i > 0) {
#pragma unroll
            for (int x = tid; x < 1024; x += NTH) {
                const int rr = x >> 4, ch = x & 15;
                cp16(sb + S_W + o256(rr, ch), w + ((r0 + rr) * HV + h) * KD + ch * 8);
                cp16(sb + S_Q + o256(rr, ch), q + ((r0 + rr) * H + hk) * KD + ch * 8);
            }
        }
#pragma unroll
        for (int x = lane; x < 128; x += 32) {
            const int rr = x >> 1, ch = x & 1;
            cp16(sU + o32(rr, ch), u + ((r0 + rr) * HV + h) * VD + v0 + ch * 8);
        }
        if (tid < 64) cp4(sb + S_G + (i & 1) * 256 + tid * 4, g + (r0 + tid) * HV + h);
    };
    auto issue_b = [&](int i) {   // Aq
        const size_t r0 = row0 + static_cast<size_t>(i) * BT;
#pragma unroll
        for (int x = tid; x < 512; x += NTH) {
            const int rr = x >> 3, ch = x & 7;
            cp16(sb + S_A + o128(rr, ch), Aq + ((r0 + rr) * HV + h) * BT + ch * 8);
        }
    };
    auto issue_c = [&](int i) {   // k (rows t, K contiguous)
        const size_t r0 = row0 + static_cast<size_t>(i) * BT;
#pragma unroll
        for (int x = tid; x < 1024; x += NTH) {
            const int rr = x >> 4, ch = x & 15;
            cp16(sb + S_K + o256(rr, ch), k + ((r0 + rr) * H + hk) * KD + ch * 8);
        }
    };

    // per-lane ldmatrix row / chunk selectors
    const int lm = lane >> 3, lr = lane & 7;
    // phase A / B (non-trans, rows = t): tile pair jj, k-chunk pair
    const int ab_t = 8 * (lm >> 1) + lr, ab_c = lm & 1;
    // phase C (trans, rows = t): t-step a, kk-tile pair
    const int c_t = 8 * (lm & 1) + lr, c_c = lm >> 1;
    // u / o (per-warp 64 x 16)
    const int uo_t = 8 * (lm >> 1) + lr, uo_c = lm & 1;

    float hacc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) hacc[j][r] = 0.f;

    issue_a(0);
    cp_commit();
    issue_b(0);
    cp_commit();
    if (NT > 1) issue_c(0);
    cp_commit();
    cp_wait<2>();
    __syncthreads();

    for (int i = 0; i < NT; ++i) {
        const float* gs = reinterpret_cast<const float*>(smem + S_G + (i & 1) * 256);
        // ------------------------------------------------ phase A: vn = u - w h, oa = q h
        float vn[8][4], oa[8][4];
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            uint32_t r0, r1, r2, r3;
            ldsm4t(sU + o32(16 * jj + uo_t, uo_c), r0, r1, r2, r3);
            vn[2 * jj][0] = blo_f(r0);
            vn[2 * jj][1] = bhi_f(r0);
            vn[2 * jj][2] = blo_f(r1);
            vn[2 * jj][3] = bhi_f(r1);
            vn[2 * jj + 1][0] = blo_f(r2);
            vn[2 * jj + 1][1] = bhi_f(r2);
            vn[2 * jj + 1][2] = blo_f(r3);
            vn[2 * jj + 1][3] = bhi_f(r3);
        }
        if (i > 0) {
            float acc[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int r = 0; r < 4; ++r) acc[j][r] = 0.f, oa[j][r] = 0.f;
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const uint32_t a0 = pk(hacc[2 * s][0], hacc[2 * s][1]);
                const uint32_t a1 = pk(hacc[2 * s][2], hacc[2 * s][3]);
                const uint32_t a2 = pk(hacc[2 * s + 1][0], hacc[2 * s + 1][1]);
                const uint32_t a3 = pk(hacc[2 * s + 1][2], hacc[2 * s + 1][3]);
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    uint32_t b0, b1, b2, b3;
                    ldsm4(sb + S_W + o256(16 * jj + ab_t, 2 * s + ab_c), b0, b1, b2, b3);
                    mma(acc[2 * jj], a0, a1, a2, a3, b0, b1);
                    mma(acc[2 * jj + 1], a0, a1, a2, a3, b2, b3);
                    ldsm4(sb + S_Q + o256(16 * jj + ab_t, 2 * s + ab_c), b0, b1, b2, b3);
                    mma(oa[2 * jj], a0, a1, a2, a3, b0, b1);
                    mma(oa[2 * jj + 1], a0, a1, a2, a3, b2, b3);
                }
            }
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int r = 0; r < 4; ++r) vn[j][r] = fsub(vn[j][r], acc[j][r]);
        }
        cp_wait<1>();
        __syncthreads();   // X1: Aq(i) visible; every warp is done with w, q, u of chunk i
        if (i + 1 < NT) issue_a(i + 1);
        cp_commit();
        // ------------------------------------------------ phase B: o = scale * Aq v_new + scale * (q h) 2^g
        {
            float d2[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int r = 0; r < 4; ++r) d2[j][r] = 0.f;
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                const uint32_t a0 = pk(vn[2 * a][0], vn[2 * a][1]);
                const uint32_t a1 = pk(vn[2 * a][2], vn[2 * a][3]);
                const uint32_t a2 = pk(vn[2 * a + 1][0], vn[2 * a + 1][1]);
                const uint32_t a3 = pk(vn[2 * a + 1][2], vn[2 * a + 1][3]);
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    if (SKIP && a > jj) continue;   // Aq[t][s] = +0 for s > t: the whole k16 x n8 B block is zero
                    uint32_t b0, b1, b2, b3;
                    ldsm4(sb + S_A + o128(16 * jj + ab_t, 2 * a + ab_c), b0, b1, b2, b3);
                    mma(d2[2 * jj], a0, a1, a2, a3, b0, b1);
                    mma(d2[2 * jj + 1], a0, a1, a2, a3, b2, b3);
                }
            }
            uint32_t ob[8][2];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float e0 = ex2(gs[8 * j + 2 * cq]), e1 = ex2(gs[8 * j + 2 * cq + 1]);
                float c0, c1, c2, c3;
                if (i > 0) {
                    c0 = fmul(scale, fmul(oa[j][0], e0));
                    c1 = fmul(scale, fmul(oa[j][1], e1));
                    c2 = fmul(scale, fmul(oa[j][2], e0));
                    c3 = fmul(scale, fmul(oa[j][3], e1));
                } else {
                    c0 = c2 = fmul(scale, fmul(e0, 0.f));
                    c1 = c3 = fmul(scale, fmul(e1, 0.f));
                }
                ob[j][0] = pk(ffma(scale, d2[j][0], c0), ffma(scale, d2[j][1], c1));
                ob[j][1] = pk(ffma(scale, d2[j][2], c2), ffma(scale, d2[j][3], c3));
            }
#pragma unroll
            for (int jj = 0; jj < 4; ++jj)
                stsm4t(sO + o32(16 * jj + uo_t, uo_c), ob[2 * jj][0], ob[2 * jj][1], ob[2 * jj + 1][0],
                       ob[2 * jj + 1][1]);
            __syncwarp();
            const size_t r0 = row0 + static_cast<size_t>(i) * BT;
#pragma unroll
            for (int x = lane; x < 128; x += 32) {
                const int rr = x >> 1, ch = x & 1;
                uint4 val;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "r"(sO + o32(rr, ch)) : "memory");
                *reinterpret_cast<uint4*>(o + ((r0 + rr) * HV + h) * VD + v0 + ch * 8) = val;
            }
            __syncwarp();
        }
        cp_wait<1>();
        __syncthreads();   // X2: k(i) visible; every warp is done with Aq(i)
        if (i + 1 < NT) issue_b(i + 1);
        cp_commit();
        // ------------------------------------------------ phase C: h = h 2^g_last + k^T (v_new 2^(g_last - g))
        if (i + 1 < NT) {
            const float gl = gs[63];
            const float el = ex2(gl);
            if (i > 0) {
#pragma unroll
                for (int j = 0; j < 16; ++j)
#pragma unroll
                    for (int r = 0; r < 4; ++r) hacc[j][r] = fmul(hacc[j][r], el);
            } else {
                const float z = fmul(el, 0.f);
#pragma unroll
                for (int j = 0; j < 16; ++j)
#pragma unroll
                    for (int r = 0; r < 4; ++r) hacc[j][r] = z;
            }
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                const float f00 = ex2(fsub(gl, gs[16 * a + 2 * cq])), f01 = ex2(fsub(gl, gs[16 * a + 2 * cq + 1]));
                const float f10 = ex2(fsub(gl, gs[16 * a + 8 + 2 * cq])),
                            f11 = ex2(fsub(gl, gs[16 * a + 8 + 2 * cq + 1]));
                const uint32_t a0 = pk(fmul(vn[2 * a][0], f00), fmul(vn[2 * a][1], f01));
                const uint32_t a1 = pk(fmul(vn[2 * a][2], f00), fmul(vn[2 * a][3], f01));
                const uint32_t a2 = pk(fmul(vn[2 * a + 1][0], f10), fmul(vn[2 * a + 1][1], f11));
                const uint32_t a3 = pk(fmul(vn[2 * a + 1][2], f10), fmul(vn[2 * a + 1][3], f11));
#pragma unroll
                for (int jj = 0; jj < 8; ++jj) {
                    uint32_t b0, b1, b2, b3;
                    ldsm4t(sb + S_K + o256(16 * a + c_t, 2 * jj + c_c), b0, b1, b2, b3);
                    mma(hacc[2 * jj], a0, a1, a2, a3, b0, b1);
                    mma(hacc[2 * jj + 1], a0, a1, a2, a3, b2, b3);
                }
            }
        }
        cp_wait<1>();
        __syncthreads();   // X3: w, q, u, g of chunk i + 1 visible; every warp is done with k(i)
        if (i + 2 < NT) issue_c(i + 1);
        cp_commit();
    }
}

// ================================================================================================================
// FFS fused: _ps_pre_kernel + _ps_seq_x_kernel in one kernel, one CTA per (sequence, value head), 8 warps (warp wi =
// value columns 16 wi .. 16 wi + 15). Per chunk i:
//   P(i): w = A kb (kb = bf16(float(bf16(k beta)) 2^g)), Aq = gated masked q k^T, u = A bf16(v beta), as Triton's
//         pass P evaluates them (u / w / Aq rounded to bf16 where pass P stores them); w, Aq go to smem, u (each warp
//         its own columns, transposed) stays in registers. A's upper 16 x 16 blocks (masked to +0 in Triton) and Aq's
//         tiles above the diagonal (where() -> +0) are skipped: trailing all-zero k16 steps leave the fp32
//         accumulator unchanged (verified bitwise), and fully masked Aq tiles are +0.
//   A(i), B(i), C(i): pass S as in seq_body.
// smem: q, k (double buffer), v, A (inputs), w, Aq (computed), g / beta (double buffer) = 99072 B.
constexpr int F_Q = 0, F_K0 = 16384, F_V = 49152, F_W = 65536, F_A = 81920, F_AQ = 90112, F_G = 98304,
              F_B = 98816, F_SMEM = 99328;

DEVI uint32_t bmul2(uint32_t a, uint32_t b) {   // mul.bf16x2 (RN), as Triton's bf16 * bf16
    uint32_t r;
    asm("mul.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
DEVI uint32_t movt(uint32_t a) {
    uint32_t r;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(r) : "r"(a));
    return r;
}
DEVI void stsm4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n" ::"r"(addr), "r"(r0), "r"(r1),
                 "r"(r2), "r"(r3) : "memory");
}
DEVI void sts32(uint32_t addr, uint32_t v) { asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(addr), "r"(v) : "memory"); }
DEVI void sts128z(uint32_t addr) {
    asm volatile("st.shared.v4.b32 [%0], {%1,%1,%1,%1};\n" ::"r"(addr), "r"(0) : "memory");
}

struct FCtx {
    const __nv_bfloat16 *q, *k, *v, *beta, *A;
    const float* g;
    int n, h, hq, hk, T, tid, warp, lane, gq, cq, lm, lr;
    uint32_t sb;
    size_t row0;
};

// chunk inputs: A, v, k (into buffer `kb`), g, beta (into slot i & 1)
DEVI void f_issue_x(const FCtx& c, int i, uint32_t kbuf) {
    const size_t r0 = c.row0 + static_cast<size_t>(i) * BT;
#pragma unroll
    for (int x = c.tid; x < 512; x += 256) {
        const int rr = x >> 3, ch = x & 7;
        cp16(c.sb + F_A + o128(rr, ch), c.A + ((r0 + rr) * HV + c.h) * BT + ch * 8);
    }
#pragma unroll
    for (int x = c.tid; x < 1024; x += 256) {
        const int rr = x >> 4, ch = x & 15;
        cp16(c.sb + F_V + o256(rr, ch), c.v + ((r0 + rr) * HV + c.h) * VD + ch * 8);
        cp16(kbuf + o256(rr, ch), c.k + ((r0 + rr) * c.hq + c.hk) * KD + ch * 8);
    }
    if (c.tid < 64) {
        cp4(c.sb + F_G + (i & 1) * 256 + c.tid * 4, c.g + (r0 + c.tid) * HV + c.h);
    } else if (c.tid < 128) {
        const int t = c.tid - 64;   // beta pair (h & ~1, h | 1) of row t
        cp4(c.sb + F_B + (i & 1) * 256 + t * 4, c.beta + (r0 + t) * HV + (c.h & ~1));
    }
}
DEVI void f_issue_q(const FCtx& c, int i) {
    const size_t r0 = c.row0 + static_cast<size_t>(i) * BT;
#pragma unroll
    for (int x = c.tid; x < 1024; x += 256) {
        const int rr = x >> 4, ch = x & 15;
        cp16(c.sb + F_Q + o256(rr, ch), c.q + ((r0 + rr) * c.hq + c.hk) * KD + ch * 8);
    }
}

// beta[s], beta[s + 1] as bf16x2 (lo = s)
DEVI uint32_t f_beta2(const FCtx& c, const uint32_t* bs, int s) {
    const uint32_t w0 = bs[s], w1 = bs[s + 1];
    return (c.h & 1) ? __byte_perm(w0, w1, 0x7632) : __byte_perm(w0, w1, 0x5410);
}

// pass P for chunk i: sW (if DO_W), sAq written; uf = this warp's u^T (bf16-rounded, as floats)
template <bool DO_W>
DEVI void f_pstage(const FCtx& c, uint32_t kbuf, const float* gs, const uint32_t* bs, const float (&e)[8][2],
                   float (&uf)[8][4]) {
    const int lane = c.lane, lm = c.lm, lr = c.lr, cq = c.cq, gq = c.gq, wi = c.warp;
    const int ab_t = 8 * (lm >> 1) + lr, ab_c = lm & 1;
    const int c_t = 8 * (lm & 1) + lr, c_c = lm >> 1;
    // ---- P1: w[:, 16 wi .. 16 wi + 15] = A kb
    if (DO_W) {
        uint32_t kb[4][4];
#pragma unroll
        for (int sb = 0; sb < 4; ++sb) {
            uint32_t r[4];
            ldsm4t(kbuf + o256(16 * sb + c_t, 2 * wi + c_c), r[0], r[1], r[2], r[3]);
            const uint32_t blo = f_beta2(c, bs, 16 * sb + 2 * cq), bhi = f_beta2(c, bs, 16 * sb + 8 + 2 * cq);
#pragma unroll
            for (int x = 0; x < 4; ++x) {
                const uint32_t p = bmul2(r[x], (x & 1) ? bhi : blo);
                const float* ee = e[2 * sb + (x & 1)];
                kb[sb][x] = pk(fmul(blo_f(p), ee[0]), fmul(bhi_f(p), ee[1]));
            }
        }
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            float acc[2][4];
#pragma unroll
            for (int nn = 0; nn < 2; ++nn)
#pragma unroll
                for (int r = 0; r < 4; ++r) acc[nn][r] = 0.f;
#pragma unroll
            for (int sb = 0; sb <= mt; ++sb) {
                uint32_t a0, a1, a2, a3;
                ldsm4(c.sb + F_A + o128(16 * mt + (lane & 15), 2 * sb + (lane >> 4)), a0, a1, a2, a3);
                mma(acc[0], a0, a1, a2, a3, kb[sb][0], kb[sb][1]);
                mma(acc[1], a0, a1, a2, a3, kb[sb][2], kb[sb][3]);
            }
            stsm4(c.sb + F_W + o256(16 * mt + c_t, 2 * wi + c_c), pk(acc[0][0], acc[0][1]), pk(acc[0][2], acc[0][3]),
                  pk(acc[1][0], acc[1][1]), pk(acc[1][2], acc[1][3]));
        }
    }
    // ---- P2: Aq tiles (mt, ns), ns <= 2 mt + 1 (20 tiles): warp wi takes tiles wi, wi + 8, wi + 16
#pragma unroll
    for (int tt = 0; tt < 3; ++tt) {
        const int id = wi + 8 * tt;
        if (id < 20) {
            const int mt = id < 2 ? 0 : id < 6 ? 1 : id < 12 ? 2 : 3;
            const int ns = id - (mt == 0 ? 0 : mt == 1 ? 2 : mt == 2 ? 6 : 12);
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int k2 = 0; k2 < 4; ++k2) {
                uint32_t b0, b1, b2, b3;
                ldsm4(kbuf + o256(8 * ns + lr, 4 * k2 + lm), b0, b1, b2, b3);
                uint32_t a0, a1, a2, a3;
                ldsm4(c.sb + F_Q + o256(16 * mt + (lane & 15), 4 * k2 + (lane >> 4)), a0, a1, a2, a3);
                mma(acc, a0, a1, a2, a3, b0, b1);
                ldsm4(c.sb + F_Q + o256(16 * mt + (lane & 15), 4 * k2 + 2 + (lane >> 4)), a0, a1, a2, a3);
                mma(acc, a0, a1, a2, a3, b2, b3);
            }
            const int t0 = 16 * mt + gq, s0 = 8 * ns + 2 * cq;
            const float gt0 = gs[t0], gt1 = gs[t0 + 8], gs0 = gs[s0], gs1 = gs[s0 + 1];
            float r0 = fmul(acc[0], ex2(fsub(gt0, gs0))), r1 = fmul(acc[1], ex2(fsub(gt0, gs1)));
            float r2 = fmul(acc[2], ex2(fsub(gt1, gs0))), r3 = fmul(acc[3], ex2(fsub(gt1, gs1)));
            if (t0 < s0) r0 = 0.f;
            if (t0 < s0 + 1) r1 = 0.f;
            if (t0 + 8 < s0) r2 = 0.f;
            if (t0 + 8 < s0 + 1) r3 = 0.f;
            sts32(c.sb + F_AQ + o128(t0, ns) + 4 * cq, pk(r0, r1));
            sts32(c.sb + F_AQ + o128(t0 + 8, ns) + 4 * cq, pk(r2, r3));
        }
    }
    // ---- P3: u^T[16 wi .., t] = (A bf16(v beta))^T for this warp's columns
    {
        float acc[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;
#pragma unroll
        for (int a = 0; a < 4; ++a) {
            uint32_t r[4];
            ldsm4t(c.sb + F_V + o256(16 * a + 8 * (lm >> 1) + lr, 2 * wi + (lm & 1)), r[0], r[1], r[2], r[3]);
            const uint32_t blo = f_beta2(c, bs, 16 * a + 2 * cq), bhi = f_beta2(c, bs, 16 * a + 8 + 2 * cq);
            const uint32_t a0 = bmul2(r[0], blo), a1 = bmul2(r[1], blo), a2 = bmul2(r[2], bhi), a3 = bmul2(r[3], bhi);
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                if (a > jj) continue;   // A[t][s] = +0 for s // 16 > t // 16
                uint32_t b0, b1, b2, b3;
                ldsm4(c.sb + F_A + o128(16 * jj + ab_t, 2 * a + ab_c), b0, b1, b2, b3);
                mma(acc[2 * jj], a0, a1, a2, a3, b0, b1);
                mma(acc[2 * jj + 1], a0, a1, a2, a3, b2, b3);
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t p0 = pk(acc[j][0], acc[j][1]), p1 = pk(acc[j][2], acc[j][3]);
            uf[j][0] = blo_f(p0);
            uf[j][1] = bhi_f(p0);
            uf[j][2] = blo_f(p1);
            uf[j][3] = bhi_f(p1);
        }
    }
}

DEVI FCtx f_ctx(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* beta,
                const __nv_bfloat16* A, const float* g, int n, int h, int T, uint8_t* smem, int hq = H) {
    FCtx c;
    c.q = q; c.k = k; c.v = v; c.beta = beta; c.A = A; c.g = g;
    c.n = n; c.h = h; c.hq = hq; c.hk = h / (HV / hq); c.T = T;
    c.tid = threadIdx.x; c.warp = c.tid >> 5; c.lane = c.tid & 31;
    c.gq = c.lane >> 2; c.cq = c.lane & 3; c.lm = c.lane >> 3; c.lr = c.lane & 7;
    c.sb = smem_u32(smem);
    c.row0 = static_cast<size_t>(n) * T;
    return c;
}

DEVI void f_exps(const float* gs, int cq, float (&e)[8][2]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        e[j][0] = ex2(gs[8 * j + 2 * cq]);
        e[j][1] = ex2(gs[8 * j + 2 * cq + 1]);
    }
}

}  // namespace

// debug / standalone pass P: grid (NT, B * HV), 256 threads; writes u, w, Aq in Triton's layout
extern "C" __global__ void __launch_bounds__(256, 1)
    ffs_pre(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* beta,
            const __nv_bfloat16* A, const float* g, __nv_bfloat16* w, __nv_bfloat16* u, __nv_bfloat16* Aq,
            const int* slen, int T) {
    extern __shared__ __align__(128) uint8_t smem[];
    const int i = blockIdx.x, n = blockIdx.y / HV, h = blockIdx.y % HV;
    if (slen[n] == 0) return;
    FCtx c = f_ctx(q, k, v, beta, A, g, n, h, T, smem);
    for (int x = c.tid; x < 512; x += 256) sts128z(c.sb + F_AQ + x * 16);
    f_issue_x(c, i, c.sb + F_K0);
    f_issue_q(c, i);
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    const float* gs = reinterpret_cast<const float*>(smem + F_G + (i & 1) * 256);
    const uint32_t* bs = reinterpret_cast<const uint32_t*>(smem + F_B + (i & 1) * 256);
    float e[8][2], uf[8][4];
    f_exps(gs, c.cq, e);
    f_pstage<true>(c, c.sb + F_K0, gs, bs, e, uf);
    __syncthreads();
    const size_t r0 = c.row0 + static_cast<size_t>(i) * BT;
    for (int x = c.tid; x < 1024; x += 256) {
        const int rr = x >> 4, ch = x & 15;
        uint4 val;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "r"(c.sb + F_W + o256(rr, ch)) : "memory");
        *reinterpret_cast<uint4*>(w + ((r0 + rr) * HV + h) * KD + ch * 8) = val;
    }
    for (int x = c.tid; x < 512; x += 256) {
        const int rr = x >> 3, ch = x & 7;
        uint4 val;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "r"(c.sb + F_AQ + o128(rr, ch)) : "memory");
        *reinterpret_cast<uint4*>(Aq + ((r0 + rr) * HV + h) * BT + ch * 8) = val;
    }
    const int v0 = 16 * c.warp;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t m0 = movt(pk(uf[j][0], uf[j][1])), m1 = movt(pk(uf[j][2], uf[j][3]));
        const size_t row = r0 + 8 * j + c.gq;
        *reinterpret_cast<uint32_t*>(u + (row * HV + h) * VD + v0 + 2 * c.cq) = m0;
        *reinterpret_cast<uint32_t*>(u + (row * HV + h) * VD + v0 + 8 + 2 * c.cq) = m1;
    }
}

namespace {
// fused P + S: grid (B * HV), 256 threads, F_SMEM dynamic smem
template <bool SKIP, bool TIMED>
DEVI void fused_body(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                     const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ beta,
                     const __nv_bfloat16* __restrict__ A, const float* __restrict__ g, __nv_bfloat16* __restrict__ o,
                     const int* __restrict__ slen, int T, float scale, unsigned long long* ts = nullptr) {
    int tsn = 0;
    auto stamp = [&]() {
        if (TIMED && (threadIdx.x & 31) == 0) {
            unsigned long long t;
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
            ts[(static_cast<size_t>(blockIdx.x) * 8 + (threadIdx.x >> 5)) * 48 + tsn] = t;
        }
        ++tsn;
    };
    stamp();
    extern __shared__ __align__(128) uint8_t smem[];
    const int n = blockIdx.x / HV, h = blockIdx.x % HV;
    if (slen[n] == 0) return;
    const FCtx c = f_ctx(q, k, v, beta, A, g, n, h, T, smem);
    const int NT = T / BT;
    const int lane = c.lane, lm = c.lm, lr = c.lr, cq = c.cq, gq = c.gq;
    const int v0 = 16 * c.warp;
    const int ab_t = 8 * (lm >> 1) + lr, ab_c = lm & 1;
    const int c_t = 8 * (lm & 1) + lr, c_c = lm >> 1;
    const uint32_t sb = c.sb;

    for (int x = c.tid; x < 512; x += 256) sts128z(sb + F_AQ + x * 16);   // Aq tiles above the diagonal stay +0
    f_issue_x(c, 0, sb + F_K0);
    f_issue_q(c, 0);
    cp_commit();

    float hacc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) hacc[j][r] = 0.f;

    cp_wait<0>();
    __syncthreads();
    stamp();
    for (int i = 0; i < NT; ++i) {
        const uint32_t kbuf = sb + F_K0 + (i & 1) * 16384;
        const float* gs = reinterpret_cast<const float*>(smem + F_G + (i & 1) * 256);
        const uint32_t* bs = reinterpret_cast<const uint32_t*>(smem + F_B + (i & 1) * 256);
        float e[8][2], vn[8][4];
        f_exps(gs, cq, e);
        // ------------------------------------------------ P(i)
        if (i > 0) f_pstage<true>(c, kbuf, gs, bs, e, vn);
        else f_pstage<false>(c, kbuf, gs, bs, e, vn);
        stamp();
        __syncthreads();   // b1: w, Aq of chunk i in smem; A, v of chunk i no longer read
        if (i + 1 < NT) f_issue_x(c, i + 1, sb + F_K0 + ((i + 1) & 1) * 16384);
        cp_commit();
        // ------------------------------------------------ A(i): vn = u - w h, oa = q h
        float oa[8][4];
        if (i > 0) {
            float acc[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int r = 0; r < 4; ++r) acc[j][r] = 0.f, oa[j][r] = 0.f;
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const uint32_t a0 = pk(hacc[2 * s][0], hacc[2 * s][1]);
                const uint32_t a1 = pk(hacc[2 * s][2], hacc[2 * s][3]);
                const uint32_t a2 = pk(hacc[2 * s + 1][0], hacc[2 * s + 1][1]);
                const uint32_t a3 = pk(hacc[2 * s + 1][2], hacc[2 * s + 1][3]);
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    uint32_t b0, b1, b2, b3;
                    ldsm4(sb + F_W + o256(16 * jj + ab_t, 2 * s + ab_c), b0, b1, b2, b3);
                    mma(acc[2 * jj], a0, a1, a2, a3, b0, b1);
                    mma(acc[2 * jj + 1], a0, a1, a2, a3, b2, b3);
                    ldsm4(sb + F_Q + o256(16 * jj + ab_t, 2 * s + ab_c), b0, b1, b2, b3);
                    mma(oa[2 * jj], a0, a1, a2, a3, b0, b1);
                    mma(oa[2 * jj + 1], a0, a1, a2, a3, b2, b3);
                }
            }
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int r = 0; r < 4; ++r) vn[j][r] = fsub(vn[j][r], acc[j][r]);
        }
        stamp();
        __syncthreads();   // b2: every warp is done with q of chunk i
        if (i + 1 < NT) f_issue_q(c, i + 1);
        cp_commit();
        // ------------------------------------------------ B(i): o (c = scale * (q h) 2^g, +0 based for chunk 0)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (i > 0) {
                oa[j][0] = fmul(scale, fmul(oa[j][0], e[j][0]));
                oa[j][1] = fmul(scale, fmul(oa[j][1], e[j][1]));
                oa[j][2] = fmul(scale, fmul(oa[j][2], e[j][0]));
                oa[j][3] = fmul(scale, fmul(oa[j][3], e[j][1]));
            } else {
                oa[j][0] = oa[j][2] = fmul(scale, fmul(e[j][0], 0.f));
                oa[j][1] = oa[j][3] = fmul(scale, fmul(e[j][1], 0.f));
            }
        }
        {
            uint32_t vs[4][4];
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                vs[a][0] = pk(vn[2 * a][0], vn[2 * a][1]);
                vs[a][1] = pk(vn[2 * a][2], vn[2 * a][3]);
                vs[a][2] = pk(vn[2 * a + 1][0], vn[2 * a + 1][1]);
                vs[a][3] = pk(vn[2 * a + 1][2], vn[2 * a + 1][3]);
            }
            const size_t r0 = c.row0 + static_cast<size_t>(i) * BT;
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                float d2[2][4];
#pragma unroll
                for (int x = 0; x < 2; ++x)
#pragma unroll
                    for (int r = 0; r < 4; ++r) d2[x][r] = 0.f;
#pragma unroll
                for (int a = 0; a < 4; ++a) {
                    if (SKIP && a > jj) continue;
                    uint32_t b0, b1, b2, b3;
                    ldsm4(sb + F_AQ + o128(16 * jj + ab_t, 2 * a + ab_c), b0, b1, b2, b3);
                    mma(d2[0], vs[a][0], vs[a][1], vs[a][2], vs[a][3], b0, b1);
                    mma(d2[1], vs[a][0], vs[a][1], vs[a][2], vs[a][3], b2, b3);
                }
#pragma unroll
                for (int x = 0; x < 2; ++x) {
                    const int j = 2 * jj + x;
                    const uint32_t m0 = movt(pk(ffma(scale, d2[x][0], oa[j][0]), ffma(scale, d2[x][1], oa[j][1])));
                    const uint32_t m1 = movt(pk(ffma(scale, d2[x][2], oa[j][2]), ffma(scale, d2[x][3], oa[j][3])));
                    const size_t row = r0 + 8 * j + gq;
                    *reinterpret_cast<uint32_t*>(o + (row * HV + h) * VD + v0 + 2 * cq) = m0;
                    *reinterpret_cast<uint32_t*>(o + (row * HV + h) * VD + v0 + 8 + 2 * cq) = m1;
                }
            }
        }
        stamp();
        // ------------------------------------------------ C(i): h update
        if (i + 1 < NT) {
            const float gl = gs[63];
            const float el = ex2(gl);
            if (i > 0) {
#pragma unroll
                for (int j = 0; j < 16; ++j)
#pragma unroll
                    for (int r = 0; r < 4; ++r) hacc[j][r] = fmul(hacc[j][r], el);
            } else {
                const float z = fmul(el, 0.f);
#pragma unroll
                for (int j = 0; j < 16; ++j)
#pragma unroll
                    for (int r = 0; r < 4; ++r) hacc[j][r] = z;
            }
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                const float f00 = ex2(fsub(gl, gs[16 * a + 2 * cq])), f01 = ex2(fsub(gl, gs[16 * a + 2 * cq + 1]));
                const float f10 = ex2(fsub(gl, gs[16 * a + 8 + 2 * cq])),
                            f11 = ex2(fsub(gl, gs[16 * a + 8 + 2 * cq + 1]));
                const uint32_t a0 = pk(fmul(vn[2 * a][0], f00), fmul(vn[2 * a][1], f01));
                const uint32_t a1 = pk(fmul(vn[2 * a][2], f00), fmul(vn[2 * a][3], f01));
                const uint32_t a2 = pk(fmul(vn[2 * a + 1][0], f10), fmul(vn[2 * a + 1][1], f11));
                const uint32_t a3 = pk(fmul(vn[2 * a + 1][2], f10), fmul(vn[2 * a + 1][3], f11));
#pragma unroll
                for (int jj = 0; jj < 8; ++jj) {
                    uint32_t b0, b1, b2, b3;
                    ldsm4t(kbuf + o256(16 * a + c_t, 2 * jj + c_c), b0, b1, b2, b3);
                    mma(hacc[2 * jj], a0, a1, a2, a3, b0, b1);
                    mma(hacc[2 * jj + 1], a0, a1, a2, a3, b2, b3);
                }
            }
        }
        stamp();
        cp_wait<0>();
        __syncthreads();   // b3: chunk i + 1 inputs landed; every warp is done with w, Aq, k of chunk i
        stamp();
    }
}

// ---------------------------------------------------------------- v2 pass P: A-tile fragments shared by w = A kb and
// u^T = (A vb)^T (the m16k16 A fragment of A[t-block, s-step] is, permuted, the two n8k16 B fragments of A^T), Aq
// tiles in pairs sharing q fragments; loads issued in slices
DEVI void f_issue_x_part(const FCtx& c, int i, uint32_t kbuf, int part) {
    const size_t r0 = c.row0 + static_cast<size_t>(i) * BT;
    if (part == 0) {
#pragma unroll
        for (int x = c.tid; x < 512; x += 256) {
            const int rr = x >> 3, ch = x & 7;
            cp16(c.sb + F_A + o128(rr, ch), c.A + ((r0 + rr) * HV + c.h) * BT + ch * 8);
        }
    } else if (part <= 2) {
#pragma unroll
        for (int jx = 0; jx < 2; ++jx) {
            const int x = c.tid + 256 * (2 * (part - 1) + jx);
            const int rr = x >> 4, ch = x & 15;
            cp16(c.sb + F_V + o256(rr, ch), c.v + ((r0 + rr) * HV + c.h) * VD + ch * 8);
        }
    } else if (part <= 4) {
#pragma unroll
        for (int jx = 0; jx < 2; ++jx) {
            const int x = c.tid + 256 * (2 * (part - 3) + jx);
            const int rr = x >> 4, ch = x & 15;
            cp16(kbuf + o256(rr, ch), c.k + ((r0 + rr) * c.hq + c.hk) * KD + ch * 8);
        }
    } else {
        if (c.tid < 64) {
            cp4(c.sb + F_G + (i & 1) * 256 + c.tid * 4, c.g + (r0 + c.tid) * HV + c.h);
        } else if (c.tid < 128) {
            const int t = c.tid - 64;
            cp4(c.sb + F_B + (i & 1) * 256 + t * 4, c.beta + (r0 + t) * HV + (c.h & ~1));
        }
    }
}
DEVI void f_issue_q_part(const FCtx& c, int i, int jx) {
    const size_t r0 = c.row0 + static_cast<size_t>(i) * BT;
    const int x = c.tid + 256 * jx;
    const int rr = x >> 4, ch = x & 15;
    cp16(c.sb + F_Q + o256(rr, ch), c.q + ((r0 + rr) * c.hq + c.hk) * KD + ch * 8);
}

DEVI void f_aq_tile_store(const FCtx& c, const float* gs, int mt, int ns, const float* acc) {
    const int t0 = 16 * mt + c.gq, s0 = 8 * ns + 2 * c.cq;
    const float gt0 = gs[t0], gt1 = gs[t0 + 8], gs0 = gs[s0], gs1 = gs[s0 + 1];
    float r0 = fmul(acc[0], ex2(fsub(gt0, gs0))), r1 = fmul(acc[1], ex2(fsub(gt0, gs1)));
    float r2 = fmul(acc[2], ex2(fsub(gt1, gs0))), r3 = fmul(acc[3], ex2(fsub(gt1, gs1)));
    if (t0 < s0) r0 = 0.f;
    if (t0 < s0 + 1) r1 = 0.f;
    if (t0 + 8 < s0) r2 = 0.f;
    if (t0 + 8 < s0 + 1) r3 = 0.f;
    sts32(c.sb + F_AQ + o128(t0, ns) + 4 * c.cq, pk(r0, r1));
    sts32(c.sb + F_AQ + o128(t0 + 8, ns) + 4 * c.cq, pk(r2, r3));
}

template <bool DO_W>
DEVI void f_pstage2(const FCtx& c, uint32_t kbuf, const float* gs, const uint32_t* bs, const float (&e)[8][2],
                    float (&uf)[8][4]) {
    const int lane = c.lane, lm = c.lm, lr = c.lr, cq = c.cq, wi = c.warp;
    const int c_t = 8 * (lm & 1) + lr, c_c = lm >> 1;
    const int ab_t = 8 * (lm >> 1) + lr, ab_c = lm & 1;
    uint32_t vb[4][4];
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        uint32_t r[4];
        ldsm4t(c.sb + F_V + o256(16 * a + 8 * (lm >> 1) + lr, 2 * wi + (lm & 1)), r[0], r[1], r[2], r[3]);
        const uint32_t blo = f_beta2(c, bs, 16 * a + 2 * cq), bhi = f_beta2(c, bs, 16 * a + 8 + 2 * cq);
        vb[a][0] = bmul2(r[0], blo);
        vb[a][1] = bmul2(r[1], blo);
        vb[a][2] = bmul2(r[2], bhi);
        vb[a][3] = bmul2(r[3], bhi);
    }
    uint32_t kb[4][4];
    if (DO_W) {
#pragma unroll
        for (int sb = 0; sb < 4; ++sb) {
            uint32_t r[4];
            ldsm4t(kbuf + o256(16 * sb + c_t, 2 * wi + c_c), r[0], r[1], r[2], r[3]);
            const uint32_t blo = f_beta2(c, bs, 16 * sb + 2 * cq), bhi = f_beta2(c, bs, 16 * sb + 8 + 2 * cq);
#pragma unroll
            for (int x = 0; x < 4; ++x) {
                const uint32_t p = bmul2(r[x], (x & 1) ? bhi : blo);
                const float* ee = e[2 * sb + (x & 1)];
                kb[sb][x] = pk(fmul(blo_f(p), ee[0]), fmul(bhi_f(p), ee[1]));
            }
        }
    }
    float uacc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) uacc[j][r] = 0.f;
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        float wacc[2][4];
#pragma unroll
        for (int nn = 0; nn < 2; ++nn)
#pragma unroll
            for (int r = 0; r < 4; ++r) wacc[nn][r] = 0.f;
#pragma unroll
        for (int sb = 0; sb <= mt; ++sb) {   // A[t][s] = +0 for s // 16 > t // 16 (trailing k16 steps skipped)
            uint32_t r0, r1, r2, r3;
            ldsm4(c.sb + F_A + o128(16 * mt + (lane & 15), 2 * sb + (lane >> 4)), r0, r1, r2, r3);
            if (DO_W) {
                mma(wacc[0], r0, r1, r2, r3, kb[sb][0], kb[sb][1]);
                mma(wacc[1], r0, r1, r2, r3, kb[sb][2], kb[sb][3]);
            }
            mma(uacc[2 * mt], vb[sb][0], vb[sb][1], vb[sb][2], vb[sb][3], r0, r2);
            mma(uacc[2 * mt + 1], vb[sb][0], vb[sb][1], vb[sb][2], vb[sb][3], r1, r3);
        }
        if (DO_W)
            stsm4(c.sb + F_W + o256(16 * mt + c_t, 2 * wi + c_c), pk(wacc[0][0], wacc[0][1]),
                  pk(wacc[0][2], wacc[0][3]), pk(wacc[1][0], wacc[1][1]), pk(wacc[1][2], wacc[1][3]));
    }
    // Aq: one tile pair (pmt, ns 2 pnp, 2 pnp + 1) per warp, + one single tile for warps 4..7
    {
        const int pmt = wi < 4 ? 3 : wi < 7 ? 2 : 1;
        const int pnp = wi < 4 ? wi : wi < 7 ? wi - 4 : 0;
        float acc[2][4];
#pragma unroll
        for (int nn = 0; nn < 2; ++nn)
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[nn][r] = 0.f;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
            ldsm4(c.sb + F_Q + o256(16 * pmt + (lane & 15), 2 * ks + (lane >> 4)), a0, a1, a2, a3);
            ldsm4(kbuf + o256(16 * pnp + ab_t, 2 * ks + ab_c), b0, b1, b2, b3);
            mma(acc[0], a0, a1, a2, a3, b0, b1);
            mma(acc[1], a0, a1, a2, a3, b2, b3);
        }
        f_aq_tile_store(c, gs, pmt, 2 * pnp, acc[0]);
        f_aq_tile_store(c, gs, pmt, 2 * pnp + 1, acc[1]);
        if (wi >= 4) {
            const int smt = wi < 6 ? 1 : 0;
            const int sns = wi == 4 ? 2 : wi == 5 ? 3 : wi == 6 ? 0 : 1;
            float a1c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int k2 = 0; k2 < 4; ++k2) {
                uint32_t b0, b1, b2, b3;
                ldsm4(kbuf + o256(8 * sns + lr, 4 * k2 + lm), b0, b1, b2, b3);
                uint32_t a0, a1, a2, a3;
                ldsm4(c.sb + F_Q + o256(16 * smt + (lane & 15), 4 * k2 + (lane >> 4)), a0, a1, a2, a3);
                mma(a1c, a0, a1, a2, a3, b0, b1);
                ldsm4(c.sb + F_Q + o256(16 * smt + (lane & 15), 4 * k2 + 2 + (lane >> 4)), a0, a1, a2, a3);
                mma(a1c, a0, a1, a2, a3, b2, b3);
            }
            f_aq_tile_store(c, gs, smt, sns, a1c);
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t p0 = pk(uacc[j][0], uacc[j][1]), p1 = pk(uacc[j][2], uacc[j][3]);
        uf[j][0] = blo_f(p0);
        uf[j][1] = bhi_f(p0);
        uf[j][2] = blo_f(p1);
        uf[j][3] = bhi_f(p1);
    }
}

template <bool SKIP, bool TIMED>
DEVI void fused2_body(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                     const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ beta,
                     const __nv_bfloat16* __restrict__ A, const float* __restrict__ g, __nv_bfloat16* __restrict__ o,
                     const int* __restrict__ slen, int T, float scale, unsigned long long* ts = nullptr) {
    int tsn = 0;
    auto stamp = [&]() {
        if (TIMED && (threadIdx.x & 31) == 0) {
            unsigned long long t;
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
            ts[(static_cast<size_t>(blockIdx.x) * 8 + (threadIdx.x >> 5)) * 48 + tsn] = t;
        }
        ++tsn;
    };
    stamp();
    extern __shared__ __align__(128) uint8_t smem[];
    const int n = blockIdx.x / HV, h = blockIdx.x % HV;
    if (slen[n] == 0) return;
    const FCtx c = f_ctx(q, k, v, beta, A, g, n, h, T, smem);
    const int NT = T / BT;
    const int lane = c.lane, lm = c.lm, lr = c.lr, cq = c.cq, gq = c.gq;
    const int v0 = 16 * c.warp;
    const int ab_t = 8 * (lm >> 1) + lr, ab_c = lm & 1;
    const int c_t = 8 * (lm & 1) + lr, c_c = lm >> 1;
    const uint32_t sb = c.sb;

    for (int x = c.tid; x < 512; x += 256) sts128z(sb + F_AQ + x * 16);   // Aq tiles above the diagonal stay +0
    f_issue_x(c, 0, sb + F_K0);
    f_issue_q(c, 0);
    cp_commit();

    float hacc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) hacc[j][r] = 0.f;

    cp_wait<0>();
    __syncthreads();
    stamp();
    for (int i = 0; i < NT; ++i) {
        const uint32_t kbuf = sb + F_K0 + (i & 1) * 16384;
        const float* gs = reinterpret_cast<const float*>(smem + F_G + (i & 1) * 256);
        const uint32_t* bs = reinterpret_cast<const uint32_t*>(smem + F_B + (i & 1) * 256);
        float e[8][2], vn[8][4];
        f_exps(gs, cq, e);
        // ------------------------------------------------ P(i)
        if (i > 0) f_pstage2<true>(c, kbuf, gs, bs, e, vn);
        else f_pstage2<false>(c, kbuf, gs, bs, e, vn);
        stamp();
        __syncthreads();   // b1: w, Aq of chunk i in smem; A, v of chunk i no longer read
        const uint32_t kbn = sb + F_K0 + ((i + 1) & 1) * 16384;
        if (i == 0 && NT > 1) f_issue_x(c, 1, kbn);
        // ------------------------------------------------ A(i): vn = u - w h, oa = q h
        float oa[8][4];
        if (i > 0) {
            float acc[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int r = 0; r < 4; ++r) acc[j][r] = 0.f, oa[j][r] = 0.f;
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const uint32_t a0 = pk(hacc[2 * s][0], hacc[2 * s][1]);
                const uint32_t a1 = pk(hacc[2 * s][2], hacc[2 * s][3]);
                const uint32_t a2 = pk(hacc[2 * s + 1][0], hacc[2 * s + 1][1]);
                const uint32_t a3 = pk(hacc[2 * s + 1][2], hacc[2 * s + 1][3]);
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    uint32_t b0, b1, b2, b3;
                    ldsm4(sb + F_W + o256(16 * jj + ab_t, 2 * s + ab_c), b0, b1, b2, b3);
                    mma(acc[2 * jj], a0, a1, a2, a3, b0, b1);
                    mma(acc[2 * jj + 1], a0, a1, a2, a3, b2, b3);
                    ldsm4(sb + F_Q + o256(16 * jj + ab_t, 2 * s + ab_c), b0, b1, b2, b3);
                    mma(oa[2 * jj], a0, a1, a2, a3, b0, b1);
                    mma(oa[2 * jj + 1], a0, a1, a2, a3, b2, b3);
                }
                if (s < 6 && i + 1 < NT) f_issue_x_part(c, i + 1, kbn, s);   // chunk i + 1 inputs, in slices
            }
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int r = 0; r < 4; ++r) vn[j][r] = fsub(vn[j][r], acc[j][r]);
        }
        stamp();
        cp_commit();
        __syncthreads();   // b2: every warp is done with q of chunk i
        // ------------------------------------------------ B(i): o (c = scale * (q h) 2^g, +0 based for chunk 0)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (i > 0) {
                oa[j][0] = fmul(scale, fmul(oa[j][0], e[j][0]));
                oa[j][1] = fmul(scale, fmul(oa[j][1], e[j][1]));
                oa[j][2] = fmul(scale, fmul(oa[j][2], e[j][0]));
                oa[j][3] = fmul(scale, fmul(oa[j][3], e[j][1]));
            } else {
                oa[j][0] = oa[j][2] = fmul(scale, fmul(e[j][0], 0.f));
                oa[j][1] = oa[j][3] = fmul(scale, fmul(e[j][1], 0.f));
            }
        }
        {
            uint32_t vs[4][4];
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                vs[a][0] = pk(vn[2 * a][0], vn[2 * a][1]);
                vs[a][1] = pk(vn[2 * a][2], vn[2 * a][3]);
                vs[a][2] = pk(vn[2 * a + 1][0], vn[2 * a + 1][1]);
                vs[a][3] = pk(vn[2 * a + 1][2], vn[2 * a + 1][3]);
            }
            const size_t r0 = c.row0 + static_cast<size_t>(i) * BT;
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                float d2[2][4];
#pragma unroll
                for (int x = 0; x < 2; ++x)
#pragma unroll
                    for (int r = 0; r < 4; ++r) d2[x][r] = 0.f;
#pragma unroll
                for (int a = 0; a < 4; ++a) {
                    if (SKIP && a > jj) continue;
                    uint32_t b0, b1, b2, b3;
                    ldsm4(sb + F_AQ + o128(16 * jj + ab_t, 2 * a + ab_c), b0, b1, b2, b3);
                    mma(d2[0], vs[a][0], vs[a][1], vs[a][2], vs[a][3], b0, b1);
                    mma(d2[1], vs[a][0], vs[a][1], vs[a][2], vs[a][3], b2, b3);
                }
#pragma unroll
                for (int x = 0; x < 2; ++x) {
                    const int j = 2 * jj + x;
                    const uint32_t m0 = movt(pk(ffma(scale, d2[x][0], oa[j][0]), ffma(scale, d2[x][1], oa[j][1])));
                    const uint32_t m1 = movt(pk(ffma(scale, d2[x][2], oa[j][2]), ffma(scale, d2[x][3], oa[j][3])));
                    const size_t row = r0 + 8 * j + gq;
                    *reinterpret_cast<uint32_t*>(o + (row * HV + h) * VD + v0 + 2 * cq) = m0;
                    *reinterpret_cast<uint32_t*>(o + (row * HV + h) * VD + v0 + 8 + 2 * cq) = m1;
                }
            }
        }
        stamp();
        // ------------------------------------------------ C(i): h update
        if (i + 1 < NT) {
            const float gl = gs[63];
            const float el = ex2(gl);
            if (i > 0) {
#pragma unroll
                for (int j = 0; j < 16; ++j)
#pragma unroll
                    for (int r = 0; r < 4; ++r) hacc[j][r] = fmul(hacc[j][r], el);
            } else {
                const float z = fmul(el, 0.f);
#pragma unroll
                for (int j = 0; j < 16; ++j)
#pragma unroll
                    for (int r = 0; r < 4; ++r) hacc[j][r] = z;
            }
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                const float f00 = ex2(fsub(gl, gs[16 * a + 2 * cq])), f01 = ex2(fsub(gl, gs[16 * a + 2 * cq + 1]));
                const float f10 = ex2(fsub(gl, gs[16 * a + 8 + 2 * cq])),
                            f11 = ex2(fsub(gl, gs[16 * a + 8 + 2 * cq + 1]));
                const uint32_t a0 = pk(fmul(vn[2 * a][0], f00), fmul(vn[2 * a][1], f01));
                const uint32_t a1 = pk(fmul(vn[2 * a][2], f00), fmul(vn[2 * a][3], f01));
                const uint32_t a2 = pk(fmul(vn[2 * a + 1][0], f10), fmul(vn[2 * a + 1][1], f11));
                const uint32_t a3 = pk(fmul(vn[2 * a + 1][2], f10), fmul(vn[2 * a + 1][3], f11));
#pragma unroll
                for (int jj = 0; jj < 8; ++jj) {
                    uint32_t b0, b1, b2, b3;
                    ldsm4t(kbuf + o256(16 * a + c_t, 2 * jj + c_c), b0, b1, b2, b3);
                    mma(hacc[2 * jj], a0, a1, a2, a3, b0, b1);
                    mma(hacc[2 * jj + 1], a0, a1, a2, a3, b2, b3);
                }
                f_issue_q_part(c, i + 1, a);   // q of chunk i + 1 (q buffer free since b2)
            }
        }
        cp_commit();
        stamp();
        cp_wait<0>();
        __syncthreads();   // b3: chunk i + 1 inputs landed; every warp is done with w, Aq, k of chunk i
        stamp();
    }
}
template <bool DO_W, bool FIRST>
DEVI void f_pstage3(const FCtx& c, uint32_t kbuf, const float* gs, const uint32_t* bs, const float (&e)[8][2],
                    float (&uf)[8][4]) {
    const int lane = c.lane, lm = c.lm, lr = c.lr, cq = c.cq, wi = c.warp;
    const int c_t = 8 * (lm & 1) + lr, c_c = lm >> 1;
    const int ab_t = 8 * (lm >> 1) + lr, ab_c = lm & 1;
    uint32_t vb[4][4];
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        uint32_t r[4];
        ldsm4t(c.sb + F_V + o256(16 * a + 8 * (lm >> 1) + lr, 2 * wi + (lm & 1)), r[0], r[1], r[2], r[3]);
        const uint32_t blo = f_beta2(c, bs, 16 * a + 2 * cq), bhi = f_beta2(c, bs, 16 * a + 8 + 2 * cq);
        vb[a][0] = bmul2(r[0], blo);
        vb[a][1] = bmul2(r[1], blo);
        vb[a][2] = bmul2(r[2], bhi);
        vb[a][3] = bmul2(r[3], bhi);
    }
    uint32_t kb[4][4];
    if (DO_W) {
#pragma unroll
        for (int sb = 0; sb < 4; ++sb) {
            uint32_t r[4];
            ldsm4t(kbuf + o256(16 * sb + c_t, 2 * wi + c_c), r[0], r[1], r[2], r[3]);
            const uint32_t blo = f_beta2(c, bs, 16 * sb + 2 * cq), bhi = f_beta2(c, bs, 16 * sb + 8 + 2 * cq);
#pragma unroll
            for (int x = 0; x < 4; ++x) {
                const uint32_t p = bmul2(r[x], (x & 1) ? bhi : blo);
                const float* ee = e[2 * sb + (x & 1)];
                kb[sb][x] = pk(fmul(blo_f(p), ee[0]), fmul(bhi_f(p), ee[1]));
            }
        }
    }
    float uacc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) uacc[j][r] = 0.f;
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        float wacc[2][4];
#pragma unroll
        for (int nn = 0; nn < 2; ++nn)
#pragma unroll
            for (int r = 0; r < 4; ++r) wacc[nn][r] = 0.f;
#pragma unroll
        for (int sb = 0; sb <= mt; ++sb) {   // A[t][s] = +0 for s // 16 > t // 16 (trailing k16 steps skipped)
            uint32_t r0, r1, r2, r3;
            ldsm4(c.sb + F_A + o128(16 * mt + (lane & 15), 2 * sb + (lane >> 4)), r0, r1, r2, r3);
            if (DO_W) {
                mma(wacc[0], r0, r1, r2, r3, kb[sb][0], kb[sb][1]);
                mma(wacc[1], r0, r1, r2, r3, kb[sb][2], kb[sb][3]);
            }
            mma(uacc[2 * mt], vb[sb][0], vb[sb][1], vb[sb][2], vb[sb][3], r0, r2);
            mma(uacc[2 * mt + 1], vb[sb][0], vb[sb][1], vb[sb][2], vb[sb][3], r1, r3);
        }
        if (DO_W)
            stsm4(c.sb + F_W + o256(16 * mt + c_t, 2 * wi + c_c), pk(wacc[0][0], wacc[0][1]),
                  pk(wacc[0][2], wacc[0][3]), pk(wacc[1][0], wacc[1][1]), pk(wacc[1][2], wacc[1][3]));
    }
    if (FIRST) {   // chunk 0: k and q (second load group) are needed from here on
        cp_wait<0>();
        __syncthreads();
    }
    // Aq: one tile pair (pmt, ns 2 pnp, 2 pnp + 1) per warp, + one single tile for warps 4..7
    {
        const int pmt = wi < 4 ? 3 : wi < 7 ? 2 : 1;
        const int pnp = wi < 4 ? wi : wi < 7 ? wi - 4 : 0;
        float acc[2][4];
#pragma unroll
        for (int nn = 0; nn < 2; ++nn)
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[nn][r] = 0.f;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
            ldsm4(c.sb + F_Q + o256(16 * pmt + (lane & 15), 2 * ks + (lane >> 4)), a0, a1, a2, a3);
            ldsm4(kbuf + o256(16 * pnp + ab_t, 2 * ks + ab_c), b0, b1, b2, b3);
            mma(acc[0], a0, a1, a2, a3, b0, b1);
            mma(acc[1], a0, a1, a2, a3, b2, b3);
        }
        f_aq_tile_store(c, gs, pmt, 2 * pnp, acc[0]);
        f_aq_tile_store(c, gs, pmt, 2 * pnp + 1, acc[1]);
        if (wi >= 4) {
            const int smt = wi < 6 ? 1 : 0;
            const int sns = wi == 4 ? 2 : wi == 5 ? 3 : wi == 6 ? 0 : 1;
            float a1c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int k2 = 0; k2 < 4; ++k2) {
                uint32_t b0, b1, b2, b3;
                ldsm4(kbuf + o256(8 * sns + lr, 4 * k2 + lm), b0, b1, b2, b3);
                uint32_t a0, a1, a2, a3;
                ldsm4(c.sb + F_Q + o256(16 * smt + (lane & 15), 4 * k2 + (lane >> 4)), a0, a1, a2, a3);
                mma(a1c, a0, a1, a2, a3, b0, b1);
                ldsm4(c.sb + F_Q + o256(16 * smt + (lane & 15), 4 * k2 + 2 + (lane >> 4)), a0, a1, a2, a3);
                mma(a1c, a0, a1, a2, a3, b2, b3);
            }
            f_aq_tile_store(c, gs, smt, sns, a1c);
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t p0 = pk(uacc[j][0], uacc[j][1]), p1 = pk(uacc[j][2], uacc[j][3]);
        uf[j][0] = blo_f(p0);
        uf[j][1] = bhi_f(p0);
        uf[j][2] = blo_f(p1);
        uf[j][3] = bhi_f(p1);
    }
}

template <bool SKIP, bool TIMED>
DEVI void fused3_body(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                     const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ beta,
                     const __nv_bfloat16* __restrict__ A, const float* __restrict__ g, __nv_bfloat16* __restrict__ o,
                     const int* __restrict__ slen, int T, float scale, int hq = H, unsigned long long* ts = nullptr) {
    int tsn = 0;
    auto stamp = [&]() {
        if (TIMED && (threadIdx.x & 31) == 0) {
            unsigned long long t;
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
            ts[(static_cast<size_t>(blockIdx.x) * 8 + (threadIdx.x >> 5)) * 48 + tsn] = t;
        }
        ++tsn;
    };
    stamp();
    extern __shared__ __align__(128) uint8_t smem[];
    const int n = blockIdx.x / HV, h = blockIdx.x % HV;
    if (slen[n] == 0) return;
    const FCtx c = f_ctx(q, k, v, beta, A, g, n, h, T, smem, hq);
    const int NT = T / BT;
    const int lane = c.lane, lm = c.lm, lr = c.lr, cq = c.cq, gq = c.gq;
    const int v0 = 16 * c.warp;
    const int ab_t = 8 * (lm >> 1) + lr, ab_c = lm & 1;
    const int c_t = 8 * (lm & 1) + lr, c_c = lm >> 1;
    const uint32_t sb = c.sb;

    for (int x = c.tid; x < 512; x += 256) sts128z(sb + F_AQ + x * 16);   // Aq tiles above the diagonal stay +0
    f_issue_x_part(c, 0, sb + F_K0, 0);   // A, v, g, beta first: pass P's u product needs only these
    f_issue_x_part(c, 0, sb + F_K0, 1);
    f_issue_x_part(c, 0, sb + F_K0, 2);
    f_issue_x_part(c, 0, sb + F_K0, 5);
    cp_commit();
    f_issue_x_part(c, 0, sb + F_K0, 3);
    f_issue_x_part(c, 0, sb + F_K0, 4);
    f_issue_q(c, 0);
    cp_commit();

    float hacc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) hacc[j][r] = 0.f;

    cp_wait<1>();
    __syncthreads();
    stamp();
    for (int i = 0; i < NT; ++i) {
        const uint32_t kbuf = sb + F_K0 + (i & 1) * 16384;
        const float* gs = reinterpret_cast<const float*>(smem + F_G + (i & 1) * 256);
        const uint32_t* bs = reinterpret_cast<const uint32_t*>(smem + F_B + (i & 1) * 256);
        float e[8][2], vn[8][4];
        f_exps(gs, cq, e);
        // ------------------------------------------------ P(i)
        if (i > 0) f_pstage3<true, false>(c, kbuf, gs, bs, e, vn);
        else f_pstage3<false, true>(c, kbuf, gs, bs, e, vn);
        stamp();
        __syncthreads();   // b1: w, Aq of chunk i in smem; A, v of chunk i no longer read
        const uint32_t kbn = sb + F_K0 + ((i + 1) & 1) * 16384;
        if (i == 0 && NT > 1) f_issue_x(c, 1, kbn);
        // ------------------------------------------------ A(i): vn = u - w h, oa = q h
        float oa[8][4];
        if (i > 0) {
            float acc[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int r = 0; r < 4; ++r) acc[j][r] = 0.f, oa[j][r] = 0.f;
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const uint32_t a0 = pk(hacc[2 * s][0], hacc[2 * s][1]);
                const uint32_t a1 = pk(hacc[2 * s][2], hacc[2 * s][3]);
                const uint32_t a2 = pk(hacc[2 * s + 1][0], hacc[2 * s + 1][1]);
                const uint32_t a3 = pk(hacc[2 * s + 1][2], hacc[2 * s + 1][3]);
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    uint32_t b0, b1, b2, b3;
                    ldsm4(sb + F_W + o256(16 * jj + ab_t, 2 * s + ab_c), b0, b1, b2, b3);
                    mma(acc[2 * jj], a0, a1, a2, a3, b0, b1);
                    mma(acc[2 * jj + 1], a0, a1, a2, a3, b2, b3);
                    ldsm4(sb + F_Q + o256(16 * jj + ab_t, 2 * s + ab_c), b0, b1, b2, b3);
                    mma(oa[2 * jj], a0, a1, a2, a3, b0, b1);
                    mma(oa[2 * jj + 1], a0, a1, a2, a3, b2, b3);
                }
                if (s < 6 && i + 1 < NT) f_issue_x_part(c, i + 1, kbn, s);   // chunk i + 1 inputs, in slices
            }
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int r = 0; r < 4; ++r) vn[j][r] = fsub(vn[j][r], acc[j][r]);
        }
        stamp();
        cp_commit();
        __syncthreads();   // b2: every warp is done with q of chunk i
        // ------------------------------------------------ B(i): o (c = scale * (q h) 2^g, +0 based for chunk 0)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (i > 0) {
                oa[j][0] = fmul(scale, fmul(oa[j][0], e[j][0]));
                oa[j][1] = fmul(scale, fmul(oa[j][1], e[j][1]));
                oa[j][2] = fmul(scale, fmul(oa[j][2], e[j][0]));
                oa[j][3] = fmul(scale, fmul(oa[j][3], e[j][1]));
            } else {
                oa[j][0] = oa[j][2] = fmul(scale, fmul(e[j][0], 0.f));
                oa[j][1] = oa[j][3] = fmul(scale, fmul(e[j][1], 0.f));
            }
        }
        {
            uint32_t vs[4][4];
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                vs[a][0] = pk(vn[2 * a][0], vn[2 * a][1]);
                vs[a][1] = pk(vn[2 * a][2], vn[2 * a][3]);
                vs[a][2] = pk(vn[2 * a + 1][0], vn[2 * a + 1][1]);
                vs[a][3] = pk(vn[2 * a + 1][2], vn[2 * a + 1][3]);
            }
            const size_t r0 = c.row0 + static_cast<size_t>(i) * BT;
            // B(i) for the t-tile pair jj: o rows 16 jj .. 16 jj + 15
            auto b_pair = [&](int jj) {
                float d2[2][4];
#pragma unroll
                for (int x = 0; x < 2; ++x)
#pragma unroll
                    for (int r = 0; r < 4; ++r) d2[x][r] = 0.f;
#pragma unroll
                for (int a = 0; a < 4; ++a) {
                    if (SKIP && a > jj) continue;
                    uint32_t b0, b1, b2, b3;
                    ldsm4(sb + F_AQ + o128(16 * jj + ab_t, 2 * a + ab_c), b0, b1, b2, b3);
                    mma(d2[0], vs[a][0], vs[a][1], vs[a][2], vs[a][3], b0, b1);
                    mma(d2[1], vs[a][0], vs[a][1], vs[a][2], vs[a][3], b2, b3);
                }
#pragma unroll
                for (int x = 0; x < 2; ++x) {
                    const int j = 2 * jj + x;
                    const uint32_t m0 = movt(pk(ffma(scale, d2[x][0], oa[j][0]), ffma(scale, d2[x][1], oa[j][1])));
                    const uint32_t m1 = movt(pk(ffma(scale, d2[x][2], oa[j][2]), ffma(scale, d2[x][3], oa[j][3])));
                    const size_t row = r0 + 8 * j + gq;
                    *reinterpret_cast<uint32_t*>(o + (row * HV + h) * VD + v0 + 2 * cq) = m0;
                    *reinterpret_cast<uint32_t*>(o + (row * HV + h) * VD + v0 + 8 + 2 * cq) = m1;
                }
            };
            if (i + 1 < NT) {
                // ------------------------------------ C(i) (h update), with B(i)'s tile pairs interleaved
                const float gl = gs[63];
                const float el = ex2(gl);
                if (i > 0) {
#pragma unroll
                    for (int j = 0; j < 16; ++j)
#pragma unroll
                        for (int r = 0; r < 4; ++r) hacc[j][r] = fmul(hacc[j][r], el);
                } else {
                    const float z = fmul(el, 0.f);
#pragma unroll
                    for (int j = 0; j < 16; ++j)
#pragma unroll
                        for (int r = 0; r < 4; ++r) hacc[j][r] = z;
                }
#pragma unroll
                for (int a = 0; a < 4; ++a) {
                    const float f00 = ex2(fsub(gl, gs[16 * a + 2 * cq])), f01 = ex2(fsub(gl, gs[16 * a + 2 * cq + 1]));
                    const float f10 = ex2(fsub(gl, gs[16 * a + 8 + 2 * cq])),
                                f11 = ex2(fsub(gl, gs[16 * a + 8 + 2 * cq + 1]));
                    const uint32_t a0 = pk(fmul(vn[2 * a][0], f00), fmul(vn[2 * a][1], f01));
                    const uint32_t a1 = pk(fmul(vn[2 * a][2], f00), fmul(vn[2 * a][3], f01));
                    const uint32_t a2 = pk(fmul(vn[2 * a + 1][0], f10), fmul(vn[2 * a + 1][1], f11));
                    const uint32_t a3 = pk(fmul(vn[2 * a + 1][2], f10), fmul(vn[2 * a + 1][3], f11));
#pragma unroll
                    for (int jj = 0; jj < 8; ++jj) {
                        uint32_t b0, b1, b2, b3;
                        ldsm4t(kbuf + o256(16 * a + c_t, 2 * jj + c_c), b0, b1, b2, b3);
                        mma(hacc[2 * jj], a0, a1, a2, a3, b0, b1);
                        mma(hacc[2 * jj + 1], a0, a1, a2, a3, b2, b3);
                    }
                    b_pair(a);
                    f_issue_q_part(c, i + 1, a);   // q of chunk i + 1 (q buffer free since b2)
                }
            } else {
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) b_pair(jj);
            }
        }
        stamp();
        cp_commit();
        stamp();
        cp_wait<0>();
        __syncthreads();   // b3: chunk i + 1 inputs landed; every warp is done with w, Aq, k of chunk i
        stamp();
    }
}

template <bool SKIP, bool TIMED>
DEVI void fused5_body(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                     const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ beta,
                     const __nv_bfloat16* __restrict__ A, const float* __restrict__ g, __nv_bfloat16* __restrict__ o,
                     const int* __restrict__ slen, int T, float scale, int hq = H, unsigned long long* ts = nullptr) {
    int tsn = 0;
    auto stamp = [&]() {
        if (TIMED && (threadIdx.x & 31) == 0) {
            unsigned long long t;
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
            ts[(static_cast<size_t>(blockIdx.x) * 8 + (threadIdx.x >> 5)) * 48 + tsn] = t;
        }
        ++tsn;
    };
    stamp();
    extern __shared__ __align__(128) uint8_t smem[];
    const int n = blockIdx.x / HV, h = blockIdx.x % HV;
    if (slen[n] == 0) return;
    const FCtx c = f_ctx(q, k, v, beta, A, g, n, h, T, smem, hq);
    const int NT = T / BT;
    const int lane = c.lane, lm = c.lm, lr = c.lr, cq = c.cq, gq = c.gq;
    const int v0 = 16 * c.warp;
    const int ab_t = 8 * (lm >> 1) + lr, ab_c = lm & 1;
    const int c_t = 8 * (lm & 1) + lr, c_c = lm >> 1;
    const uint32_t sb = c.sb;

    for (int x = c.tid; x < 512; x += 256) sts128z(sb + F_AQ + x * 16);   // Aq tiles above the diagonal stay +0
    f_issue_x_part(c, 0, sb + F_K0, 0);   // A, v, g, beta first: pass P's u product needs only these
    f_issue_x_part(c, 0, sb + F_K0, 1);
    f_issue_x_part(c, 0, sb + F_K0, 2);
    f_issue_x_part(c, 0, sb + F_K0, 5);
    cp_commit();
    f_issue_x_part(c, 0, sb + F_K0, 3);
    f_issue_x_part(c, 0, sb + F_K0, 4);
    f_issue_q(c, 0);
    cp_commit();

    float hacc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) hacc[j][r] = 0.f;

    cp_wait<1>();
    __syncthreads();
    stamp();
    for (int i = 0; i < NT; ++i) {
        const uint32_t kbuf = sb + F_K0 + (i & 1) * 16384;
        const float* gs = reinterpret_cast<const float*>(smem + F_G + (i & 1) * 256);
        const uint32_t* bs = reinterpret_cast<const uint32_t*>(smem + F_B + (i & 1) * 256);
        float vn[8][4];
        float e[8][2];
        f_exps(gs, cq, e);   // pass P's 2^g (recomputed for B below: not live across A)
        // ------------------------------------------------ P(i)
        if (i > 0) f_pstage3<true, false>(c, kbuf, gs, bs, e, vn);
        else f_pstage3<false, true>(c, kbuf, gs, bs, e, vn);
        stamp();
        __syncthreads();   // b1: w, Aq of chunk i in smem; A, v of chunk i no longer read
        const uint32_t kbn = sb + F_K0 + ((i + 1) & 1) * 16384;
        if (i == 0 && NT > 1) f_issue_x(c, 1, kbn);
        // ------------------------------------------------ A(i): vn = u - w h, oa = q h
        float oa[8][4];
        if (i > 0) {
            float acc[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int r = 0; r < 4; ++r) acc[j][r] = 0.f, oa[j][r] = 0.f;
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const uint32_t a0 = pk(hacc[2 * s][0], hacc[2 * s][1]);
                const uint32_t a1 = pk(hacc[2 * s][2], hacc[2 * s][3]);
                const uint32_t a2 = pk(hacc[2 * s + 1][0], hacc[2 * s + 1][1]);
                const uint32_t a3 = pk(hacc[2 * s + 1][2], hacc[2 * s + 1][3]);
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    uint32_t b0, b1, b2, b3;
                    ldsm4(sb + F_W + o256(16 * jj + ab_t, 2 * s + ab_c), b0, b1, b2, b3);
                    mma(acc[2 * jj], a0, a1, a2, a3, b0, b1);
                    mma(acc[2 * jj + 1], a0, a1, a2, a3, b2, b3);
                    ldsm4(sb + F_Q + o256(16 * jj + ab_t, 2 * s + ab_c), b0, b1, b2, b3);
                    mma(oa[2 * jj], a0, a1, a2, a3, b0, b1);
                    mma(oa[2 * jj + 1], a0, a1, a2, a3, b2, b3);
                }
                if (s < 6 && i + 1 < NT) f_issue_x_part(c, i + 1, kbn, s);   // chunk i + 1 inputs, in slices
            }
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int r = 0; r < 4; ++r) vn[j][r] = fsub(vn[j][r], acc[j][r]);
        }
        stamp();
        cp_commit();
        __syncthreads();   // b2: every warp is done with q of chunk i
        // ------------------------------------------------ B(i): o (c = scale * (q h) 2^g, +0 based for chunk 0)
        f_exps(gs, cq, e);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (i > 0) {
                oa[j][0] = fmul(scale, fmul(oa[j][0], e[j][0]));
                oa[j][1] = fmul(scale, fmul(oa[j][1], e[j][1]));
                oa[j][2] = fmul(scale, fmul(oa[j][2], e[j][0]));
                oa[j][3] = fmul(scale, fmul(oa[j][3], e[j][1]));
            } else {
                oa[j][0] = oa[j][2] = fmul(scale, fmul(e[j][0], 0.f));
                oa[j][1] = oa[j][3] = fmul(scale, fmul(e[j][1], 0.f));
            }
        }
        {
            uint32_t vs[4][4];
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                vs[a][0] = pk(vn[2 * a][0], vn[2 * a][1]);
                vs[a][1] = pk(vn[2 * a][2], vn[2 * a][3]);
                vs[a][2] = pk(vn[2 * a + 1][0], vn[2 * a + 1][1]);
                vs[a][3] = pk(vn[2 * a + 1][2], vn[2 * a + 1][3]);
            }
            const size_t r0 = c.row0 + static_cast<size_t>(i) * BT;
            // B(i) for the t-tile pair jj: o rows 16 jj .. 16 jj + 15
            auto b_pair = [&](int jj) {
                float d2[2][4];
#pragma unroll
                for (int x = 0; x < 2; ++x)
#pragma unroll
                    for (int r = 0; r < 4; ++r) d2[x][r] = 0.f;
#pragma unroll
                for (int a = 0; a < 4; ++a) {
                    if (SKIP && a > jj) continue;
                    uint32_t b0, b1, b2, b3;
                    ldsm4(sb + F_AQ + o128(16 * jj + ab_t, 2 * a + ab_c), b0, b1, b2, b3);
                    mma(d2[0], vs[a][0], vs[a][1], vs[a][2], vs[a][3], b0, b1);
                    mma(d2[1], vs[a][0], vs[a][1], vs[a][2], vs[a][3], b2, b3);
                }
#pragma unroll
                for (int x = 0; x < 2; ++x) {
                    const int j = 2 * jj + x;
                    const uint32_t m0 = movt(pk(ffma(scale, d2[x][0], oa[j][0]), ffma(scale, d2[x][1], oa[j][1])));
                    const uint32_t m1 = movt(pk(ffma(scale, d2[x][2], oa[j][2]), ffma(scale, d2[x][3], oa[j][3])));
                    const size_t row = r0 + 8 * j + gq;
                    *reinterpret_cast<uint32_t*>(o + (row * HV + h) * VD + v0 + 2 * cq) = m0;
                    *reinterpret_cast<uint32_t*>(o + (row * HV + h) * VD + v0 + 8 + 2 * cq) = m1;
                }
            };
            if (i + 1 < NT) {
                // ------------------------------------ C(i) (h update), with B(i)'s tile pairs interleaved
                const float gl = gs[63];
                const float el = ex2(gl);
                if (i > 0) {
#pragma unroll
                    for (int j = 0; j < 16; ++j)
#pragma unroll
                        for (int r = 0; r < 4; ++r) hacc[j][r] = fmul(hacc[j][r], el);
                } else {
                    const float z = fmul(el, 0.f);
#pragma unroll
                    for (int j = 0; j < 16; ++j)
#pragma unroll
                        for (int r = 0; r < 4; ++r) hacc[j][r] = z;
                }
#pragma unroll
                for (int a = 0; a < 4; ++a) {
                    const float f00 = ex2(fsub(gl, gs[16 * a + 2 * cq])), f01 = ex2(fsub(gl, gs[16 * a + 2 * cq + 1]));
                    const float f10 = ex2(fsub(gl, gs[16 * a + 8 + 2 * cq])),
                                f11 = ex2(fsub(gl, gs[16 * a + 8 + 2 * cq + 1]));
                    const uint32_t a0 = pk(fmul(vn[2 * a][0], f00), fmul(vn[2 * a][1], f01));
                    const uint32_t a1 = pk(fmul(vn[2 * a][2], f00), fmul(vn[2 * a][3], f01));
                    const uint32_t a2 = pk(fmul(vn[2 * a + 1][0], f10), fmul(vn[2 * a + 1][1], f11));
                    const uint32_t a3 = pk(fmul(vn[2 * a + 1][2], f10), fmul(vn[2 * a + 1][3], f11));
#pragma unroll
                    for (int jj = 0; jj < 8; ++jj) {
                        uint32_t b0, b1, b2, b3;
                        ldsm4t(kbuf + o256(16 * a + c_t, 2 * jj + c_c), b0, b1, b2, b3);
                        mma(hacc[2 * jj], a0, a1, a2, a3, b0, b1);
                        mma(hacc[2 * jj + 1], a0, a1, a2, a3, b2, b3);
                    }
                    b_pair(a);
                    f_issue_q_part(c, i + 1, a);   // q of chunk i + 1 (q buffer free since b2)
                }
            } else {
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) b_pair(jj);
            }
        }
        stamp();
        cp_commit();
        stamp();
        cp_wait<0>();
        __syncthreads();   // b3: chunk i + 1 inputs landed; every warp is done with w, Aq, k of chunk i
        stamp();
    }
}

// ---------------------------------------------------------------- v4: pass P split. Aq(i) at the start of chunk i;
// w(i + 1) and u(i + 1) (h-independent, inputs loaded during A(i)) interleaved with C(i) (the h update), whose
// tensor work they fill in. Per chunk: P2(i) | b1 | A(i) [+ chunk i + 1 inputs] | b2 | B(i) | C(i) + P13(i + 1)
// [+ q(i + 1)] | b3.
struct P13 {
    uint32_t vb[4][4], kb[4][4];
    float uacc[8][4];
};

template <bool DO_W>
DEVI void p13_setup(const FCtx& c, uint32_t kbuf, const float* gs, const uint32_t* bs, P13& p) {
    const int lm = c.lm, lr = c.lr, cq = c.cq, wi = c.warp;
    const int c_t = 8 * (lm & 1) + lr, c_c = lm >> 1;
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        uint32_t r[4];
        ldsm4t(c.sb + F_V + o256(16 * a + 8 * (lm >> 1) + lr, 2 * wi + (lm & 1)), r[0], r[1], r[2], r[3]);
        const uint32_t blo = f_beta2(c, bs, 16 * a + 2 * cq), bhi = f_beta2(c, bs, 16 * a + 8 + 2 * cq);
        p.vb[a][0] = bmul2(r[0], blo);
        p.vb[a][1] = bmul2(r[1], blo);
        p.vb[a][2] = bmul2(r[2], bhi);
        p.vb[a][3] = bmul2(r[3], bhi);
    }
    if (DO_W) {
#pragma unroll
        for (int sb = 0; sb < 4; ++sb) {
            uint32_t r[4];
            ldsm4t(kbuf + o256(16 * sb + c_t, 2 * wi + c_c), r[0], r[1], r[2], r[3]);
            const uint32_t blo = f_beta2(c, bs, 16 * sb + 2 * cq), bhi = f_beta2(c, bs, 16 * sb + 8 + 2 * cq);
            const float e00 = ex2(gs[16 * sb + 2 * cq]), e01 = ex2(gs[16 * sb + 2 * cq + 1]);
            const float e10 = ex2(gs[16 * sb + 8 + 2 * cq]), e11 = ex2(gs[16 * sb + 8 + 2 * cq + 1]);
#pragma unroll
            for (int x = 0; x < 4; ++x) {
                const uint32_t pr = bmul2(r[x], (x & 1) ? bhi : blo);
                p.kb[sb][x] = (x & 1) ? pk(fmul(blo_f(pr), e10), fmul(bhi_f(pr), e11))
                                      : pk(fmul(blo_f(pr), e00), fmul(bhi_f(pr), e01));
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) p.uacc[j][r] = 0.f;
}

// rows mt of w (if DO_W) and the u^T tile pair mt
template <bool DO_W>
DEVI void p13_row(const FCtx& c, int mt, P13& p) {
    const int lane = c.lane, lm = c.lm, lr = c.lr, wi = c.warp;
    const int c_t = 8 * (lm & 1) + lr, c_c = lm >> 1;
    float wacc[2][4];
#pragma unroll
    for (int nn = 0; nn < 2; ++nn)
#pragma unroll
        for (int r = 0; r < 4; ++r) wacc[nn][r] = 0.f;
#pragma unroll
    for (int sb = 0; sb < 4; ++sb) {
        if (sb > mt) continue;
        uint32_t r0, r1, r2, r3;
        ldsm4(c.sb + F_A + o128(16 * mt + (lane & 15), 2 * sb + (lane >> 4)), r0, r1, r2, r3);
        if (DO_W) {
            mma(wacc[0], r0, r1, r2, r3, p.kb[sb][0], p.kb[sb][1]);
            mma(wacc[1], r0, r1, r2, r3, p.kb[sb][2], p.kb[sb][3]);
        }
        mma(p.uacc[2 * mt], p.vb[sb][0], p.vb[sb][1], p.vb[sb][2], p.vb[sb][3], r0, r2);
        mma(p.uacc[2 * mt + 1], p.vb[sb][0], p.vb[sb][1], p.vb[sb][2], p.vb[sb][3], r1, r3);
    }
    if (DO_W)
        stsm4(c.sb + F_W + o256(16 * mt + c_t, 2 * wi + c_c), pk(wacc[0][0], wacc[0][1]), pk(wacc[0][2], wacc[0][3]),
              pk(wacc[1][0], wacc[1][1]), pk(wacc[1][2], wacc[1][3]));
}

DEVI void p13_finish(P13& p, float (&uf)[8][4]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t p0 = pk(p.uacc[j][0], p.uacc[j][1]), p1 = pk(p.uacc[j][2], p.uacc[j][3]);
        uf[j][0] = blo_f(p0);
        uf[j][1] = bhi_f(p0);
        uf[j][2] = blo_f(p1);
        uf[j][3] = bhi_f(p1);
    }
}

DEVI void p2_aq(const FCtx& c, uint32_t kbuf, const float* gs) {
    const int lane = c.lane, lm = c.lm, lr = c.lr, wi = c.warp;
    const int ab_t = 8 * (lm >> 1) + lr, ab_c = lm & 1;
    const int pmt = wi < 4 ? 3 : wi < 7 ? 2 : 1;
    const int pnp = wi < 4 ? wi : wi < 7 ? wi - 4 : 0;
    float acc[2][4];
#pragma unroll
    for (int nn = 0; nn < 2; ++nn)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[nn][r] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
        ldsm4(c.sb + F_Q + o256(16 * pmt + (lane & 15), 2 * ks + (lane >> 4)), a0, a1, a2, a3);
        ldsm4(kbuf + o256(16 * pnp + ab_t, 2 * ks + ab_c), b0, b1, b2, b3);
        mma(acc[0], a0, a1, a2, a3, b0, b1);
        mma(acc[1], a0, a1, a2, a3, b2, b3);
    }
    f_aq_tile_store(c, gs, pmt, 2 * pnp, acc[0]);
    f_aq_tile_store(c, gs, pmt, 2 * pnp + 1, acc[1]);
    if (wi >= 4) {
        const int smt = wi < 6 ? 1 : 0;
        const int sns = wi == 4 ? 2 : wi == 5 ? 3 : wi == 6 ? 0 : 1;
        float a1c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k2 = 0; k2 < 4; ++k2) {
            uint32_t b0, b1, b2, b3;
            ldsm4(kbuf + o256(8 * sns + lr, 4 * k2 + lm), b0, b1, b2, b3);
            uint32_t a0, a1, a2, a3;
            ldsm4(c.sb + F_Q + o256(16 * smt + (lane & 15), 4 * k2 + (lane >> 4)), a0, a1, a2, a3);
            mma(a1c, a0, a1, a2, a3, b0, b1);
            ldsm4(c.sb + F_Q + o256(16 * smt + (lane & 15), 4 * k2 + 2 + (lane >> 4)), a0, a1, a2, a3);
            mma(a1c, a0, a1, a2, a3, b2, b3);
        }
        f_aq_tile_store(c, gs, smt, sns, a1c);
    }
}

template <bool TIMED>
DEVI void fused4_body(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                      const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ beta,
                      const __nv_bfloat16* __restrict__ A, const float* __restrict__ g, __nv_bfloat16* __restrict__ o,
                      const int* __restrict__ slen, int T, float scale, int hq, unsigned long long* ts = nullptr) {
    int tsn = 0;
    auto stamp = [&]() {
        if (TIMED && (threadIdx.x & 31) == 0) {
            unsigned long long t;
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)::"memory");
            ts[(static_cast<size_t>(blockIdx.x) * 8 + (threadIdx.x >> 5)) * 48 + tsn] = t;
        }
        ++tsn;
    };
    stamp();
    extern __shared__ __align__(128) uint8_t smem[];
    const int n = blockIdx.x / HV, h = blockIdx.x % HV;
    if (slen[n] == 0) return;
    const FCtx c = f_ctx(q, k, v, beta, A, g, n, h, T, smem, hq);
    const int NT = T / BT;
    const int lm = c.lm, lr = c.lr, cq = c.cq, gq = c.gq;
    const int v0 = 16 * c.warp;
    const int ab_t = 8 * (lm >> 1) + lr, ab_c = lm & 1;
    const int c_t = 8 * (lm & 1) + lr, c_c = lm >> 1;
    const uint32_t sb = c.sb;

    for (int x = c.tid; x < 512; x += 256) sts128z(sb + F_AQ + x * 16);   // Aq tiles above the diagonal stay +0
    f_issue_x_part(c, 0, sb + F_K0, 0);
    f_issue_x_part(c, 0, sb + F_K0, 1);
    f_issue_x_part(c, 0, sb + F_K0, 2);
    f_issue_x_part(c, 0, sb + F_K0, 5);
    cp_commit();
    f_issue_x_part(c, 0, sb + F_K0, 3);
    f_issue_x_part(c, 0, sb + F_K0, 4);
    f_issue_q(c, 0);
    cp_commit();

    float hacc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) hacc[j][r] = 0.f;
    float uf[8][4];
    cp_wait<1>();
    __syncthreads();
    {   // u of chunk 0 (no w: chunk 0's v_new = u)
        P13 p;
        p13_setup<false>(c, sb + F_K0, reinterpret_cast<const float*>(smem + F_G),
                         reinterpret_cast<const uint32_t*>(smem + F_B), p);
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) p13_row<false>(c, mt, p);
        p13_finish(p, uf);
    }
    cp_wait<0>();
    __syncthreads();   // k, q of chunk 0 visible; A, v of chunk 0 no longer read
    if (NT > 1) {
        const uint32_t kb1 = sb + F_K0 + 16384;
#pragma unroll
        for (int part = 0; part < 6; ++part) f_issue_x_part(c, 1, kb1, part);
    }
    cp_commit();
    stamp();
    for (int i = 0; i < NT; ++i) {
        const uint32_t kbuf = sb + F_K0 + (i & 1) * 16384;
        const uint32_t kbn = sb + F_K0 + ((i + 1) & 1) * 16384;
        const float* gs = reinterpret_cast<const float*>(smem + F_G + (i & 1) * 256);
        // ------------------------------------------------ P2(i): Aq
        p2_aq(c, kbuf, gs);
        stamp();
        __syncthreads();   // b1: Aq(i) (and w(i), written during C(i - 1)) in smem
        // ------------------------------------------------ A(i): vn = u - w h, oa = q h
        float vn[8][4], oa[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r) vn[j][r] = uf[j][r];
        if (i > 0) {
            float acc[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int r = 0; r < 4; ++r) acc[j][r] = 0.f, oa[j][r] = 0.f;
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const uint32_t a0 = pk(hacc[2 * s][0], hacc[2 * s][1]);
                const uint32_t a1 = pk(hacc[2 * s][2], hacc[2 * s][3]);
                const uint32_t a2 = pk(hacc[2 * s + 1][0], hacc[2 * s + 1][1]);
                const uint32_t a3 = pk(hacc[2 * s + 1][2], hacc[2 * s + 1][3]);
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    uint32_t b0, b1, b2, b3;
                    ldsm4(sb + F_W + o256(16 * jj + ab_t, 2 * s + ab_c), b0, b1, b2, b3);
                    mma(acc[2 * jj], a0, a1, a2, a3, b0, b1);
                    mma(acc[2 * jj + 1], a0, a1, a2, a3, b2, b3);
                    ldsm4(sb + F_Q + o256(16 * jj + ab_t, 2 * s + ab_c), b0, b1, b2, b3);
                    mma(oa[2 * jj], a0, a1, a2, a3, b0, b1);
                    mma(oa[2 * jj + 1], a0, a1, a2, a3, b2, b3);
                }
                if (s < 6 && i + 1 < NT) f_issue_x_part(c, i + 1, kbn, s);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int r = 0; r < 4; ++r) vn[j][r] = fsub(vn[j][r], acc[j][r]);
            cp_commit();
        }
        stamp();
        cp_wait<0>();
        __syncthreads();   // b2: q(i), w(i) no longer read; chunk i + 1 inputs (A, v, k, g, beta) visible
        // ------------------------------------------------ B(i): o
        {
            float e[8][2];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                e[j][0] = ex2(gs[8 * j + 2 * cq]);
                e[j][1] = ex2(gs[8 * j + 2 * cq + 1]);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (i > 0) {
                    oa[j][0] = fmul(scale, fmul(oa[j][0], e[j][0]));
                    oa[j][1] = fmul(scale, fmul(oa[j][1], e[j][1]));
                    oa[j][2] = fmul(scale, fmul(oa[j][2], e[j][0]));
                    oa[j][3] = fmul(scale, fmul(oa[j][3], e[j][1]));
                } else {
                    oa[j][0] = oa[j][2] = fmul(scale, fmul(e[j][0], 0.f));
                    oa[j][1] = oa[j][3] = fmul(scale, fmul(e[j][1], 0.f));
                }
            }
            uint32_t vs[4][4];
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                vs[a][0] = pk(vn[2 * a][0], vn[2 * a][1]);
                vs[a][1] = pk(vn[2 * a][2], vn[2 * a][3]);
                vs[a][2] = pk(vn[2 * a + 1][0], vn[2 * a + 1][1]);
                vs[a][3] = pk(vn[2 * a + 1][2], vn[2 * a + 1][3]);
            }
            const size_t r0 = c.row0 + static_cast<size_t>(i) * BT;
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                float d2[2][4];
#pragma unroll
                for (int x = 0; x < 2; ++x)
#pragma unroll
                    for (int r = 0; r < 4; ++r) d2[x][r] = 0.f;
#pragma unroll
                for (int a = 0; a < 4; ++a) {
                    if (a > jj) continue;
                    uint32_t b0, b1, b2, b3;
                    ldsm4(sb + F_AQ + o128(16 * jj + ab_t, 2 * a + ab_c), b0, b1, b2, b3);
                    mma(d2[0], vs[a][0], vs[a][1], vs[a][2], vs[a][3], b0, b1);
                    mma(d2[1], vs[a][0], vs[a][1], vs[a][2], vs[a][3], b2, b3);
                }
#pragma unroll
                for (int x = 0; x < 2; ++x) {
                    const int j = 2 * jj + x;
                    const uint32_t m0 = movt(pk(ffma(scale, d2[x][0], oa[j][0]), ffma(scale, d2[x][1], oa[j][1])));
                    const uint32_t m1 = movt(pk(ffma(scale, d2[x][2], oa[j][2]), ffma(scale, d2[x][3], oa[j][3])));
                    const size_t row = r0 + 8 * j + gq;
                    *reinterpret_cast<uint32_t*>(o + (row * HV + h) * VD + v0 + 2 * cq) = m0;
                    *reinterpret_cast<uint32_t*>(o + (row * HV + h) * VD + v0 + 8 + 2 * cq) = m1;
                }
            }
        }
        stamp();
        // ------------------------------------------------ C(i) (h update) + w, u of chunk i + 1
        if (i + 1 < NT) {
            const float gl = gs[63];
            const float el = ex2(gl);
            uint32_t vd[4][4];
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                const float f00 = ex2(fsub(gl, gs[16 * a + 2 * cq])), f01 = ex2(fsub(gl, gs[16 * a + 2 * cq + 1]));
                const float f10 = ex2(fsub(gl, gs[16 * a + 8 + 2 * cq])),
                            f11 = ex2(fsub(gl, gs[16 * a + 8 + 2 * cq + 1]));
                vd[a][0] = pk(fmul(vn[2 * a][0], f00), fmul(vn[2 * a][1], f01));
                vd[a][1] = pk(fmul(vn[2 * a][2], f00), fmul(vn[2 * a][3], f01));
                vd[a][2] = pk(fmul(vn[2 * a + 1][0], f10), fmul(vn[2 * a + 1][1], f11));
                vd[a][3] = pk(fmul(vn[2 * a + 1][2], f10), fmul(vn[2 * a + 1][3], f11));
            }
            if (i > 0) {
#pragma unroll
                for (int j = 0; j < 16; ++j)
#pragma unroll
                    for (int r = 0; r < 4; ++r) hacc[j][r] = fmul(hacc[j][r], el);
            } else {
                const float z = fmul(el, 0.f);
#pragma unroll
                for (int j = 0; j < 16; ++j)
#pragma unroll
                    for (int r = 0; r < 4; ++r) hacc[j][r] = z;
            }
            P13 p;
            p13_setup<true>(c, kbn, reinterpret_cast<const float*>(smem + F_G + ((i + 1) & 1) * 256),
                            reinterpret_cast<const uint32_t*>(smem + F_B + ((i + 1) & 1) * 256), p);
#pragma unroll
            for (int a = 0; a < 4; ++a) {
#pragma unroll
                for (int jj = 0; jj < 8; ++jj) {
                    uint32_t b0, b1, b2, b3;
                    ldsm4t(kbuf + o256(16 * a + c_t, 2 * jj + c_c), b0, b1, b2, b3);
                    mma(hacc[2 * jj], vd[a][0], vd[a][1], vd[a][2], vd[a][3], b0, b1);
                    mma(hacc[2 * jj + 1], vd[a][0], vd[a][1], vd[a][2], vd[a][3], b2, b3);
                }
                p13_row<true>(c, a, p);
                f_issue_q_part(c, i + 1, a);   // q of chunk i + 1 (q buffer free since b2)
            }
            p13_finish(p, uf);
        }
        cp_commit();
        stamp();
        cp_wait<0>();
        __syncthreads();   // b3: q(i + 1), w(i + 1) visible; Aq(i), k(i) no longer read
        stamp();
    }
}
}  // namespace

extern "C" __global__ void __launch_bounds__(256, 1)
    ffs_fused(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* beta,
              const __nv_bfloat16* A, const float* g, __nv_bfloat16* o, const int* slen, int T, float scale) {
    fused_body<true, false>(q, k, v, beta, A, g, o, slen, T, scale);
}

extern "C" __global__ void __launch_bounds__(256, 1)
    ffs_fused_ts(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* beta,
                 const __nv_bfloat16* A, const float* g, __nv_bfloat16* o, const int* slen, int T, float scale,
                 unsigned long long* ts) {
    fused_body<true, true>(q, k, v, beta, A, g, o, slen, T, scale, ts);
}

#define SEQ_KERNEL(NAME, WPC, SKIP)                                                                                  \
    extern "C" __global__ void __launch_bounds__(WPC * 32, 1)                                                        \
        NAME(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* w, const __nv_bfloat16* u,         \
             const __nv_bfloat16* Aq, const float* g, __nv_bfloat16* o, const int* slen, int T, float scale) {       \
        seq_body<WPC, SKIP>(q, k, w, u, Aq, g, o, slen, T, scale);                                                     \
    }

SEQ_KERNEL(ffs_seq_w4, 4, false)
SEQ_KERNEL(ffs_seq_w4s, 4, true)
SEQ_KERNEL(ffs_seq_w8, 8, false)
SEQ_KERNEL(ffs_seq_w8s, 8, true)
SEQ_KERNEL(ffs_seq_w2s, 2, true)

extern "C" __global__ void __launch_bounds__(256, 1)
    ffs_fused2(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* beta,
               const __nv_bfloat16* A, const float* g, __nv_bfloat16* o, const int* slen, int T, float scale) {
    fused2_body<true, false>(q, k, v, beta, A, g, o, slen, T, scale);
}

extern "C" __global__ void __launch_bounds__(256, 1)
    ffs_fused2_ts(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* beta,
                  const __nv_bfloat16* A, const float* g, __nv_bfloat16* o, const int* slen, int T, float scale,
                  unsigned long long* ts) {
    fused2_body<true, true>(q, k, v, beta, A, g, o, slen, T, scale, ts);
}

extern "C" __global__ void __launch_bounds__(256, 1)
    ffs_fused3(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* beta,
               const __nv_bfloat16* A, const float* g, __nv_bfloat16* o, const int* slen, int T, float scale, int hq) {
    fused3_body<true, false>(q, k, v, beta, A, g, o, slen, T, scale, hq);
}

extern "C" __global__ void __launch_bounds__(256, 1)
    ffs_fused3_ts(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* beta,
                  const __nv_bfloat16* A, const float* g, __nv_bfloat16* o, const int* slen, int T, float scale,
                  unsigned long long* ts) {
    fused3_body<true, true>(q, k, v, beta, A, g, o, slen, T, scale, H, ts);
}

extern "C" __global__ void __launch_bounds__(256, 1)
    ffs_fused4(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* beta,
               const __nv_bfloat16* A, const float* g, __nv_bfloat16* o, const int* slen, int T, float scale, int hq) {
    fused4_body<false>(q, k, v, beta, A, g, o, slen, T, scale, hq);
}

extern "C" __global__ void __launch_bounds__(256, 1)
    ffs_fused4_ts(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* beta,
                  const __nv_bfloat16* A, const float* g, __nv_bfloat16* o, const int* slen, int T, float scale,
                  unsigned long long* ts) {
    fused4_body<true>(q, k, v, beta, A, g, o, slen, T, scale, H, ts);
}

extern "C" __global__ void __launch_bounds__(256, 1)
    ffs_fused5(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* beta,
               const __nv_bfloat16* A, const float* g, __nv_bfloat16* o, const int* slen, int T, float scale, int hq) {
    fused5_body<true, false>(q, k, v, beta, A, g, o, slen, T, scale, hq);
}
