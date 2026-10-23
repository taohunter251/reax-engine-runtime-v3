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

#ifndef ONORM_V
#define ONORM_V 2
#endif

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
constexpr int F_N = 99328;   // ONORM_SMS: norm weight (256 B) + plen / start / slen (16 B); smem 99328 + 512
DEVI int lds_s32(uint32_t a) {
    int v;
    asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(v) : "r"(a) : "memory");
    return v;
}

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
#ifndef FFS_SHIP
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
#endif

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

template <bool DO_W, bool FIRST, class Hook>
DEVI void f_pstage3h(const FCtx& c, uint32_t kbuf, const float* gs, const uint32_t* bs, const float (&e)[8][2],
                     float (&uf)[8][4], Hook&& hook) {
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
        hook(mt);
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
    hook(4);
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

// ---------------------------------------------------------------- ONORM: gated RMSNorm epilogue (fused.py
// _gated_norm_packed_kernel, dumped PTX of the e4c build: dev_ax_onorm/dump/gn.ptx), per row of 128 values:
//   L_t = ((sq[4t] + sq[4t+1]) + sq[4t+2]) + sq[4t+3]  (sq = f32(o)^2, lane t = cols 4t..4t+3)
//   s   = shfl-down tree 16, 8, 4, 2, 1 over L_0..L_31 (== xor butterfly: every add commutative)
//   r   = rsqrt.approx.f32(eps + s * fl(1/128))
//   hn  = bf16(o * r); hw = mul.rn.bf16(w, hn); zf = f32(z)
//   out = bf16(hw * div.rn(zf, 1 + __nv_expf(0 - zf)))   (__nv_expf: the exact PTX sequence of libdevice)
// every op an explicit-rounding PTX instruction (never contracted).
struct NormArgs {
    const __nv_bfloat16* z;    // packed z rows, row stride zs (elements); head h at column h * 128
    int zs;
    const __nv_bfloat16* w;    // norm weight [128]
    float eps;
    const int* start;          // packed row of position plen[n] of sequence n
    const int* plen;           // PV prefix length of sequence n (rows p < plen have no packed row)
    __nv_bfloat16* out;        // [R * 32, 128]
};

DEVI float n_rsqrt(float x) {
    float y;
    asm("rsqrt.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}
DEVI float n_div(float a, float b) {
    float y;
    asm("div.rn.f32 %0, %1, %2;\n" : "=f"(y) : "f"(a), "f"(b));
    return y;
}
DEVI float n_add(float a, float b) {
    float y;
    asm("add.rn.f32 %0, %1, %2;\n" : "=f"(y) : "f"(a), "f"(b));
    return y;
}
DEVI float n_mul(float a, float b) {
    float y;
    asm("mul.rn.f32 %0, %1, %2;\n" : "=f"(y) : "f"(a), "f"(b));
    return y;
}
DEVI float n_sub(float a, float b) {
    float y;
    asm("sub.rn.f32 %0, %1, %2;\n" : "=f"(y) : "f"(a), "f"(b));
    return y;
}
// libdevice __nv_expf (non-ftz build), instruction for instruction as in the Triton PTX
DEVI float n_expf(float x) {
    float y;
    asm("{\n"
        ".reg .f32 t1, t2, t3, t4, t5, t6, t7, t9;\n"
        ".reg .b32 t8;\n"
        "fma.rn.f32 t1, %1, 0f3BBB989D, 0f3F000000;\n"
        "cvt.sat.f32.f32 t2, t1;\n"
        "fma.rm.f32 t3, t2, 0f437C0000, 0f4B400001;\n"
        "add.rn.f32 t4, t3, 0fCB40007F;\n"
        "neg.f32 t5, t4;\n"
        "fma.rn.f32 t6, %1, 0f3FB8AA3B, t5;\n"
        "fma.rn.f32 t7, %1, 0f32A57060, t6;\n"
        "mov.b32 t8, t3;\n"
        "shl.b32 t8, t8, 23;\n"
        "ex2.approx.ftz.f32 t9, t7;\n"
        "mul.rn.f32 %0, t9, t8;\n"
        "}\n"
        : "=f"(y) : "f"(x));
    return y;
}
DEVI uint32_t n_bmul2(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
DEVI float n_out(float hw, float zf) {
    const float e = n_expf(n_sub(0.f, zf));
    return n_mul(n_div(zf, n_add(e, 1.f)), hw);
}
// FDIV: div.rn(z, d) without the FCHK slow-path branch for z in (-87, +inf) (d = 1 + expf(-z) in [1, 2^126)): MUFU
// rcp + one Newton step + residual correction (ptxas's own fast path), zeros passed through; equal to div.rn for every
// such bf16 z (exhaustive check silu_ex.cu). Rows with any other z take n_out (warp-uniform branch).
DEVI float n_fdiv(float z, float d) {
    float q;
    asm("{\n.reg .f32 y0, e, y1, q0, r;\n"
        "rcp.approx.ftz.f32 y0, %2;\n"
        "neg.f32 e, %2;\n"
        "fma.rn.f32 e, e, y0, 0f3F800000;\n"
        "fma.rn.f32 y1, y0, e, y0;\n"
        "mul.rn.f32 q0, %1, y1;\n"
        "neg.f32 r, %2;\n"
        "fma.rn.f32 r, r, q0, %1;\n"
        "fma.rn.f32 %0, y1, r, q0;\n}\n" : "=f"(q) : "f"(z), "f"(d));
    return z == 0.f ? z : q;
}
DEVI float n_out_f(float hw, float zf) {
    const float e = n_expf(n_sub(0.f, zf));
    return n_mul(n_fdiv(zf, n_add(e, 1.f)), hw);
}
DEVI bool n_special(float z) { return !(z > -87.f && z < 3.0e38f); }
// SPEC2: div.rn(z, d) (d = 1 + expf(-z) as computed above) per element for a warp with special z: the fast division
// where it is exact (z in (-87, inf)); for finite z whose d overflowed to +inf the IEEE quotient is -0 (z < 0) - no
// division needed; anything else (z in [-88.72, -87] with finite d, z = +-inf, NaN) the real div.rn
DEVI float n_silu_sp(float z, float d) {
    float q = n_fdiv(z, d);
    if (n_special(z)) q = (d == __int_as_float(0x7f800000) && z > -3.0e38f && z < 0.f) ? -0.0f : n_div(z, d);
    return q;
}

// rows t = warp + 8 rr of chunk i: o (bf16, swizzled 64 x 256 B tile at `tile`) -> gated norm -> out (packed rows)
DEVI void n_rows(const NormArgs& na, uint32_t tile, int i, int h, int warp, int lane, int pl, int sl, int st,
                 uint32_t w01, uint32_t w23, float inv_d) {
#pragma unroll 2
    for (int rr = 0; rr < 8; ++rr) {
        const int t = warp + 8 * rr;
        const int p = i * BT + t;
        if (p < pl || p >= sl) continue;   // warp-uniform
        const size_t row = static_cast<size_t>(st + p - pl);
        uint32_t o01, o23;
        asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];\n" : "=r"(o01), "=r"(o23)
                     : "r"(tile + o256(t, lane >> 1) + 8 * (lane & 1)));
        const uint2 zv = *reinterpret_cast<const uint2*>(na.z + row * na.zs + h * VD + 4 * lane);
        const float f0 = blo_f(o01), f1 = bhi_f(o01), f2 = blo_f(o23), f3 = bhi_f(o23);
        float s = n_add(n_add(n_add(n_mul(f0, f0), n_mul(f1, f1)), n_mul(f2, f2)), n_mul(f3, f3));
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1) s = n_add(s, __shfl_xor_sync(0xffffffffu, s, m));
        const float r = n_rsqrt(n_add(na.eps, n_mul(inv_d, s)));
        const uint32_t hw01 = n_bmul2(w01, pk(n_mul(r, f0), n_mul(r, f1)));
        const uint32_t hw23 = n_bmul2(w23, pk(n_mul(r, f2), n_mul(r, f3)));
        const uint32_t y01 = pk(n_out(blo_f(hw01), blo_f(zv.x)), n_out(bhi_f(hw01), bhi_f(zv.x)));
        const uint32_t y23 = pk(n_out(blo_f(hw23), blo_f(zv.y)), n_out(bhi_f(hw23), bhi_f(zv.y)));
        *reinterpret_cast<uint2*>(na.out + (row * HV + h) * VD + 4 * lane) = make_uint2(y01, y23);
    }
}

// V2 / V3: chunk i's valid rows [lo, lo + nv) (positions plen .. slen - 1), row t = lo + warp + 8 rr for rr < cnt
// (balanced over the 8 warps, no work on prefix / padding rows); z of those rows loaded up front (zv)
struct NRows {
    int lo, cnt;
    uint2 zv[8];
};
DEVI void n_prep(const NormArgs& na, int i, int h, int warp, int lane, int pl, int sl, int st, NRows& nr) {
    const int lo = max(pl - i * BT, 0), hi = min(sl - i * BT, BT);
    const int nv = hi - lo;
    nr.lo = lo;
    nr.cnt = nv > warp ? (nv - warp + 7) >> 3 : 0;
    const __nv_bfloat16* zb = na.z + static_cast<size_t>(st + i * BT + lo + warp - pl) * na.zs + h * VD + 4 * lane;
#pragma unroll
    for (int rr = 0; rr < 8; ++rr) {
        if (rr < nr.cnt) nr.zv[rr] = *reinterpret_cast<const uint2*>(zb + static_cast<size_t>(8 * rr) * na.zs);
        else nr.zv[rr] = make_uint2(0u, 0u);
    }
}
DEVI void n_row(const NormArgs& na, uint32_t tile, int i, int h, int warp, int lane, int pl, int st, const NRows& nr,
                int rr, uint32_t w01, uint32_t w23) {
    const int t = nr.lo + warp + 8 * rr;
    const size_t row = static_cast<size_t>(st + i * BT + t - pl);
    uint32_t o01, o23;
    asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];\n" : "=r"(o01), "=r"(o23)
                 : "r"(tile + o256(t, lane >> 1) + 8 * (lane & 1)));
    const uint2 zv = nr.zv[rr];
    const float f0 = blo_f(o01), f1 = bhi_f(o01), f2 = blo_f(o23), f3 = bhi_f(o23);
    float s = n_add(n_add(n_add(n_mul(f0, f0), n_mul(f1, f1)), n_mul(f2, f2)), n_mul(f3, f3));
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) s = n_add(s, __shfl_xor_sync(0xffffffffu, s, m));
    const float r = n_rsqrt(n_add(na.eps, n_mul(0.0078125f, s)));
    const uint32_t hw01 = n_bmul2(w01, pk(n_mul(r, f0), n_mul(r, f1)));
    const uint32_t hw23 = n_bmul2(w23, pk(n_mul(r, f2), n_mul(r, f3)));
#ifdef ONORM_FDIV
    const float z0 = blo_f(zv.x), z1 = bhi_f(zv.x), z2 = blo_f(zv.y), z3 = bhi_f(zv.y);
    uint32_t y01 = pk(n_out_f(blo_f(hw01), z0), n_out_f(bhi_f(hw01), z1));
    uint32_t y23 = pk(n_out_f(blo_f(hw23), z2), n_out_f(bhi_f(hw23), z3));
    if (__any_sync(0xffffffffu, n_special(z0) | n_special(z1) | n_special(z2) | n_special(z3))) {
        y01 = pk(n_out(blo_f(hw01), z0), n_out(bhi_f(hw01), z1));
        y23 = pk(n_out(blo_f(hw23), z2), n_out(bhi_f(hw23), z3));
    }
#else
    const uint32_t y01 = pk(n_out(blo_f(hw01), blo_f(zv.x)), n_out(bhi_f(hw01), bhi_f(zv.x)));
    const uint32_t y23 = pk(n_out(blo_f(hw23), blo_f(zv.y)), n_out(bhi_f(hw23), bhi_f(zv.y)));
#endif
    *reinterpret_cast<uint2*>(na.out + (row * HV + h) * VD + 4 * lane) = make_uint2(y01, y23);
}

// V5: silu(z) (the expensive, o-independent part) computed early - inside C(i)'s mma stream - with the FDIV fast
// division; after bN only the norm itself (sum, rsqrt, bf16(o r), w, x silu) remains, branch-free in groups of 4 rows
DEVI void n_silu(const NRows& nr, int rr, float (&sg)[8][4], bool& spec) {
    const uint2 zv = nr.zv[rr];
    const float z0 = blo_f(zv.x), z1 = bhi_f(zv.x), z2 = blo_f(zv.y), z3 = bhi_f(zv.y);
#if defined(ONORM_DIAG) && ONORM_DIAG >= 1   // timing diagnostics only (not exact): no silu
    sg[rr][0] = z0, sg[rr][1] = z1, sg[rr][2] = z2, sg[rr][3] = z3;
    return;
#endif
    sg[rr][0] = n_fdiv(z0, n_add(n_expf(n_sub(0.f, z0)), 1.f));
    sg[rr][1] = n_fdiv(z1, n_add(n_expf(n_sub(0.f, z1)), 1.f));
    sg[rr][2] = n_fdiv(z2, n_add(n_expf(n_sub(0.f, z2)), 1.f));
    sg[rr][3] = n_fdiv(z3, n_add(n_expf(n_sub(0.f, z3)), 1.f));
    spec |= n_special(z0) | n_special(z1) | n_special(z2) | n_special(z3);
}
#if defined(ONORM_NOINL) || defined(ONORM_FINROLL)
// ONORM_NOINL: the rare exact path's 4 div.rn silus live in one out-of-line function (smaller hot code)
__device__ __noinline__ float4 n_silu4_exact(uint32_t zx, uint32_t zy) {
    const float zz[4] = {__uint_as_float(zx << 16), __uint_as_float(zx & 0xffff0000u), __uint_as_float(zy << 16),
                         __uint_as_float(zy & 0xffff0000u)};
    float r[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) r[k] = n_div(zz[k], n_add(n_expf(n_sub(0.f, zz[k])), 1.f));
    return make_float4(r[0], r[1], r[2], r[3]);
}
#endif
// rare path (some z outside the FDIV range in this warp): every silu of the warp's rows again with div.rn
DEVI void n_silu_exact(const NormArgs& na, int i, int h, int warp, int lane, int pl, int st, const NRows& nr,
                       float (&sg)[8][4]) {
#pragma unroll
    for (int rr = 0; rr < 8; ++rr) {
        if (rr < nr.cnt) {
            const size_t row = static_cast<size_t>(st + i * BT + nr.lo + warp + 8 * rr - pl);
            const uint2 zv = *reinterpret_cast<const uint2*>(na.z + row * na.zs + h * VD + 4 * lane);
#ifdef ONORM_NOINL
            const float4 r4 = n_silu4_exact(zv.x, zv.y);
            sg[rr][0] = r4.x, sg[rr][1] = r4.y, sg[rr][2] = r4.z, sg[rr][3] = r4.w;
#else
            const float zz[4] = {blo_f(zv.x), bhi_f(zv.x), blo_f(zv.y), bhi_f(zv.y)};
#pragma unroll
            for (int k = 0; k < 4; ++k) sg[rr][k] = n_div(zz[k], n_add(n_expf(n_sub(0.f, zz[k])), 1.f));
#endif
        }
    }
}
// ONORM_FIN8: rows 0 .. NR - 1 of the warp in one batch (one interleaved shfl tree; t clamped, stores for rr < cnt)
template <int NR>
DEVI void n_finN(const NormArgs& na, uint32_t tile, int i, int h, int warp, int lane, int pl, int st, const NRows& nr,
                 const float (&sg)[8][4], uint32_t w01, uint32_t w23) {
    float f[NR][4], s[NR];
#pragma unroll
    for (int x = 0; x < NR; ++x) {
        const int t = min(nr.lo + warp + 8 * x, BT - 1);
        uint32_t o01, o23;
        asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];\n" : "=r"(o01), "=r"(o23)
                     : "r"(tile + o256(t, lane >> 1) + 8 * (lane & 1)));
        f[x][0] = blo_f(o01), f[x][1] = bhi_f(o01), f[x][2] = blo_f(o23), f[x][3] = bhi_f(o23);
        s[x] = n_add(n_add(n_add(n_mul(f[x][0], f[x][0]), n_mul(f[x][1], f[x][1])), n_mul(f[x][2], f[x][2])),
                     n_mul(f[x][3], f[x][3]));
    }
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1)
#pragma unroll
        for (int x = 0; x < NR; ++x) s[x] = n_add(s[x], __shfl_xor_sync(0xffffffffu, s[x], m));
#pragma unroll
    for (int x = 0; x < NR; ++x) {
        const float r = n_rsqrt(n_add(na.eps, n_mul(0.0078125f, s[x])));
        const uint32_t hw01 = n_bmul2(w01, pk(n_mul(r, f[x][0]), n_mul(r, f[x][1])));
        const uint32_t hw23 = n_bmul2(w23, pk(n_mul(r, f[x][2]), n_mul(r, f[x][3])));
        const uint32_t y01 = pk(n_mul(sg[x][0], blo_f(hw01)), n_mul(sg[x][1], bhi_f(hw01)));
        const uint32_t y23 = pk(n_mul(sg[x][2], blo_f(hw23)), n_mul(sg[x][3], bhi_f(hw23)));
        const size_t row = static_cast<size_t>(st + i * BT + nr.lo + warp + 8 * x - pl);
        if (x < nr.cnt) *reinterpret_cast<uint2*>(na.out + (row * HV + h) * VD + 4 * lane) = make_uint2(y01, y23);
    }
}
#ifdef ONORM_SKIPFLAG
// g_skip (diagnostics only, defined extern "C" by the including file): 1 = finS stores nothing
__device__ __forceinline__ int skip_flag() { return *reinterpret_cast<volatile int*>(&g_skip); }
#define g_skip_st skip_flag()
#endif
// ONORM_SILU_FIN: n_finN with silu(z) of the rows computed here (NR x 4 independent chains per lane after bN, no
// silu registers live across C(i)); EX: div.rn (some z of the warp outside the fast division's range)
template <int NR, bool EX>
DEVI void n_finS(const NormArgs& na, uint32_t tile, int i, int h, int warp, int lane, int pl, int st, const NRows& nr,
                 uint32_t w01, uint32_t w23) {
    float f[NR][4], s[NR];
#pragma unroll
    for (int x = 0; x < NR; ++x) {
        const int t = min(nr.lo + warp + 8 * x, BT - 1);
        uint32_t o01, o23;
        asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];\n" : "=r"(o01), "=r"(o23)
                     : "r"(tile + o256(t, lane >> 1) + 8 * (lane & 1)));
        f[x][0] = blo_f(o01), f[x][1] = bhi_f(o01), f[x][2] = blo_f(o23), f[x][3] = bhi_f(o23);
        s[x] = n_add(n_add(n_add(n_mul(f[x][0], f[x][0]), n_mul(f[x][1], f[x][1])), n_mul(f[x][2], f[x][2])),
                     n_mul(f[x][3], f[x][3]));
    }
#ifdef ONORM_SKIPFLAG
    const int skip_once = skip_flag();
#undef g_skip_st
#define g_skip_st skip_once
#endif
#ifdef ONORM_BULKST
    uint32_t yy[NR][2];
#endif
    float sg[NR][4];
#pragma unroll
    for (int x = 0; x < NR; ++x) {
        const float zz[4] = {blo_f(nr.zv[x].x), bhi_f(nr.zv[x].x), blo_f(nr.zv[x].y), bhi_f(nr.zv[x].y)};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float d = n_add(n_expf(n_sub(0.f, zz[k])), 1.f);
#ifdef ONORM_SPEC2
            sg[x][k] = EX ? n_silu_sp(zz[k], d) : n_fdiv(zz[k], d);
#else
            sg[x][k] = EX ? n_div(zz[k], d) : n_fdiv(zz[k], d);
#endif
        }
    }
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1)
#pragma unroll
        for (int x = 0; x < NR; ++x) s[x] = n_add(s[x], __shfl_xor_sync(0xffffffffu, s[x], m));
#pragma unroll
    for (int x = 0; x < NR; ++x) {
        const float r = n_rsqrt(n_add(na.eps, n_mul(0.0078125f, s[x])));
        const uint32_t hw01 = n_bmul2(w01, pk(n_mul(r, f[x][0]), n_mul(r, f[x][1])));
        const uint32_t hw23 = n_bmul2(w23, pk(n_mul(r, f[x][2]), n_mul(r, f[x][3])));
        const uint32_t y01 = pk(n_mul(sg[x][0], blo_f(hw01)), n_mul(sg[x][1], bhi_f(hw01)));
        const uint32_t y23 = pk(n_mul(sg[x][2], blo_f(hw23)), n_mul(sg[x][3], bhi_f(hw23)));
        const size_t row = static_cast<size_t>(st + i * BT + nr.lo + warp + 8 * x - pl);
#if defined(ONORM_BULKST)
        yy[x][0] = y01, yy[x][1] = y23;
        (void)row;
#elif defined(ONORM_SKIPFLAG)
        if (x < nr.cnt && !g_skip_st) *reinterpret_cast<uint2*>(na.out + (row * HV + h) * VD + 4 * lane) = make_uint2(y01, y23);
#else
        if (x < nr.cnt) *reinterpret_cast<uint2*>(na.out + (row * HV + h) * VD + 4 * lane) = make_uint2(y01, y23);
#endif
    }
#ifdef ONORM_BULKST
    // ONORM_BULKST: the warp's finished rows overwrite their own o rows in the tile (natural column order; every lane's
    // reads of them are done: syncwarp), then one bulk async copy (TMA, 256 B) per row to the packed out row; the warp
    // never waits on global stores (the tile is reused only after cp.async.bulk.wait_group.read, before b3)
    __syncwarp();
#pragma unroll
    for (int x = 0; x < NR; ++x)
        if (x < nr.cnt)
            asm volatile("st.shared.v2.b32 [%0], {%1, %2};\n" ::"r"(tile + (nr.lo + warp + 8 * x) * 256 + 8 * lane),
                         "r"(yy[x][0]), "r"(yy[x][1]) : "memory");
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncwarp();
    if (lane == 0) {
#pragma unroll
        for (int x = 0; x < NR; ++x)
            if (x < nr.cnt) {
                const size_t row = static_cast<size_t>(st + i * BT + nr.lo + warp + 8 * x - pl);
                asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 256;\n"
                             ::"l"(na.out + (row * HV + h) * VD), "r"(tile + (nr.lo + warp + 8 * x) * 256) : "memory");
            }
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
    }
#endif
}
#ifdef ONORM_FINROLL
// ONORM_FINROLL: rows 4 g .. 4 g + 3 of the warp, silu included; called from a rolled loop over g, so the whole per-chunk
// norm epilogue is one ~4-row code body (fetched once per launch: the unrolled 8-row body is cold instruction fetch on
// every launch in the engine, where other kernels evict the icache between GDN layers); EX: some z of the warp is
// outside the fast division's range -> the out-of-line div.rn silus
DEVI void n_finR(const NormArgs& na, uint32_t tile, int i, int h, int warp, int lane, int pl, int st, const NRows& nr,
                 int g, uint32_t w01, uint32_t w23, bool ex) {
    float f[4][4], s[4], sg[4][4];
    uint2 zl[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        zl[k].x = g ? nr.zv[4 + k].x : nr.zv[k].x;
        zl[k].y = g ? nr.zv[4 + k].y : nr.zv[k].y;
    }
#pragma unroll
    for (int x = 0; x < 4; ++x) {
        const int t = min(nr.lo + warp + 8 * (4 * g + x), BT - 1);
        uint32_t o01, o23;
        asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];\n" : "=r"(o01), "=r"(o23)
                     : "r"(tile + o256(t, lane >> 1) + 8 * (lane & 1)));
        f[x][0] = blo_f(o01), f[x][1] = bhi_f(o01), f[x][2] = blo_f(o23), f[x][3] = bhi_f(o23);
        s[x] = n_add(n_add(n_add(n_mul(f[x][0], f[x][0]), n_mul(f[x][1], f[x][1])), n_mul(f[x][2], f[x][2])),
                     n_mul(f[x][3], f[x][3]));
    }
    if (ex) {
#pragma unroll
        for (int x = 0; x < 4; ++x) {
            const float4 r4 = n_silu4_exact(zl[x].x, zl[x].y);
            sg[x][0] = r4.x, sg[x][1] = r4.y, sg[x][2] = r4.z, sg[x][3] = r4.w;
        }
    } else {
#pragma unroll
        for (int x = 0; x < 4; ++x) {
            const float zz[4] = {blo_f(zl[x].x), bhi_f(zl[x].x), blo_f(zl[x].y), bhi_f(zl[x].y)};
#pragma unroll
            for (int k = 0; k < 4; ++k) sg[x][k] = n_fdiv(zz[k], n_add(n_expf(n_sub(0.f, zz[k])), 1.f));
        }
    }
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1)
#pragma unroll
        for (int x = 0; x < 4; ++x) s[x] = n_add(s[x], __shfl_xor_sync(0xffffffffu, s[x], m));
#pragma unroll
    for (int x = 0; x < 4; ++x) {
        const float r = n_rsqrt(n_add(na.eps, n_mul(0.0078125f, s[x])));
        const uint32_t hw01 = n_bmul2(w01, pk(n_mul(r, f[x][0]), n_mul(r, f[x][1])));
        const uint32_t hw23 = n_bmul2(w23, pk(n_mul(r, f[x][2]), n_mul(r, f[x][3])));
        const uint32_t y01 = pk(n_mul(sg[x][0], blo_f(hw01)), n_mul(sg[x][1], bhi_f(hw01)));
        const uint32_t y23 = pk(n_mul(sg[x][2], blo_f(hw23)), n_mul(sg[x][3], bhi_f(hw23)));
        const size_t row = static_cast<size_t>(st + i * BT + nr.lo + warp + 8 * (4 * g + x) - pl);
        if (4 * g + x < nr.cnt)
            *reinterpret_cast<uint2*>(na.out + (row * HV + h) * VD + 4 * lane) = make_uint2(y01, y23);
    }
}
#endif
// rows 4 g .. 4 g + 3 of the warp (t clamped into the tile; stores only for rr < cnt)
DEVI void n_fin4(const NormArgs& na, uint32_t tile, int i, int h, int warp, int lane, int pl, int st, const NRows& nr,
                 int g, const float (&sg)[8][4], uint32_t w01, uint32_t w23) {
    float f[4][4], s[4];
#pragma unroll
    for (int x = 0; x < 4; ++x) {
        const int t = min(nr.lo + warp + 8 * (4 * g + x), BT - 1);
        uint32_t o01, o23;
        asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];\n" : "=r"(o01), "=r"(o23)
                     : "r"(tile + o256(t, lane >> 1) + 8 * (lane & 1)));
        f[x][0] = blo_f(o01), f[x][1] = bhi_f(o01), f[x][2] = blo_f(o23), f[x][3] = bhi_f(o23);
        s[x] = n_add(n_add(n_add(n_mul(f[x][0], f[x][0]), n_mul(f[x][1], f[x][1])), n_mul(f[x][2], f[x][2])),
                     n_mul(f[x][3], f[x][3]));
    }
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1)
#pragma unroll
        for (int x = 0; x < 4; ++x) s[x] = n_add(s[x], __shfl_xor_sync(0xffffffffu, s[x], m));
#pragma unroll
    for (int x = 0; x < 4; ++x) {
        const int rr = 4 * g + x;
        const float r = n_rsqrt(n_add(na.eps, n_mul(0.0078125f, s[x])));
        const uint32_t hw01 = n_bmul2(w01, pk(n_mul(r, f[x][0]), n_mul(r, f[x][1])));
        const uint32_t hw23 = n_bmul2(w23, pk(n_mul(r, f[x][2]), n_mul(r, f[x][3])));
        const uint32_t y01 = pk(n_mul(sg[rr][0], blo_f(hw01)), n_mul(sg[rr][1], bhi_f(hw01)));
        const uint32_t y23 = pk(n_mul(sg[rr][2], blo_f(hw23)), n_mul(sg[rr][3], bhi_f(hw23)));
        const size_t row = static_cast<size_t>(st + i * BT + nr.lo + warp + 8 * rr - pl);
        if (rr < nr.cnt) *reinterpret_cast<uint2*>(na.out + (row * HV + h) * VD + 4 * lane) = make_uint2(y01, y23);
    }
}

template <bool SKIP, bool TIMED, bool NORM = false>
DEVI void fused5_body(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                     const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ beta,
                     const __nv_bfloat16* __restrict__ A, const float* __restrict__ g, __nv_bfloat16* __restrict__ o,
                     const int* __restrict__ slen, int T, float scale, int hq = H, unsigned long long* ts = nullptr,
                     NormArgs na = NormArgs{}) {
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
    // ONORM: this sequence's packed rows (positions pl .. sl - 1 -> rows st ..), the norm weight of lane's 4 columns
    int n_pl = 0, n_sl = 0, n_st = 0;
    uint32_t n_w01 = 0, n_w23 = 0;
#ifdef ONORM_SMS
    // ONORM_SMS: this sequence's plen / start / slen and the norm weight staged in shared memory (F_N) once the first
    // inputs are in flight; read where used (no long-lived registers, no spill of a value waiting on a global load)
    int sms_pl = 0, sms_st = 0, sms_sl = 0;
    uint2 sms_w = make_uint2(0u, 0u);
    if (NORM) {
        if (threadIdx.x < 32) sms_w = *reinterpret_cast<const uint2*>(na.w + 4 * threadIdx.x);
        if (threadIdx.x == 32) sms_pl = na.plen[n], sms_st = na.start[n], sms_sl = slen[n];
    }
#else
    if (NORM) {
        n_pl = na.plen[n];
        n_sl = slen[n];
        n_st = na.start[n];
        const uint2 wv = *reinterpret_cast<const uint2*>(na.w + 4 * lane);
        n_w01 = wv.x;
        n_w23 = wv.y;
    }
#endif
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
#ifdef ONORM_WARM2
    if (NORM && c.tid < 16) {   // ONORM_WARM2: one 16-B read per 64 rows of this CTA's z and out ranges (TLB + L2 warm)
        const int nrow = n_sl - n_pl;
        if (nrow > 0) {
            const int r = min((c.tid & 7) * BT, nrow - 1);
            const void* src = c.tid < 8 ? static_cast<const void*>(na.z + static_cast<size_t>(n_st + r) * na.zs + h * VD)
                                        : static_cast<const void*>(na.out + (static_cast<size_t>(n_st + r) * HV + h) * VD);
            cp16(sb + F_SMEM + 16 * c.tid, src);
        }
    }
#endif
#ifdef ONORM_WARM
    if (NORM && c.tid < 2 * BT) {   // ONORM_WARM: z rows of chunk 0 fetched (TLB + L2) with k, q (2 KB smem scratch)
        const int t = c.tid >> 1, p = t;
        if (p >= n_pl && p < n_sl)
            cp16(sb + F_SMEM + 16 * c.tid,
                 na.z + static_cast<size_t>(n_st + p - n_pl) * na.zs + h * VD + 64 * (c.tid & 1));
    }
#endif
    cp_commit();

    float hacc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) hacc[j][r] = 0.f;

#ifdef ONORM_SMS
    if (NORM) {
        if (threadIdx.x < 32) asm volatile("st.shared.v2.b32 [%0], {%1, %2};\n" ::"r"(sb + F_N + 8 * threadIdx.x),
                                           "r"(sms_w.x), "r"(sms_w.y) : "memory");
        if (threadIdx.x == 32) asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %3};\n" ::"r"(sb + F_N + 256),
                                            "r"(sms_pl), "r"(sms_st), "r"(sms_sl) : "memory");
    }
#define N_PL lds_s32(sb + F_N + 256)
#define N_ST lds_s32(sb + F_N + 260)
#define N_SL lds_s32(sb + F_N + 264)
#else
#define N_PL n_pl
#define N_ST n_st
#define N_SL n_sl
#endif
    cp_wait<1>();
    __syncthreads();
    stamp();
    for (int i = 0; i < NT; ++i) {
#ifdef ONORM_PFL2N
        if (NORM) {   // ONORM_PFL2N: z rows of chunk i + 1 (and of chunk 0 at i = 0) -> L2, a whole chunk ahead
            const int t = c.tid >> 1, pl_ = N_PL, sl_ = N_SL;
            const int p = (i + (c.tid >= 2 * BT ? 0 : 1)) * BT + t - (c.tid >= 2 * BT ? BT : 0);
            if ((c.tid < 2 * BT || i == 0) && p >= pl_ && p < sl_) {
                const __nv_bfloat16* zp =
                    na.z + static_cast<size_t>(N_ST + p - pl_) * na.zs + h * VD + 64 * (c.tid & 1);
                asm volatile("prefetch.global.L2 [%0];\n" ::"l"(zp));
            }
        }
#endif
#ifdef ONORM_ZEARLY
        NRows nr_e;   // ONORM_ZEARLY: z of chunk i's rows loaded at the top of the chunk (registers through P, A, C)
        if (NORM) n_prep(na, i, h, c.warp, lane, N_PL, N_SL, N_ST, nr_e);
#endif
#ifdef ONORM_PF2
        if (NORM && c.tid < 2 * BT) {   // ONORM_PF2: z rows of chunk i -> L1, P(i) + A(i) ahead of their first use
            const int t = c.tid >> 1, p = i * BT + t, pl_ = N_PL;
            if (p >= pl_ && p < N_SL) {
                const __nv_bfloat16* zp =
                    na.z + static_cast<size_t>(N_ST + p - pl_) * na.zs + h * VD + 64 * (c.tid & 1);
                asm volatile("prefetch.global.L1 [%0];\n" ::"l"(zp));
            }
        }
#endif
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
#ifdef ONORM_PF
        if (NORM && c.tid < 2 * BT) {   // ONORM: z rows of chunk i into L1 (read after bN)
            const int t = c.tid >> 1, p = i * BT + t;
            if (p >= n_pl && p < n_sl) {
                const __nv_bfloat16* zp = na.z + static_cast<size_t>(n_st + p - n_pl) * na.zs + h * VD + 64 * (c.tid & 1);
                asm volatile("prefetch.global.L1 [%0];\n" ::"l"(zp));
            }
        }
#endif
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
                    if (NORM) {   // ONORM: o tile of chunk i into the free w buffer (64 rows x 256 B, swizzled)
                        const int t = 8 * j + gq;
                        sts32(sb + F_W + o256(t, 2 * c.warp) + 4 * cq, m0);
                        sts32(sb + F_W + o256(t, 2 * c.warp + 1) + 4 * cq, m1);
                    } else {
                        const size_t row = r0 + 8 * j + gq;
                        *reinterpret_cast<uint32_t*>(o + (row * HV + h) * VD + v0 + 2 * cq) = m0;
                        *reinterpret_cast<uint32_t*>(o + (row * HV + h) * VD + v0 + 8 + 2 * cq) = m1;
                    }
                }
            };
#if ONORM_V == 5
            if (NORM) {   // ONORM V5: B(i) first (o tile -> w buffer); silu(z) inside C(i)'s mma stream; bN; norm
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) b_pair(jj);
#ifdef ONORM_FINESTAMP
                stamp();
#endif
#ifdef ONORM_ZEARLY
                const NRows& nr = nr_e;
#else
                NRows nr;
                n_prep(na, i, h, c.warp, lane, N_PL, N_SL, N_ST, nr);
#endif
                float sg[8][4];
                bool spec = false;
#ifdef ONORM_FINESTAMP
                stamp();
#endif
                if (i + 1 < NT) {
                    const float gl = gs[63];
                    const float el = ex2(gl);
                    if (i > 0) {
#pragma unroll
                        for (int j = 0; j < 16; ++j)
#pragma unroll
                            for (int r = 0; r < 4; ++r) hacc[j][r] = fmul(hacc[j][r], el);
                    } else {
                        const float zz = fmul(el, 0.f);
#pragma unroll
                        for (int j = 0; j < 16; ++j)
#pragma unroll
                            for (int r = 0; r < 4; ++r) hacc[j][r] = zz;
                    }
#pragma unroll
                    for (int a = 0; a < 4; ++a) {
                        const float f00 = ex2(fsub(gl, gs[16 * a + 2 * cq])),
                                    f01 = ex2(fsub(gl, gs[16 * a + 2 * cq + 1]));
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
#ifndef ONORM_SILU_FIN
                            if (jj == 2) n_silu(nr, 2 * a, sg, spec);
                            if (jj == 5) n_silu(nr, 2 * a + 1, sg, spec);
#endif
                        }
                        f_issue_q_part(c, i + 1, a);   // q of chunk i + 1 (q buffer free since b2)
                    }
                } else {
#ifndef ONORM_SILU_FIN
#pragma unroll
                    for (int g4 = 0; g4 < 2; ++g4)
                        if (4 * g4 < nr.cnt) {
#pragma unroll
                            for (int x = 0; x < 4; ++x) n_silu(nr, 4 * g4 + x, sg, spec);
                        }
#endif
                }
#ifdef ONORM_FINESTAMP
                stamp();
#endif
#ifdef ONORM_SILU_FIN
#pragma unroll
                for (int x = 0; x < 8; ++x)
                    spec |= n_special(blo_f(nr.zv[x].x)) | n_special(bhi_f(nr.zv[x].x)) |
                            n_special(blo_f(nr.zv[x].y)) | n_special(bhi_f(nr.zv[x].y));
                const bool n_ex = __any_sync(0xffffffffu, spec);
#else
                if (__any_sync(0xffffffffu, spec)) n_silu_exact(na, i, h, c.warp, lane, N_PL, N_ST, nr, sg);
#endif
                __syncthreads();   // bN: chunk i's o tile complete (w buffer is free from b2 to b3)
#ifdef ONORM_FINESTAMP
                stamp();
#endif
#if !defined(ONORM_DIAG) || ONORM_DIAG < 2
#ifdef ONORM_SMS
                {
                    uint32_t w01, w23;
                    asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];\n" : "=r"(w01), "=r"(w23) : "r"(sb + F_N + 8 * lane));
                    const int pl_ = N_PL, st_ = N_ST;
#ifdef ONORM_FIN8
                    if (nr.cnt > 4) n_finN<8>(na, sb + F_W, i, h, c.warp, lane, pl_, st_, nr, sg, w01, w23);
                    else if (nr.cnt > 0) n_finN<4>(na, sb + F_W, i, h, c.warp, lane, pl_, st_, nr, sg, w01, w23);
#else
#pragma unroll
                    for (int g4 = 0; g4 < 2; ++g4)
                        if (4 * g4 < nr.cnt) n_fin4(na, sb + F_W, i, h, c.warp, lane, pl_, st_, nr, g4, sg, w01, w23);
#endif
                }
#else
#if defined(ONORM_FINROLL)
#pragma unroll 1
                for (int g4 = 0; g4 < 2; ++g4) {
                    if (4 * g4 >= nr.cnt) break;
                    n_finR(na, sb + F_W, i, h, c.warp, lane, n_pl, n_st, nr, g4, n_w01, n_w23, n_ex);
                }
#elif defined(ONORM_SILU_FIN)
                if (nr.cnt > 4) {
                    if (n_ex) n_finS<8, true>(na, sb + F_W, i, h, c.warp, lane, n_pl, n_st, nr, n_w01, n_w23);
                    else n_finS<8, false>(na, sb + F_W, i, h, c.warp, lane, n_pl, n_st, nr, n_w01, n_w23);
                } else if (nr.cnt > 0) {
                    if (n_ex) n_finS<4, true>(na, sb + F_W, i, h, c.warp, lane, n_pl, n_st, nr, n_w01, n_w23);
                    else n_finS<4, false>(na, sb + F_W, i, h, c.warp, lane, n_pl, n_st, nr, n_w01, n_w23);
                }
#elif defined(ONORM_FIN8)
                if (nr.cnt > 4) n_finN<8>(na, sb + F_W, i, h, c.warp, lane, n_pl, n_st, nr, sg, n_w01, n_w23);
                else if (nr.cnt > 0) n_finN<4>(na, sb + F_W, i, h, c.warp, lane, n_pl, n_st, nr, sg, n_w01, n_w23);
#else
#pragma unroll
                for (int g4 = 0; g4 < 2; ++g4)
                    if (4 * g4 < nr.cnt) n_fin4(na, sb + F_W, i, h, c.warp, lane, n_pl, n_st, nr, g4, sg, n_w01, n_w23);
#endif
#endif
#else
                if (nr.cnt > 0 && lane == 0 && sg[0][0] == 12345.f) na.out[0] = __float2bfloat16(sg[1][1]);
#endif
            } else
#endif
#if ONORM_V == 3
            if (NORM) {   // ONORM V3: B(i) first (o tile -> w buffer), bN, then C(i) with the norm rows interleaved
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) b_pair(jj);
                NRows nr;
                n_prep(na, i, h, c.warp, lane, n_pl, n_sl, n_st, nr);
                __syncthreads();   // bN: chunk i's o tile complete (w buffer is free from b2 to b3)
                if (i + 1 < NT) {
                    const float gl = gs[63];
                    const float el = ex2(gl);
                    if (i > 0) {
#pragma unroll
                        for (int j = 0; j < 16; ++j)
#pragma unroll
                            for (int r = 0; r < 4; ++r) hacc[j][r] = fmul(hacc[j][r], el);
                    } else {
                        const float zz = fmul(el, 0.f);
#pragma unroll
                        for (int j = 0; j < 16; ++j)
#pragma unroll
                            for (int r = 0; r < 4; ++r) hacc[j][r] = zz;
                    }
#pragma unroll
                    for (int a = 0; a < 4; ++a) {
                        const float f00 = ex2(fsub(gl, gs[16 * a + 2 * cq])),
                                    f01 = ex2(fsub(gl, gs[16 * a + 2 * cq + 1]));
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
                            if (jj == 3 && 2 * a < nr.cnt)
                                n_row(na, sb + F_W, i, h, c.warp, lane, n_pl, n_st, nr, 2 * a, n_w01, n_w23);
                        }
                        if (2 * a + 1 < nr.cnt)
                            n_row(na, sb + F_W, i, h, c.warp, lane, n_pl, n_st, nr, 2 * a + 1, n_w01, n_w23);
                        f_issue_q_part(c, i + 1, a);   // q of chunk i + 1 (q buffer free since b2)
                    }
                } else {
#pragma unroll
                    for (int rr = 0; rr < 8; ++rr)
                        if (rr < nr.cnt) n_row(na, sb + F_W, i, h, c.warp, lane, n_pl, n_st, nr, rr, n_w01, n_w23);
                }
            } else
#endif
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
#if ONORM_V == 1
        if (NORM) {
            __syncthreads();   // bN: chunk i's o tile complete (w buffer is free from b2 to b3)
            n_rows(na, sb + F_W, i, h, c.warp, lane, n_pl, n_sl, n_st, n_w01, n_w23, 0.0078125f);
        }
#elif ONORM_V == 2
        if (NORM) {
            NRows nr;
            n_prep(na, i, h, c.warp, lane, n_pl, n_sl, n_st, nr);
            __syncthreads();   // bN: chunk i's o tile complete (w buffer is free from b2 to b3)
#pragma unroll
            for (int rr = 0; rr < 8; ++rr)
                if (rr < nr.cnt) n_row(na, sb + F_W, i, h, c.warp, lane, n_pl, n_st, nr, rr, n_w01, n_w23);
        }
#endif
        stamp();
#ifdef ONORM_BULKST
        if (NORM && lane == 0) asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");   // tile reads done
#endif
        cp_wait<0>();
        __syncthreads();   // b3: chunk i + 1 inputs landed; every warp is done with w, Aq, k of chunk i
        stamp();
    }
#ifdef ONORM_BULKST
    if (NORM && lane == 0) asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");   // out rows written
#endif
}
#undef N_PL
#undef N_ST
#undef N_SL

// ---------------------------------------------------------------- ONORM V6: chunk i = P(i) [+ norm of chunk i - 1
// interleaved] | b1 | A(i) | b2 | C(i) [+ silu(z) of chunk i's rows interleaved] | b2' | B(i) (o tile -> k buffer of
// chunk i, free once every warp passed b2') | b3. The norm of chunk i - 1 reads its o tile in P(i) (k(i + 1) is loaded
// into that buffer only after b1(i)); the last chunk's norm runs after its b3.
DEVI void fused6n_body(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                       const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ beta,
                       const __nv_bfloat16* __restrict__ A, const float* __restrict__ g, const int* __restrict__ slen,
                       int T, float scale, int hq, const NormArgs na) {
    extern __shared__ __align__(128) uint8_t smem[];
    const int n = blockIdx.x / HV, h = blockIdx.x % HV;
    if (slen[n] == 0) return;
    const FCtx c = f_ctx(q, k, v, beta, A, g, n, h, T, smem, hq);
    const int NT = T / BT;
    const int lane = c.lane, lm = c.lm, lr = c.lr, cq = c.cq, gq = c.gq;
    const int ab_t = 8 * (lm >> 1) + lr, ab_c = lm & 1;
    const int c_t = 8 * (lm & 1) + lr, c_c = lm >> 1;
    const uint32_t sb = c.sb;
    const int n_pl = na.plen[n], n_sl = slen[n], n_st = na.start[n];
    uint32_t n_w01, n_w23;
    {
        const uint2 wv = *reinterpret_cast<const uint2*>(na.w + 4 * lane);
        n_w01 = wv.x;
        n_w23 = wv.y;
    }

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
    NRows nr;            // rows / z of the chunk being finished
    float sg[8][4];      // silu(z) of those rows
    nr.lo = 0;
    nr.cnt = 0;

    cp_wait<1>();
    __syncthreads();
    for (int i = 0; i < NT; ++i) {
        const uint32_t kbuf = sb + F_K0 + (i & 1) * 16384;
        const uint32_t kbn = sb + F_K0 + ((i + 1) & 1) * 16384;   // = the o tile of chunk i - 1
        const float* gs = reinterpret_cast<const float*>(smem + F_G + (i & 1) * 256);
        const uint32_t* bs = reinterpret_cast<const uint32_t*>(smem + F_B + (i & 1) * 256);
        float vn[8][4];
        float e[8][2];
        f_exps(gs, cq, e);
        // ------------------------------------------------ P(i) [+ norm of chunk i - 1: groups of 4 rows after mt 1, 3]
        if (i > 0) {
            const int ip = i - 1;
            f_pstage3h<true, false>(c, kbuf, gs, bs, e, vn, [&](int mt) {
                if (mt == 1) n_fin4(na, kbn, ip, h, c.warp, lane, n_pl, n_st, nr, 0, sg, n_w01, n_w23);
                if (mt == 3) n_fin4(na, kbn, ip, h, c.warp, lane, n_pl, n_st, nr, 1, sg, n_w01, n_w23);
            });
        } else {
            f_pstage3h<false, true>(c, kbuf, gs, bs, e, vn, [&](int) {});
        }
        __syncthreads();   // b1: w, Aq of chunk i in smem; A, v of chunk i no longer read; chunk i - 1's o tile read
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
                if (s < 6 && i + 1 < NT) f_issue_x_part(c, i + 1, kbn, s);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int r = 0; r < 4; ++r) vn[j][r] = fsub(vn[j][r], acc[j][r]);
        }
        cp_commit();
        __syncthreads();   // b2: every warp is done with q of chunk i
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
        uint32_t vs[4][4];
#pragma unroll
        for (int a = 0; a < 4; ++a) {
            vs[a][0] = pk(vn[2 * a][0], vn[2 * a][1]);
            vs[a][1] = pk(vn[2 * a][2], vn[2 * a][3]);
            vs[a][2] = pk(vn[2 * a + 1][0], vn[2 * a + 1][1]);
            vs[a][3] = pk(vn[2 * a + 1][2], vn[2 * a + 1][3]);
        }
        n_prep(na, i, h, c.warp, lane, n_pl, n_sl, n_st, nr);
        bool spec = false;
        if (i + 1 < NT) {
            // -------------------------------------------- C(i) (h update) + silu(z) of chunk i's rows
            const float gl = gs[63];
            const float el = ex2(gl);
            if (i > 0) {
#pragma unroll
                for (int j = 0; j < 16; ++j)
#pragma unroll
                    for (int r = 0; r < 4; ++r) hacc[j][r] = fmul(hacc[j][r], el);
            } else {
                const float zz = fmul(el, 0.f);
#pragma unroll
                for (int j = 0; j < 16; ++j)
#pragma unroll
                    for (int r = 0; r < 4; ++r) hacc[j][r] = zz;
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
                    if (jj == 2) n_silu(nr, 2 * a, sg, spec);
                    if (jj == 5) n_silu(nr, 2 * a + 1, sg, spec);
                }
                f_issue_q_part(c, i + 1, a);   // q of chunk i + 1 (q buffer free since b2)
            }
            __syncthreads();   // b2': every warp is done with k(i) (C(i)): its buffer takes chunk i's o tile
        } else {
#pragma unroll
            for (int rr = 0; rr < 8; ++rr) n_silu(nr, rr, sg, spec);
        }
        if (__any_sync(0xffffffffu, spec)) n_silu_exact(na, i, h, c.warp, lane, n_pl, n_st, nr, sg);
        // ------------------------------------------------ B(i): o -> the o tile in kbuf (64 rows x 256 B, swizzled)
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            float d2[2][4];
#pragma unroll
            for (int x = 0; x < 2; ++x)
#pragma unroll
                for (int r = 0; r < 4; ++r) d2[x][r] = 0.f;
#pragma unroll
            for (int a = 0; a <= jj; ++a) {
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
                const int t = 8 * j + gq;
                sts32(kbuf + o256(t, 2 * c.warp) + 4 * cq, m0);
                sts32(kbuf + o256(t, 2 * c.warp + 1) + 4 * cq, m1);
            }
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();   // b3: chunk i + 1 inputs landed; chunk i's o tile complete; w, Aq of chunk i no longer read
    }
    {   // the last chunk's norm (its o tile is in the k buffer of chunk NT - 1)
        const int il = NT - 1;
        const uint32_t kl = sb + F_K0 + (il & 1) * 16384;
#pragma unroll
        for (int g4 = 0; g4 < 2; ++g4)
            if (4 * g4 < nr.cnt) n_fin4(na, kl, il, h, c.warp, lane, n_pl, n_st, nr, g4, sg, n_w01, n_w23);
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

#ifndef FFS_SHIP
extern "C" __global__ void __launch_bounds__(256, 1)
    ffs_fused(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* beta,
              const __nv_bfloat16* A, const float* g, __nv_bfloat16* o, const int* slen, int T, float scale) {
    fused_body<true, false>(q, k, v, beta, A, g, o, slen, T, scale);
}
#endif

#ifndef FFS_SHIP
extern "C" __global__ void __launch_bounds__(256, 1)
    ffs_fused_ts(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* beta,
                 const __nv_bfloat16* A, const float* g, __nv_bfloat16* o, const int* slen, int T, float scale,
                 unsigned long long* ts) {
    fused_body<true, true>(q, k, v, beta, A, g, o, slen, T, scale, ts);
}
#endif

#define SEQ_KERNEL(NAME, WPC, SKIP)                                                                                  \
    extern "C" __global__ void __launch_bounds__(WPC * 32, 1)                                                        \
        NAME(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* w, const __nv_bfloat16* u,         \
             const __nv_bfloat16* Aq, const float* g, __nv_bfloat16* o, const int* slen, int T, float scale) {       \
        seq_body<WPC, SKIP>(q, k, w, u, Aq, g, o, slen, T, scale);                                                     \
    }

#ifndef FFS_SHIP
SEQ_KERNEL(ffs_seq_w4, 4, false)
#endif
#ifndef FFS_SHIP
SEQ_KERNEL(ffs_seq_w4s, 4, true)
#endif
#ifndef FFS_SHIP
SEQ_KERNEL(ffs_seq_w8, 8, false)
#endif
#ifndef FFS_SHIP
SEQ_KERNEL(ffs_seq_w8s, 8, true)
#endif
#ifndef FFS_SHIP
SEQ_KERNEL(ffs_seq_w2s, 2, true)
#endif

#ifndef FFS_SHIP
extern "C" __global__ void __launch_bounds__(256, 1)
    ffs_fused2(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* beta,
               const __nv_bfloat16* A, const float* g, __nv_bfloat16* o, const int* slen, int T, float scale) {
    fused2_body<true, false>(q, k, v, beta, A, g, o, slen, T, scale);
}
#endif

#ifndef FFS_SHIP
extern "C" __global__ void __launch_bounds__(256, 1)
    ffs_fused2_ts(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* beta,
                  const __nv_bfloat16* A, const float* g, __nv_bfloat16* o, const int* slen, int T, float scale,
                  unsigned long long* ts) {
    fused2_body<true, true>(q, k, v, beta, A, g, o, slen, T, scale, ts);
}
#endif

#ifndef FFS_SHIP
extern "C" __global__ void __launch_bounds__(256, 1)
    ffs_fused3(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* beta,
               const __nv_bfloat16* A, const float* g, __nv_bfloat16* o, const int* slen, int T, float scale, int hq) {
    fused3_body<true, false>(q, k, v, beta, A, g, o, slen, T, scale, hq);
}
#endif

#ifndef FFS_SHIP
extern "C" __global__ void __launch_bounds__(256, 1)
    ffs_fused3_ts(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* beta,
                  const __nv_bfloat16* A, const float* g, __nv_bfloat16* o, const int* slen, int T, float scale,
                  unsigned long long* ts) {
    fused3_body<true, true>(q, k, v, beta, A, g, o, slen, T, scale, H, ts);
}
#endif

#ifndef FFS_SHIP
extern "C" __global__ void __launch_bounds__(256, 1)
    ffs_fused4(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* beta,
               const __nv_bfloat16* A, const float* g, __nv_bfloat16* o, const int* slen, int T, float scale, int hq) {
    fused4_body<false>(q, k, v, beta, A, g, o, slen, T, scale, hq);
}
#endif

#ifndef FFS_SHIP
extern "C" __global__ void __launch_bounds__(256, 1)
    ffs_fused4_ts(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* beta,
                  const __nv_bfloat16* A, const float* g, __nv_bfloat16* o, const int* slen, int T, float scale,
                  unsigned long long* ts) {
    fused4_body<true>(q, k, v, beta, A, g, o, slen, T, scale, H, ts);
}
#endif

extern "C" __global__ void __launch_bounds__(256, 1)
    ffs_fused5(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* beta,
               const __nv_bfloat16* A, const float* g, __nv_bfloat16* o, const int* slen, int T, float scale, int hq) {
    fused5_body<true, false>(q, k, v, beta, A, g, o, slen, T, scale, hq);
}

extern "C" __global__ void __launch_bounds__(256, 1)
    ffs_fused6n(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* beta,
                const __nv_bfloat16* A, const float* g, const int* slen, int T, float scale, int hq,
                const __nv_bfloat16* z, int zs, const __nv_bfloat16* w, float eps, const int* start, const int* plen,
                __nv_bfloat16* out) {
    fused6n_body(q, k, v, beta, A, g, slen, T, scale, hq, NormArgs{z, zs, w, eps, start, plen, out});
}

// ONORM: ffs_fused5 + the gated RMSNorm of fused.py _gated_norm_packed_kernel; out = [R * 32, 128] packed rows
// (positions plen[n] .. slen[n] - 1 of sequence n at rows start[n] ..); no padded o is written
extern "C" __global__ void __launch_bounds__(256, 1)
    ffs_fused5n(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* beta,
                const __nv_bfloat16* A, const float* g, const int* slen, int T, float scale, int hq,
                const __nv_bfloat16* z, int zs, const __nv_bfloat16* w, float eps, const int* start, const int* plen,
                __nv_bfloat16* out) {
    fused5_body<true, false, true>(q, k, v, beta, A, g, nullptr, slen, T, scale, hq, nullptr,
                                   NormArgs{z, zs, w, eps, start, plen, out});
}
