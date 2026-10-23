// onormpv.cu - PROFUSE-v on ONORM (ruj, r3 2026-10-23): ffs_pv5n = ONORM's ffs_fused5n (onorm.cu, rgh's
// onorm_v17, included below byte for byte unchanged) with the GDN value rows v computed INSIDE the kernel from the packed
// qkv GEMM rows, so the GDN prologue (pron.prologue(qk_only=True): PRON's q / k head blocks and gate units only) no
// longer writes v and nothing reads v from global memory.
//
// Credits: the v arithmetic in a FFS kernel, the raw-row / halo staging (pv_issue_raw, pv_convert) and the alternating
// R / V buffer scheme are rgh's dev_profuse ffsp.cu (ffs_pv5, built on ffs.cu's fused5 body, never shipped); this file
// ports them onto ONORM's fused5n body (fused5_body<SKIP = true, TIMED = false, NORM = true> with onorm.cubin's flags
// ONORM_V=5 ONORM_PF2 ONORM_SILU_FIN ONORM_FINROLL), which the engine runs for packed graphs.
//
// v (PRON's conv_tile4 in pron.cu, op for op; pron.cubin is unchanged and still computes q / k):
//   raw row x_p of position p of slot n = the prefix-variant cache row pvar[n] * pmax + p for p < plen[n] (pfx > 0),
//   else the packed row start[n] + p - plen[n]; +0 for p < 0 and for p >= slen[n]
//   acc = +0; acc = fma.rn.f32.bf16(x_{t-3+kk}[c], w[c][kk], acc) for kk = 0..3 (FHFMA, exact bf16 products)
//   v_t[c] = bf16_rn(silu_f9(f32(bf16_rn(acc))))  (silu_f9 = PRON's F9 formula); v_t = +0 for t >= slen[n]
// Schedule per chunk i (64 rows): the raw rows of positions 64 i .. 64 i + 63 land in R_i (swizzled 64 x 256 B, as v),
// positions 64 i - 3 .. 64 i - 1 in the halo (3 x 256 B); rows of positions p < 0 or p >= slen are zero-filled
// (cp.async src-size 0, no global read). X(i) converts R_i + halo -> V_i at the top of chunk i by column slab (warp wi: v columns 16 wi .. 16 wi + 15 for all rows - exactly the columns
// its own P(i) reads and its own w columns overwrite in R_i - so a __syncwarp replaces a CTA barrier); P(i) reads v from
// V_i and writes w to W_i = R_i; B(i) stages the o tile in W_i (w(i) is consumed by A(i) before b2, as in ONORM where the
// o tile goes to the w buffer); raw(i + 1) is issued during A(i) into V_i (free after P(i), b1); R_i = (i even) ? F_W :
// F_V, V_i = the other one. Everything else (P, A, B, C, the z prefetch, the gated norm epilogue, the barriers b1 / b2 /
// bN / b3, every arithmetic line) is fused5_body's NORM path as onorm.cubin compiles it. The per-CTA raw-row constants
// are ints; pointers and row strides are read from the kernel parameters (no spill beyond ffs_fused5n's 8 bytes).
// Dynamic smem 99328 + 768 (halo) + 1024 (conv weights) = 101120 B.
//
// Kernels: ffs_pv5n (the lever) and ffs_pvx (start-up gate helper: the same raw-row staging + conversion per chunk,
// v written to global memory, for the byte check against pron.prologue's v). onorm.cu's own kernels are compiled into
// this cubin as well (unused; the engine loads them from onorm.cubin).
// Build (in /opt/s1eng/s1eng): nvcc -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -DFFS_SHIP -DONORM_V=5 -DONORM_PF2
//   -DONORM_SILU_FIN -DONORM_FINROLL -o onormpv.cubin onormpv.cu   (module id = CRC32 of /opt/s1eng/s1eng/onormpv.cu)
#include "onorm.cu"

namespace {
constexpr int F_HALO = F_SMEM, F_CW = F_SMEM + 768, F_SMEM_PV = F_SMEM + 768 + 1024;

struct PVNArgs {
    const __nv_bfloat16 *q, *k, *beta, *A;     // PRON (qk_only) q / k / beta, kkt2n's A
    const float* g;                            // PRON's chunk-cumulated g
    const int *slen, *start, *plen, *pvar;     // packed layout (plen / pvar: prefix variants, read only if pfx)
    const __nv_bfloat16 *z, *nw;               // ONORM: packed z rows (row stride zs), norm weight [128]
    __nv_bfloat16* out;                        // ONORM: [R * 32, 128] packed gated-norm rows
    const __nv_bfloat16 *x, *cx, *cw;          // packed qkv GEMM rows (row stride XS), prefix cache rows (CXS), conv w [C, 4]
    int T, hq, zs, XS, CXS, pfx, pmax;
    float scale, eps;
};

DEVI void cp16z(uint32_t dst, const void* src, int bytes) {   // cp.async 16 B, bytes past `bytes` zero-filled
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
DEVI uint4 lds16(uint32_t a) {
    uint4 r;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(a));
    return r;
}
DEVI void sts16(uint32_t a, uint4 v) {
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
DEVI float pv_mul(float a, float b) { float r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
DEVI float pv_ex2(float x) { float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
DEVI float pv_rcp(float x) { float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
DEVI float silu_f9(float cb) {   // pron.cu silu_f9_raw (exhaustively equal to the Triton chain after bf16 rounding)
    const float e = pv_ex2(pv_mul(cb, -1.4426950408889634f));
    return pv_mul(pv_mul(cb, 0.0625f), pv_rcp(__fmaf_rn(e, 0.0625f, 0.0625f)));
}
// acc = fma.rn.f32.bf16(x half XH of xw, w half WH of ww, acc) (pron.cu fmab: FHFMA, no unpack)
template <int XH, int WH>
DEVI float fmab(uint32_t xw, uint32_t ww, float acc) {
    float r;
    if (XH == 0 && WH == 0)
        asm("{\n\t.reg .b16 xl, xh, wl, wh;\n\tmov.b32 {xl, xh}, %1;\n\tmov.b32 {wl, wh}, %2;\n\t"
            "fma.rn.f32.bf16 %0, xl, wl, %3;\n\t}" : "=f"(r) : "r"(xw), "r"(ww), "f"(acc));
    else if (XH == 0 && WH == 1)
        asm("{\n\t.reg .b16 xl, xh, wl, wh;\n\tmov.b32 {xl, xh}, %1;\n\tmov.b32 {wl, wh}, %2;\n\t"
            "fma.rn.f32.bf16 %0, xl, wh, %3;\n\t}" : "=f"(r) : "r"(xw), "r"(ww), "f"(acc));
    else if (XH == 1 && WH == 0)
        asm("{\n\t.reg .b16 xl, xh, wl, wh;\n\tmov.b32 {xl, xh}, %1;\n\tmov.b32 {wl, wh}, %2;\n\t"
            "fma.rn.f32.bf16 %0, xh, wl, %3;\n\t}" : "=f"(r) : "r"(xw), "r"(ww), "f"(acc));
    else
        asm("{\n\t.reg .b16 xl, xh, wl, wh;\n\tmov.b32 {xl, xh}, %1;\n\tmov.b32 {wl, wh}, %2;\n\t"
            "fma.rn.f32.bf16 %0, xh, wh, %3;\n\t}" : "=f"(r) : "r"(xw), "r"(ww), "f"(acc));
    return r;
}

struct PVSeq {   // per-CTA constants of the raw-row addressing, ints only (the pointers and row strides are read
    int xr, cr, n, pl, col0;   // from the kernel parameters where used); xr = start - plen, cr = pvar * pmax
};

DEVI PVSeq pv_seq(const PVNArgs& P, int n, int h) {
    PVSeq s;
    s.col0 = 2 * KD * H + h * VD;   // v column of value head h in the packed qkv row (q | k | v)
    s.pl = P.pfx ? P.plen[n] : 0;
    s.xr = P.start[n] - s.pl;
    s.cr = (P.pfx ? P.pvar[n] : 0) * P.pmax;
    s.n = P.slen[n];
    return s;
}

// raw rows of chunk i: part 0 = rows 0..31, part 1 = rows 32..63 + the 3 halo rows (positions 64 i - 3 ..). Rows of
// positions p < 0 or p >= slen are zero-filled (cp.async with src-size 0: no global read; the source operand is then
// the always valid P.cx), as PRON's raw rows are +0 there.
DEVI void pv_issue_raw(const FCtx& c, const PVNArgs& P, const PVSeq& s, int i, uint32_t rbuf, int part) {
    const int lo = part == 0 ? 0 : 512, hi = part == 0 ? 512 : 1024 + 48;
    for (int x = lo + c.tid; x < hi; x += 256) {
        int p, ch;
        uint32_t dst;
        if (x < 1024) {
            const int rr = x >> 4;
            ch = x & 15;
            p = i * BT + rr;
            dst = rbuf + o256(rr, ch);
        } else {
            const int y = x - 1024, hh = y >> 4;
            ch = y & 15;
            p = i * BT - 3 + hh;
            dst = c.sb + F_HALO + hh * 256 + ch * 16;
        }
        const bool ok = p >= 0 && p < s.n;
        const __nv_bfloat16* src = p < s.pl ? P.cx + static_cast<long long>(s.cr + p) * P.CXS
                                            : P.x + static_cast<long long>(s.xr + p) * P.XS;
        cp16z(dst, ok ? static_cast<const void*>(src + s.col0 + ch * 8) : static_cast<const void*>(P.cx), ok ? 16 : 0);
    }
}


// X(i): raw rows (rbuf + halo) -> v (vbuf, swizzled as F_V) by column slab: warp wi converts v columns 16 wi ..
// 16 wi + 15 - the columns its own P(i) reads and its own w columns overwrite in rbuf - for all 64 rows, so X(i) -> P(i)
// needs only __syncwarp, no CTA barrier; lane: column chunk j = 2 wi + (lane & 1), rows 4 rg .. 4 rg + 3 (rg = lane / 2).
// Per element PRON's conv_tile4 arithmetic (header).
DEVI void pv_convert(const FCtx& c, int i, int n, uint32_t rbuf, uint32_t vbuf) {
    const int j = 2 * c.warp + (c.lane & 1), rg = c.lane >> 1;
    uint32_t ww[16];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const uint4 x = lds16(c.sb + F_CW + j * 64 + u * 16);
        ww[4 * u] = x.x; ww[4 * u + 1] = x.y; ww[4 * u + 2] = x.z; ww[4 * u + 3] = x.w;
    }
    uint32_t raw[7][4];
#pragma unroll
    for (int r = 0; r < 7; ++r) {
        const int rr = 4 * rg - 3 + r;
        const uint32_t a = rr >= 0 ? rbuf + o256(rr, j) : c.sb + F_HALO + (rr + 3) * 256 + j * 16;
        const uint4 v = lds16(a);
        raw[r][0] = v.x; raw[r][1] = v.y; raw[r][2] = v.z; raw[r][3] = v.w;
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        float acc[8];
#pragma unroll
        for (int cc = 0; cc < 8; ++cc) acc[cc] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
            for (int c2 = 0; c2 < 4; ++c2) {
                const uint32_t xw = raw[r + kk][c2];
                if (kk & 1) {
                    acc[2 * c2] = fmab<0, 1>(xw, ww[2 * (2 * c2) + (kk >> 1)], acc[2 * c2]);
                    acc[2 * c2 + 1] = fmab<1, 1>(xw, ww[2 * (2 * c2 + 1) + (kk >> 1)], acc[2 * c2 + 1]);
                } else {
                    acc[2 * c2] = fmab<0, 0>(xw, ww[2 * (2 * c2) + (kk >> 1)], acc[2 * c2]);
                    acc[2 * c2 + 1] = fmab<1, 0>(xw, ww[2 * (2 * c2 + 1) + (kk >> 1)], acc[2 * c2 + 1]);
                }
            }
        }
        uint32_t yp[4];
#pragma unroll
        for (int cc = 0; cc < 8; cc += 2) {
            const uint32_t cbp2 = pk(acc[cc], acc[cc + 1]);
            yp[cc >> 1] = pk(silu_f9(blo_f(cbp2)), silu_f9(bhi_f(cbp2)));
        }
        uint4 o = make_uint4(yp[0], yp[1], yp[2], yp[3]);
        if (i * BT + 4 * rg + r >= n) o = make_uint4(0u, 0u, 0u, 0u);
        sts16(vbuf + o256(4 * rg + r, j), o);
    }
    __syncwarp();
}

// f_pstage3 (onorm.cu) with the v / w buffers as arguments (every other line as there)
template <bool DO_W, bool FIRST>
DEVI void pv_pstage(const FCtx& c, uint32_t kbuf, uint32_t vbuf, uint32_t wbuf, const float* gs, const uint32_t* bs,
                    const float (&e)[8][2], float (&uf)[8][4]) {
    const int lane = c.lane, lm = c.lm, lr = c.lr, cq = c.cq, wi = c.warp;
    const int c_t = 8 * (lm & 1) + lr, c_c = lm >> 1;
    const int ab_t = 8 * (lm >> 1) + lr, ab_c = lm & 1;
    uint32_t vb[4][4];
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        uint32_t r[4];
        ldsm4t(vbuf + o256(16 * a + 8 * (lm >> 1) + lr, 2 * wi + (lm & 1)), r[0], r[1], r[2], r[3]);
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
            stsm4(wbuf + o256(16 * mt + c_t, 2 * wi + c_c), pk(wacc[0][0], wacc[0][1]),
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

// fused5_body<true, false, true> (onorm.cu) with onorm.cubin's flags, v from the raw rows (header)
DEVI void fused5pvn_body(const PVNArgs& P) {
    extern __shared__ __align__(128) uint8_t smem[];
    const int n = blockIdx.x / HV, h = blockIdx.x % HV;
    if (P.slen[n] == 0) return;
    const int T = P.T;
    const float scale = P.scale;
    const NormArgs na{P.z, P.zs, P.nw, P.eps, P.start, P.plen, P.out};
    const FCtx c = f_ctx(P.q, P.k, nullptr, P.beta, P.A, P.g, n, h, T, smem, P.hq);
    const int NT = T / BT;
    const int lane = c.lane, lm = c.lm, lr = c.lr, cq = c.cq, gq = c.gq;
    // ONORM: this sequence's packed rows (positions pl .. sl - 1 -> rows st ..), the norm weight of lane's 4 columns
    int n_pl = 0, n_sl = 0, n_st = 0;
    uint32_t n_w01 = 0, n_w23 = 0;
    {
        n_pl = na.plen[n];
        n_sl = P.slen[n];
        n_st = na.start[n];
        const uint2 wv = *reinterpret_cast<const uint2*>(na.w + 4 * lane);
        n_w01 = wv.x;
        n_w23 = wv.y;
    }
    const int ab_t = 8 * (lm >> 1) + lr, ab_c = lm & 1;
    const int c_t = 8 * (lm & 1) + lr, c_c = lm >> 1;
    const uint32_t sb = c.sb;
    const PVSeq s = pv_seq(P, n, h);
    auto rbuf_of = [&](int i) -> uint32_t { return sb + ((i & 1) ? F_V : F_W); };   // R_i (= W_i)
    auto vbuf_of = [&](int i) -> uint32_t { return sb + ((i & 1) ? F_W : F_V); };   // V_i

    for (int x = c.tid; x < 512; x += 256) sts128z(sb + F_AQ + x * 16);   // Aq tiles above the diagonal stay +0
    if (c.tid < 64)   // conv weights of this head's 128 v columns (8 B each)
        cp16(sb + F_CW + c.tid * 16, P.cw + static_cast<size_t>(2 * KD * H + h * VD) * 4 + c.tid * 8);
    f_issue_x_part(c, 0, sb + F_K0, 0);   // A, raw rows (for v), g, beta first: X(0) and pass P's u need only these
    pv_issue_raw(c, P, s, 0, rbuf_of(0), 0);
    pv_issue_raw(c, P, s, 0, rbuf_of(0), 1);
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
    for (int i = 0; i < NT; ++i) {
        if (c.tid < 2 * BT) {   // ONORM_PF2: z rows of chunk i -> L1, P(i) + A(i) ahead of their first use
            const int t = c.tid >> 1, p = i * BT + t, pl_ = n_pl;
            if (p >= pl_ && p < n_sl) {
                const __nv_bfloat16* zp =
                    na.z + static_cast<size_t>(n_st + p - pl_) * na.zs + h * VD + 64 * (c.tid & 1);
                asm volatile("prefetch.global.L1 [%0];\n" ::"l"(zp));
            }
        }
        const uint32_t kbuf = sb + F_K0 + (i & 1) * 16384;
        const uint32_t vbuf = vbuf_of(i), wbuf = rbuf_of(i);
        const float* gs = reinterpret_cast<const float*>(smem + F_G + (i & 1) * 256);
        const uint32_t* bs = reinterpret_cast<const uint32_t*>(smem + F_B + (i & 1) * 256);
        // ------------------------------------------------ X(i): v of chunk i from its raw rows
        pv_convert(c, i, n_sl, rbuf_of(i), vbuf);   // each warp: its own v / w column slab (no CTA barrier)
        float vn[8][4];
        float e[8][2];
        f_exps(gs, cq, e);   // pass P's 2^g (recomputed for B below: not live across A)
        // ------------------------------------------------ P(i)
        if (i > 0) pv_pstage<true, false>(c, kbuf, vbuf, wbuf, gs, bs, e, vn);
        else pv_pstage<false, true>(c, kbuf, vbuf, wbuf, gs, bs, e, vn);
        __syncthreads();   // b1: w, Aq of chunk i in smem; A, v of chunk i no longer read
        const uint32_t kbn = sb + F_K0 + ((i + 1) & 1) * 16384;
        if (i == 0 && NT > 1) {   // chunk 1's inputs: A, raw rows (into V_0), k, g / beta
            f_issue_x_part(c, 1, kbn, 0);
            pv_issue_raw(c, P, s, 1, vbuf, 0);
            pv_issue_raw(c, P, s, 1, vbuf, 1);
            f_issue_x_part(c, 1, kbn, 3);
            f_issue_x_part(c, 1, kbn, 4);
            f_issue_x_part(c, 1, kbn, 5);
        }
        // ------------------------------------------------ A(i): vn = u - w h, oa = q h
        float oa[8][4];
        if (i > 0) {
            float acc[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int r = 0; r < 4; ++r) acc[j][r] = 0.f, oa[j][r] = 0.f;
#pragma unroll
            for (int s2 = 0; s2 < 8; ++s2) {
                const uint32_t a0 = pk(hacc[2 * s2][0], hacc[2 * s2][1]);
                const uint32_t a1 = pk(hacc[2 * s2][2], hacc[2 * s2][3]);
                const uint32_t a2 = pk(hacc[2 * s2 + 1][0], hacc[2 * s2 + 1][1]);
                const uint32_t a3 = pk(hacc[2 * s2 + 1][2], hacc[2 * s2 + 1][3]);
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    uint32_t b0, b1, b2, b3;
                    ldsm4(wbuf + o256(16 * jj + ab_t, 2 * s2 + ab_c), b0, b1, b2, b3);
                    mma(acc[2 * jj], a0, a1, a2, a3, b0, b1);
                    mma(acc[2 * jj + 1], a0, a1, a2, a3, b2, b3);
                    ldsm4(sb + F_Q + o256(16 * jj + ab_t, 2 * s2 + ab_c), b0, b1, b2, b3);
                    mma(oa[2 * jj], a0, a1, a2, a3, b0, b1);
                    mma(oa[2 * jj + 1], a0, a1, a2, a3, b2, b3);
                }
                if (i + 1 < NT) {   // chunk i + 1 inputs, in slices (raw rows into V_i, free since b1)
                    if (s2 == 0 || (s2 >= 3 && s2 <= 5)) f_issue_x_part(c, i + 1, kbn, s2);
                    if (s2 == 1) pv_issue_raw(c, P, s, i + 1, vbuf, 0);
                    if (s2 == 2) pv_issue_raw(c, P, s, i + 1, vbuf, 1);
                }
            }
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int r = 0; r < 4; ++r) vn[j][r] = fsub(vn[j][r], acc[j][r]);
        }
        cp_commit();
        __syncthreads();   // b2: every warp is done with q and w of chunk i
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
            // B(i) for the t-tile pair jj: o rows 16 jj .. 16 jj + 15 -> the o tile in W_i (64 rows x 256 B, swizzled)
            auto b_pair = [&](int jj) {
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
                    const int t = 8 * j + gq;
                    sts32(wbuf + o256(t, 2 * c.warp) + 4 * cq, m0);
                    sts32(wbuf + o256(t, 2 * c.warp + 1) + 4 * cq, m1);
                }
            };
            // ONORM V5: B(i) first (o tile -> W_i); C(i); silu(z) range check; bN; the norm rows (FINROLL)
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) b_pair(jj);
            NRows nr;
            n_prep(na, i, h, c.warp, lane, n_pl, n_sl, n_st, nr);
            bool spec = false;
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
                    }
                    f_issue_q_part(c, i + 1, a);   // q of chunk i + 1 (q buffer free since b2)
                }
            }
#pragma unroll
            for (int x = 0; x < 8; ++x)
                spec |= n_special(blo_f(nr.zv[x].x)) | n_special(bhi_f(nr.zv[x].x)) |
                        n_special(blo_f(nr.zv[x].y)) | n_special(bhi_f(nr.zv[x].y));
            const bool n_ex = __any_sync(0xffffffffu, spec);
            __syncthreads();   // bN: chunk i's o tile complete
#pragma unroll 1
            for (int g4 = 0; g4 < 2; ++g4) {
                if (4 * g4 >= nr.cnt) break;
                n_finR(na, wbuf, i, h, c.warp, lane, n_pl, n_st, nr, g4, n_w01, n_w23, n_ex);
            }
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();   // b3: chunk i + 1 inputs landed; every warp is done with W_i (o tile), Aq, k of chunk i
    }
}

// start-up gate helper: per (slot, value head) CTA, every chunk's v through pv_issue_raw + pv_convert -> vout
// [B, T, 32, 128] (rows >= slen and unused slots: +0, as pron.prologue writes them)
DEVI void pvx_body(const PVNArgs& P, __nv_bfloat16* vout) {
    extern __shared__ __align__(128) uint8_t smem[];
    const int n = blockIdx.x / HV, h = blockIdx.x % HV;
    const int T = P.T, NT = T / BT;
    const FCtx c = f_ctx(P.q, P.k, nullptr, P.beta, P.A, P.g, n, h, T, smem, P.hq);
    const PVSeq s = pv_seq(P, n, h);
    if (c.tid < 64)
        cp16(c.sb + F_CW + c.tid * 16, P.cw + static_cast<size_t>(2 * KD * H + h * VD) * 4 + c.tid * 8);
    for (int i = 0; i < NT; ++i) {
        pv_issue_raw(c, P, s, i, c.sb + F_W, 0);
        pv_issue_raw(c, P, s, i, c.sb + F_W, 1);
        cp_commit();
        cp_wait<0>();
        __syncthreads();
        pv_convert(c, i, s.n, c.sb + F_W, c.sb + F_V);
        __syncthreads();   // every slab converted
        for (int x = c.tid; x < 1024; x += 256) {
            const int rr = x >> 4, ch = x & 15;
            const uint4 v = lds16(c.sb + F_V + o256(rr, ch));
            *reinterpret_cast<uint4*>(vout + ((static_cast<size_t>(n) * T + i * BT + rr) * HV + h) * VD + ch * 8) = v;
        }
        __syncthreads();
    }
}
}  // namespace

extern "C" __global__ void __launch_bounds__(256, 1) ffs_pv5n(const PVNArgs P) { fused5pvn_body(P); }

extern "C" __global__ void __launch_bounds__(256, 1) ffs_pvx(const PVNArgs P, __nv_bfloat16* vout) {
    pvx_body(P, vout);
}

// smem bytes the PV kernels need (the loader reads it once)
extern "C" __global__ void ffs_pv_smem(int* out) { *out = F_SMEM_PV; }
