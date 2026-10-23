// PRON (rgh 2026-10-21): native sm_120 GDN prologue, bit-identical to fused._gdn_prologue_kernel (E4 PRO): causal
// depthwise conv (4 taps) + SiLU on the packed qkv rows (prefix-variant cache rows below plen), split into q / k / v,
// fla l2norm of q and k (eps), beta = sigmoid(b) and g = chunk_local_cumsum(negexpA * softplus(a + dt_bias)) * scale.
//
// Arithmetic = the Triton kernel's PTX (Triton 3.7, EXACT: no FMA contraction, no ftz in libdevice), op for op:
//   conv   : acc = +0; acc = add.rn(acc, mul.rn(w[c][kk], x[t - 3 + kk][c])) for kk = 0..3 (x = +0 for src < 0)
//   silu   : cb = f32(bf16_rn(acc)); y = bf16_rn(div.rn(cb, add.rn(expf(sub.rn(0, cb)), 1)))
//            (expf = libdevice __nv_expf, non-ftz, transcribed below; output rows t >= slen are +0)
//   l2norm : (q, k only) per row of 128 columns: lane j of a half warp holds columns 8 j .. 8 j + 7 (Triton's
//            blocked [1, 8] x [2, 16] layout): s = sequential add.rn of the 8 squares (mul.rn), then
//            s = add.rn(s, shfl.bfly(s, m)) for m = 8, 4, 2, 1; rstd = div.full(1, sqrt.approx(add.rn(eps, s)));
//            yn = bf16_rn(mul.rn(rstd, f32(y)))
//   gate   : beta = valid ? bf16_rn(div.rn(1, add.rn(expf(sub.rn(0, b)), 1))) : 0
//            t = add.rn(f32(a), f32(dt_bias)); sp = t > 20 ? t : div.rn(log1pf(expf(t)), 1)
//            s = valid ? mul.rn(sp, negexpA) : 0; inclusive scan of the 64 positions of a chunk as Triton's tl.cumsum
//            (Hillis-Steele shfl.up 1..16 per 32-position half, v = add.rn(v, up); second half = add.rn(T0, v) with
//            T0 the first half's total); g = mul.rn(scale, scan)
// One launch: CTAs [0, NCONV) = conv tiles (64 rows x one 128-column head of one slot, 128 threads: thread
// (row group rg = tid / 16, column chunk j = tid % 16) computes rows 8 rg .. 8 rg + 7, columns 8 j .. 8 j + 7, from
// 11 raw rows loaded once); CTAs [NCONV, ..) = gate units (one warp per (chunk, slot, value head)).
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

namespace {
constexpr int KD = 2048, VD = 4096, C = 2 * KD + VD, NH = 32, BT = 64;

DEVI float f_mul(float a, float b) { float r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
DEVI float f_add(float a, float b) { float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
DEVI float f_sub(float a, float b) { float r; asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
DEVI float f_div_rn(float a, float b) { float r; asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
DEVI float f_div_full(float a, float b) {
    float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
DEVI float f_sqrt_approx(float a) { float r; asm("sqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
DEVI float bf_rn(float x) {   // f32 -> bf16 (RN) -> f32
    uint16_t h; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(x));
    return __uint_as_float(static_cast<uint32_t>(h) << 16);
}
DEVI uint32_t pk2(float lo, float hi) {
    uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
DEVI float lo_f(uint32_t x) { return __uint_as_float(x << 16); }
DEVI float hi_f(uint32_t x) { return __uint_as_float(x & 0xffff0000u); }

// libdevice __nv_expf (non-ftz), as Triton's PTX: fma.rn / cvt.sat / fma.rm / add.rn / neg / fma.rn x2 / shl / ex2 / mul
DEVI float f_expf(float x) {
    float r;
    asm("{\n\t.reg .b32 t, s, j, k, e, q, sh;\n\t"
        "fma.rn.f32 t, %1, 0f3BBB989D, 0f3F000000;\n\t"
        "cvt.sat.f32.f32 s, t;\n\t"
        "fma.rm.f32 j, s, 0f437C0000, 0f4B400001;\n\t"
        "add.rn.f32 k, j, 0fCB40007F;\n\t"
        "neg.f32 k, k;\n\t"
        "fma.rn.f32 e, %1, 0f3FB8AA3B, k;\n\t"
        "fma.rn.f32 e, %1, 0f32A57060, e;\n\t"
        "shl.b32 sh, j, 23;\n\t"
        "ex2.approx.ftz.f32 q, e;\n\t"
        "mul.rn.f32 %0, q, sh;\n\t}"
        : "=f"(r) : "f"(x));
    return r;
}

// libdevice __nv_log1pf (non-ftz), as Triton's PTX (the special-case branches as selects)
DEVI float f_log1pf(float x) {
    float r;
    asm("{\n\t.reg .b32 xb, a, b, c, d, f, m, ce, p, sc, res;\n\t.reg .pred p8, p9, p10;\n\t"
        "mov.b32 xb, %1;\n\t"
        "add.rz.f32 a, xb, 0f3F800000;\n\t"
        "add.s32 b, a, -1061158912;\n\t"
        "and.b32 b, b, -8388608;\n\t"
        "sub.s32 c, xb, b;\n\t"
        "sub.s32 d, 1082130432, b;\n\t"
        "fma.rn.f32 f, 0f3E800000, d, 0fBF800000;\n\t"
        "add.rn.f32 m, f, c;\n\t"
        "cvt.rn.f32.s32 ce, b;\n\t"
        "mul.rn.f32 ce, ce, 0f34000000;\n\t"
        "fma.rn.f32 p, 0fBD39BF78, m, 0f3DD80012;\n\t"
        "fma.rn.f32 p, p, m, 0fBE0778E0;\n\t"
        "fma.rn.f32 p, p, m, 0f3E146475;\n\t"
        "fma.rn.f32 p, p, m, 0fBE2A68DD;\n\t"
        "fma.rn.f32 p, p, m, 0f3E4CAF9E;\n\t"
        "fma.rn.f32 p, p, m, 0fBE800042;\n\t"
        "fma.rn.f32 p, p, m, 0f3EAAAAE6;\n\t"
        "fma.rn.f32 p, p, m, 0fBF000000;\n\t"
        "mul.rn.f32 p, m, p;\n\t"
        "fma.rn.f32 p, p, m, m;\n\t"
        "fma.rn.f32 res, ce, 0f3F317218, p;\n\t"
        "setp.lt.u32 p8, xb, 2139095040;\n\t"
        "setp.lt.s32 p9, xb, -1082130431;\n\t"
        "fma.rn.f32 sc, xb, 0f7F800000, 0f7F800000;\n\t"
        "selp.b32 a, res, sc, p9;\n\t"
        "setp.eq.f32 p10, xb, 0f00000000;\n\t"
        "selp.b32 a, 0f80000000, a, p10;\n\t"
        "selp.b32 res, res, a, p8;\n\t"
        "mov.b32 %0, res;\n\t}"
        : "=f"(r) : "f"(x));
    return r;
}

// SiLU of a bf16-rounded conv sum (cb is exactly representable in bf16) -> bf16-rounded result as f32
DEVI float silu_bf(float cb) { return bf_rn(f_div_rn(cb, f_add(f_expf(f_sub(0.f, cb)), 1.f))); }

// FS: the same bf16 as silu_bf for every one of the 65536 bf16 inputs (silu_probe formula 2, exhaustive check at
// start-up): e = ex2.approx.ftz(cb * -log2(e)); y = bf16_rn(cb * rcp.approx(e + 1)) (rcp non-ftz)
DEVI float rcp_approx(float x) { float r; asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
DEVI float ex2_ftz(float x) { float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
DEVI float silu_fast_raw(float cb) {   // f32, before the bf16 rounding
    return f_mul(cb, rcp_approx(f_add(ex2_ftz(f_mul(cb, -1.4426950408889634f)), 1.f)));
}

// F9: the same bf16 as silu_bf for all 65536 bf16 inputs (silu_probe formula 9, exhaustive check at start-up):
// e = ex2.approx.ftz(cb * -log2(e)); y = bf16_rn((cb * 1/16) * rcp.approx.ftz(fma(e, 1/16, 1/16)))
DEVI float rcp_ftz(float x) { float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
DEVI float silu_f9_raw(float cb) {
    const float e = ex2_ftz(f_mul(cb, -1.4426950408889634f));
    return f_mul(f_mul(cb, 0.0625f), rcp_ftz(__fmaf_rn(e, 0.0625f, 0.0625f)));
}
// acc = fma.rn.f32.bf16(x half XH of xw, w half WH of ww, acc) (FHFMA: no unpack; = fma.rn on the exact f32 values)
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
// s = fma.rn.f32.bf16(y half H, y half H, s): the l2norm square-sum step (y^2 exact unless |y| < 2^-63, where the
// row's sum is then far below eps and add.rn(eps, s) is eps either way)
template <int H>
DEVI float sqab(uint32_t yw, float s) {
    float r;
    if (H == 0)
        asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tfma.rn.f32.bf16 %0, l, l, %2;\n\t}" : "=f"(r) : "r"(yw), "f"(s));
    else
        asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tfma.rn.f32.bf16 %0, h, h, %2;\n\t}" : "=f"(r) : "r"(yw), "f"(s));
    return r;
}

DEVI uint4 ldg16(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
DEVI void stg16(void* p, uint4 v) {
    asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

struct Args {
    const __nv_bfloat16 *x, *w, *cx, *b, *a, *cb, *ca, *dt_bias;
    const float* negexpA;
    const int *start, *slen, *plen, *pvar;
    __nv_bfloat16 *q, *k, *v, *beta;
    float* g;
    int L, XS, CXS, BS, AS, CBS, CAS, pfx, pmax, B, nconv, nhb;
    float eps, scale;
};

// raw row p (position in the sequence) of slot bb, columns col0 .. col0 + 7
DEVI uint4 raw_row(const Args& A, int p, int n, int s0, int pl, int pv, int col0) {
    if (p < 0 || p >= n) return make_uint4(0u, 0u, 0u, 0u);
    if (p < pl) return ldg16(A.cx + (static_cast<size_t>(pv) * A.pmax + p) * A.CXS + col0);
    return ldg16(A.x + (static_cast<size_t>(s0) + p - pl) * A.XS + col0);
}

// conv tile: TR = 8 RPT rows x one 128-column head of one slot; thread (rg = tid / 16, j = tid % 16): rows
// RPT rg .. RPT rg + RPT - 1, columns 8 j .. 8 j + 7, from RPT + 3 raw rows loaded once.
// FFMA: acc = fma.rn(w, x, acc) equals add.rn(acc, mul.rn(w, x)) whenever w x is exact in f32, which holds for
// bf16 w, x unless |w x| < 2^-110 (nonzero) or overflows (8 + 8 significant bits, f32 exponent range).
template <int RPT, bool FMA>
DEVI void conv_tile(const Args& A, int blk) {
    constexpr int TR = 8 * RPT;
    const int L = A.L, NCH = L / TR;
    const int ci = blk % NCH;
    const int hb = (blk / NCH) % (C / 128);
    const int bb = blk / (NCH * (C / 128));
    const int tid = threadIdx.x, j = tid & 15, rg = tid >> 4;
    const int n = A.slen[bb];
    const int t0 = ci * TR + rg * RPT;
    const int col0 = hb * 128 + j * 8;
    __nv_bfloat16* dst;
    size_t rs;   // row stride of the destination (elements)
    if (hb < 16) {
        dst = A.q + ((static_cast<size_t>(bb) * L) * 16 + hb) * 128 + j * 8;
        rs = 16 * 128;
    } else if (hb < 32) {
        dst = A.k + ((static_cast<size_t>(bb) * L) * 16 + (hb - 16)) * 128 + j * 8;
        rs = 16 * 128;
    } else {
        dst = A.v + ((static_cast<size_t>(bb) * L) * 32 + (hb - 32)) * 128 + j * 8;
        rs = 32 * 128;
    }
    if (ci * TR >= n) {   // whole tile past the sequence: +0 (all taps masked in Triton)
#pragma unroll
        for (int r = 0; r < RPT; ++r) stg16(dst + (t0 + r) * rs, make_uint4(0u, 0u, 0u, 0u));
        return;
    }
    const int s0 = A.start[bb];
    const int pl = A.pfx ? A.plen[bb] : 0;
    const int pv = A.pfx ? A.pvar[bb] : 0;
    uint4 raw[RPT + 3];
#pragma unroll
    for (int r = 0; r < RPT + 3; ++r) raw[r] = raw_row(A, t0 - 3 + r, n, s0, pl, pv, col0);
    float wf[8][4];
    {
        const uint4* wp = reinterpret_cast<const uint4*>(A.w + static_cast<size_t>(col0) * 4);
#pragma unroll
        for (int u = 0; u < 4; ++u) {   // 16 B = columns 2u, 2u + 1, 4 taps each
            const uint4 x = __ldg(wp + u);
            wf[2 * u][0] = lo_f(x.x); wf[2 * u][1] = hi_f(x.x); wf[2 * u][2] = lo_f(x.y); wf[2 * u][3] = hi_f(x.y);
            wf[2 * u + 1][0] = lo_f(x.z); wf[2 * u + 1][1] = hi_f(x.z);
            wf[2 * u + 1][2] = lo_f(x.w); wf[2 * u + 1][3] = hi_f(x.w);
        }
    }
    const bool l2 = hb < 32;
#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        float acc[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) acc[c] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const uint4 x = raw[r + kk];
            const uint32_t xw[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const float xv = (c & 1) ? hi_f(xw[c >> 1]) : lo_f(xw[c >> 1]);
                acc[c] = FMA ? __fmaf_rn(wf[c][kk], xv, acc[c]) : f_add(acc[c], f_mul(wf[c][kk], xv));
            }
        }
        const bool valid = t0 + r < n;
        float y[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) y[c] = silu_bf(bf_rn(acc[c]));
        uint4 o;
        if (l2) {
            float s = f_mul(y[0], y[0]);
#pragma unroll
            for (int c = 1; c < 8; ++c) s = f_add(s, f_mul(y[c], y[c]));
            s = f_add(s, __shfl_xor_sync(0xffffffffu, s, 8));
            s = f_add(s, __shfl_xor_sync(0xffffffffu, s, 4));
            s = f_add(s, __shfl_xor_sync(0xffffffffu, s, 2));
            s = f_add(s, __shfl_xor_sync(0xffffffffu, s, 1));
            const float rstd = f_div_full(1.f, f_sqrt_approx(f_add(A.eps, s)));
            o.x = pk2(f_mul(rstd, y[0]), f_mul(rstd, y[1]));
            o.y = pk2(f_mul(rstd, y[2]), f_mul(rstd, y[3]));
            o.z = pk2(f_mul(rstd, y[4]), f_mul(rstd, y[5]));
            o.w = pk2(f_mul(rstd, y[6]), f_mul(rstd, y[7]));
        } else {
            o.x = pk2(y[0], y[1]);
            o.y = pk2(y[2], y[3]);
            o.z = pk2(y[4], y[5]);
            o.w = pk2(y[6], y[7]);
        }
        if (!valid) o = make_uint4(0u, 0u, 0u, 0u);
        stg16(dst + (t0 + r) * rs, o);
    }
}

// v3: FFMA conv, fast SiLU (FS), branch-free raw row addressing; tile TR = 8 RPT rows (128 threads)
template <int RPT, bool FS>
DEVI void conv_tile3(const Args& A, int blk) {
    constexpr int TR = 8 * RPT;
    const int L = A.L, NCH = L / TR;
    const int ci = blk % NCH;
    const int hb = (blk / NCH) % (C / 128);
    const int bb = blk / (NCH * (C / 128));
    const int tid = threadIdx.x, j = tid & 15, rg = tid >> 4;
    const int n = A.slen[bb];
    const int t0 = ci * TR + rg * RPT;
    const int col0 = hb * 128 + j * 8;
    const bool isq = hb < 16, isk = hb >= 16 && hb < 32;
    const int hd = isq ? hb : (isk ? hb - 16 : hb - 32);
    const int rsi = hb < 32 ? 16 * 128 : 32 * 128;
    __nv_bfloat16* dst = (isq ? A.q : (isk ? A.k : A.v)) + (static_cast<size_t>(bb) * L * (rsi / 128) + hd) * 128
                         + j * 8 + static_cast<size_t>(t0) * rsi;
    if (ci * TR >= n) {
#pragma unroll
        for (int r = 0; r < RPT; ++r) stg16(dst + r * rsi, make_uint4(0u, 0u, 0u, 0u));
        return;
    }
    const int s0 = A.start[bb];
    const int pl = A.pfx ? A.plen[bb] : 0;
    const int pv = A.pfx ? A.pvar[bb] : 0;
    const __nv_bfloat16* xb = A.x + (static_cast<long long>(s0) - pl) * A.XS + col0;
    const __nv_bfloat16* cbp = A.cx + static_cast<long long>(pv) * A.pmax * A.CXS + col0;
    uint4 raw[RPT + 3];
#pragma unroll
    for (int r = 0; r < RPT + 3; ++r) {
        const int p = t0 - 3 + r;
        const __nv_bfloat16* src = p < pl ? cbp + static_cast<long long>(p) * A.CXS : xb + static_cast<long long>(p) * A.XS;
        raw[r] = (p >= 0 && p < n) ? ldg16(src) : make_uint4(0u, 0u, 0u, 0u);
    }
    float wf[8][4];
    {
        const uint4* wp = reinterpret_cast<const uint4*>(A.w + static_cast<size_t>(col0) * 4);
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const uint4 x = __ldg(wp + u);
            wf[2 * u][0] = lo_f(x.x); wf[2 * u][1] = hi_f(x.x); wf[2 * u][2] = lo_f(x.y); wf[2 * u][3] = hi_f(x.y);
            wf[2 * u + 1][0] = lo_f(x.z); wf[2 * u + 1][1] = hi_f(x.z);
            wf[2 * u + 1][2] = lo_f(x.w); wf[2 * u + 1][3] = hi_f(x.w);
        }
    }
    const bool l2 = hb < 32;
#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        float acc[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) acc[c] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const uint4 x = raw[r + kk];
            const uint32_t xw[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const float xv = (c & 1) ? hi_f(xw[c >> 1]) : lo_f(xw[c >> 1]);
                acc[c] = __fmaf_rn(wf[c][kk], xv, acc[c]);
            }
        }
        float y[8];
        uint32_t yp[4];
#pragma unroll
        for (int c = 0; c < 8; c += 2) {
            const uint32_t cbp2 = pk2(acc[c], acc[c + 1]);
            if (FS) {
                yp[c >> 1] = pk2(silu_fast_raw(lo_f(cbp2)), silu_fast_raw(hi_f(cbp2)));
            } else {
                yp[c >> 1] = pk2(silu_bf(lo_f(cbp2)), silu_bf(hi_f(cbp2)));
            }
            y[c] = lo_f(yp[c >> 1]);
            y[c + 1] = hi_f(yp[c >> 1]);
        }
        uint4 o;
        if (l2) {
            float s = f_mul(y[0], y[0]);
#pragma unroll
            for (int c = 1; c < 8; ++c) s = f_add(s, f_mul(y[c], y[c]));
            s = f_add(s, __shfl_xor_sync(0xffffffffu, s, 8));
            s = f_add(s, __shfl_xor_sync(0xffffffffu, s, 4));
            s = f_add(s, __shfl_xor_sync(0xffffffffu, s, 2));
            s = f_add(s, __shfl_xor_sync(0xffffffffu, s, 1));
            const float rstd = f_div_full(1.f, f_sqrt_approx(f_add(A.eps, s)));
            o.x = pk2(f_mul(rstd, y[0]), f_mul(rstd, y[1]));
            o.y = pk2(f_mul(rstd, y[2]), f_mul(rstd, y[3]));
            o.z = pk2(f_mul(rstd, y[4]), f_mul(rstd, y[5]));
            o.w = pk2(f_mul(rstd, y[6]), f_mul(rstd, y[7]));
        } else {
            o = make_uint4(yp[0], yp[1], yp[2], yp[3]);
        }
        if (t0 + r >= n) o = make_uint4(0u, 0u, 0u, 0u);
        stg16(dst + r * rsi, o);
    }
}

// v4: FHFMA conv on the raw bf16 halves, F9 SiLU, FHFMA squares; tile TR = 8 RPT rows (128 threads)
template <int RPT>
DEVI void conv_tile4(const Args& A, int blk) {
    constexpr int TR = 8 * RPT;
    const int L = A.L, NCH = L / TR;
    const int ci = blk % NCH;
    const int hb = (blk / NCH) % A.nhb;   // nhb = 64 (q, k, v) or 32 (q, k only: PROFUSE-v computes v in FFS)
    const int bb = blk / (NCH * A.nhb);
    const int tid = threadIdx.x, j = tid & 15, rg = tid >> 4;
    const int n = A.slen[bb];
    const int t0 = ci * TR + rg * RPT;
    const int col0 = hb * 128 + j * 8;
    const bool isq = hb < 16, isk = hb >= 16 && hb < 32;
    const int hd = isq ? hb : (isk ? hb - 16 : hb - 32);
    const int rsi = hb < 32 ? 16 * 128 : 32 * 128;
    __nv_bfloat16* dst = (isq ? A.q : (isk ? A.k : A.v)) + (static_cast<size_t>(bb) * L * (rsi / 128) + hd) * 128
                         + j * 8 + static_cast<size_t>(t0) * rsi;
    if (ci * TR >= n) {
#pragma unroll
        for (int r = 0; r < RPT; ++r) stg16(dst + r * rsi, make_uint4(0u, 0u, 0u, 0u));
        return;
    }
    const int s0 = A.start[bb];
    const int pl = A.pfx ? A.plen[bb] : 0;
    const int pv = A.pfx ? A.pvar[bb] : 0;
    const __nv_bfloat16* xb = A.x + (static_cast<long long>(s0) - pl) * A.XS + col0;
    const __nv_bfloat16* cbp = A.cx + static_cast<long long>(pv) * A.pmax * A.CXS + col0;
    uint32_t raw[RPT + 3][4];
#pragma unroll
    for (int r = 0; r < RPT + 3; ++r) {
        const int p = t0 - 3 + r;
        const __nv_bfloat16* src = p < pl ? cbp + static_cast<long long>(p) * A.CXS : xb + static_cast<long long>(p) * A.XS;
        const uint4 v = (p >= 0 && p < n) ? ldg16(src) : make_uint4(0u, 0u, 0u, 0u);
        raw[r][0] = v.x; raw[r][1] = v.y; raw[r][2] = v.z; raw[r][3] = v.w;
    }
    uint32_t ww[16];   // column c's taps (0, 1) in ww[2 c], taps (2, 3) in ww[2 c + 1]
    {
        const uint4* wp = reinterpret_cast<const uint4*>(A.w + static_cast<size_t>(col0) * 4);
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const uint4 x = __ldg(wp + u);
            ww[4 * u] = x.x; ww[4 * u + 1] = x.y; ww[4 * u + 2] = x.z; ww[4 * u + 3] = x.w;
        }
    }
    const bool l2 = hb < 32;
#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        float acc[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) acc[c] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
            for (int c2 = 0; c2 < 4; ++c2) {   // columns 2 c2 (low half of raw word c2), 2 c2 + 1 (high half)
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
        for (int c = 0; c < 8; c += 2) {
            const uint32_t cbp2 = pk2(acc[c], acc[c + 1]);
            yp[c >> 1] = pk2(silu_f9_raw(lo_f(cbp2)), silu_f9_raw(hi_f(cbp2)));
        }
        uint4 o;
        if (l2) {
            float s = sqab<0>(yp[0], 0.f);
            s = sqab<1>(yp[0], s);
#pragma unroll
            for (int c2 = 1; c2 < 4; ++c2) {
                s = sqab<0>(yp[c2], s);
                s = sqab<1>(yp[c2], s);
            }
            s = f_add(s, __shfl_xor_sync(0xffffffffu, s, 8));
            s = f_add(s, __shfl_xor_sync(0xffffffffu, s, 4));
            s = f_add(s, __shfl_xor_sync(0xffffffffu, s, 2));
            s = f_add(s, __shfl_xor_sync(0xffffffffu, s, 1));
            const float rstd = f_div_full(1.f, f_sqrt_approx(f_add(A.eps, s)));
            o.x = pk2(f_mul(rstd, lo_f(yp[0])), f_mul(rstd, hi_f(yp[0])));
            o.y = pk2(f_mul(rstd, lo_f(yp[1])), f_mul(rstd, hi_f(yp[1])));
            o.z = pk2(f_mul(rstd, lo_f(yp[2])), f_mul(rstd, hi_f(yp[2])));
            o.w = pk2(f_mul(rstd, lo_f(yp[3])), f_mul(rstd, hi_f(yp[3])));
        } else {
            o = make_uint4(yp[0], yp[1], yp[2], yp[3]);
        }
        if (t0 + r >= n) o = make_uint4(0u, 0u, 0u, 0u);
        stg16(dst + r * rsi, o);
    }
}

DEVI float ld_bf(const __nv_bfloat16* p) {
    return __uint_as_float(static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(p)) << 16);
}

DEVI void gate_unit(const Args& A, int u) {
    const int L = A.L, NCH = L / BT;
    const int ci = u % NCH;
    const int hh = (u / NCH) % NH;
    const int bb = u / (NCH * NH);
    if (bb >= A.B) return;
    const int lane = threadIdx.x & 31;
    const int n = A.slen[bb];
    const int s0 = A.start[bb];
    const int pl = A.pfx ? A.plen[bb] : 0;
    const int pv = A.pfx ? A.pvar[bb] : 0;
    const float dtb = ld_bf(A.dt_bias + hh), nea = A.negexpA[hh];
    float val[2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int p = ci * BT + lane + 32 * half;
        const bool valid = p < n;
        float bf = 0.f, af = 0.f;
        if (valid) {
            if (p < pl) {
                const size_t cr = static_cast<size_t>(pv) * A.pmax + p;
                bf = ld_bf(A.cb + cr * A.CBS + hh);
                af = ld_bf(A.ca + cr * A.CAS + hh);
            } else {
                const size_t row = static_cast<size_t>(s0) + p - pl;
                bf = ld_bf(A.b + row * A.BS + hh);
                af = ld_bf(A.a + row * A.AS + hh);
            }
        }
        const float sg = f_div_rn(1.f, f_add(f_expf(f_sub(0.f, bf)), 1.f));
        const uint16_t bh = valid ? static_cast<uint16_t>(__float_as_uint(bf_rn(sg)) >> 16) : static_cast<uint16_t>(0);
        const size_t offs = (static_cast<size_t>(bb) * L + p) * NH + hh;
        reinterpret_cast<uint16_t*>(A.beta)[offs] = bh;
        const float t = f_add(af, dtb);
        const float sp = (t > 20.f) ? t : f_div_rn(f_log1pf(f_expf(t)), 1.f);
        val[half] = valid ? f_mul(sp, nea) : 0.f;
    }
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const float u0 = __shfl_up_sync(0xffffffffu, val[0], d);
        const float u1 = __shfl_up_sync(0xffffffffu, val[1], d);
        if (lane >= d) {
            val[0] = f_add(val[0], u0);
            val[1] = f_add(val[1], u1);
        }
    }
    const float t0 = __shfl_sync(0xffffffffu, val[0], 31);
    val[1] = f_add(t0, val[1]);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int p = ci * BT + lane + 32 * half;
        A.g[(static_cast<size_t>(bb) * L + p) * NH + hh] = f_mul(A.scale, val[half]);
    }
}
}  // namespace

#ifndef PRON_SHIP
#define PRON_KERNEL(NAME, RPT, FMA)                                                                         \
    extern "C" __global__ void __launch_bounds__(128) NAME(Args A) {                                                \
        const int blk = blockIdx.x;                                                                                 \
        if (blk < A.nconv) conv_tile<RPT, FMA>(A, blk);                                                             \
        else gate_unit(A, (blk - A.nconv) * 4 + (threadIdx.x >> 5));                                               \
    }
PRON_KERNEL(pron, 8, false)
PRON_KERNEL(pron_r4, 4, false)
PRON_KERNEL(pron_r4f, 4, true)
PRON_KERNEL(pron_r2f, 2, true)
PRON_KERNEL(pron_r8f, 8, true)
#define PRON3_KERNEL(NAME, RPT, FS)                                                                         \
    extern "C" __global__ void __launch_bounds__(128) NAME(Args A) {                                                \
        const int blk = blockIdx.x;                                                                                 \
        if (blk < A.nconv) conv_tile3<RPT, FS>(A, blk);                                                             \
        else gate_unit(A, (blk - A.nconv) * 4 + (threadIdx.x >> 5));                                               \
    }
PRON3_KERNEL(pron3_r2, 2, true)
PRON3_KERNEL(pron3_r4, 4, true)
PRON3_KERNEL(pron3_r4x, 4, false)
PRON3_KERNEL(pron3_r8, 8, true)
#endif
#define PRON4_KERNEL(NAME, RPT)                                                                             \
    extern "C" __global__ void __launch_bounds__(128) NAME(Args A) {                                                \
        const int blk = blockIdx.x;                                                                                 \
        if (blk < A.nconv) conv_tile4<RPT>(A, blk);                                                                 \
        else gate_unit(A, (blk - A.nconv) * 4 + (threadIdx.x >> 5));                                               \
    }
PRON4_KERNEL(pron4_r4, 4)
#ifndef PRON_SHIP
PRON4_KERNEL(pron4_r2, 2)
PRON4_KERNEL(pron4_r8, 8)
#endif
