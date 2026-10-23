// rn.cu - r2 hotspot (ruj, 2026-10-21): exact single-pass row-norm kernels for the s1eng packed forward.
//
// Replaces, byte for byte (start-up gated in rn.py against the kernels they replace):
//   fused._rmsnorm_kernel (HAS_Y = add + RMSNorm, and plain RMSNorm), rarn._rarn_kernel (split-K slice fold +
//   residual add + RMSNorm) and fused._gated_norm_packed_kernel (GDN gated RMSNorm, D = 128).
// Arithmetic = the Triton kernels' (fused.py / rarn.py, EXACT flags = no FMA contraction, no FTZ reflect):
//   * RMSNorm (D = 2560): one warp per row; lane t owns elements 128 j + 4 t + i (i = 0..3, j = 0..19), the
//     [32 lanes, 4] tile of the Triton kernel; per (lane, i) a running fp32 sum of squares over j ascending (mul and
//     add rounded separately), lane value ((s0 + s1) + s2) + s3, then the shfl_down tree with offsets 16, 8, 4, 2, 1
//     (_warp_tree32); r = rsqrt.approx.f32(sum * inv_d + eps); out = bf16((v * r) * w) with w = 1 + weight (fp32).
//     HAS_Y: v = bf16(h + y); fold: d = p0 + p1 + ... (fp32, slice order), v = bf16(h + bf16(d)).
//   * gated norm (D = 128): one warp per (packed row, head); lane value ((sq0 + sq1) + sq2) + sq3, the same tree,
//     r = rsqrt(sum * inv_128 + eps); hn = bf16(o * r); hw = bf16(w * hn); out = bf16(hw * z / (1 + expf(-z))).
// Unlike the Triton kernels (pass 1 sums, pass 2 re-reads every input), the row stays in registers: each input is
// read once from global memory, and the loads use 8-byte vectors (4 bf16 per lane and step).
//
// Build (nvcc 13.4): nvcc -cubin -arch=sm_120a -O3 -fmad=false -o rn.cubin rn.cu
#include <stdint.h>

#define FULL 0xffffffffu

__device__ __forceinline__ float bf2f(uint32_t b16) { return __uint_as_float(b16 << 16); }
__device__ __forceinline__ float lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ uint32_t f2bf(float f) {
    unsigned short r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f));
    return (uint32_t)r;
}
__device__ __forceinline__ uint32_t pack2(float a, float b) { return f2bf(a) | (f2bf(b) << 16); }
__device__ __forceinline__ float rsqrt_approx(float x) {
    float r;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ float fm(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float fa(float a, float b) { return __fadd_rn(a, b); }
// round to bf16 and back (the .to(tl.bfloat16).to(tl.float32) of the Triton kernels)
__device__ __forceinline__ float rbf(float f) { return bf2f(f2bf(f)); }

// _warp_tree32: shfl_down with offsets 16, 8, 4, 2, 1; lane 0's value broadcast
__device__ __forceinline__ float warp_tree(float s) {
    s = fa(s, __shfl_down_sync(FULL, s, 16));
    s = fa(s, __shfl_down_sync(FULL, s, 8));
    s = fa(s, __shfl_down_sync(FULL, s, 4));
    s = fa(s, __shfl_down_sync(FULL, s, 2));
    s = fa(s, __shfl_down_sync(FULL, s, 1));
    return __shfl_sync(FULL, s, 0);
}

__device__ __forceinline__ uint2 ld2(const uint16_t* p) { return *reinterpret_cast<const uint2*>(p); }

// MODE 0: out = rmsnorm(x); MODE 1: hout = bf16(x + y), out = rmsnorm(hout); MODE 2: y = fold of S slice partials
// (slice s at y + s * pstride), hout = bf16(x + bf16(y)), out = rmsnorm(hout).
template <int MODE, int S, int ROWS>
__device__ __forceinline__ void rn_row(const uint16_t* __restrict__ x, const uint16_t* __restrict__ y,
                                       long long pstride, uint16_t* __restrict__ hout, uint16_t* __restrict__ out,
                                       const float* __restrict__ w, float eps, float inv_d, int R) {
    const int lane = threadIdx.x & 31;
    const long long row = (long long)blockIdx.x * ROWS + (threadIdx.x >> 5);
    if (row >= R) return;   // warp-uniform
    const long long base = row * 2560 + 4 * lane;
    uint32_t hb[40];
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
    for (int j = 0; j < 20; ++j) {
        const long long e = base + 128 * j;
        const uint2 xv = ld2(x + e);
        float v0, v1, v2, v3;
        if (MODE == 0) {
            v0 = lo(xv.x); v1 = hi(xv.x); v2 = lo(xv.y); v3 = hi(xv.y);
        } else if (MODE == 1) {
            const uint2 yv = ld2(y + e);
            v0 = rbf(fa(lo(xv.x), lo(yv.x)));
            v1 = rbf(fa(hi(xv.x), hi(yv.x)));
            v2 = rbf(fa(lo(xv.y), lo(yv.y)));
            v3 = rbf(fa(hi(xv.y), hi(yv.y)));
        } else {
            const uint2 p = ld2(y + e);
            float d0 = lo(p.x), d1 = hi(p.x), d2 = lo(p.y), d3 = hi(p.y);
#pragma unroll
            for (int s = 1; s < S; ++s) {
                const uint2 q = ld2(y + s * pstride + e);
                d0 = fa(d0, lo(q.x)); d1 = fa(d1, hi(q.x)); d2 = fa(d2, lo(q.y)); d3 = fa(d3, hi(q.y));
            }
            v0 = rbf(fa(lo(xv.x), rbf(d0)));
            v1 = rbf(fa(hi(xv.x), rbf(d1)));
            v2 = rbf(fa(lo(xv.y), rbf(d2)));
            v3 = rbf(fa(hi(xv.y), rbf(d3)));
        }
        hb[2 * j] = (MODE == 0) ? xv.x : pack2(v0, v1);
        hb[2 * j + 1] = (MODE == 0) ? xv.y : pack2(v2, v3);
        a0 = fa(a0, fm(v0, v0));
        a1 = fa(a1, fm(v1, v1));
        a2 = fa(a2, fm(v2, v2));
        a3 = fa(a3, fm(v3, v3));
    }
    const float s = warp_tree(fa(fa(fa(a0, a1), a2), a3));
    const float r = rsqrt_approx(fa(fm(s, inv_d), eps));
#pragma unroll
    for (int j = 0; j < 20; ++j) {
        const long long e = base + 128 * j;
        const float4 wv = *reinterpret_cast<const float4*>(w + 4 * lane + 128 * j);
        const uint32_t p0 = hb[2 * j], p1 = hb[2 * j + 1];
        uint2 o;
        o.x = pack2(fm(fm(lo(p0), r), wv.x), fm(fm(hi(p0), r), wv.y));
        o.y = pack2(fm(fm(lo(p1), r), wv.z), fm(fm(hi(p1), r), wv.w));
        *reinterpret_cast<uint2*>(out + e) = o;
        if (MODE != 0) *reinterpret_cast<uint2*>(hout + e) = make_uint2(p0, p1);
    }
}

#define RN_ROWS 4

extern "C" __global__ void __launch_bounds__(32 * RN_ROWS) rn_plain(const uint16_t* x, uint16_t* out, const float* w,
                                                                   float eps, float inv_d, int R) {
    rn_row<0, 1, RN_ROWS>(x, x, 0, nullptr, out, w, eps, inv_d, R);
}

extern "C" __global__ void __launch_bounds__(32 * RN_ROWS) rn_add(const uint16_t* h, const uint16_t* y, uint16_t* hout,
                                                                 uint16_t* out, const float* w, float eps, float inv_d,
                                                                 int R) {
    rn_row<1, 1, RN_ROWS>(h, y, 0, hout, out, w, eps, inv_d, R);
}

#define RN_FOLD(SS)                                                                                                   \
    extern "C" __global__ void __launch_bounds__(32 * RN_ROWS) rn_fold##SS(                                           \
        const uint16_t* part, long long pstride, const uint16_t* h, uint16_t* hout, uint16_t* out, const float* w,    \
        float eps, float inv_d, int R) {                                                                              \
        rn_row<2, SS, RN_ROWS>(h, part, pstride, hout, out, w, eps, inv_d, R);                                        \
    }
RN_FOLD(2)
RN_FOLD(3)
RN_FOLD(4)
RN_FOLD(5)
RN_FOLD(6)
RN_FOLD(7)
RN_FOLD(8)

// ---------------------------------------------------------------------------------------- compact variants (rnc_*)
// The same arithmetic in the same order with small code (the engine's GEMMs evict the instruction cache between
// layers; rn_add is 19 KB and rn_fold4 30 KB of SASS fully unrolled): j-loops unrolled 5x (4 rounds of loads in
// flight), the bf16 row h' staged in shared memory between the two passes (no dynamic register indexing), the slice
// fold a rolled loop over a runtime S.
template <int MODE>
__device__ __forceinline__ void rnc_row(const uint16_t* __restrict__ x, const uint16_t* __restrict__ y,
                                        long long pstride, int S, uint16_t* __restrict__ hout,
                                        uint16_t* __restrict__ out, const float* __restrict__ w, float eps,
                                        float inv_d, int R) {
    __shared__ uint2 sm[RN_ROWS * 640];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const long long row = (long long)blockIdx.x * RN_ROWS + wid;
    if (row >= R) return;   // warp-uniform; no block-wide sync below
    const long long base = row * 2560 + 4 * lane;
    uint2* hs = sm + wid * 640 + lane;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll 5
    for (int j = 0; j < 20; ++j) {
        const long long e = base + 128 * j;
        const uint2 xv = ld2(x + e);
        float v0, v1, v2, v3;
        if (MODE == 0) {
            v0 = lo(xv.x); v1 = hi(xv.x); v2 = lo(xv.y); v3 = hi(xv.y);
        } else if (MODE == 1) {
            const uint2 yv = ld2(y + e);
            v0 = rbf(fa(lo(xv.x), lo(yv.x)));
            v1 = rbf(fa(hi(xv.x), hi(yv.x)));
            v2 = rbf(fa(lo(xv.y), lo(yv.y)));
            v3 = rbf(fa(hi(xv.y), hi(yv.y)));
        } else {
            const uint2 p = ld2(y + e);
            float d0 = lo(p.x), d1 = hi(p.x), d2 = lo(p.y), d3 = hi(p.y);
#pragma unroll 1
            for (int s = 1; s < S; ++s) {
                const uint2 q = ld2(y + s * pstride + e);
                d0 = fa(d0, lo(q.x)); d1 = fa(d1, hi(q.x)); d2 = fa(d2, lo(q.y)); d3 = fa(d3, hi(q.y));
            }
            v0 = rbf(fa(lo(xv.x), rbf(d0)));
            v1 = rbf(fa(hi(xv.x), rbf(d1)));
            v2 = rbf(fa(lo(xv.y), rbf(d2)));
            v3 = rbf(fa(hi(xv.y), rbf(d3)));
        }
        if (MODE != 0) hs[32 * j] = make_uint2(pack2(v0, v1), pack2(v2, v3));
        a0 = fa(a0, fm(v0, v0));
        a1 = fa(a1, fm(v1, v1));
        a2 = fa(a2, fm(v2, v2));
        a3 = fa(a3, fm(v3, v3));
    }
    const float s = warp_tree(fa(fa(fa(a0, a1), a2), a3));
    const float r = rsqrt_approx(fa(fm(s, inv_d), eps));
#pragma unroll 5
    for (int j = 0; j < 20; ++j) {
        const long long e = base + 128 * j;
        const float4 wv = *reinterpret_cast<const float4*>(w + 4 * lane + 128 * j);
        const uint2 hv = (MODE == 0) ? ld2(x + e) : hs[32 * j];
        uint2 o;
        o.x = pack2(fm(fm(lo(hv.x), r), wv.x), fm(fm(hi(hv.x), r), wv.y));
        o.y = pack2(fm(fm(lo(hv.y), r), wv.z), fm(fm(hi(hv.y), r), wv.w));
        *reinterpret_cast<uint2*>(out + e) = o;
        if (MODE != 0) *reinterpret_cast<uint2*>(hout + e) = hv;
    }
}

extern "C" __global__ void __launch_bounds__(32 * RN_ROWS) rnc_plain(const uint16_t* x, uint16_t* out, const float* w,
                                                                    float eps, float inv_d, int R) {
    rnc_row<0>(x, x, 0, 1, nullptr, out, w, eps, inv_d, R);
}

extern "C" __global__ void __launch_bounds__(32 * RN_ROWS) rnc_add(const uint16_t* h, const uint16_t* y,
                                                                  uint16_t* hout, uint16_t* out, const float* w,
                                                                  float eps, float inv_d, int R) {
    rnc_row<1>(h, y, 0, 1, hout, out, w, eps, inv_d, R);
}

extern "C" __global__ void __launch_bounds__(32 * RN_ROWS) rnc_fold(const uint16_t* part, long long pstride, int S,
                                                                   const uint16_t* h, uint16_t* hout, uint16_t* out,
                                                                   const float* w, float eps, float inv_d, int R) {
    rnc_row<2>(h, part, pstride, S, hout, out, w, eps, inv_d, R);
}

// ---------------------------------------------------------------------------------------- wide variants (rnw_*)
// One row per CTA of 128 threads, thread k owning the element 128 j + k of every 128-column step (the Triton kernel's
// own [32 lanes, 4] tile: lane-slot t = k / 4, slot i = k % 4), so each thread carries a quarter of the row (20
// elements, kept in registers) and the GPU gets 4x the warps of rn_* (rn_* is latency-bound: ~4 warps per SM at the
// served R). Same arithmetic and order: per thread the running sum over j ascending, ((c0 + c1) + c2) + c3 over the
// 4 threads of a lane-slot (shfl_down 1, 2, 3), the 32 lane values through shared memory into warp 0's shfl_down tree
// (16, 8, 4, 2, 1), r broadcast through shared memory; each input read once, 2-byte accesses (64 B per warp, as Triton).
__device__ __forceinline__ uint32_t ld1(const uint16_t* p) { return (uint32_t)*p; }

template <int MODE, int S>
__device__ __forceinline__ void rnw_row(const uint16_t* __restrict__ x, const uint16_t* __restrict__ y,
                                        long long pstride, uint16_t* __restrict__ hout, uint16_t* __restrict__ out,
                                        const float* __restrict__ w, float eps, float inv_d) {
    __shared__ float red[32];
    __shared__ float rsh;
    const int k = threadIdx.x;
    const long long base = (long long)blockIdx.x * 2560 + k;
    uint32_t hb[20];
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 20; ++j) {
        const long long e = base + 128 * j;
        const uint32_t xv = ld1(x + e);
        uint32_t b;
        if (MODE == 0) {
            b = xv;
        } else if (MODE == 1) {
            b = f2bf(fa(bf2f(xv), bf2f(ld1(y + e))));
        } else {
            float d = bf2f(ld1(y + e));
#pragma unroll
            for (int s = 1; s < S; ++s) d = fa(d, bf2f(ld1(y + s * pstride + e)));
            b = f2bf(fa(bf2f(xv), rbf(d)));
        }
        hb[j] = b;
        const float v = bf2f(b);
        acc = fa(acc, fm(v, v));
    }
    // ((c0 + c1) + c2) + c3 over the lane-slot's 4 threads (k = 4 t + i); valid at i = 0
    const float c1 = __shfl_down_sync(FULL, acc, 1);
    const float c2 = __shfl_down_sync(FULL, acc, 2);
    const float c3 = __shfl_down_sync(FULL, acc, 3);
    if ((k & 3) == 0) red[k >> 2] = fa(fa(fa(acc, c1), c2), c3);
    __syncthreads();
    if (k < 32) {
        const float s = warp_tree(red[k]);
        if (k == 0) rsh = rsqrt_approx(fa(fm(s, inv_d), eps));
    }
    __syncthreads();
    const float r = rsh;
#pragma unroll
    for (int j = 0; j < 20; ++j) {
        const long long e = base + 128 * j;
        const float v = bf2f(hb[j]);
        out[e] = (uint16_t)f2bf(fm(fm(v, r), w[k + 128 * j]));
        if (MODE != 0) hout[e] = (uint16_t)hb[j];
    }
}

extern "C" __global__ void __launch_bounds__(128) rnw_plain(const uint16_t* x, uint16_t* out, const float* w,
                                                            float eps, float inv_d, int R) {
    rnw_row<0, 1>(x, x, 0, nullptr, out, w, eps, inv_d);
}

extern "C" __global__ void __launch_bounds__(128) rnw_add(const uint16_t* h, const uint16_t* y, uint16_t* hout,
                                                          uint16_t* out, const float* w, float eps, float inv_d,
                                                          int R) {
    rnw_row<1, 1>(h, y, 0, hout, out, w, eps, inv_d);
}

#define RNW_FOLD(SS)                                                                                                  \
    extern "C" __global__ void __launch_bounds__(128) rnw_fold##SS(                                                   \
        const uint16_t* part, long long pstride, const uint16_t* h, uint16_t* hout, uint16_t* out, const float* w,    \
        float eps, float inv_d, int R) {                                                                              \
        rnw_row<2, SS>(h, part, pstride, hout, out, w, eps, inv_d);                                                   \
    }
RNW_FOLD(2)
RNW_FOLD(3)
RNW_FOLD(4)
RNW_FOLD(5)
RNW_FOLD(6)
RNW_FOLD(7)
RNW_FOLD(8)

// gated norm: N head-rows n = r * 32 + hh (packed row r, head hh; NH = 32); o_pad row of n = row_pos[r] * 32 + hh
// (padded [B, L, 32, 128] layout); z row r at z + r * ZS + hh * 128; out [N, 128]; w bf16 [128]. HPW consecutive
// head-rows per warp (HPW divides 32: one packed row, one row_pos load).
template <int HPW>
__device__ __forceinline__ void gnorm_body(const uint16_t* __restrict__ o, const uint16_t* __restrict__ z,
                                           const uint16_t* __restrict__ w, uint16_t* __restrict__ out,
                                           const int* __restrict__ row_pos, int N, float eps, float inv_d, int ZS) {
    const int lane = threadIdx.x & 31;
    const int n0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * HPW;
    if (n0 >= N) return;   // warp-uniform (N is a multiple of 32, HPW divides 32)
    const int r = n0 >> 5;
    const int h0i = n0 & 31;
    const long long pr = row_pos[r];
    const uint2 wq = ld2(w + 4 * lane);
    const float w0 = lo(wq.x), w1 = hi(wq.x), w2 = lo(wq.y), w3 = hi(wq.y);
    uint2 ov[HPW], zv[HPW];
#pragma unroll
    for (int k = 0; k < HPW; ++k) {
        ov[k] = ld2(o + (pr * 32 + h0i + k) * 128 + 4 * lane);
        zv[k] = ld2(z + (long long)r * ZS + (h0i + k) * 128 + 4 * lane);
    }
#pragma unroll
    for (int k = 0; k < HPW; ++k) {
        const float o0 = lo(ov[k].x), o1 = hi(ov[k].x), o2 = lo(ov[k].y), o3 = hi(ov[k].y);
        const float s = warp_tree(fa(fa(fa(fm(o0, o0), fm(o1, o1)), fm(o2, o2)), fm(o3, o3)));
        const float rr = rsqrt_approx(fa(fm(s, inv_d), eps));
        const float q0 = rbf(fm(w0, rbf(fm(o0, rr)))), q1 = rbf(fm(w1, rbf(fm(o1, rr))));
        const float q2 = rbf(fm(w2, rbf(fm(o2, rr)))), q3 = rbf(fm(w3, rbf(fm(o3, rr))));
        const float z0 = lo(zv[k].x), z1 = hi(zv[k].x), z2 = lo(zv[k].y), z3 = hi(zv[k].y);
        const float g0 = __fdiv_rn(z0, fa(1.0f, expf(-z0)));
        const float g1 = __fdiv_rn(z1, fa(1.0f, expf(-z1)));
        const float g2 = __fdiv_rn(z2, fa(1.0f, expf(-z2)));
        const float g3 = __fdiv_rn(z3, fa(1.0f, expf(-z3)));
        uint2 res;
        res.x = pack2(fm(q0, g0), fm(q1, g1));
        res.y = pack2(fm(q2, g2), fm(q3, g3));
        *reinterpret_cast<uint2*>(out + (long long)(n0 + k) * 128 + 4 * lane) = res;
    }
}

#define RN_GNORM(HP)                                                                                                  \
    extern "C" __global__ void __launch_bounds__(128) rn_gnorm##HP(const uint16_t* o, const uint16_t* z,              \
                                                                   const uint16_t* w, uint16_t* out,                  \
                                                                   const int* row_pos, int N, float eps, float inv_d, \
                                                                   int ZS) {                                          \
        gnorm_body<HP>(o, z, w, out, row_pos, N, eps, inv_d, ZS);                                                     \
    }
RN_GNORM(1)
RN_GNORM(2)
RN_GNORM(4)
RN_GNORM(8)
