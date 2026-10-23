// [rgh 10-21: + 4 nsw7l tall-tail instantiations (NG7T), see below; everything else = pjs e5y cc128692]
// E6 NG7 (own code, e6 GEMM prototype proto/gemm7/ng7.cu reduced to what ships): exact bf16 GEMMs for sm_120a
// (RTX PRO 6000) with a lean BK 64 two-stage mainloop and two epilogues:
//   * nsw7l_*: gate|up GEMM + SwiGLU with the SiLU taken from a 64K-entry table (EP_LUT) that the host fills from
//     torch's own F.silu over every bf16 bit pattern (ng.check_ng7 builds it and checks the kernels' epilogue
//     arithmetic against torch F.silu(g) * u and the engine's silu_mul for every bf16 g at start-up);
//   * np7_*: split-K partials in the E5-4 NP-FOLD layout (P [S, M, N], the pin's ceil64 slices; the consumer folds
//     them in slice order), for down (K 9216), out_proj / o_proj (K 4096) and the fused k|v block (K 2560).
//   * swiglu7_elem_lut: the SwiGLU epilogue arithmetic elementwise (the start-up table check).
//
// Arithmetic (the same bit class as NG3 / NG5 / NG9B / the cuBLASLt pins; nothing here changes it):
//   every output element (or slice partial) is one fp32 chain of mma.sync.m16n8k16 (bf16 x bf16 -> fp32) over the
//   k16 blocks of its K range in ascending order, started from zero, rounded once to bf16 (RN);
//   SwiGLU: g = bf16(acc_g), u = bf16(acc_u), s = lut[bits(g)] (= bf16(F.silu(g)) as torch computes it on this GPU),
//   out = bf16(float(s) * float(u)) (exact fp32 product, RN) = the reference's act_fn(gate) * up;
//   np7: P[s] = bf16(acc over K slice s = [s * chunk, min((s + 1) * chunk, K))); the consumer folds
//   y = bf16(sum_s float(P[s])) in slice order (E5-4), the pin's OUTPUT_TYPE reduction.
//
// Mainloop (e6): K is a template parameter (KC; the A and W row strides must equal K: lda is not read): the 12
// cp.async sources of a k-tile are one per-thread base pointer per operand plus compile-time row offsets; ldmatrix
// addresses one register per (operand, k16 step) plus the stage offset; one k-tile in flight (stage kt+1 issued at
// the top of k-tile kt, waited before its last k16 step), fragments double-buffered across k16 steps; 16-byte chunk c
// of a 128-byte smem row stored at c ^ (row & 7). 8-warp CTAs, 128x256 big tiles (64x64 warp tiles) with 64- or
// 32-row small tiles in one launch (rows [0, mb) big, the rest small), BK 64, two stages.
// Epilogues: plain (stmatrix into a padded bf16 staging tile, 16-byte row stores of rows < M); SwiGLU (a tile loads
// its gate rows and the matching up rows as [gate BN/2 | up BN/2], so g and u of an output sit in the same thread;
// stmatrix staging, 16-byte stores).
// Kernel signature = ngemm5 / NG9B's (A, B, C, M, N, K, lda, ldc, chunk, mb, ws, flags); nsw7l: the SiLU table in ws.
// Build (reproducible): nvcc 13.4 -O3 -arch=sm_120a -cubin -o ngemm7.cubin ngemm7.cu. Every kernel's SASS is identical
// to the e6 r3 prototype cubin's kernel of the same name (tools/cubin_fn_diff.py), the build that was checked bit for
// bit (94,500 checks) and timed on veda1.
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__
typedef __nv_bfloat16 bf16;

DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
DEVI void cp16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
// rows past M: ignore-src form (zeros written, the source is not read; its address may lie past the tensor)
DEVI void cp16z(uint32_t dst, const void* src, bool ok) {
    asm volatile("{\n .reg .pred p;\n setp.eq.u32 p, %2, 0;\n cp.async.cg.shared.global [%0], [%1], 16, p;\n}\n"
                 ::"r"(dst), "l"(src), "r"((uint32_t)ok));
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
DEVI void cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
DEVI void ldsm4(uint32_t a, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
DEVI void stsm4(uint32_t a, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n" ::"r"(a), "r"(r0), "r"(r1), "r"(r2),
                 "r"(r3) : "memory");
}
DEVI void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI uint32_t pack_bf16(float lo, float hi) {   // RN, lo in bits 0-15
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
DEVI float bf_lo(uint32_t p) { return __uint_as_float(p << 16); }
DEVI float bf_hi(uint32_t p) { return __uint_as_float(p & 0xffff0000u); }

// ------------------------------------------------------------------------------------------------ tile configs
// 8 warps; BK = 64 (one 128-byte smem row per tile row and stage, 16-byte chunk c stored at c ^ (row & 7)); 2 stages.
template <int BM_, int BN_, int WM_, int WN_>
struct T7 {
    static constexpr int BM = BM_, BN = BN_, WM = WM_, WN = WN_;
    static_assert(WM * WN == 8, "8 warps");
    static constexpr int WTM = BM / WM, WTN = BN / WN, MT = WTM / 16, NT = WTN / 8;
    static constexpr int AI = BM / 32, BI = BN / 32;                 // 16-byte chunks per thread per operand and stage
    static constexpr int A_ST = BM * 128, B_ST = BN * 128, STAGE = A_ST + B_ST;
    static constexpr int PITCH = (BN + 8) * 2;                       // plain staging row (bytes): conflict-free stsm
    static constexpr int PITCH_SW = (BN / 2 + 8) * 2;                // SwiGLU staging row (BN / 2 outputs)
    static constexpr int SMEM = 2 * STAGE > BM * PITCH ? 2 * STAGE : BM * PITCH;
    static_assert(MT >= 1 && NT % 2 == 0 && AI >= 1, "tile");
};
using B7 = T7<128, 256, 2, 4>;      // 64x64 warp tiles
using S7a = T7<64, 256, 2, 4>;      // 32x64
using S7b = T7<32, 256, 1, 8>;      // 32x32
// NG7T tall tails (rgh 10-21, the COV9b ngemm9t geometry on this mainloop + LUT epilogue): the last row band of 160 /
// 192 / 224 rows x 128 columns (drop 1: 128 + R mod 128 rows) or 96x256 (drop 0: R mod 128 = 96). Tile geometry never
// changes an element's ascending-k mma.sync chain (same bit class); every use is byte-checked at its R.
using T7a = T7<160, 128, 2, 4>;     // 80x32 warp tiles
using T7b = T7<192, 128, 2, 4>;     // 96x32
using T7c = T7<224, 128, 2, 4>;     // 112x32
using S7c = T7<96, 256, 2, 4>;      // 48x64

// ------------------------------------------------------------------------------------------------ mainloop
// acc = A[m0:m0+BM, kb:kb+64*KT] @ W[rows, same]^T over ascending k. MODE 0: smem B rows = W rows n0 .. n0+BN-1.
// MODE 1 (SwiGLU, n0 = output column o0): smem B rows [0, BN/2) = gate rows o0 .. o0+BN/2-1, rows [BN/2, BN) = the
// same up rows (W rows H + o0 ..); warp wn reads gate rows wn*WTN/2 + [0, WTN/2) as its n8 tiles j < NT/2 and the
// matching up rows as tiles j >= NT/2, so acc[i][j] (j < NT/2) is a gate and acc[i][j + NT/2] the matching up column.
// cp.async: all threads, 12 x 16 B each per k-tile.
template <class T, int MODE>
DEVI uint32_t boff_of(int ks, int lane, int wn, uint32_t sm) {
    const int r = (MODE == 0 ? wn * T::WTN : wn * (T::WTN / 2)) + ((lane >> 4) << 3) + (lane & 7);
    return sm + T::A_ST + r * 128 + (((2 * ks + ((lane >> 3) & 1)) ^ (lane & 7)) << 4);
}
template <class T, int MODE>
__host__ __device__ constexpr int bpair_off(int j) {   // byte offset of n8 tile pair (j, j+1) from the warp's B base
    return (MODE == 0 || j < T::NT / 2) ? j * 1024 : (T::BN / 2) * 128 + (j - T::NT / 2) * 1024;
}
template <class T, int KC, int MODE>
DEVI void ml7(const bf16* __restrict__ A, const bf16* __restrict__ W, int M, int m0, int n0, int H, int kb, int KT,
              uint32_t sm, float (&acc)[T::MT][T::NT][4]) {
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int wm = warp / T::WN, wn = warp % T::WN;
    const int r0 = t >> 3, c = t & 7;
    const uint32_t dst = sm + r0 * 128 + ((c ^ (r0 & 7)) << 4);
    const bf16* gA = A + (size_t)(m0 + r0) * KC + kb + c * 8;
    uint32_t okm = 0;
#pragma unroll
    for (int i = 0; i < T::AI; ++i) okm |= (m0 + r0 + 32 * i < M) ? (1u << i) : 0u;
    const bf16* gB0 = W + (size_t)(n0 + r0) * KC + kb + c * 8;      // MODE 1: gate rows o0 + r0 + 32 i
    const bf16* gB1 = gB0 + (size_t)H * KC;                           // MODE 1: the up rows (+H)
    auto load = [&](int st, int kt) {
        const uint32_t d = dst + st * T::STAGE;
        const int ko = kt * 64;
#pragma unroll
        for (int i = 0; i < T::AI; ++i) cp16z(d + i * 4096, gA + i * 32 * KC + ko, (okm >> i) & 1u);
#pragma unroll
        for (int i = 0; i < T::BI; ++i) {
            const bf16* p;
            if constexpr (MODE == 0) p = gB0 + i * 32 * KC;
            else p = (i < T::BI / 2) ? gB0 + i * 32 * KC : gB1 + (i - T::BI / 2) * 32 * KC;
            cp16(d + T::A_ST + i * 4096, p + ko);
        }
        cp_commit();
    };
    uint32_t aoff[4], boff[4];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        aoff[ks] = sm + (wm * T::WTM + (lane & 15)) * 128 + (((2 * ks + (lane >> 4)) ^ (lane & 7)) << 4);
        boff[ks] = boff_of<T, MODE>(ks, lane, wn, sm);
    }
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
    uint32_t af[2][T::MT][4];
    uint32_t bfr[2][T::NT][2];
    auto frags = [&](int buf, uint32_t so, int ks) {
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
            ldsm4(aoff[ks] + so + i * 2048, af[buf][i][0], af[buf][i][1], af[buf][i][2], af[buf][i][3]);
#pragma unroll
        for (int j = 0; j < T::NT; j += 2)
            ldsm4(boff[ks] + so + bpair_off<T, MODE>(j), bfr[buf][j][0], bfr[buf][j][1], bfr[buf][j + 1][0],
                  bfr[buf][j + 1][1]);
    };
    load(0, 0);
    cp_wait0();
    __syncthreads();
    frags(0, 0, 0);
#pragma unroll 1
    for (int kt = 0; kt < KT; ++kt) {
        const uint32_t so = (kt & 1) * T::STAGE, sn = so ^ T::STAGE;
        const bool more = kt + 1 < KT;
        if (more) load((kt & 1) ^ 1, kt + 1);
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            if (ks < 3) {
                frags((ks + 1) & 1, so, ks + 1);
            } else {
                if (more) cp_wait0();
                __syncthreads();
                if (more) frags(0, sn, 0);
            }
#pragma unroll
            for (int i = 0; i < T::MT; ++i)
#pragma unroll
                for (int j = 0; j < T::NT; ++j) mma16816(acc[i][j], af[ks & 1][i], bfr[ks & 1][j][0], bfr[ks & 1][j][1]);
        }
    }
}

// ------------------------------------------------------------------------------------------------ plain epilogue
// fp32 -> bf16 (RN) -> stmatrix into the padded staging tile -> 16-byte stores of rows < M (C row stride ldc)
template <class T>
DEVI void store7(const float (&acc)[T::MT][T::NT][4], bf16* __restrict__ C, int M, int ldc, int m0, int n0,
                 uint32_t sm) {
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int wm = warp / T::WN, wn = warp % T::WN;
    __syncthreads();   // every warp is past its last ldmatrix of the stages
    const uint32_t base = sm + (wm * T::WTM + ((lane >> 3) & 1) * 8 + (lane & 7)) * T::PITCH +
                          (wn * T::WTN + (lane >> 4) * 8) * 2;
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; j += 2)
            stsm4(base + i * 16 * T::PITCH + j * 16, pack_bf16(acc[i][j][0], acc[i][j][1]),
                  pack_bf16(acc[i][j][2], acc[i][j][3]), pack_bf16(acc[i][j + 1][0], acc[i][j + 1][1]),
                  pack_bf16(acc[i][j + 1][2], acc[i][j + 1][3]));
    __syncthreads();
    constexpr int CPR = T::BN / 8, RPI = 256 / CPR;   // 16-byte chunks per row, rows per pass
    const int cc = t % CPR, rr = t / CPR;
#pragma unroll
    for (int q = 0; q < T::BM / RPI; ++q) {
        const int r = rr + q * RPI;
        uint4 v;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "r"(sm + r * T::PITCH + cc * 16));
        if (m0 + r < M) *reinterpret_cast<uint4*>(C + (size_t)(m0 + r) * ldc + n0 + cc * 8) = v;
    }
}

// ------------------------------------------------------------------------------------------------ SwiGLU epilogue
// out bf16 pair (columns c, c+1) from the gate pair (a0, a1) and up pair (b0, b1) accumulators: g and u rounded to
// bf16 (RN), s = lut[bits(g)] (torch's F.silu of that bf16 value), out = bf16(float(s) * float(u)) (RN)
DEVI uint32_t swiglu_pair(float a0, float a1, float b0, float b1, const uint16_t* __restrict__ lut) {
    const uint32_t gp = pack_bf16(a0, a1), up = pack_bf16(b0, b1);
    const float u0 = bf_lo(up), u1 = bf_hi(up);
    const float s0 = __uint_as_float(uint32_t(__ldg(lut + (gp & 0xffffu))) << 16);
    const float s1 = __uint_as_float(uint32_t(__ldg(lut + (gp >> 16))) << 16);
    return pack_bf16(__fmul_rn(s0, u0), __fmul_rn(s1, u1));
}

// SwiGLU epilogue: outputs of the tile = BM rows x BN/2 columns starting at output column o0; staging pitch PITCH_SW
template <class T>
DEVI void store7_swiglu(const float (&acc)[T::MT][T::NT][4], bf16* __restrict__ O, int M, int ldo, int m0, int o0,
                        uint32_t sm, const uint16_t* __restrict__ lut) {
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int wm = warp / T::WN, wn = warp % T::WN;
    constexpr int NH = T::NT / 2;                     // gate n8 tiles per warp
    constexpr int P = T::PITCH_SW;
    __syncthreads();
    const uint32_t base = sm + (wm * T::WTM + ((lane >> 3) & 1) * 8 + (lane & 7)) * P + (wn * (T::WTN / 2) +
                                                                                          (lane >> 4) * 8) * 2;
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < NH; j += 2) {
            const uint32_t r0 = swiglu_pair(acc[i][j][0], acc[i][j][1], acc[i][j + NH][0], acc[i][j + NH][1], lut);
            const uint32_t r1 = swiglu_pair(acc[i][j][2], acc[i][j][3], acc[i][j + NH][2], acc[i][j + NH][3], lut);
            const uint32_t r2 = swiglu_pair(acc[i][j + 1][0], acc[i][j + 1][1], acc[i][j + 1 + NH][0],
                                            acc[i][j + 1 + NH][1], lut);
            const uint32_t r3 = swiglu_pair(acc[i][j + 1][2], acc[i][j + 1][3], acc[i][j + 1 + NH][2],
                                            acc[i][j + 1 + NH][3], lut);
            stsm4(base + i * 16 * P + j * 16, r0, r1, r2, r3);
        }
    __syncthreads();
    constexpr int CPR = T::BN / 16, RPI = 256 / CPR;   // 16-byte chunks per output row, rows per pass
    const int cc = t % CPR, rr = t / CPR;
#pragma unroll
    for (int q = 0; q < T::BM / RPI; ++q) {
        const int r = rr + q * RPI;
        uint4 v;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "r"(sm + r * P + cc * 16));
        if (m0 + r < M) *reinterpret_cast<uint4*>(O + (size_t)(m0 + r) * ldo + o0 + cc * 8) = v;
    }
}

// ------------------------------------------------------------------------------------------------ kernels
#define ARGS                                                                                                       \
    const bf16 *__restrict__ A, const bf16 *__restrict__ B, bf16 *__restrict__ Cm, int M, int N, int K, int lda,   \
        int ldc, int chunk, int mb, void *__restrict__ ws, int *__restrict__ flags

// SwiGLU: W = [Wg; Wu] (N = 2H rows); out [M, H] (ldc = its row stride); a TB tile covers TB::BN / 2 output columns;
// rows [0, mb) in TB tiles (CTAs 0 .. tb-1, row-tile fastest), rows [mb, M) in TS tiles (the CTAs after); ws = the
// 64K-entry SiLU table (uint16, index = the bf16 bits of g)
template <class TB, class TS, int KC>
DEVI void nsw7_body(ARGS) {
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    const uint32_t sm = smem_u32(smem_raw);
    const int H = N / 2;
    const uint16_t* lut = reinterpret_cast<const uint16_t*>(ws);
    const int mtb = mb / TB::BM, tb = mtb * (H / (TB::BN / 2));
    int b = blockIdx.x;
    if (b < tb) {
        float acc[TB::MT][TB::NT][4];
        const int m0 = (b % mtb) * TB::BM, o0 = (b / mtb) * (TB::BN / 2);
        ml7<TB, KC, 1>(A, B, M, m0, o0, H, 0, KC / 64, sm, acc);
        store7_swiglu<TB>(acc, Cm, M, ldc, m0, o0, sm, lut);
    } else {
        b -= tb;
        const int mts = (M - mb + TS::BM - 1) / TS::BM;
        float acc[TS::MT][TS::NT][4];
        const int m0 = mb + (b % mts) * TS::BM, o0 = (b / mts) * (TS::BN / 2);
        ml7<TS, KC, 1>(A, B, M, m0, o0, H, 0, KC / 64, sm, acc);
        store7_swiglu<TS>(acc, Cm, M, ldc, m0, o0, sm, lut);
    }
}

// split-K partials (E5-4 NP-FOLD layout): grid = S slices x tiles (slice-major); P[s] = Cm + s * M * ldc
template <class TB, class TS, int KC>
DEVI void np7_body(ARGS) {
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    const uint32_t sm = smem_u32(smem_raw);
    const int mtb = mb / TB::BM, tb = mtb * (N / TB::BN);
    const int mts = (M - mb + TS::BM - 1) / TS::BM;
    const int tiles = tb + mts * (N / TS::BN);
    const int s = blockIdx.x / tiles;
    int b = blockIdx.x % tiles;
    const int kb = s * chunk, kt = (min(KC, kb + chunk) - kb) / 64;
    bf16* Ps = Cm + (size_t)s * M * ldc;
    if (b < tb) {
        float acc[TB::MT][TB::NT][4];
        const int m0 = (b % mtb) * TB::BM, n0 = (b / mtb) * TB::BN;
        ml7<TB, KC, 0>(A, B, M, m0, n0, 0, kb, kt, sm, acc);
        store7<TB>(acc, Ps, M, ldc, m0, n0, sm);
    } else {
        b -= tb;
        float acc[TS::MT][TS::NT][4];
        const int m0 = mb + (b % mts) * TS::BM, n0 = (b / mts) * TS::BN;
        ml7<TS, KC, 0>(A, B, M, m0, n0, 0, kb, kt, sm, acc);
        store7<TS>(acc, Ps, M, ldc, m0, n0, sm);
    }
}

#define NSW7(NAME, TB, TS, KC)                                                                                      \
    extern "C" __global__ void __launch_bounds__(256, 1) NAME(ARGS) {                                               \
        nsw7_body<TB, TS, KC>(A, B, Cm, M, N, K, lda, ldc, chunk, mb, ws, flags);                                   \
    }
#define NP7(NAME, TB, TS, KC)                                                                                       \
    extern "C" __global__ void __launch_bounds__(256, 1) NAME(ARGS) {                                               \
        np7_body<TB, TS, KC>(A, B, Cm, M, N, K, lda, ldc, chunk, mb, ws, flags);                                    \
    }

// gate|up + SwiGLU (18432 x 2560 -> 9216): e6 r3 held +1.4 / +1.7 % over NG9B's nsw9 at R 896 / 640
NSW7(nsw7l_b128x256_s64x256_k2560, B7, S7a, 2560)
NSW7(nsw7l_b128x256_s32x256_k2560, B7, S7b, 2560)
// NG7T tall tails (rgh): drop 1 for t160 / t192 / t224, drop 0 for s96
NSW7(nsw7l_b128x256_t160x128_k2560, B7, T7a, 2560)
NSW7(nsw7l_b128x256_t192x128_k2560, B7, T7b, 2560)
NSW7(nsw7l_b128x256_t224x128_k2560, B7, T7c, 2560)
NSW7(nsw7l_b128x256_s96x256_k2560, B7, S7c, 2560)
// split-K partials: down (2560 x 9216), out_proj / o_proj (2560 x 4096), k|v (2048 x 2560); e6 r3 held +3 % (down,
// out_proj) and +11-14 % (k|v) over E5-4's np_* kernels (before the consumer fold)
NP7(np7_b128x256_s64x256_k9216, B7, S7a, 9216)
NP7(np7_b128x256_s64x256_k4096, B7, S7a, 4096)
NP7(np7_b128x256_s64x256_k2560, B7, S7a, 2560)

// ------------------------------------------------------------------------------------------------ table check
// out[i] = the SwiGLU epilogue of (g[i], u[i]) (bf16 bit patterns) with the kernels' arithmetic: ng.check_ng7 runs it
// over every bf16 g x 8 u values at start-up against torch F.silu(g) * u and the engine's silu_mul
extern "C" __global__ void swiglu7_elem_lut(const uint16_t* g, const uint16_t* u, uint16_t* o, const uint16_t* lut,
                                            int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gf = __uint_as_float(uint32_t(g[i]) << 16), uf = __uint_as_float(uint32_t(u[i]) << 16);
    o[i] = uint16_t(swiglu_pair(gf, 0.f, uf, 0.f, lut) & 0xffffu);
}
