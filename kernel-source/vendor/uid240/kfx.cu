// KFX (kai/kx-k2, backlog #6 KF-MX, 2026-10-22): KF's mainloop and epilogue (kf.cu's kf_tile, unchanged) for the MX
// two-class GEMMs that skx6 / skx6t serve in the mixed-bucket graphs (128 -> 192, 192 -> 256): down 2560x9216,
// GDN out_proj / attention o_proj 2560x4096, k|v (2048x2560, k / v 1024x2560), z 4096x2560.
//   C[M, N] = A[M, K] @ B[N, K]^T, A = activations (lda), B = weight [N, K] (ld K), C bf16 (ldc).
// Rows [0, M1) take class 1 (slice width chunk1), rows [M1, M) class 2 (chunk2); M1 = *m1p (int32 on the device: the
// graph's meta blob, so one captured graph serves every split), clamped to [0, M]; m1p == nullptr -> M1 = m1s.
// Arithmetic per stored element = skx6's / skx6t's / KF's (the pin's bit class): per class, S' = ceil(K / chunk) ceil64
// slices, each ONE ascending-k chain of mma.sync m16n8k16 (bf16 in, fp32 from +0) rounded once to bf16, folded in
// slice order with fp32 RN adds from the first partial, rounded once (chunk = K: y = bf16(acc)). KF's fold starts from
// -0.0f (-0 + f(p_0) == f(p_0) bit for bit, +-0 included) and runs each slice's first k16 step as a zero-C MMA.
// Tile map (kfx_map.h): each segment in KF's 128x128 tiles plus ONE row-tail tile of 32 / 64 / 96 rows (128 for
// 97..127), big tiles of both segments first; a tile never mixes classes (rows past its segment end: zero-filled
// loads, no stores). Grid (N / 128, ceil(M / 128) + 1): enough row tiles for every M1, the spare ones exit at once.
// What changes against skx6t: KF's continuous cp.async pipeline over all K, zero-C fold, 16-byte smem epilogue (_e),
// in-launch row tails per class segment.
// Needs N % 128 == 0, K % 64 == 0, chunk1 % 64 == 0, chunk2 % 64 == 0, 16-byte aligned rows. Kernel args = skx6t's:
// (A, B, C, M, N, K, lda, ldc, chunk1, chunk2, m1p, m1s).
// Build (nvcc 13.4, V13.4.92; kfx.cu, kfx_map.h and kf.cu in this directory):
//   nvcc -arch=sm_120a -O3 -cubin -Xptxas -v -o kfx.cubin kfx.cu
#define KF_NO_KERNELS
#include "kf.cu"
#include "kfx_map.h"

// M1 from the device (an asm volatile load: never merged with another load of it), clamped, warp-uniform for ptxas
// (the shuffle from lane 0: no WARPSYNC around the tiles' ldmatrix / mma.sync)
DEVI int kfx_m1(const int* __restrict__ m1p, int m1s, int M) {
    int M1 = m1s;
    if (m1p != nullptr) asm volatile("ld.global.nc.b32 %0, [%1];\n" : "=r"(M1) : "l"(m1p));
    M1 = M1 < 0 ? 0 : (M1 > M ? M : M1);
    return __shfl_sync(0xffffffffu, M1, 0);
}

// _e: the tile's m0 / mend re-derived after the mainloop (M1 loaded again, the same map), so they hold no register
// across it: KF's _e tile sits at 254 registers and a kept m0 / mend spills (ptxas: 28 bytes); the plain-epilogue
// kernel keeps them (with the re-derivation it is the one that spills)
struct KfxLate {
    const int* m1p;
    int m1s, M;
    DEVI void operator()(int& m0, int& mend) const {
        int cls, h;
        kfx_tile(blockIdx.y, M, kfx_m1(m1p, m1s, M), &m0, &mend, &cls, &h);
    }
};

template <class C, bool EPI_SMEM>
DEVI void kfx_tile_run(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
                       __nv_bfloat16* __restrict__ Cm, int m0, int mend, int n0, int K, int lda, int ldc, int KC,
                       const int* __restrict__ m1p, int m1s, int M, unsigned char* smem_raw) {
    if constexpr (EPI_SMEM)
        kf_tile<C, true, KfxLate>(A, B, Cm, m0, mend, n0, K, lda, ldc, KC, smem_raw, KfxLate{m1p, m1s, M});
    else
        kf_tile<C, false>(A, B, Cm, m0, mend, n0, K, lda, ldc, KC, smem_raw);
}

template <bool EPI_SMEM>
DEVI void kfx_main(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
                   __nv_bfloat16* __restrict__ Cm, int M, int K, int lda, int ldc, int chunk1, int chunk2,
                   const int* __restrict__ m1p, int m1s, unsigned char* smem_raw) {
    const int M1 = kfx_m1(m1p, m1s, M);
    int m0, mend, cls, h;
    if (!kfx_tile(blockIdx.y, M, M1, &m0, &mend, &cls, &h)) return;
    const int n0 = blockIdx.x * B128::BN;
    const int KC = (cls == 1 ? chunk1 : chunk2) / B128::BK;   // every config has BK = 64
    if (h == 32)
        kfx_tile_run<S32, EPI_SMEM>(A, B, Cm, m0, mend, n0, K, lda, ldc, KC, m1p, m1s, M, smem_raw);
    else if (h == 64)
        kfx_tile_run<S64, EPI_SMEM>(A, B, Cm, m0, mend, n0, K, lda, ldc, KC, m1p, m1s, M, smem_raw);
    else if (h == 96)
        kfx_tile_run<S96, EPI_SMEM>(A, B, Cm, m0, mend, n0, K, lda, ldc, KC, m1p, m1s, M, smem_raw);
    else
        kfx_tile_run<B128, EPI_SMEM>(A, B, Cm, m0, mend, n0, K, lda, ldc, KC, m1p, m1s, M, smem_raw);
}

#define KFX(NAME, EPI)                                                                                             \
    extern "C" __global__ void __launch_bounds__(B128::T, 1)                                                       \
        NAME(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Cm, \
             int M, int N, int K, int lda, int ldc, int chunk1, int chunk2, const int* __restrict__ m1p, int m1s) { \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        kfx_main<EPI>(A, B, Cm, M, K, lda, ldc, chunk1, chunk2, m1p, m1s, smem_raw);                                \
    }

static_assert(B128::T == S32::T && B128::T == S64::T && B128::T == S96::T, "one block size for every tile config");
static_assert(B128::BN == S32::BN && B128::BN == S64::BN && B128::BN == S96::BN, "one N tile for every tile config");
static_assert(B128::BM == KFX_BM, "the tile map's big tile is B128");
static_assert(B128::BK == 64 && S32::BK == 64 && S64::BK == 64 && S96::BK == 64, "one k-tile width");
KFX(gemm_kfx_b128, false)
KFX(gemm_kfx_b128_e, true)
