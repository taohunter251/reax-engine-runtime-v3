// NG3: bf16 GEMMs for sm_120 (RTX PRO 6000), C[M, N] = A[M, K] @ B[N, K]^T, bit-identical to cuBLASLt pins.
// Arithmetic (byte-exact vs cuBLASLt / torch F.linear, ngtest2 2026-10-19):
//   unsplit: fp32 accumulation over K in ascending order through mma.sync m16n8k16, one RN round to bf16;
//   split (S ceil64 slices of `chunk`, the last the rest): each slice accumulated as unsplit, slices folded in slice
//   order by the pin's reduction scheme: 1 INPLACE o = bf16(acc_0); o = bf16(float(o) + acc_s); 2 COMPUTE_TYPE
//   bf16(sum acc_s) (fp32); 4 OUTPUT_TYPE bf16(sum_s float(bf16(acc_s))) (fp32).
// Kernels:
//   nh_*  unsplit, two tile shapes in one launch: rows [0, mb) in big tiles (CTAs 0 .. Tb-1), rows [mb, M) in small
//         tiles (the CTAs after them). CTAs dispatch roughly in index order, so the small tiles fill the last wave.
//   nf_*  split, serial fixup: one CTA per (slice, tile), slice-major order. A slice CTA accumulates its K range, waits
//         until the previous slice of its tile has folded (flag == s), folds its sum into the fp32 running total
//         (workspace) or, for the last slice, writes the bf16 output and resets the flag. Fold order = slice order.
//   ns_*  split inside one CTA (all slices in order), for row counts whose tiles fill about one wave.
// Mainloop: cp.async ring, XOR-swizzled smem + ldmatrix, fragments double-buffered across k-steps and k-tiles.
// Needs N % BN == 0, K % BK == 0 (slices end on 64-wide boundaries), 16-byte aligned rows.
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
DEVI void cp_async16(uint32_t dst, const void* src, bool pred) {
    int sz = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
DEVI void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
DEVI void mma16816(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
template <int MODE>
DEVI float combine(float total, float acc, bool first) {
    if constexpr (MODE == 2) return first ? acc : total + acc;
    else if constexpr (MODE == 1) return first ? rbf(acc) : rbf(total + acc);
    else return first ? rbf(acc) : total + rbf(acc);
}
DEVI int ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
    return v;
}
DEVI void st_release(int* p, int v) { asm volatile("st.release.gpu.global.b32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory"); }

template <int BK>
DEVI int swz_f(int r) {
    if constexpr (BK == 32) return (r >> 1) & 3;
    else return r & 7;
}

template <int BM_, int BN_, int BK_, int WARPS_M, int WARPS_N, int STAGES_>
struct Cfg {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, STAGES = STAGES_, WM = WARPS_M, WN = WARPS_N;
    static constexpr int THREADS = WARPS_M * WARPS_N * 32;
    static constexpr int WTM = BM / WARPS_M, WTN = BN / WARPS_N;
    static constexpr int MT = WTM / 16, NT = WTN / 8, KS = BK / 16, CH = BK / 8;
    static constexpr int A_STAGE = BM * BK, B_STAGE = BN * BK;
    static constexpr int SMEM_PIPE = STAGES * (A_STAGE + B_STAGE) * 2;
    static constexpr int CPAD = 8;
    static constexpr int SMEM_EPI = BM * (BN + CPAD) * 2;
    static constexpr int SMEM = SMEM_PIPE > SMEM_EPI ? SMEM_PIPE : SMEM_EPI;
    static_assert(MT >= 1 && NT % 2 == 0 && KS % 2 == 0, "tile");
};

template <class C>
DEVI void load_stage(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, uint32_t sA,
                     uint32_t sB, int m0, int n0, int k0, int M, int K, int lda, int n1 = -1) {
    const int tid = threadIdx.x;
#pragma unroll
    for (int i = tid; i < C::A_STAGE / 8; i += C::THREADS) {
        int r = i / C::CH, c = i % C::CH;
        int gm = m0 + r;
        bool ok = gm < M;
        cp_async16(sA + (r * C::BK + ((c ^ swz_f<C::BK>(r)) * 8)) * 2, A + (size_t)(ok ? gm : 0) * lda + k0 + c * 8, ok);
    }
#pragma unroll
    for (int i = tid; i < C::B_STAGE / 8; i += C::THREADS) {
        int r = i / C::CH, c = i % C::CH;
        // n1 >= 0 (SwiGLU): rows [0, BN/2) are W rows n0.., rows [BN/2, BN) are W rows n1..
        int wr = (n1 < 0 || r < C::BN / 2) ? n0 + r : n1 + r - C::BN / 2;
        cp_async16(sB + (r * C::BK + ((c ^ swz_f<C::BK>(r)) * 8)) * 2, B + (size_t)wr * K + k0 + c * 8, true);
    }
}

// acc (+)= A[m0:m0+BM, kb:ke] @ B[n0:n0+BN, kb:ke]^T, k ascending. FOLD: fold acc into tot every `chunk` columns
// (in-CTA split); otherwise acc holds the plain sum over [kb, ke).
template <class C, int FOLD>
DEVI void mainloop(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, int M, int K, int lda,
                   int m0, int n0, int kb, int ke, int chunk, __nv_bfloat16* smem, float (&acc)[C::MT][C::NT][4],
                   float (&tot)[FOLD ? C::MT : 1][FOLD ? C::NT : 1][4], int n1 = -1) {
    constexpr int MT = C::MT, NT = C::NT, KS = C::KS, BK = C::BK, STAGES = C::STAGES;
    const uint32_t sA0 = smem_u32(smem);
    const uint32_t sB0 = sA0 + STAGES * C::A_STAGE * 2;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int KT = (ke - kb) / BK;
    const int KT_CH = FOLD ? chunk / BK : KT;
    const int a_r = wm * C::WTM + (lane % 16);
    const int a_f = swz_f<BK>(lane % 16);
    const int a_kc = lane / 16;
    const int b_r = wn * C::WTN + (lane / 16) * 8 + (lane % 8);
    const int b_f = swz_f<BK>((lane / 16) * 8 + (lane % 8));
    const int b_kc = (lane / 8) % 2;
    uint32_t a_off[KS], b_off[KS];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
        a_off[ks] = (a_r * BK + (((ks * 2 + a_kc) ^ a_f) * 8)) * 2;
        b_off[ks] = (b_r * BK + (((ks * 2 + b_kc) ^ b_f) * 8)) * 2;
    }
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT)
            load_stage<C>(A, B, sA0 + s * C::A_STAGE * 2, sB0 + s * C::B_STAGE * 2, m0, n0, kb + s * BK, M, K, lda, n1);
        cp_commit();
    }
    cp_wait<STAGES - 2>();
    __syncthreads();
    uint32_t af[2][MT][4];
    uint32_t bf[2][NT][2];
    auto load_frags = [&](int buf, int stage, int ks) {
        const uint32_t a = sA0 + stage * C::A_STAGE * 2 + a_off[ks];
        const uint32_t b = sB0 + stage * C::B_STAGE * 2 + b_off[ks];
#pragma unroll
        for (int i = 0; i < MT; ++i)
            ldsm_x4(a + i * 16 * BK * 2, af[buf][i][0], af[buf][i][1], af[buf][i][2], af[buf][i][3]);
#pragma unroll
        for (int j = 0; j < NT; j += 2)
            ldsm_x4(b + j * 8 * BK * 2, bf[buf][j][0], bf[buf][j][1], bf[buf][j + 1][0], bf[buf][j + 1][1]);
    };
    load_frags(0, 0, 0);
    int in_slice = 0;
    bool first = true;
    for (int kt = 0; kt < KT; ++kt) {
        const int stage = kt % STAGES;
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            if (ks == KS - 1) {
                cp_wait<STAGES - 2>();
                __syncthreads();
            }
            if (ks < KS - 1)
                load_frags((ks + 1) & 1, stage, ks + 1);
            else if (kt + 1 < KT)
                load_frags((ks + 1) & 1, (kt + 1) % STAGES, 0);
            if (ks == 0) {
                const int nk = kt + STAGES - 1;
                if (nk < KT)
                    load_stage<C>(A, B, sA0 + (nk % STAGES) * C::A_STAGE * 2, sB0 + (nk % STAGES) * C::B_STAGE * 2, m0,
                                  n0, kb + nk * BK, M, K, lda, n1);
                cp_commit();
            }
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j) mma16816(acc[i][j], af[ks & 1][i], bf[ks & 1][j][0], bf[ks & 1][j][1]);
        }
        if constexpr (FOLD != 0) {
            if (++in_slice == KT_CH || kt == KT - 1) {
#pragma unroll
                for (int i = 0; i < MT; ++i)
#pragma unroll
                    for (int j = 0; j < NT; ++j)
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            tot[i][j][q] = combine<FOLD>(tot[i][j][q], acc[i][j][q], first);
                            acc[i][j][q] = 0.f;
                        }
                first = false;
                in_slice = 0;
            }
        }
    }
    cp_wait<0>();
    __syncthreads();
}

// fragments (fp32) -> bf16 smem tile -> 16-byte stores of rows < M
template <class C>
DEVI void store_tile(const float (&v)[C::MT][C::NT][4], __nv_bfloat16* __restrict__ Cm, int M, int ldc, int m0, int n0,
                     __nv_bfloat16* smem) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int g = lane / 4, t4 = lane % 4;
    constexpr int LD = C::BN + C::CPAD;
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int j = 0; j < C::NT; ++j) {
            int r = wm * C::WTM + i * 16 + g;
            int c = wn * C::WTN + j * 8 + t4 * 2;
            *reinterpret_cast<__nv_bfloat162*>(smem + r * LD + c) = __floats2bfloat162_rn(v[i][j][0], v[i][j][1]);
            *reinterpret_cast<__nv_bfloat162*>(smem + (r + 8) * LD + c) = __floats2bfloat162_rn(v[i][j][2], v[i][j][3]);
        }
    __syncthreads();
    constexpr int CPR = C::BN / 8;
    for (int i = threadIdx.x; i < C::BM * CPR; i += C::THREADS) {
        int r = i / CPR, c = i % CPR;
        int gm = m0 + r;
        if (gm < M)
            *reinterpret_cast<uint4*>(Cm + (size_t)gm * ldc + n0 + c * 8) = *reinterpret_cast<const uint4*>(smem + r * LD + c * 8);
    }
}

template <class C>
DEVI void unsplit_tile(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
                       __nv_bfloat16* __restrict__ Cm, int M, int K, int lda, int ldc, int m0, int n0,
                       __nv_bfloat16* smem) {
    float acc[C::MT][C::NT][4];
    float tot[1][1][4];
    mainloop<C, 0>(A, B, M, K, lda, m0, n0, 0, K, K, smem, acc, tot);
    store_tile<C>(acc, Cm, M, ldc, m0, n0, smem);
}

// unsplit, big tiles over rows [0, mb), small tiles over rows [mb, M)
template <class CB, class CS>
DEVI void hetero(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Cm,
                 int M, int N, int K, int lda, int ldc, int mb, __nv_bfloat16* smem) {
    static_assert(CB::THREADS == CS::THREADS, "threads");
    const int mtb = mb / CB::BM;
    const int tb = mtb * (N / CB::BN);
    int b = blockIdx.x;
    if (b < tb) {
        unsplit_tile<CB>(A, B, Cm, M, K, lda, ldc, (b % mtb) * CB::BM, (b / mtb) * CB::BN, smem);
    } else {
        b -= tb;
        const int mts = (M - mb + CS::BM - 1) / CS::BM;
        unsplit_tile<CS>(A, B, Cm, M, K, lda, ldc, mb + (b % mts) * CS::BM, (b / mts) * CS::BN, smem);
    }
}

// split, serial fixup across CTAs; ws: fp32 [M, N] running totals, flags: int per tile (zero between launches)
template <class C, int MODE>
DEVI void fixup(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Cm,
                int M, int N, int K, int lda, int ldc, int chunk, float* __restrict__ ws, int* __restrict__ flags,
                __nv_bfloat16* smem) {
    const int mt = (M + C::BM - 1) / C::BM;
    const int tiles = mt * (N / C::BN);
    const int S = (K + chunk - 1) / chunk;
    const int s = blockIdx.x / tiles;
    const int t = blockIdx.x % tiles;
    const int m0 = (t % mt) * C::BM, n0 = (t / mt) * C::BN;
    const int kb = s * chunk, ke = min(K, kb + chunk);
    float acc[C::MT][C::NT][4];
    float tot[1][1][4];
    mainloop<C, 0>(A, B, M, K, lda, m0, n0, kb, ke, chunk, smem, acc, tot);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int g = lane / 4, t4 = lane % 4;
    if (s > 0) {
        if (threadIdx.x == 0)
            while (ld_acquire(flags + t) != s) {
            }
        __syncthreads();
    }
    // fold: tot = combine(previous total, acc) element-wise, in the fragment layout (rows >= M are never read back)
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int j = 0; j < C::NT; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                int r = m0 + wm * C::WTM + i * 16 + g + h * 8;
                int c = n0 + wn * C::WTN + j * 8 + t4 * 2;
                float p0 = 0.f, p1 = 0.f;
                if (s > 0 && r < M) {
                    float2 pv = __ldcg(reinterpret_cast<const float2*>(ws + (size_t)r * N + c));
                    p0 = pv.x, p1 = pv.y;
                }
                acc[i][j][2 * h] = combine<MODE>(p0, acc[i][j][2 * h], s == 0);
                acc[i][j][2 * h + 1] = combine<MODE>(p1, acc[i][j][2 * h + 1], s == 0);
                if (s < S - 1 && r < M)
                    __stcg(reinterpret_cast<float2*>(ws + (size_t)r * N + c), make_float2(acc[i][j][2 * h], acc[i][j][2 * h + 1]));
            }
    if (s < S - 1) {
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) st_release(flags + t, s + 1);
    } else {
        store_tile<C>(acc, Cm, M, ldc, m0, n0, smem);
        if (threadIdx.x == 0) flags[t] = 0;
    }
}

template <class C, int MODE>
DEVI void insplit(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Cm,
                  int M, int N, int K, int lda, int ldc, int chunk, __nv_bfloat16* smem) {
    const int mt = (M + C::BM - 1) / C::BM;
    const int m0 = (blockIdx.x % mt) * C::BM, n0 = (blockIdx.x / mt) * C::BN;
    float acc[C::MT][C::NT][4];
    float tot[C::MT][C::NT][4];
    mainloop<C, MODE>(A, B, M, K, lda, m0, n0, 0, K, chunk, smem, acc, tot);
    store_tile<C>(tot, Cm, M, ldc, m0, n0, smem);
}

#define ARGS                                                                                                       \
    const __nv_bfloat16 *__restrict__ A, const __nv_bfloat16 *__restrict__ B, __nv_bfloat16 *__restrict__ Cm, int M, \
        int N, int K, int lda, int ldc, int chunk, int mb, float *__restrict__ ws, int *__restrict__ flags
#define SMEMP reinterpret_cast<__nv_bfloat16*>(smem_raw)

using B128x128w8 = Cfg<128, 128, 32, 4, 2, 3>;
using B128x256 = Cfg<128, 256, 32, 2, 4, 4>;
using B128x256s3 = Cfg<128, 256, 32, 2, 4, 3>;
using S32x256 = Cfg<32, 256, 32, 1, 8, 4>;
using S64x256 = Cfg<64, 256, 32, 2, 4, 4>;
using S32x128 = Cfg<32, 128, 32, 1, 8, 4>;
using S64x128w8 = Cfg<64, 128, 32, 2, 4, 4>;
using F128x128w8 = Cfg<128, 128, 32, 4, 2, 3>;
using F128x256 = Cfg<128, 256, 32, 2, 4, 3>;
using F64x128 = Cfg<64, 128, 32, 2, 2, 4>;
using F64x256 = Cfg<64, 256, 32, 2, 4, 4>;

#define NH(NAME, CB, CS)                                                                                           \
    extern "C" __global__ void __launch_bounds__(256, 1) NAME(ARGS) {                                              \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        hetero<CB, CS>(A, B, Cm, M, N, K, lda, ldc, mb, SMEMP);                                                     \
    }
#define NF(NAME, CF, MODE, TH, MINB)                                                                               \
    extern "C" __global__ void __launch_bounds__(TH, MINB) NAME(ARGS) {                                            \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        fixup<CF, MODE>(A, B, Cm, M, N, K, lda, ldc, chunk, ws, flags, SMEMP);                                      \
    }
#define NS(NAME, CF, MODE, TH, MINB)                                                                               \
    extern "C" __global__ void __launch_bounds__(TH, MINB) NAME(ARGS) {                                            \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        insplit<CF, MODE>(A, B, Cm, M, N, K, lda, ldc, chunk, SMEMP);                                               \
    }

NH(nh_b128x128w8_s32x256, B128x128w8, S32x256)
NH(nh_b128x128w8_s64x256, B128x128w8, S64x256)
NH(nh_b128x128w8_s32x128, B128x128w8, S32x128)
NH(nh_b128x128w8_s64x128, B128x128w8, S64x128w8)
NH(nh_b128x256_s32x256, B128x256, S32x256)
NH(nh_b128x256_s64x256, B128x256, S64x256)
NH(nh_b128x256s3_s32x256, B128x256s3, S32x256)
NF(nf4_128x128w8, F128x128w8, 4, 256, 1)
NF(nf4_128x256, F128x256, 4, 256, 1)
NF(nf4_64x128, F64x128, 4, 128, 2)
NF(nf4_64x256, F64x256, 4, 256, 1)
NF(nf1_128x128w8, F128x128w8, 1, 256, 1)
NF(nf1_128x256, F128x256, 1, 256, 1)
NF(nf1_64x128, F64x128, 1, 128, 2)
NF(nf2_128x128w8, F128x128w8, 2, 256, 1)
NS(ns4_128x128w8, F128x128w8, 4, 256, 1)
NS(ns1_128x128w8, F128x128w8, 1, 256, 1)

// v4: 4-warp 128x128 big tiles (64x64 warp tiles) at 2 CTAs per SM (3 stages = 48 KB smem each), with 4-warp small
// tiles; one CTA's epilogue / barrier stalls overlap the other CTA's mainloop
using B128x128w4 = Cfg<128, 128, 32, 2, 2, 3>;
using S32x128w4 = Cfg<32, 128, 32, 1, 4, 3>;
using S64x128w4 = Cfg<64, 128, 32, 2, 2, 3>;
using B64x128w4 = Cfg<64, 128, 32, 2, 2, 4>;
#define NH2(NAME, CB, CS, TH, MINB)                                                                                \
    extern "C" __global__ void __launch_bounds__(TH, MINB) NAME(ARGS) {                                            \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        hetero<CB, CS>(A, B, Cm, M, N, K, lda, ldc, mb, SMEMP);                                                     \
    }
NH2(nh4_b128x128_s32x128, B128x128w4, S32x128w4, 128, 2)
NH2(nh4_b128x128_s64x128, B128x128w4, S64x128w4, 128, 2)
NH2(nh4_b64x128_s32x128, B64x128w4, S32x128w4, 128, 2)
NS(ns4_128x128w4, B128x128w4, 4, 128, 2)
NS(ns1_128x128w4, B128x128w4, 1, 128, 2)
NF(nf4_128x128w4, B128x128w4, 4, 128, 2)

// ------------------------------------------------------------------ NG5: gate|up GEMM with the SwiGLU epilogue
// Reference: g = bf16(x @ Wg^T), u = bf16(x @ Wu^T) (unsplit pins), F.silu(g) -> bf16 (opmath f32: x / (1 + expf(-x)),
// libdevice expf, IEEE division), * u -> bf16. W = [Wg; Wu] stacked ([2H, K]); a tile of BN columns computes BN/2 gate
// columns and the same BN/2 up columns, rounds both to bf16 in smem, and writes BN/2 SwiGLU outputs (out [M, H]).
DEVI float silu_f32(float x) { return __fdiv_rn(x, 1.0f + expf(-x)); }
DEVI __nv_bfloat16 swiglu1(__nv_bfloat16 g, __nv_bfloat16 u) {
    float s = __bfloat162float(__float2bfloat16_rn(silu_f32(__bfloat162float(g))));
    return __float2bfloat16_rn(s * __bfloat162float(u));
}

template <class C>
DEVI void store_tile_swiglu(const float (&v)[C::MT][C::NT][4], __nv_bfloat16* __restrict__ Out, int M, int ldo, int m0,
                            int o0, __nv_bfloat16* smem) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int g = lane / 4, t4 = lane % 4;
    constexpr int LD = C::BN + C::CPAD;
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int j = 0; j < C::NT; ++j) {
            int r = wm * C::WTM + i * 16 + g;
            int c = wn * C::WTN + j * 8 + t4 * 2;
            *reinterpret_cast<__nv_bfloat162*>(smem + r * LD + c) = __floats2bfloat162_rn(v[i][j][0], v[i][j][1]);
            *reinterpret_cast<__nv_bfloat162*>(smem + (r + 8) * LD + c) = __floats2bfloat162_rn(v[i][j][2], v[i][j][3]);
        }
    __syncthreads();
    constexpr int H = C::BN / 2, CPR = H / 8;
    for (int i = threadIdx.x; i < C::BM * CPR; i += C::THREADS) {
        int r = i / CPR, c = (i % CPR) * 8;
        int gm = m0 + r;
        if (gm >= M) continue;
        uint4 gv = *reinterpret_cast<const uint4*>(smem + r * LD + c);
        uint4 uv = *reinterpret_cast<const uint4*>(smem + r * LD + H + c);
        const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
        const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&uv);
        uint4 ov;
        __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(&ov);
#pragma unroll
        for (int e = 0; e < 8; ++e) op[e] = swiglu1(gp[e], up[e]);
        *reinterpret_cast<uint4*>(Out + (size_t)gm * ldo + o0 + c) = ov;
    }
}

template <class C>
DEVI void swiglu_tile(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
                      __nv_bfloat16* __restrict__ Out, int M, int N, int K, int lda, int ldo, int m0, int t,
                      __nv_bfloat16* smem) {
    const int H = C::BN / 2;
    const int o0 = t * H;                  // output column = gate row index
    float acc[C::MT][C::NT][4];
    float tot[1][1][4];
    mainloop<C, 0>(A, B, M, K, lda, m0, o0, 0, K, K, smem, acc, tot, N / 2 + o0);
    store_tile_swiglu<C>(acc, Out, M, ldo, m0, o0, smem);
}

// N = 2H rows of W; rows [0, mb) in big tiles, [mb, M) in small tiles (as nh_*); Cm = out [M, H] (ldc = its row stride)
template <class CB, class CS>
DEVI void swiglu_hetero(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
                        __nv_bfloat16* __restrict__ Out, int M, int N, int K, int lda, int ldo, int mb,
                        __nv_bfloat16* smem) {
    static_assert(CB::THREADS == CS::THREADS, "threads");
    const int H = N / 2;
    const int mtb = mb / CB::BM;
    const int tb = mtb * (H / (CB::BN / 2));
    int b = blockIdx.x;
    if (b < tb) {
        swiglu_tile<CB>(A, B, Out, M, N, K, lda, ldo, (b % mtb) * CB::BM, b / mtb, smem);
    } else {
        b -= tb;
        const int mts = (M - mb + CS::BM - 1) / CS::BM;
        swiglu_tile<CS>(A, B, Out, M, N, K, lda, ldo, mb + (b % mts) * CS::BM, b / mts, smem);
    }
}

#define NSW(NAME, CB, CS, TH, MINB)                                                                                \
    extern "C" __global__ void __launch_bounds__(TH, MINB) NAME(ARGS) {                                            \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        swiglu_hetero<CB, CS>(A, B, Cm, M, N, K, lda, ldc, mb, SMEMP);                                              \
    }
NSW(nsw_b128x256_s32x256, B128x256, S32x256, 256, 1)
NSW(nsw_b128x256_s64x256, B128x256, S64x256, 256, 1)
NSW(nsw_b128x256s3_s32x256, B128x256s3, S32x256, 256, 1)
NSW(nsw_b128x128w8_s32x256, B128x128w8, S32x256, 256, 1)
NSW(nsw_b128x128w8_s32x128, B128x128w8, S32x128, 256, 1)
NSW(nsw_b128x128w8_s64x128, B128x128w8, S64x128w8, 256, 1)

// exhaustive check of the epilogue arithmetic: out[i] = swiglu1(g[i], u[i])
extern "C" __global__ void swiglu_check(const __nv_bfloat16* __restrict__ g, const __nv_bfloat16* __restrict__ u,
                                        __nv_bfloat16* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = swiglu1(g[i], u[i]);
}
