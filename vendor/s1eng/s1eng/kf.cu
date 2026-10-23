// KF (ruj gemm2/k2, 2026-10-21): EXACT in-CTA split-K GEMM for the split-K pins of the N = 2560 GEMMs (MLP down
// 2560x9216, GDN out_proj / attention o_proj 2560x4096) and attention k|v (2048x2560), OUTPUT_TYPE scheme (4).
//   C[M, N] = A[M, K] @ B[N, K]^T, A = activations (lda), B = weight [N, K] (ld K), C bf16 (ldc).
// Arithmetic (= the pin's bit class, same as our SKX5 / SKX6): S' = ceil(K / chunk) ceil64 slices; each slice is ONE
// ascending-k chain of mma.sync m16n8k16 (bf16 in, fp32 accumulate) starting from +0 over its k16 steps, rounded once
// to bf16 (p_s); the slices are folded in slice order with fp32 RN adds starting from the first partial
// (t = f(p_0); t = t + f(p_s)), and t is rounded once to bf16. chunk = K: the unsplit class (y = bf16(acc)).
// What is new against SKX6 (all ours):
//   * row tail in the same launch: rows [0, mb) in 128x128 tiles, rows [mb, M) in small tiles (32 / 64 / 96 x 128),
//     big tiles first in blockIdx order (no MMA work on the padded rows of a last 128-row band);
//   * a fold that costs no extra tensor-pipe time: t starts at -0.0f (so t + f(p_0) == f(p_0) bit for bit), the first
//     k16 step of every slice is a zero-C MMA (C = RZ: no accumulator clearing), and the fold of a cell (one F2FP that
//     packs bf16(acc) over a zero low half = f(bf16(acc)) as an fp32, one FADD) is issued right before that cell's
//     zero-C MMA, so the folds interleave with the next slice's first MMAs instead of stalling the pipe;
//   * one continuous cp.async pipeline over all K (slice bounds never drain it; the load stream equals the unsplit one);
//   * optional smem-staged epilogue (16-byte coalesced stores).
// Needs N % 128 == 0, K % 64 == 0, chunk % 64 == 0, 16-byte aligned rows. Kernel args: (A, B, C, M, N, K, lda, ldc,
// chunk, mb); mb = rows done in big tiles (a multiple of 128; mb >= M = big tiles only, rows padded).
#include <cuda_bf16.h>
#include <stdint.h>

#include <type_traits>

#define DEVI __device__ __forceinline__

DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
DEVI void cp16(uint32_t dst, const void* src, uint32_t sz) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
DEVI void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
DEVI void mma_acc(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// first k16 step of a chain: C = +0 (ptxas uses RZ, no clearing moves)
DEVI void mma_zero(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%10,%10,%10,%10};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.f));
}
// f(bf16_rn(x)) as an fp32 in ONE instruction: pack {bf16(x), bf16(+0)} -> high half = bf16(x), low half = 0
DEVI float bfr(float x) {
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(x), "f"(0.f));
    return __uint_as_float(r);
}
DEVI uint32_t pack_bf2(float lo, float hi) {
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

template <int BK>
DEVI int swz_f(int r) {
    if constexpr (BK == 32) return (r >> 1) & 3;
    else return r & 7;
}

template <int BM_, int BN_, int BK_, int WM_, int WN_, int ST_>
struct Cfg {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, WM = WM_, WN = WN_, ST = ST_;
    static constexpr int T = WM * WN * 32;
    static constexpr int WTM = BM / WM, WTN = BN / WN;
    static constexpr int MT = WTM / 16, NT = WTN / 8, KS = BK / 16, CH = BK / 8;
    static constexpr int A_ST = BM * BK, B_ST = BN * BK;   // bf16 elements per stage
    static constexpr int AI = BM * CH / T, BI = BN * CH / T;   // 16-byte chunks per thread per stage
    static constexpr int RS = T / CH;                         // row step between one thread's chunks
    static constexpr int SMEM_PIPE = ST * (A_ST + B_ST) * 2;
    static constexpr int SMEM_EPI = BM * (BN + 8) * 2;
    static constexpr int SMEM = SMEM_PIPE > SMEM_EPI ? SMEM_PIPE : SMEM_EPI;
    static_assert(MT >= 1 && NT % 2 == 0 && KS % 2 == 0, "tile");
    static_assert((BM * CH) % T == 0 && (BN * CH) % T == 0 && RS % 8 == 0, "loader");
};

// KF: m0 / mend as passed (kfx.cu: re-derived after the mainloop, so they need no register across it)
struct KfKeep {
    DEVI void operator()(int&, int&) const {}
};

// one output tile [m0, m0 + BM) x [n0, n0 + BN), rows >= mend neither loaded (zero-filled) nor stored; late(m0, mend)
// runs once after the mainloop, before the epilogue
template <class C, bool EPI_SMEM, class LATE = KfKeep>
DEVI void kf_tile(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
                  __nv_bfloat16* __restrict__ Cm, int m0, int mend, const int n0, const int K,
                  const int lda, const int ldc, const int KC, unsigned char* smem_raw, const LATE late = LATE{}) {
    constexpr int BM = C::BM, BN = C::BN, BK = C::BK, ST = C::ST, MT = C::MT, NT = C::NT, KS = C::KS;
    const int tid = threadIdx.x;
    const int KT = K / BK;

    // ---- loader: thread's 16-byte chunk column and first row; its rows are lr0 + c * RS
    const uint32_t sA0 = smem_u32(smem_raw);
    const uint32_t sB0 = sA0 + ST * C::A_ST * 2;
    const int lr0 = tid / C::CH, lcc = tid % C::CH;
    const uint32_t lso = (lr0 * BK + ((lcc ^ swz_f<BK>(lr0)) * 8)) * 2;
    const __nv_bfloat16* pa[C::AI];
    const __nv_bfloat16* pb[C::BI];
    uint32_t asz[C::AI];
#pragma unroll
    for (int c = 0; c < C::AI; ++c) {
        const int gm = m0 + lr0 + c * C::RS;
        asz[c] = gm < mend ? 16u : 0u;
        pa[c] = A + (size_t)(gm < mend ? gm : mend - 1) * lda + lcc * 8;
    }
#pragma unroll
    for (int c = 0; c < C::BI; ++c) pb[c] = B + (size_t)(n0 + lr0 + c * C::RS) * K + lcc * 8;
    int lrem = KT;   // k-tiles still to load
    auto issue_a = [&](int st) {
        const uint32_t a = sA0 + st * C::A_ST * 2 + lso;
#pragma unroll
        for (int c = 0; c < C::AI; ++c) cp16(a + c * C::RS * BK * 2, pa[c], asz[c]);
#pragma unroll
        for (int c = 0; c < C::AI; ++c) pa[c] += BK;
    };
    auto issue_b = [&](int st) {
        const uint32_t b = sB0 + st * C::B_ST * 2 + lso;
#pragma unroll
        for (int c = 0; c < C::BI; ++c) cp16(b + c * C::RS * BK * 2, pb[c], 16u);
#pragma unroll
        for (int c = 0; c < C::BI; ++c) pb[c] += BK;
    };

    // ---- fragments
    const int warp = tid / 32, lane = tid % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
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
    uint32_t af[2][MT][4];
    uint32_t bfr2[2][NT][2];
    auto load_frags = [&](int buf, int stage, int ks) {
        const uint32_t a = sA0 + stage * C::A_ST * 2 + a_off[ks];
        const uint32_t b = sB0 + stage * C::B_ST * 2 + b_off[ks];
#pragma unroll
        for (int i = 0; i < MT; ++i)
            ldsm_x4(a + i * 16 * BK * 2, af[buf][i][0], af[buf][i][1], af[buf][i][2], af[buf][i][3]);
#pragma unroll
        for (int j = 0; j < NT; j += 2)
            ldsm_x4(b + j * 8 * BK * 2, bfr2[buf][j][0], bfr2[buf][j][1], bfr2[buf][j + 1][0], bfr2[buf][j + 1][1]);
    };

    // ---- prologue: ST - 1 k-tiles in flight
#pragma unroll
    for (int s = 0; s < ST - 1; ++s) {
        if (lrem > 0) {
            issue_a(s);
            issue_b(s);
            --lrem;
        }
        cp_commit();
    }
    cp_wait<ST - 2>();
    __syncthreads();
    load_frags(0, 0, 0);
    int cst = 0, lst = ST - 1;   // compute stage, load stage

    float acc[MT][NT][4];
    float tot[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) tot[i][j][q] = -0.0f;

    // one k-tile. MODE 0: plain; 1: first k-tile of slice 0 (k-step 0 = zero-C MMAs); 2: first k-tile of a later slice
    // (k-step 0 = fold the cell's finished chain into tot, then its zero-C MMA). FAST: the load ST - 1 ahead is due.
    auto ktile = [&](auto mode_c, auto fast_c) {
        constexpr int MODE = decltype(mode_c)::value;
        constexpr bool FAST = decltype(fast_c)::value;
        const int nst = cst + 1 == ST ? 0 : cst + 1;
        const bool ld = FAST || lrem > 0;
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            if (ks == KS - 1) {
                cp_wait<ST - 2>();
                __syncthreads();
            }
            if (ks < KS - 1) load_frags((ks + 1) & 1, cst, ks + 1);
            else load_frags(0, nst, 0);
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int jj = 0; jj < NT; ++jj) {
                    const int j = (i & 1) ? NT - 1 - jj : jj;   // serpentine: B operand reuse
                    if (MODE != 0 && ks == 0) {
                        if (MODE == 2) {
#pragma unroll
                            for (int q = 0; q < 4; ++q) tot[i][j][q] = tot[i][j][q] + bfr(acc[i][j][q]);
                        }
                        mma_zero(acc[i][j], af[0][i], bfr2[0][j][0], bfr2[0][j][1]);
                    } else {
                        mma_acc(acc[i][j], af[ks & 1][i], bfr2[ks & 1][j][0], bfr2[ks & 1][j][1]);
                    }
                }
            if (ks == 0 && ld) issue_a(lst);
            if (ks == (KS > 2 ? 1 : 0)) {
                if (ld) {
                    issue_b(lst);
                    --lrem;
                }
                cp_commit();
                lst = lst + 1 == ST ? 0 : lst + 1;
            }
        }
        cst = nst;
    };

    using I0 = std::integral_constant<int, 0>;
    using I1 = std::integral_constant<int, 1>;
    using I2 = std::integral_constant<int, 2>;
    const int kfast = KT - (ST - 1);   // k-tiles < kfast: the load ST - 1 ahead is inside K
    int kt = 0;
    while (kt < KT) {
        const int k1 = min(kt + KC, KT);
        if (kt == 0) {
            if (kt < kfast) ktile(I1{}, std::true_type{});
            else ktile(I1{}, std::false_type{});
        } else {
            if (kt < kfast) ktile(I2{}, std::true_type{});
            else ktile(I2{}, std::false_type{});
        }
        ++kt;
        const int f = min(k1, kfast);
        for (; kt < f; ++kt) ktile(I0{}, std::true_type{});
        for (; kt < k1; ++kt) ktile(I0{}, std::false_type{});
    }
    cp_wait<0>();
    late(m0, mend);
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) tot[i][j][q] = tot[i][j][q] + bfr(acc[i][j][q]);

    const int g = lane / 4, t4 = lane % 4;
    if constexpr (!EPI_SMEM) {
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int r = m0 + wm * C::WTM + i * 16 + g;
                const int c = n0 + wn * C::WTN + j * 8 + t4 * 2;
                if (r < mend)
                    *reinterpret_cast<uint32_t*>(Cm + (size_t)r * ldc + c) = pack_bf2(tot[i][j][0], tot[i][j][1]);
                if (r + 8 < mend)
                    *reinterpret_cast<uint32_t*>(Cm + (size_t)(r + 8) * ldc + c) = pack_bf2(tot[i][j][2], tot[i][j][3]);
            }
    } else {
        constexpr int LDS = BN + 8;   // bf16 elements per staged row (16-byte pad: conflict-free 4-byte writes)
        __syncthreads();              // every warp is done with the pipeline buffers
        uint32_t* st32 = reinterpret_cast<uint32_t*>(smem_raw);
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int r = wm * C::WTM + i * 16 + g;
                const int c = wn * C::WTN + j * 8 + t4 * 2;
                st32[(r * LDS + c) / 2] = pack_bf2(tot[i][j][0], tot[i][j][1]);
                st32[((r + 8) * LDS + c) / 2] = pack_bf2(tot[i][j][2], tot[i][j][3]);
            }
        __syncthreads();
        constexpr int CPR = BN / 8;   // 16-byte chunks per row
#pragma unroll
        for (int e = tid; e < BM * CPR; e += C::T) {
            const int r = e / CPR, cc = e % CPR;
            if (m0 + r < mend) {
                const uint4 v = *reinterpret_cast<const uint4*>(smem_raw + (r * LDS + cc * 8) * 2);
                *reinterpret_cast<uint4*>(Cm + (size_t)(m0 + r) * ldc + n0 + cc * 8) = v;
            }
        }
    }
}

template <class BIG, class SMALL, bool EPI_SMEM>
DEVI void kf_main(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
                  __nv_bfloat16* __restrict__ Cm, int M, int N, int K, int lda, int ldc, int chunk, int mb,
                  unsigned char* smem_raw) {
    const int nb = N / BIG::BN;
    const int bt = (mb < M ? mb : ((M + BIG::BM - 1) / BIG::BM) * BIG::BM) / BIG::BM * nb;
    const int KC = chunk / BIG::BK;
    int t = blockIdx.x;
    if (t < bt) {
        kf_tile<BIG, EPI_SMEM>(A, B, Cm, (t / nb) * BIG::BM, M, (t % nb) * BIG::BN, K, lda, ldc, KC, smem_raw);
    } else {
        t -= bt;
        const int ns = N / SMALL::BN;
        kf_tile<SMALL, EPI_SMEM>(A, B, Cm, mb + (t / ns) * SMALL::BM, M, (t % ns) * SMALL::BN, K, lda, ldc,
                                 chunk / SMALL::BK, smem_raw);
    }
}

#define KF(NAME, BIGC, SMALLC, EPI)                                                                                \
    extern "C" __global__ void __launch_bounds__(BIGC::T, 1)                                                       \
        NAME(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Cm, \
             int M, int N, int K, int lda, int ldc, int chunk, int mb) {                                           \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        kf_main<BIGC, SMALLC, EPI>(A, B, Cm, M, N, K, lda, ldc, chunk, mb, smem_raw);                               \
    }

// (BM, BN, BK, warps M, warps N, stages)
using B128 = Cfg<128, 128, 64, 2, 4, 3>;   // 96 KB, warp tile 64x32
using S32 = Cfg<32, 128, 64, 2, 4, 4>;     // 80 KB, warp tile 16x32
using S64 = Cfg<64, 128, 64, 2, 4, 4>;     // 96 KB, warp tile 32x32
using S96 = Cfg<96, 128, 64, 2, 4, 3>;     // 84 KB, warp tile 48x32
#ifndef KF_NO_KERNELS   // kfx.cu includes this file for kf_tile and the configs only (kf.cubin: unchanged bytes)
KF(gemm_kf_b128_s32, B128, S32, false)
KF(gemm_kf_b128_s64, B128, S64, false)
KF(gemm_kf_b128_s96, B128, S96, false)
KF(gemm_kf_b128_s32_e, B128, S32, true)
KF(gemm_kf_b128_s64_e, B128, S64, true)
KF(gemm_kf_b128_s96_e, B128, S96, true)
#endif
