// SKX6 (ruj r2 mixbatch, 2026-10-21): EXACT two-class GEMM for cross-bucket ("mixed") batches, one CTA per output
// tile with the K slices folded in registers, the segment boundary read from DEVICE memory (so one captured CUDA graph
// serves every split of its rows between the two buckets).
//   C[M, N] = A[M, K] @ B[N, K]^T, A = activations (lda), B = weight [N, K] (ld K), C bf16 (ldc).
// Rows [0, M1) are computed in class 1 (K slices of chunk1 columns), rows [M1, M) in class 2 (chunk2); chunk = K is
// the unsplit class. M1 = *m1p (int32 on the device, clamped to [0, M]); m1p == nullptr -> M1 = m1s.
// Arithmetic of a class = the pin's bit class, as SKX5 (n2560 track, mc_test exact on every JOIN shape) and rgh's
// ns9 / RARN nsq9 in-CTA split: S = ceil(K / chunk) slices, the last one the rest; each slice is one ascending-k chain
// of mma.sync m16n8k16 (fp32) from +0 over its k16 steps, rounded once to bf16; the slices are folded in slice order
// with fp32 RN adds starting from the first partial itself (tot = f(b(acc_0)); tot = tot + f(b(acc_s))), rounded once:
// scheme 4 (OUTPUT_TYPE). S = 1: bf16(acc_0).
// Schedule: grid (N / BN, ceil(M / BM) + 1). Row tile t < mt1 = ceil(M1 / BM) covers rows [t BM, min(t BM + BM, M1));
// the others cover segment 2 from row M1 on; tiles past both segments exit at once. Each CTA runs its tile's whole K
// through one continuous cp.async pipeline (SKX3's branch-free hot loop; loads checked only in the last ST k-tiles)
// and folds a slice's chain into the fp32 total in registers at every slice end (no partial buffers, no tickets).
// Needs N % BN == 0, K % BK == 0, chunk % BK == 0, 16-byte aligned rows.
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
DEVI void mma16816(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI uint32_t pack_bf2(float a, float b) {
    __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&v);
}
DEVI float bf_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

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
    static constexpr int SMEM = ST * (A_ST + B_ST) * 2;
    static constexpr int E = MT * NT * 4;    // fp32 accumulators per thread
    static_assert(MT >= 1 && NT % 2 == 0 && KS % 2 == 0, "tile");
    static_assert((BM * CH) % T == 0 && (BN * CH) % T == 0 && RS % 8 == 0, "loader");
};

template <class C>
DEVI void skx6(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Cm,
               int M, int N, int K, int lda, int ldc, int chunk1, int chunk2, const int* __restrict__ m1p, int m1s,
               unsigned char* smem_raw) {
    constexpr int BM = C::BM, BN = C::BN, BK = C::BK, ST = C::ST, MT = C::MT, NT = C::NT, KS = C::KS;
    const int tid = threadIdx.x;
    int M1 = m1p != nullptr ? __ldg(m1p) : m1s;
    M1 = M1 < 0 ? 0 : (M1 > M ? M : M1);
    const int mt1 = (M1 + BM - 1) / BM, mt2 = (M - M1 + BM - 1) / BM;
    const int t = blockIdx.y;
    int m0, mend, chunk;
    if (t < mt1) {
        m0 = t * BM; mend = M1; chunk = chunk1;
    } else if (t - mt1 < mt2) {
        m0 = M1 + (t - mt1) * BM; mend = M; chunk = chunk2;
    } else {
        return;   // past both segments (the grid has one spare row tile for a boundary inside a tile)
    }
    const int n0 = blockIdx.x * BN;
    const int KT = K / BK;
    const int KC = chunk / BK;
    const int S = (KT + KC - 1) / KC;

    // ---- loader: thread's chunk column and first row; its rows are r0 + c * RS
    const uint32_t sA0 = smem_u32(smem_raw);
    const uint32_t sB0 = sA0 + ST * C::A_ST * 2;
    const int lr0 = tid / C::CH, lcc = tid % C::CH;
    const uint32_t lso = (lr0 * BK + ((lcc ^ swz_f<BK>(lr0)) * 8)) * 2;   // + c * RS * BK * 2 per chunk
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
    uint32_t bfr[2][NT][2];
    auto load_frags = [&](int buf, int stage, int ks) {
        const uint32_t a = sA0 + stage * C::A_ST * 2 + a_off[ks];
        const uint32_t b = sB0 + stage * C::B_ST * 2 + b_off[ks];
#pragma unroll
        for (int i = 0; i < MT; ++i)
            ldsm_x4(a + i * 16 * BK * 2, af[buf][i][0], af[buf][i][1], af[buf][i][2], af[buf][i][3]);
#pragma unroll
        for (int j = 0; j < NT; j += 2)
            ldsm_x4(b + j * 8 * BK * 2, bfr[buf][j][0], bfr[buf][j][1], bfr[buf][j + 1][0], bfr[buf][j + 1][1]);
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
    float tot[C::E];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

    // one k-tile: MMAs of k-step ks, then this k-tile's share of the loads ST - 1 k-tiles ahead (A after step 0, B after
    // step 1, committed before the wait of the last step); FAST: a load is due (no check)
    auto ktile = [&](auto fast) {
        constexpr bool FAST = decltype(fast)::value;
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
                    mma16816(acc[i][j], af[ks & 1][i], bfr[ks & 1][j][0], bfr[ks & 1][j][1]);
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
    // k-tiles [k0, k1): FAST while the load ST - 1 ahead is surely inside K
    auto run_range = [&](int k0, int k1) {
        const int f = min(k1, KT - ST);
        int k = k0;
        for (; k < f; ++k) ktile(std::true_type{});
        for (; k < k1; ++k) ktile(std::false_type{});
    };

    const int g = lane / 4, t4 = lane % 4;
    auto store_c = [&](const float* v) {
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int r = m0 + wm * C::WTM + i * 16 + g;
                const int c = n0 + wn * C::WTN + j * 8 + t4 * 2;
                const float* q = v + (i * NT + j) * 4;
                if (r < mend) *reinterpret_cast<uint32_t*>(Cm + (size_t)r * ldc + c) = pack_bf2(q[0], q[1]);
                if (r + 8 < mend) *reinterpret_cast<uint32_t*>(Cm + (size_t)(r + 8) * ldc + c) = pack_bf2(q[2], q[3]);
            }
    };

    if (S <= 1) {
        run_range(0, KT);
        cp_wait<0>();
        store_c(&acc[0][0][0]);
        return;
    }
    for (int s = 0; s < S; ++s) {
        const int k0 = s * KC;
        const int k1 = s + 1 < S ? k0 + KC : KT;
        if (s > 0) {
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
#pragma unroll
                    for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
        }
        run_range(k0, k1);
        const float* a = &acc[0][0][0];
        if (s == 0) {
#pragma unroll
            for (int e = 0; e < C::E; ++e) tot[e] = bf_round(a[e]);
        } else {
#pragma unroll
            for (int e = 0; e < C::E; ++e) tot[e] = tot[e] + bf_round(a[e]);
        }
    }
    cp_wait<0>();
    store_c(tot);
}

#define SKX6(NAME, CF)                                                                                             \
    extern "C" __global__ void __launch_bounds__(CF::T, 1)                                                         \
        NAME(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Cm, \
             int M, int N, int K, int lda, int ldc, int chunk1, int chunk2, const int* __restrict__ m1p, int m1s) { \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        skx6<CF>(A, B, Cm, M, N, K, lda, ldc, chunk1, chunk2, m1p, m1s, smem_raw);                                  \
    }

// (BM, BN, BK, warps M, warps N, stages)
using Y128x128k64 = Cfg<128, 128, 64, 2, 4, 3>;     // 96 KB, warp tile 64x32 (= nh9 / ns9 128x128k64)
using Y128x128k32 = Cfg<128, 128, 32, 2, 4, 4>;     // 64 KB
using Y64x256k32 = Cfg<64, 256, 32, 1, 8, 4>;       // 80 KB, 64-row tiles
using Y128x64k64 = Cfg<128, 64, 64, 4, 2, 3>;       // 72 KB, warp tile 32x32 (small N: more CTAs)
using Y64x128k64 = Cfg<64, 128, 64, 2, 4, 3>;       // 48 KB, warp tile 32x32 (64-row tiles)
SKX6(gemm_skx6_128x128k64, Y128x128k64)
SKX6(gemm_skx6_128x128k32, Y128x128k32)
SKX6(gemm_skx6_64x256k32, Y64x256k32)
SKX6(gemm_skx6_128x64k64, Y128x64k64)
SKX6(gemm_skx6_64x128k64, Y64x128k64)
