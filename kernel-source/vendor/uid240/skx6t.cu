// SKX6T (ruj skx6t track, 2026-10-21): skx6 (skx6.cu, the MX two-class GEMM of mixed-bucket graphs) without the padded
// tail work. Every stored output element keeps skx6's arithmetic exactly: per class, S = ceil(K / chunk) slices (the
// last one the rest); each slice is one ascending-k chain of mma.sync m16n8k16 (fp32) from +0 over its k16 steps,
// rounded once to bf16; the slices are folded in slice order with fp32 RN adds starting from the first partial itself
// (tot = f(b(acc_0)); tot = tot + f(b(acc_s))), rounded once (S = 1: bf16(acc_0)). Rows [0, M1) take class 1 (chunk1),
// rows [M1, M) class 2 (chunk2); M1 = *m1p (int32 on the device, clamped to [0, M]), m1p == nullptr -> M1 = m1s.
//   C[M, N] = A[M, K] @ B[N, K]^T, A = activations (lda), B = weight [N, K] (ld K), C bf16 (ldc).
// Only the work on rows that are never stored changes:
//   gemm_skx6t_<cfg>  skx6's schedule: grid (N / BN, ceil(M / BM) + 1), class-1 tiles over [0, M1), class-2 tiles from
//                     row M1 on, tiles past both segments exit at once; in the last (partial) tile of a segment every
//                     warp skips its m16 row sub-tiles that lie wholly past the segment end (their ldmatrix + mma.sync;
//                     a warp with no row left skips its B fragments too, but still issues its share of the CTA's
//                     cp.async loads and joins every barrier).
//   gemm_skx6m_<cfg>  one tile grid over [0, M): grid (N / BN, ceil(M / BM)). The tile that holds row M1 carries rows of
//                     both classes: the k-loop runs through the union of both classes' slice ends and each accumulator
//                     row folds only at its own class's slice ends, so every row's chain is the one skx6 gives it (the
//                     MMA rows of mma.sync are independent: a row's sums never see another row's data). The last tile
//                     skips the m16 sub-tiles past M as above.
// The hot loop (SKX3's branch-free k-tile with loads ST - 1 ahead, checked only in the last ST k-tiles) is skx6's; full
// tiles run it without any skip predicate (template FULL).
// Needs N % BN == 0, K % BK == 0, chunk % BK == 0, 16-byte aligned rows.
// Build (nvcc 13.4, V13.4.92, reproduces the shipped cubin byte for byte):
//   nvcc -arch=sm_120a -O3 -cubin -Xptxas -v -o skx6t.cubin skx6t.cu
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

// one output tile, the part of ONE warp: rows [m0, min(m0 + BM, mend)) are stored; the warp's m16 row sub-tiles
// i < ACT hold rows below mend (ACT is a compile-time count, so a warp's hot loop has no skip predicate: predicated
// ldmatrix / mma.sync groups make ptxas fence each group with WARPSYNC and lose the load / MMA interleave); warps with
// other ACT values of the same CTA run their own instantiation, with the same cp.async loads and barriers per k-tile.
// has1 / has2 = rows of class 1 / class 2 in the tile (rows below M1 are class 1); KC1 / KC2 = the classes' slice
// widths in k-tiles (>= KT: unsplit). MIXED (skx6m's boundary tile): the k-loop runs to the union of both classes'
// slice ends and each m16 sub-tile folds at its own class's ends (a sub-tile that straddles M1 per row).
template <class C, int ACT, bool MIXED>
DEVI void tile(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Cm,
               int K, int lda, int ldc, int m0, int mend, int M1, bool has1, int KC1, int KC2,
               unsigned char* smem_raw) {
    constexpr int BN = C::BN, BK = C::BK, ST = C::ST, NT = C::NT, KS = C::KS;
    constexpr int AT = ACT > 0 ? ACT : 1;   // fragment / accumulator arrays (unused at ACT = 0)
    const int tid = threadIdx.x;
    const int n0 = blockIdx.x * BN;
    const int KT = K / BK;

    // ---- loader: thread's chunk column and first row; its rows are r0 + c * RS (skx6's, rows past mend zero-filled)
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
    uint32_t af[2][AT][4];
    uint32_t bfr[2][NT][2];
    auto load_frags = [&](int buf, int stage, int ks) {
        if constexpr (ACT > 0) {
            const uint32_t a = sA0 + stage * C::A_ST * 2 + a_off[ks];
            const uint32_t b = sB0 + stage * C::B_ST * 2 + b_off[ks];
#pragma unroll
            for (int i = 0; i < ACT; ++i)
                ldsm_x4(a + i * 16 * BK * 2, af[buf][i][0], af[buf][i][1], af[buf][i][2], af[buf][i][3]);
#pragma unroll
            for (int j = 0; j < NT; j += 2)
                ldsm_x4(b + j * 8 * BK * 2, bfr[buf][j][0], bfr[buf][j][1], bfr[buf][j + 1][0], bfr[buf][j + 1][1]);
        }
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

    float acc[AT][NT][4];
    float tot[AT * NT * 4];
#pragma unroll
    for (int i = 0; i < AT; ++i)
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
            for (int i = 0; i < ACT; ++i)
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
        for (int i = 0; i < ACT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int r = m0 + wm * C::WTM + i * 16 + g;
                const int c = n0 + wn * C::WTN + j * 8 + t4 * 2;
                const float* q = v + (i * NT + j) * 4;
                if (r < mend) *reinterpret_cast<uint32_t*>(Cm + (size_t)r * ldc + c) = pack_bf2(q[0], q[1]);
                if (r + 8 < mend) *reinterpret_cast<uint32_t*>(Cm + (size_t)(r + 8) * ldc + c) = pack_bf2(q[2], q[3]);
            }
    };

    if constexpr (!MIXED) {   // a one-class tile: skx6's slice loop and fold, unchanged
        const int KC = has1 ? KC1 : KC2;
        const int S = (KT + KC - 1) / KC;
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
                for (int i = 0; i < AT; ++i)
#pragma unroll
                    for (int j = 0; j < NT; ++j)
#pragma unroll
                        for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
            }
            run_range(k0, k1);
            const float* a = &acc[0][0][0];
            if (s == 0) {
#pragma unroll
                for (int e = 0; e < ACT * NT * 4; ++e) tot[e] = bf_round(a[e]);
            } else {
#pragma unroll
                for (int e = 0; e < ACT * NT * 4; ++e) tot[e] = tot[e] + bf_round(a[e]);
            }
        }
        cp_wait<0>();
        store_c(tot);
    } else {
        // skx6m's boundary tile (rows of both classes): the k-loop runs to the union of both classes' slice ends
        // (class c: KC_c, 2 KC_c, .., KT); at each end the sub-tiles of the class(es) ending there fold (first slice:
        // tot = b(acc); later: tot = tot + b(acc); unsplit class: tot = acc at KT) and restart their chain from +0. A
        // sub-tile wholly on one side of M1 decides with warp-uniform branches; the one that straddles M1 per row.
        int e1 = min(KC1, KT), e2 = min(KC2, KT);
        bool p1 = true, p2 = true;   // the class's first slice is still open
        int k = 0;
        while (k < KT) {
            const int e = min(e1, e2);
            run_range(k, e);
            k = e;
            const bool f1 = e == e1, f2 = e == e2;
            const bool u1 = KC1 >= KT, u2 = KC2 >= KT;
#pragma unroll
            for (int i = 0; i < ACT; ++i) {
                const int rb = m0 + wm * C::WTM + i * 16;   // the sub-tile's first row (warp-uniform)
                float* a = &acc[i][0][0];
                float* t = &tot[i * NT * 4];
                if (rb + 16 <= M1 || rb >= M1) {   // one class
                    const bool c1 = rb + 16 <= M1;
                    if (c1 ? f1 : f2) {
                        const bool un = c1 ? u1 : u2, first = c1 ? p1 : p2;
                        if (un) {
#pragma unroll
                            for (int x = 0; x < NT * 4; ++x) t[x] = a[x];
                        } else if (first) {
#pragma unroll
                            for (int x = 0; x < NT * 4; ++x) {
                                t[x] = bf_round(a[x]);
                                a[x] = 0.f;
                            }
                        } else {
#pragma unroll
                            for (int x = 0; x < NT * 4; ++x) {
                                t[x] = t[x] + bf_round(a[x]);
                                a[x] = 0.f;
                            }
                        }
                    }
                } else {   // rows g / g + 8 of this thread on either side of M1
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const bool c1 = rb + g + 8 * h < M1;
                        if (c1 ? f1 : f2) {
                            const bool un = c1 ? u1 : u2, first = c1 ? p1 : p2;
#pragma unroll
                            for (int j = 0; j < NT; ++j)
#pragma unroll
                                for (int q2 = 0; q2 < 2; ++q2) {
                                    float& av = a[j * 4 + h * 2 + q2];
                                    float& tv = t[j * 4 + h * 2 + q2];
                                    tv = un ? av : (first ? bf_round(av) : tv + bf_round(av));
                                    av = 0.f;
                                }
                        }
                    }
                }
            }
            if (f1) {
                p1 = false;
                e1 = min(e1 + KC1, KT);
            }
            if (f2) {
                p2 = false;
                e2 = min(e2 + KC2, KT);
            }
        }
        cp_wait<0>();
        store_c(tot);
    }
}

// the warp's active m16 sub-tile count (rows below mend) as a template argument
template <class C, bool MIXED, int A = C::MT>
DEVI void tile_act(int act, const __nv_bfloat16* __restrict__ Ap, const __nv_bfloat16* __restrict__ B,
                   __nv_bfloat16* __restrict__ Cm, int K, int lda, int ldc, int m0, int mend, int M1, bool has1,
                   int KC1, int KC2, unsigned char* smem_raw) {
    if constexpr (A > 0) {
        if (act < A) {
            tile_act<C, MIXED, A - 1>(act, Ap, B, Cm, K, lda, ldc, m0, mend, M1, has1, KC1, KC2, smem_raw);
            return;
        }
    }
    tile<C, A, MIXED>(Ap, B, Cm, K, lda, ldc, m0, mend, M1, has1, KC1, KC2, smem_raw);
}

// MERGE = false: skx6's two segment grids (gemm_skx6t_*); true: one grid over [0, M) (gemm_skx6m_*)
template <class C, bool MERGE>
DEVI void skx6x(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Cm,
                int M, int N, int K, int lda, int ldc, int chunk1, int chunk2, const int* __restrict__ m1p, int m1s,
                unsigned char* smem_raw) {
    constexpr int BM = C::BM, BK = C::BK;
    int M1 = m1p != nullptr ? __ldg(m1p) : m1s;
    M1 = M1 < 0 ? 0 : (M1 > M ? M : M1);
    const int t = blockIdx.y;
    int m0, mend;
    bool has1, has2;
    if constexpr (MERGE) {
        m0 = t * BM;
        if (m0 >= M) return;
        mend = M;
        has1 = m0 < M1;
        has2 = min(m0 + BM, M) > M1;
    } else {
        const int mt1 = (M1 + BM - 1) / BM, mt2 = (M - M1 + BM - 1) / BM;
        if (t < mt1) {
            m0 = t * BM; mend = M1; has1 = true; has2 = false;
        } else if (t - mt1 < mt2) {
            m0 = M1 + (t - mt1) * BM; mend = M; has1 = false; has2 = true;
        } else {
            return;   // past both segments (the grid has one spare row tile for a boundary inside a tile)
        }
    }
    const int KC1 = chunk1 / BK, KC2 = chunk2 / BK;
    const int wm = (threadIdx.x / 32) / C::WN;
    const int left = mend - (m0 + wm * C::WTM);   // rows of this warp below mend
    // warp-uniform by construction (the shuffle from lane 0 lets ptxas prove it: no WARPSYNC around the per-warp
    // instantiations' ldmatrix / mma.sync)
    const int act = __shfl_sync(0xffffffffu, left <= 0 ? 0 : (left >= C::WTM ? C::MT : (left + 15) / 16), 0);
    if (MERGE && has1 && has2)   // skx6m's boundary tile
        tile_act<C, true>(act, A, B, Cm, K, lda, ldc, m0, mend, M1, has1, KC1, KC2, smem_raw);
    else
        tile_act<C, false>(act, A, B, Cm, K, lda, ldc, m0, mend, M1, has1, KC1, KC2, smem_raw);
}

#define SKX6X(NAME, CF, MERGE)                                                                                     \
    extern "C" __global__ void __launch_bounds__(CF::T, 1)                                                         \
        NAME(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Cm, \
             int M, int N, int K, int lda, int ldc, int chunk1, int chunk2, const int* __restrict__ m1p, int m1s) { \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        skx6x<CF, MERGE>(A, B, Cm, M, N, K, lda, ldc, chunk1, chunk2, m1p, m1s, smem_raw);                          \
    }

// (BM, BN, BK, warps M, warps N, stages): skx6's five configs
using Y128x128k64 = Cfg<128, 128, 64, 2, 4, 3>;     // 96 KB, warp tile 64x32 (= nh9 / ns9 128x128k64)
using Y128x128k32 = Cfg<128, 128, 32, 2, 4, 4>;     // 64 KB
using Y64x256k32 = Cfg<64, 256, 32, 1, 8, 4>;       // 80 KB, 64-row tiles
using Y128x64k64 = Cfg<128, 64, 64, 4, 2, 3>;       // 72 KB, warp tile 32x32 (small N: more CTAs)
using Y64x128k64 = Cfg<64, 128, 64, 2, 4, 3>;       // 48 KB, warp tile 32x32 (64-row tiles)
SKX6X(gemm_skx6t_128x128k64, Y128x128k64, false)
SKX6X(gemm_skx6t_128x128k32, Y128x128k32, false)
SKX6X(gemm_skx6t_64x256k32, Y64x256k32, false)
SKX6X(gemm_skx6t_128x64k64, Y128x64k64, false)
SKX6X(gemm_skx6t_64x128k64, Y64x128k64, false)
SKX6X(gemm_skx6m_128x128k64, Y128x128k64, true)
SKX6X(gemm_skx6m_128x128k32, Y128x128k32, true)
SKX6X(gemm_skx6m_64x256k32, Y64x256k32, true)
SKX6X(gemm_skx6m_128x64k64, Y128x64k64, true)
SKX6X(gemm_skx6m_64x128k64, Y64x128k64, true)
