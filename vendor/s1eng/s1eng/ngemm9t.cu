// ngemm9t.cu (COV9b, rgh 2026-10-21): NG9B "tall tail" GEMMs for sm_120 (RTX PRO 6000): the ngemm9b kernels (same
// mainloop, epilogues and arithmetic: fp32 accumulation over K in ascending k16 order through mma.sync m16n8k16, one RN
// round to bf16; SwiGLU = swiglu1) with a different small-tile shape for the last row band.
// Why: the paired GEMM cost is ~linear in the number of 128-row bands (COV9b graph-paired tables 2026-10-21: one more
// band of 32 rows costs 84 % of a 128-row band: a 32x256 / 64x256 tile streams the whole 256-row weight panel for few
// rows), and served row counts are 32 / 64 / 96 mod 128 for 78 % of the rows (5.8 % padded rows with 128-row bands).
// Tall tails cover the last 128 + (R mod 128) rows with ONE band of 160 / 192 / 224-row x 128-column tiles (drop 1),
// or the last R mod 128 = 96 rows with one band of 96x256 tiles (drop 0), at a load-per-FLOP close to 128x256's.
// Kernels (arg order = ngemm5 / ng.py: A, B, C, M, N, K, lda, ldc, chunk, mb, ws, flags; chunk / ws / flags unused):
//   nh9_b128x256k64_t<H>x128k64 / nh9_b128x256k64_s96x256k64: unsplit, rows [0, mb) 128x256 tiles, [mb, M) tail tiles
//   nsw9_b128x256k64_t<H>x128k64 / nsw9_b128x256k64_s96x256k64: N = 2H stacked [Wg; Wu], out [M, H] (as nsw9_*)
// Tile geometry never changes the bits (each output element is the same ascending-k mma.sync chain). Needs N % 256 == 0
// (the tails need N % 128), K % 64 == 0, 16-byte aligned rows.
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

namespace ng9t {

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
DEVI void named_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
DEVI uint32_t pack_bf2(float a, float b) {
    __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}
DEVI float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
// split-K fold of slice sums (cuBLASLt reduction schemes): 1 INPLACE, 2 COMPUTE_TYPE, 4 OUTPUT_TYPE
template <int MODE>
DEVI float combine(float total, float acc, bool first) {
    if constexpr (MODE == 2) return first ? acc : total + acc;
    else if constexpr (MODE == 1) return first ? rbf(acc) : rbf(total + acc);
    else return first ? rbf(acc) : total + rbf(acc);
}
DEVI float silu_f32(float x) { return __fdiv_rn(x, 1.0f + expf(-x)); }
DEVI __nv_bfloat16 swiglu1(__nv_bfloat16 g, __nv_bfloat16 u) {
    float s = __bfloat162float(__float2bfloat16_rn(silu_f32(__bfloat162float(g))));
    return __float2bfloat16_rn(s * __bfloat162float(u));
}

template <int BK>
DEVI int swz_f(int r) {
    if constexpr (BK == 32) return (r >> 1) & 3;
    else return r & 7;
}

// EPI: 2 = plain stmatrix epilogue, 1 = SwiGLU register epilogue (interleaved B rows)
template <int BM_, int BN_, int BK_, int WARPS_M, int WARPS_N, int STAGES_, int EPI_>
struct Cfg {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, STAGES = STAGES_, WM = WARPS_M, WN = WARPS_N, EPI = EPI_;
    static constexpr int CW = WARPS_M * WARPS_N;
    static constexpr int THREADS = CW * 32;
    static constexpr int WTM = BM / WARPS_M, WTN = BN / WARPS_N;
    static constexpr int MT = WTM / 16, NT = WTN / 8, KS = BK / 16, CH = BK / 8;
    static constexpr int A_STAGE = BM * BK, B_STAGE = BN * BK;
    static constexpr int SMEM_PIPE = STAGES * (A_STAGE + B_STAGE) * 2;
    static constexpr int CPAD = 8;
    static constexpr int SMEM_EPI = BM * (BN + CPAD) * 2;
    static constexpr int SMEM = SMEM_PIPE > SMEM_EPI ? SMEM_PIPE : SMEM_EPI;
    static_assert(MT >= 1 && NT % 2 == 0 && KS % 2 == 0 && STAGES >= 2, "tile");
};

// per-thread chunk loader: chunk q < NA = A rows r0 + q RS, q >= NA = B rows r0 + (q - NA) RS (RS = threads / chunks
// per row); global element offsets precomputed once
template <class C>
struct Loader {
    static constexpr int RS = C::THREADS / C::CH;
    static constexpr int NA = (C::BM + RS - 1) / RS;
    static constexpr int NB = (C::BN + RS - 1) / RS;
    static constexpr int NQ = NA + NB;
    const __nv_bfloat16* pa;
    const __nv_bfloat16* pb;
    uint32_t offa[NA], offb[NB];
    uint32_t dst0;
    uint32_t amask;
    int r0;
    DEVI void init(const __nv_bfloat16* A, const __nv_bfloat16* B, int m0, int n0, int n1, int M, int K, int lda) {
        const int tid = threadIdx.x;
        r0 = tid / C::CH;
        const int c = tid % C::CH;
        dst0 = (r0 * C::BK + ((c ^ swz_f<C::BK>(r0)) * 8)) * 2;
        pa = A;
        pb = B;
        amask = 0;
#pragma unroll
        for (int q = 0; q < NA; ++q) {
            const int r = r0 + q * RS;
            const int gm = m0 + r;
            const bool ok = gm < M;
            offa[q] = (uint32_t)((ok ? gm : 0) * lda + c * 8);
            amask |= (ok ? 1u : 0u) << q;
        }
#pragma unroll
        for (int q = 0; q < NB; ++q) {
            const int r = r0 + q * RS;
            int wr;
            if (C::EPI == 1 && n1 >= 0) {   // warp column wn: gate rows wn*WTN/2.. then the matching up rows
                const int wn = r / C::WTN, h = (r % C::WTN) >= C::WTN / 2, cc = r % (C::WTN / 2);
                wr = (h ? n1 : n0) + wn * (C::WTN / 2) + cc;
            } else {
                wr = (n1 < 0 || r < C::BN / 2) ? n0 + r : n1 + r - C::BN / 2;
            }
            offb[q] = (uint32_t)(wr * K + c * 8);
        }
    }
    DEVI void issue_all(uint32_t sA, uint32_t sB, int k0) {
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            if (q < NA) {
                if (C::BM % RS == 0 || r0 + q * RS < C::BM)
                    cp_async16(sA + dst0 + q * RS * C::BK * 2, pa + offa[q] + k0, (amask >> q) & 1);
            } else {
                const int qb = q - NA;
                if (C::BN % RS == 0 || r0 + qb * RS < C::BN)
                    cp_async16(sB + dst0 + qb * RS * C::BK * 2, pb + offb[qb] + k0, true);
            }
        }
    }
};

template <class C>
struct Frag {
    uint32_t a_off[C::KS], b_off[C::KS];
    uint32_t af[2][C::MT][4];
    uint32_t bf[2][C::NT][2];
    DEVI void init(int wm, int wn, int lane) {
        constexpr int BK = C::BK;
        const int a_r = wm * C::WTM + (lane % 16);
        const int a_f = swz_f<BK>(lane % 16);
        const int a_kc = lane / 16;
        const int b_r = wn * C::WTN + (lane / 16) * 8 + (lane % 8);
        const int b_f = swz_f<BK>((lane / 16) * 8 + (lane % 8));
        const int b_kc = (lane / 8) % 2;
#pragma unroll
        for (int ks = 0; ks < C::KS; ++ks) {
            a_off[ks] = (a_r * BK + (((ks * 2 + a_kc) ^ a_f) * 8)) * 2;
            b_off[ks] = (b_r * BK + (((ks * 2 + b_kc) ^ b_f) * 8)) * 2;
        }
    }
    DEVI void load(int buf, uint32_t sA, uint32_t sB, int ks) {
        const uint32_t a = sA + a_off[ks];
        const uint32_t b = sB + b_off[ks];
#pragma unroll
        for (int i = 0; i < C::MT; ++i)
            ldsm_x4(a + i * 16 * C::BK * 2, af[buf][i][0], af[buf][i][1], af[buf][i][2], af[buf][i][3]);
#pragma unroll
        for (int j = 0; j < C::NT; j += 2)
            ldsm_x4(b + j * 8 * C::BK * 2, bf[buf][j][0], bf[buf][j][1], bf[buf][j + 1][0], bf[buf][j + 1][1]);
    }
};

// acc = A[m0:m0+BM, :] @ B-tile^T over all of K, k ascending
// FOLD != 0 (in-CTA split-K, ns9_*): every `chunk` columns of K (slice boundaries, multiples of 64) acc is folded into
// tot by the pin's reduction scheme and restarted; otherwise acc holds the plain sum
template <class C, int FOLD = 0>
DEVI void mainloop(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, int M, int K, int lda,
                   int m0, int n0, int n1, __nv_bfloat16* smem, float (&acc)[C::MT][C::NT][4],
                   float (*tot)[C::NT][4] = nullptr, int chunk = 0) {
    constexpr int KS = C::KS, BK = C::BK, S = C::STAGES, MT = C::MT, NT = C::NT;
    const uint32_t sA0 = smem_u32(smem);
    const uint32_t sB0 = sA0 + S * C::A_STAGE * 2;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int KT = K / BK;
    auto sa = [&](int s) { return sA0 + s * C::A_STAGE * 2; };
    auto sb = [&](int s) { return sB0 + s * C::B_STAGE * 2; };
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
    Loader<C> ld;
    ld.init(A, B, m0, n0, n1, M, K, lda);
#pragma unroll
    for (int t = 0; t < S - 1; ++t) {
        if (t < KT) ld.issue_all(sa(t), sb(t), t * BK);
        cp_commit();
    }
    Frag<C> fr;
    fr.init(warp / C::WN, warp % C::WN, lane);
    cp_wait<S - 2>();
    __syncthreads();
    fr.load(0, sa(0), sb(0), 0);
    int s = 0;   // slot of tile kt
    const int kt_ch = FOLD ? chunk / BK : KT;
    int in_slice = 0;
    bool first = true;
    for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            if (ks == KS - 1) {
                cp_wait<S - 2>();
                __syncthreads();
            }
            if (ks == 0) {   // refill the slot tile kt-1 used (free since the barrier of its last k-step)
                const int tt = kt + S - 1;
                if (tt < KT) ld.issue_all(sa(tt % S), sb(tt % S), tt * BK);
            }
            if (ks < KS - 1 || kt + 1 < KT) {
                const int ns = ks < KS - 1 ? s : (s + 1 == S ? 0 : s + 1);
                fr.load((ks + 1) & 1, sa(ns), sb(ns), ks < KS - 1 ? ks + 1 : 0);
            }
            const int cb = ks & 1;
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int jj = 0; jj < NT; ++jj) {
                    const int j = (i & 1) ? NT - 1 - jj : jj;
                    mma16816(acc[i][j], fr.af[cb][i], fr.bf[cb][j][0], fr.bf[cb][j][1]);
                }
            if (ks == 0) cp_commit();
        }
        if (++s == S) s = 0;
        if constexpr (FOLD != 0) {
            if (++in_slice == kt_ch || kt == KT - 1) {
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

// plain epilogue: stmatrix.x4 (n-tiles j, j+1, both 8-row halves) -> padded smem tile -> 16-byte stores of rows < M
template <class C>
DEVI void store_tile(const float (&v)[C::MT][C::NT][4], __nv_bfloat16* __restrict__ Cm, int M, int ldc, int m0, int n0,
                     __nv_bfloat16* smem) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    constexpr int LD = C::BN + C::CPAD;
    const int mi = lane / 8, rr = lane % 8;
    const uint32_t base = smem_u32(smem) +
        (uint32_t)(((wm * C::WTM + (mi & 1) * 8 + rr) * LD + wn * C::WTN + (mi >> 1) * 8) * 2);
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int j = 0; j < C::NT; j += 2) {
            const uint32_t a = base + (uint32_t)((i * 16 * LD + j * 8) * 2);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(a),
                         "r"(pack_bf2(v[i][j][0], v[i][j][1])), "r"(pack_bf2(v[i][j][2], v[i][j][3])),
                         "r"(pack_bf2(v[i][j + 1][0], v[i][j + 1][1])), "r"(pack_bf2(v[i][j + 1][2], v[i][j + 1][3]))
                         : "memory");
        }
    named_sync(1, C::THREADS);
    constexpr int CPR = C::BN / 8;
    for (int i = threadIdx.x; i < C::BM * CPR; i += C::THREADS) {
        int r = i / CPR, c = i % CPR;
        int gm = m0 + r;
        if (gm < M)
            *reinterpret_cast<uint4*>(Cm + (size_t)gm * ldc + n0 + c * 8) = *reinterpret_cast<const uint4*>(smem + r * LD + c * 8);
    }
}

// SwiGLU register epilogue: n-tile j < NT/2 holds gate columns, j + NT/2 the matching up columns
template <class C>
DEVI void store_swiglu(const float (&v)[C::MT][C::NT][4], __nv_bfloat16* __restrict__ Out, int M, int ldo, int m0,
                       int o0) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int g = lane / 4, t4 = lane % 4;
    constexpr int NH = C::NT / 2;
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = m0 + wm * C::WTM + i * 16 + g + h * 8;
            if (r >= M) continue;
            __nv_bfloat16* row = Out + (size_t)r * ldo + o0 + wn * (C::WTN / 2) + t4 * 2;
#pragma unroll
            for (int j = 0; j < NH; ++j) {
                const __nv_bfloat162 gg = __floats2bfloat162_rn(v[i][j][2 * h], v[i][j][2 * h + 1]);
                const __nv_bfloat162 uu = __floats2bfloat162_rn(v[i][j + NH][2 * h], v[i][j + NH][2 * h + 1]);
                __nv_bfloat162 o;
                o.x = swiglu1(gg.x, uu.x);
                o.y = swiglu1(gg.y, uu.y);
                *reinterpret_cast<__nv_bfloat162*>(row + j * 8) = o;
            }
        }
}

template <class C, int SW>
DEVI void tile(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Cm,
               int M, int N, int K, int lda, int ldc, int m0, int nt, __nv_bfloat16* smem) {
    float acc[C::MT][C::NT][4];
    if constexpr (SW == 0) {
        static_assert(C::EPI == 2, "plain tiles use the stmatrix epilogue");
        mainloop<C>(A, B, M, K, lda, m0, nt * C::BN, -1, smem, acc);
        store_tile<C>(acc, Cm, M, ldc, m0, nt * C::BN, smem);
    } else {
        static_assert(C::EPI == 1, "SwiGLU tiles use the interleaved register epilogue");
        const int o0 = nt * (C::BN / 2);
        mainloop<C>(A, B, M, K, lda, m0, o0, N / 2 + o0, smem, acc);
        store_swiglu<C>(acc, Cm, M, ldc, m0, o0);
    }
}

// rows [0, mb) big tiles (CTAs 0 .. Tb-1, m fastest), rows [mb, M) small tiles; SW: N = 2H stacked gate|up rows
template <class CB, class CS, int SW>
DEVI void hetero(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Cm,
                 int M, int N, int K, int lda, int ldc, int mb, __nv_bfloat16* smem) {
    static_assert(CB::THREADS == CS::THREADS, "threads");
    const int ncols = SW ? N / 2 : N;
    const int mtb = mb / CB::BM;
    const int tb = mtb * (ncols / (SW ? CB::BN / 2 : CB::BN));
    int b = blockIdx.x;
    if (b < tb) {
        tile<CB, SW>(A, B, Cm, M, N, K, lda, ldc, (b % mtb) * CB::BM, b / mtb, smem);
    } else {
        b -= tb;
        const int mts = (M - mb + CS::BM - 1) / CS::BM;
        tile<CS, SW>(A, B, Cm, M, N, K, lda, ldc, mb + (b % mts) * CS::BM, b / mts, smem);
    }
}

// in-CTA split-K (all slices of a tile in one CTA, folded in slice order), plain output
template <class C, int MODE>
DEVI void insplit(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Cm,
                  int M, int N, int K, int lda, int ldc, int chunk, __nv_bfloat16* smem) {
    static_assert(C::EPI == 2, "plain tiles use the stmatrix epilogue");
    const int mt = (M + C::BM - 1) / C::BM;
    const int m0 = (blockIdx.x % mt) * C::BM, n0 = (blockIdx.x / mt) * C::BN;
    float acc[C::MT][C::NT][4];
    float tot[C::MT][C::NT][4];
    mainloop<C, MODE>(A, B, M, K, lda, m0, n0, -1, smem, acc, tot, chunk);
    store_tile<C>(tot, Cm, M, ldc, m0, n0, smem);
}

}  // namespace ng9t

#define NG9_ARGS                                                                                                   \
    const __nv_bfloat16 *__restrict__ A, const __nv_bfloat16 *__restrict__ B, __nv_bfloat16 *__restrict__ Cm, int M, \
        int N, int K, int lda, int ldc, int chunk, int mb, float *__restrict__ ws, int *__restrict__ flags
#define NG9_KERNEL(NAME, CB, CS, SW)                                                                               \
    extern "C" __global__ void __launch_bounds__(CB::THREADS, 1) NAME(NG9_ARGS) {                                  \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        ng9t::hetero<CB, CS, SW>(A, B, Cm, M, N, K, lda, ldc, mb, reinterpret_cast<__nv_bfloat16*>(smem_raw));     \
    }

// (BM, BN, BK, warps M, warps N, stages, epilogue)
using P_B128x256k64 = ng9t::Cfg<128, 256, 64, 2, 4, 2, 2>;
using P_T160x128k64 = ng9t::Cfg<160, 128, 64, 2, 4, 2, 2>;
using P_T192x128k64 = ng9t::Cfg<192, 128, 64, 2, 4, 2, 2>;
using P_T224x128k64 = ng9t::Cfg<224, 128, 64, 2, 4, 2, 2>;
using P_S96x256k64 = ng9t::Cfg<96, 256, 64, 2, 4, 2, 2>;
using W_B128x256k64 = ng9t::Cfg<128, 256, 64, 2, 4, 2, 1>;
using W_T160x128k64 = ng9t::Cfg<160, 128, 64, 2, 4, 2, 1>;
using W_T192x128k64 = ng9t::Cfg<192, 128, 64, 2, 4, 2, 1>;
using W_T224x128k64 = ng9t::Cfg<224, 128, 64, 2, 4, 2, 1>;
using W_S96x256k64 = ng9t::Cfg<96, 256, 64, 2, 4, 2, 1>;

NG9_KERNEL(nh9_b128x256k64_t160x128k64, P_B128x256k64, P_T160x128k64, 0)
NG9_KERNEL(nh9_b128x256k64_t192x128k64, P_B128x256k64, P_T192x128k64, 0)
NG9_KERNEL(nh9_b128x256k64_t224x128k64, P_B128x256k64, P_T224x128k64, 0)
NG9_KERNEL(nh9_b128x256k64_s96x256k64, P_B128x256k64, P_S96x256k64, 0)
NG9_KERNEL(nsw9_b128x256k64_t160x128k64, W_B128x256k64, W_T160x128k64, 1)
NG9_KERNEL(nsw9_b128x256k64_t192x128k64, W_B128x256k64, W_T192x128k64, 1)
NG9_KERNEL(nsw9_b128x256k64_t224x128k64, W_B128x256k64, W_T224x128k64, 1)
NG9_KERNEL(nsw9_b128x256k64_s96x256k64, W_B128x256k64, W_S96x256k64, 1)
