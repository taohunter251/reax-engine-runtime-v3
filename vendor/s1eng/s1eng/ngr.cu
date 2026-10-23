// NGR (rgh 2026-10-20, RARN): split-K GEMM kernels for sm_120 used by s1eng/rarn.py, C = A[M, K] @ B[N, K]^T with the
// arithmetic of ngemm9b.cu's ns9_* (byte-exact vs the split cuBLASLt pins): K cut into `chunk`-wide slices (ceil64, the
// pin's split), each slice accumulated in fp32 in ascending k16 order through mma.sync m16n8k16 starting from 0, the
// slices folded in slice order by the OUTPUT_TYPE scheme (tot = bf16(acc_0); tot += bf16(acc_s); one RN round).
//   nr9p_*: one CTA per (output tile, K slice) (grid = tiles * S, the S slices of a tile adjacent); every CTA stores its
//           bf16 slice partial to part[s][M][N]; rarn._rarn_kernel then folds the S partials in slice order, adds the
//           residual and applies the RMSNorm in one pass (no cuBLASLt splitKreduce launch, no y round trip).
//   nr9h_*: as nr9p_* with rows [0, mb) in the big tiles and the last row block in small tiles (32 / 64 rows: no
//           MMA work on the empty rows of a 128-row tile past R); mb rides in the ldh argument.
//   nsq9_*: ns9_* (all slices of a tile in one CTA, folded in slice order) whose epilogue writes
//           Hout = bf16(float(Hres) + float(bf16(fold))) (add_rmsnorm's residual add); a plain RMSNorm follows.
// Args: A, B, Cm (Hout), M, N, K, lda, ldc, chunk, S, part (bf16 [S][M][N]), flags (unused), Hres, ldh.
// Needs N % BN == 0, chunk % BK == 0, 16-byte aligned rows. Build: nvcc 13.4 -cubin -O3 -std=c++17 -arch=sm_120a.
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

namespace ngr {

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
template <int MODE>
DEVI float combine(float total, float acc, bool first) {
    if constexpr (MODE == 2) return first ? acc : total + acc;
    else if constexpr (MODE == 1) return first ? rbf(acc) : rbf(total + acc);
    else return first ? rbf(acc) : total + rbf(acc);
}
DEVI uint4 ldcg16(const void* p) { return __ldcg(reinterpret_cast<const uint4*>(p)); }
DEVI void bf8_to_f(const uint4& v, float (&f)[8]) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __bfloat1622float2(h[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}
DEVI uint4 f_to_bf8(const float (&f)[8]) {
    uint4 v;
    v.x = pack_bf2(f[0], f[1]);
    v.y = pack_bf2(f[2], f[3]);
    v.z = pack_bf2(f[4], f[5]);
    v.w = pack_bf2(f[6], f[7]);
    return v;
}

template <int BK>
DEVI int swz_f(int r) {
    if constexpr (BK == 32) return (r >> 1) & 3;
    else return r & 7;
}

template <int BM_, int BN_, int BK_, int WARPS_M, int WARPS_N, int STAGES_>
struct Cfg {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, STAGES = STAGES_, WM = WARPS_M, WN = WARPS_N;
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

// per-thread chunk loader (ngemm9b's), with B's row stride ldb and the K window starting at kbeg
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
    DEVI void init(const __nv_bfloat16* A, const __nv_bfloat16* B, int m0, int n0, int M, int lda, int ldb, int kbeg) {
        const int tid = threadIdx.x;
        r0 = tid / C::CH;
        const int c = tid % C::CH;
        dst0 = (r0 * C::BK + ((c ^ swz_f<C::BK>(r0)) * 8)) * 2;
        pa = A + kbeg;
        pb = B + kbeg;
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
            offb[q] = (uint32_t)((n0 + r) * ldb + c * 8);
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

// ngemm9b's mainloop over the K window [kbeg, kbeg + klen) (klen % BK == 0); FOLD != 0: in-CTA split-K, every `chunk`
// columns acc is folded into tot by the scheme and restarted (ns9_*); else acc = the window's plain ascending sum
template <class C, int FOLD = 0>
DEVI void mainloop(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, int M, int lda, int ldb,
                   int kbeg, int klen, int m0, int n0, __nv_bfloat16* smem, float (&acc)[C::MT][C::NT][4],
                   float (*tot)[C::NT][4] = nullptr, int chunk = 0) {
    constexpr int KS = C::KS, BK = C::BK, S = C::STAGES, MT = C::MT, NT = C::NT;
    const uint32_t sA0 = smem_u32(smem);
    const uint32_t sB0 = sA0 + S * C::A_STAGE * 2;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int KT = klen / BK;
    auto sa = [&](int s) { return sA0 + s * C::A_STAGE * 2; };
    auto sb = [&](int s) { return sB0 + s * C::B_STAGE * 2; };
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
    Loader<C> ld;
    ld.init(A, B, m0, n0, M, lda, ldb, kbeg);
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
    int s = 0;
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
            if (ks == 0) {
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

// fragments -> bf16 (RN) -> padded smem tile [BM][BN + CPAD] (stmatrix.x4), then a CTA barrier
template <class C>
DEVI void stage_tile(const float (&v)[C::MT][C::NT][4], __nv_bfloat16* smem) {
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
}

// staged tile -> global rows < M (16-byte stores)
template <class C>
DEVI void write_tile(const __nv_bfloat16* smem, __nv_bfloat16* __restrict__ Out, int ldo, int M, int m0, int n0) {
    constexpr int LD = C::BN + C::CPAD;
    constexpr int CPR = C::BN / 8;
    for (int i = threadIdx.x; i < C::BM * CPR; i += C::THREADS) {
        const int r = i / CPR, c = i % CPR;
        const int gm = m0 + r;
        if (gm < M)
            *reinterpret_cast<uint4*>(Out + (size_t)gm * ldo + n0 + c * 8) = *reinterpret_cast<const uint4*>(smem + r * LD + c * 8);
    }
}

// residual write: every h chunk of the thread loaded first (one latency), then add + store
template <class C>
DEVI void write_tile_res2(const __nv_bfloat16* smem, __nv_bfloat16* __restrict__ Out, int ldo,
                          const __nv_bfloat16* __restrict__ H, int ldh, int M, int m0, int n0) {
    constexpr int LD = C::BN + C::CPAD;
    constexpr int CPR = C::BN / 8;
    constexpr int IT = C::BM * CPR / C::THREADS;
    static_assert(C::BM * CPR % C::THREADS == 0, "tile chunks");
    uint4 hv[IT];
#pragma unroll
    for (int u = 0; u < IT; ++u) {
        const int i = threadIdx.x + u * C::THREADS;
        const int r = i / CPR, c = i % CPR;
        const int gm = m0 + r;
        if (gm < M) hv[u] = *reinterpret_cast<const uint4*>(H + (size_t)gm * ldh + n0 + c * 8);
    }
#pragma unroll
    for (int u = 0; u < IT; ++u) {
        const int i = threadIdx.x + u * C::THREADS;
        const int r = i / CPR, c = i % CPR;
        const int gm = m0 + r;
        if (gm >= M) continue;
        float y[8], h[8];
        bf8_to_f(*reinterpret_cast<const uint4*>(smem + r * LD + c * 8), y);
        bf8_to_f(hv[u], h);
#pragma unroll
        for (int e = 0; e < 8; ++e) y[e] = h[e] + y[e];
        *reinterpret_cast<uint4*>(Out + (size_t)gm * ldo + n0 + c * 8) = f_to_bf8(y);
    }
}

// one CTA per (tile, slice): the bf16 slice partial of the tile -> part[s]
template <class C>
DEVI void kslice(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, int M, int N, int K, int lda,
                 int chunk, int S, __nv_bfloat16* __restrict__ part, __nv_bfloat16* smem) {
    const int b = blockIdx.x;
    const int s = b % S;
    const int t = b / S;
    const int mt = (M + C::BM - 1) / C::BM;
    const int m0 = (t % mt) * C::BM, n0 = (t / mt) * C::BN;
    const int kbeg = s * chunk;
    const int klen = min(chunk, K - kbeg);
    float acc[C::MT][C::NT][4];
    mainloop<C>(A, B, M, lda, K, kbeg, klen, m0, n0, smem, acc);
    stage_tile<C>(acc, smem);   // bf16(acc_s): the OUTPUT_TYPE partial
    write_tile<C>(smem, part + s * (size_t)M * N, N, M, m0, n0);
}

// hetero partial-only variant: rows [0, mb) in CB tiles, rows [mb, M) in CS tiles (finer last row block, no MMA work
// on rows past M beyond the small tile), every tile x S slices; partials as kslice EPI 0
template <class CB, class CS>
DEVI void kslice_h(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, int M, int N, int K,
                   int lda, int chunk, int S, int mb, __nv_bfloat16* __restrict__ part, __nv_bfloat16* smem) {
    static_assert(CB::THREADS == CS::THREADS, "threads");
    const int b = blockIdx.x;
    const int s = b % S;
    const int t = b / S;
    const int mtb = mb / CB::BM;
    const int tb = mtb * (N / CB::BN);
    const int kbeg = s * chunk;
    const int klen = min(chunk, K - kbeg);
    const size_t pstride = (size_t)M * N;
    if (t < tb) {
        const int m0 = (t % mtb) * CB::BM, n0 = (t / mtb) * CB::BN;
        float acc[CB::MT][CB::NT][4];
        mainloop<CB>(A, B, M, lda, K, kbeg, klen, m0, n0, smem, acc);
        stage_tile<CB>(acc, smem);
        write_tile<CB>(smem, part + s * pstride, N, M, m0, n0);
    } else {
        const int t2 = t - tb;
        const int mts = (M - mb + CS::BM - 1) / CS::BM;
        const int m0 = mb + (t2 % mts) * CS::BM, n0 = (t2 / mts) * CS::BN;
        float acc[CS::MT][CS::NT][4];
        mainloop<CS>(A, B, M, lda, K, kbeg, klen, m0, n0, smem, acc);
        stage_tile<CS>(acc, smem);
        write_tile<CS>(smem, part + s * pstride, N, M, m0, n0);
    }
}

// ns9_* in-CTA split (all slices of a tile in one CTA, folded in slice order) with the residual epilogue
template <class C, int MODE>
DEVI void insplit_res(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Cm,
                      int M, int N, int K, int lda, int ldc, int chunk, const __nv_bfloat16* __restrict__ H, int ldh,
                      __nv_bfloat16* smem) {
    const int mt = (M + C::BM - 1) / C::BM;
    const int m0 = (blockIdx.x % mt) * C::BM, n0 = (blockIdx.x / mt) * C::BN;
    float acc[C::MT][C::NT][4];
    float tot[C::MT][C::NT][4];
    mainloop<C, MODE>(A, B, M, lda, K, 0, K, m0, n0, smem, acc, tot, chunk);
    stage_tile<C>(tot, smem);
    write_tile_res2<C>(smem, Cm, ldc, H, ldh, M, m0, n0);
}

}  // namespace ngr

#define NGR_ARGS                                                                                                   \
    const __nv_bfloat16 *__restrict__ A, const __nv_bfloat16 *__restrict__ B, __nv_bfloat16 *__restrict__ Cm, int M, \
        int N, int K, int lda, int ldc, int chunk, int S, __nv_bfloat16 *__restrict__ part, int *__restrict__ flags, \
        const __nv_bfloat16 *__restrict__ H, int ldh

#define NGR_PART(NAME, CF)                                                                                         \
    extern "C" __global__ void __launch_bounds__(CF::THREADS, 1) NAME(NGR_ARGS) {                                  \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        ngr::kslice<CF>(A, B, M, N, K, lda, chunk, S, part, reinterpret_cast<__nv_bfloat16*>(smem_raw));           \
    }
#define NGR_INRES(NAME, CF, MODE)                                                                                  \
    extern "C" __global__ void __launch_bounds__(CF::THREADS, 1) NAME(NGR_ARGS) {                                  \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        ngr::insplit_res<CF, MODE>(A, B, Cm, M, N, K, lda, ldc, chunk, H, ldh,                                     \
                                   reinterpret_cast<__nv_bfloat16*>(smem_raw));                                     \
    }

// hetero partials: the ldh argument carries mb (rows in big tiles)
#define NGR_PARTH(NAME, CB, CS)                                                                                    \
    extern "C" __global__ void __launch_bounds__(CB::THREADS, 1) NAME(NGR_ARGS) {                                  \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        ngr::kslice_h<CB, CS>(A, B, M, N, K, lda, chunk, S, ldh, part, reinterpret_cast<__nv_bfloat16*>(smem_raw)); \
    }

// (BM, BN, BK, warps M, warps N, stages)
using P128x128 = ngr::Cfg<128, 128, 64, 2, 4, 3>;
using P128x256 = ngr::Cfg<128, 256, 64, 2, 4, 2>;
using P64x256 = ngr::Cfg<64, 256, 64, 2, 4, 2>;
using P32x256 = ngr::Cfg<32, 256, 64, 1, 8, 2>;
using P64x128 = ngr::Cfg<64, 128, 64, 2, 4, 3>;
using P32x128 = ngr::Cfg<32, 128, 64, 1, 8, 3>;

NGR_PART(nr9p_128x256k64, P128x256)
NGR_PART(nr9p_128x128k64, P128x128)
NGR_PARTH(nr9h_b128x256_s32x256k64, P128x256, P32x256)
NGR_PARTH(nr9h_b128x256_s64x256k64, P128x256, P64x256)
NGR_PARTH(nr9h_b128x128_s32x128k64, P128x128, P32x128)
NGR_PARTH(nr9h_b128x128_s64x128k64, P128x128, P64x128)
NGR_INRES(nsq9_4_128x128k64, P128x128, 4)
