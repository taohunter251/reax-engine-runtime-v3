// ngemm9l.cu (COV9b, rgh 2026-10-21): grouped "lastfold" split-K GEMMs for sm_120 (RTX PRO 6000), C[M, N] =
// A[M, K] @ B[N, K]^T, for the split-K cuBLASLt pins with reduction scheme 4 (OUTPUT_TYPE) that the frozen h8 plan
// still runs on cuBLAS (cutlass_80 split-K + splitKreduce): down 2560x9216, out_proj / o_proj 2560x4096, k|v 2048 /
// 1024 x 2560.
// Arithmetic = the pin's (decoded byte-exact, spike rules.py): K cut into S slices of `chunk` (ceil64 of K / split,
// the last one the rest); slice s = fp32 accumulation from +0 over its K range in ascending k16 order through
// mma.sync m16n8k16 (= ngemm9b's mainloop), rounded once to bf16 (b_s); out = bf16_rn(((f32(b_0) + f32(b_1)) + ...)
// + f32(b_{S-1})) (fp32 RN adds in slice order).
// Grid: NG groups x T tiles (group-major, blockIdx = g * T + t). Group g computes the G = ceil(S / NG) consecutive
// slices [gG, min(S, gG + G)) of tile t with ONE continuous cp.async pipeline (no drain at slice boundaries); at each
// boundary but the last its fp32 sums are rounded to bf16, stored as that slice's partial and the accumulators restart
// at +0. G = 1 is the plain lastfold (one CTA per slice, as cuBLASLt's split-K grid), G = S one CTA per tile (in-CTA
// split with 128x256 tiles, which ns9_* cannot hold in registers). The CTA of a tile that finishes last (tile counter;
// it peeks the counter before storing its own last partial) folds all S partials IN SLICE ORDER: its own last slice
// from shared memory, the others from the workspace (L2), and writes the bf16 tile; it resets the counter (counters
// are zero between launches). No spin-wait, no reduce kernel.
// Rows: [0, mb) in big tiles, [mb, M) in small tiles (mb >= M: big tiles only, rows padded), as nh9_*.
// Workspace (ws, per lane): tile-major partial slots [tile][slice][V][THREADS] uint4 (big tiles first), bf16 words in
// a thread-native layout (16-byte coalesced); flags: one int counter per tile.
// Arg order = ng.py: A, B, C, M, N, K, lda, ldc, chunk, mb, ws, flags. NG = gridDim.x / T (the host launches NG * T
// CTAs). Needs N % BN == 0 (big and small), chunk % 64 == 0, K % 64 == 0, 16-byte aligned rows.
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

namespace ng9l {

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
DEVI float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
DEVI float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
DEVI int ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
    return v;
}
DEVI uint4 ldcg16(const void* p) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)
                 : "memory");
    return v;
}
DEVI void stcg16(void* p, uint4 v) {
    asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
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
    static constexpr int WORDS = MT * NT * 2;   // 32-bit bf16-pair words of one thread's partial
    static constexpr int V = WORDS / 4;         // 16-byte vectors of one thread's partial
    static constexpr int SMEM_OWN = V * THREADS * 16;
    static constexpr int SMEM_A = SMEM_PIPE > SMEM_EPI ? SMEM_PIPE : SMEM_EPI;
    static constexpr int SMEM = SMEM_A > SMEM_OWN ? SMEM_A : SMEM_OWN;
    static_assert(MT >= 1 && NT % 2 == 0 && KS % 2 == 0 && STAGES >= 2 && WORDS % 4 == 0, "tile");
};

// per-thread chunk loader over a K range starting at kb (A row stride lda, B row stride ldb)
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
    DEVI void init(const __nv_bfloat16* A, const __nv_bfloat16* B, int m0, int n0, int M, int lda, int ldb, int kb) {
        const int tid = threadIdx.x;
        r0 = tid / C::CH;
        const int c = tid % C::CH;
        dst0 = (r0 * C::BK + ((c ^ swz_f<C::BK>(r0)) * 8)) * 2;
        pa = A + kb;
        pb = B + kb;
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

// one thread's fp32 sums as bf16 vector v (words 4v .. 4v+3 = flat elements 8v .. 8v+7, RN)
template <class C>
DEVI uint4 pvec(const float (&acc)[C::MT][C::NT][4], int v) {
    const float* af = &acc[0][0][0];
    return make_uint4(pack_bf2(af[8 * v], af[8 * v + 1]), pack_bf2(af[8 * v + 2], af[8 * v + 3]),
                      pack_bf2(af[8 * v + 4], af[8 * v + 5]), pack_bf2(af[8 * v + 6], af[8 * v + 7]));
}

// ngemm9b's mainloop (XOR-swizzled smem + ldmatrix, fragments double-buffered, one barrier per k-tile, refill of tile
// kt+STAGES-1 at k-step 0 of tile kt, serpentine mma order) over the K range [kb, kb + KT * BK); every KC k-tiles
// (a slice boundary) except at the end: dst + slice * V * THREADS <- bf16(acc) and acc restarts at +0
template <class C>
DEVI void mainloop(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, int M, int lda, int ldb,
                   int m0, int n0, int kb, int KT, int KC, uint4* __restrict__ dst, __nv_bfloat16* smem,
                   float (&acc)[C::MT][C::NT][4]) {
    constexpr int KS = C::KS, BK = C::BK, S = C::STAGES, MT = C::MT, NT = C::NT;
    const uint32_t sA0 = smem_u32(smem);
    const uint32_t sB0 = sA0 + S * C::A_STAGE * 2;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    auto sa = [&](int s) { return sA0 + s * C::A_STAGE * 2; };
    auto sb = [&](int s) { return sB0 + s * C::B_STAGE * 2; };
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
    Loader<C> ld;
    ld.init(A, B, m0, n0, M, lda, ldb, kb);
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
    int in_sl = 0;
    uint4* d = dst;
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
        if (++in_sl == KC && kt + 1 < KT) {   // slice boundary inside the group: flush this slice's partial
#pragma unroll
            for (int v = 0; v < C::V; ++v) {
                stcg16(d + v * C::THREADS, pvec<C>(acc, v));
            }
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
#pragma unroll
                    for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
            d += C::V * C::THREADS;
            in_sl = 0;
        }
    }
    cp_wait<0>();
    __syncthreads();
}

// plain epilogue: stmatrix.x4 -> padded smem tile -> 16-byte stores of rows < M
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

// fold step over one 16-byte vector of bf16 partials: t = first ? b : t + b (fp32 RN), 8 elements
DEVI void fold8(float* t, const uint4 x, bool first) {
    const uint32_t w[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const float lo = bf_lo(w[u]), hi = bf_hi(w[u]);
        t[2 * u] = first ? lo : __fadd_rn(t[2 * u], lo);
        t[2 * u + 1] = first ? hi : __fadd_rn(t[2 * u + 1], hi);
    }
}

// one (group, tile) CTA: slices [s0, s1) of the tile at (m0, n0); wsl = the tile's partial slots, cnt its counter
template <class C>
DEVI void lg_tile(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Cm,
                  int M, int K, int lda, int ldc, int chunk, int S, int NG, int s0, int s1, int m0, int n0,
                  uint4* __restrict__ wsl, int* __restrict__ cnt, __nv_bfloat16* smem, int* s_flag) {
    constexpr int V = C::V, TH = C::THREADS;
    const int tid = threadIdx.x;
    const int kb = s0 * chunk;
    const int ke = min(K, s1 * chunk);
    float acc[C::MT][C::NT][4];
    uint4* const mine = wsl + tid;
    mainloop<C>(A, B, M, lda, K, m0, n0, kb, (ke - kb) / C::BK, chunk / C::BK, mine + (size_t)s0 * V * TH, smem, acc);
    if (S == 1) {
        store_tile<C>(acc, Cm, M, ldc, m0, n0, smem);
        return;
    }
    const int own = s1 - 1;   // acc = slice own (fp32 sums)
    if (NG > 1) {
        if (tid == 0) *s_flag = ld_acquire(cnt) == NG - 1;
        __syncthreads();
        if (!*s_flag) {
            uint4* dst = mine + (size_t)own * V * TH;
#pragma unroll
            for (int v = 0; v < V; ++v) stcg16(dst + v * TH, pvec<C>(acc, v));
            __threadfence();
            __syncthreads();
            if (tid == 0) *s_flag = atomicAdd(cnt, 1) == NG - 1;
            __syncthreads();
            if (!*s_flag) return;
        }
        __threadfence();
    }
    // last CTA of the tile: own partial -> smem (thread-native), then fold all S partials in slice order into acc
    uint4* so = reinterpret_cast<uint4*>(smem) + tid;
#pragma unroll
    for (int v = 0; v < V; ++v) so[v * TH] = pvec<C>(acc, v);
    float* tf = &acc[0][0][0];
    for (int s2 = 0; s2 < S; ++s2) {
        const bool first = s2 == 0;
        if (s2 == own) {
#pragma unroll
            for (int v = 0; v < V; ++v) fold8(tf + 8 * v, so[v * TH], first);
        } else {
            const uint4* src = mine + (size_t)s2 * V * TH;
            uint4 x[V];
#pragma unroll
            for (int v = 0; v < V; ++v) x[v] = ldcg16(src + v * TH);
#pragma unroll
            for (int v = 0; v < V; ++v) fold8(tf + 8 * v, x[v], first);
        }
    }
    if (NG > 1 && tid == 0) *cnt = 0;
    __syncthreads();   // every thread has read its own vectors from smem before store_tile reuses it
    store_tile<C>(acc, Cm, M, ldc, m0, n0, smem);
}

template <class CB, class CS>
DEVI void nlg(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Cm,
              int M, int N, int K, int lda, int ldc, int chunk, int mb, uint4* __restrict__ ws, int* __restrict__ flags,
              __nv_bfloat16* smem, int* s_flag) {
    static_assert(CB::THREADS == CS::THREADS, "threads");
    const int S = (K + chunk - 1) / chunk;
    const int mtb = mb / CB::BM;
    const int Tb = mtb * (N / CB::BN);
    const int mts = M > mb ? (M - mb + CS::BM - 1) / CS::BM : 0;
    const int T = Tb + mts * (N / CS::BN);
    const int NG = gridDim.x / T;
    const int G = (S + NG - 1) / NG;
    const int g = blockIdx.x / T;
    const int t = blockIdx.x % T;
    const int s0 = g * G;
    const int s1 = min(S, s0 + G);
    if (s0 >= S) return;   // (cannot happen for NG = ceil(S / G); kept as a guard)
    const int NGe = (S + G - 1) / G;   // groups that actually hold slices
    if (t < Tb) {
        const int m0 = (t % mtb) * CB::BM, n0 = (t / mtb) * CB::BN;
        lg_tile<CB>(A, B, Cm, M, K, lda, ldc, chunk, S, NGe, s0, s1, m0, n0,
                    ws + (size_t)t * S * CB::V * CB::THREADS, flags + t, smem, s_flag);
    } else {
        const int t2 = t - Tb;
        const int m0 = mb + (t2 % mts) * CS::BM, n0 = (t2 / mts) * CS::BN;
        lg_tile<CS>(A, B, Cm, M, K, lda, ldc, chunk, S, NGe, s0, s1, m0, n0,
                    ws + (size_t)Tb * S * CB::V * CB::THREADS + (size_t)t2 * S * CS::V * CS::THREADS, flags + t, smem,
                    s_flag);
    }
}
}  // namespace ng9l

#define NLG_ARGS                                                                                                   \
    const __nv_bfloat16 *__restrict__ A, const __nv_bfloat16 *__restrict__ B, __nv_bfloat16 *__restrict__ Cm, int M, \
        int N, int K, int lda, int ldc, int chunk, int mb, float *__restrict__ ws, int *__restrict__ flags
#define NLG9(NAME, CB, CS)                                                                                         \
    extern "C" __global__ void __launch_bounds__(CB::THREADS, 1) NAME(NLG_ARGS) {                                  \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        __shared__ int s_flag;                                                                                      \
        ng9l::nlg<CB, CS>(A, B, Cm, M, N, K, lda, ldc, chunk, mb, reinterpret_cast<uint4*>(ws), flags,             \
                          reinterpret_cast<__nv_bfloat16*>(smem_raw), &s_flag);                                     \
    }

// (BM, BN, BK, warps M, warps N, stages)
using G128x256 = ng9l::Cfg<128, 256, 64, 2, 4, 2>;
using G64x256 = ng9l::Cfg<64, 256, 64, 2, 4, 2>;
using G32x256 = ng9l::Cfg<32, 256, 64, 1, 8, 2>;
using G128x128 = ng9l::Cfg<128, 128, 64, 2, 4, 3>;
using G64x128 = ng9l::Cfg<64, 128, 64, 2, 4, 3>;
using G32x128 = ng9l::Cfg<32, 128, 64, 1, 8, 2>;

NLG9(nlg9_4_b128x256_s32x256, G128x256, G32x256)
NLG9(nlg9_4_b128x256_s64x256, G128x256, G64x256)
NLG9(nlg9_4_b128x128_s32x128, G128x128, G32x128)
NLG9(nlg9_4_b128x128_s64x128, G128x128, G64x128)

// tall tails (COV9b 2026-10-21, see ngemm9t.cu): the last 128 + (R mod 128) rows in one band of 160 / 192 / 224-row x
// 128-column tiles (drop 1), or the last 96 rows in one band of 96x256 tiles (drop 0)
using G160x128 = ng9l::Cfg<160, 128, 64, 2, 4, 2>;
using G192x128 = ng9l::Cfg<192, 128, 64, 2, 4, 2>;
using G224x128 = ng9l::Cfg<224, 128, 64, 2, 4, 2>;
using G96x256 = ng9l::Cfg<96, 256, 64, 2, 4, 2>;
NLG9(nlg9_4_b128x256_t160x128, G128x256, G160x128)
NLG9(nlg9_4_b128x256_t192x128, G128x256, G192x128)
NLG9(nlg9_4_b128x256_t224x128, G128x256, G224x128)
NLG9(nlg9_4_b128x256_s96x256, G128x256, G96x256)
