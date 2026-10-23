// KKT2N v7 (= v6 + masked-zero terms left out of the substitution; v6 = v4 + bank-conflict-free swizzled fp32
// blocks, vector row loads in the substitution): native sm_120 replacement of fla 0.5.2 chunk_gated_delta_rule_fwd_kkt_solve_kernel (e4c pin BK 64,
// 1 warp; every fla config gives the same bits) for grouped values HV = 2 H (S1ENG_GVA=1): one CTA (4 warps) per
// (64-row chunk, batch row, KEY head) computes the raw K K^T blocks once and, for both value heads of that key head,
// the gate / beta scaling, the forward substitution of the four diagonal 16 x 16 blocks and the block merge, and
// stores A on the entries fla stores (rows < T, the lower block triangle, diagonal blocks whole).
//
// Arithmetic = the Triton kernel's (PTX of the pinned build), op for op:
//   R_pq   = mma.m16n8k16.bf16 chain over k16 = 0..7 ascending from +0 (k rows of sub-chunk p times those of q)
//   w      = mask ? ex2.approx.ftz(g_i - g_j) : 0             (diag: i > j; rows / cols < T)
//   diag   : nA = fma.rn(-(R * w), beta_i, +0)                ( = Triton's 0 - (R w) beta, contracted)
//   offdiag: A  = (R * w) * beta_i
//   subst  : for i = 2 .. min(16, T - i_tc) - 1, per column j of Ai (initially nA):
//              b_k = k < i ? (nA[i][k] + 0) : +0
//              s_r = fma.rn(Ai[r][j], b_r, Ai[r+8][j] * b_{r+8})            r = 0..7 (Triton's in-thread step)
//              Ai[i][j] = b_j + (((s0+s4)+(s2+s6)) + ((s1+s5)+(s3+s7)))     (its xor-16/8/4 butterfly)
//            then Ai += I (0.0 / 1.0 added to every element)
//   merge  : mma.m16n8k8.tf32 (fp32 bits passed as is) in Triton's operand layouts: kWidth 2 dots take k-step s over
//            logical k {2t+s, 8+2t+s} (slots t, t+4), kWidth 1 dots over {8s+t, 8s+t+4}; "dot + dot" chains the
//            accumulator; -dot = 0 - D
//   store  : cvt.rn.bf16
// v2 layout: phase 1 static block lists per warp (A/B fragments shared, chains interleaved); gate operands loaded to
// registers before any shared store; branch-free substitution for whole chunks; two warps per value head for the
// merge, split by dependency chain (no exchange), C -> A / B operand moves by warp shuffles (no scratch shared
// memory: 22.8 KB, 4 CTAs / SM).
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

namespace {
constexpr int H = 16, HV = 32, KD = 128, BT = 64;
// fp32 16 x 16 block: element (r, c) at prow(r) * 16 + (c ^ swz(r)); blocks 272 floats apart (alternate 16-bank
// halves). prow swaps rows 2/3 (mod 4) so rows r, r + 2 sit in opposite bank halves; swz is linear over GF(2) (rows
// 1, 2, 4, 8 -> 1, 9, 13, 2). Every warp access (mma A / B fragments of both k widths, C-layout stores, the
// per-column substitution, the row-wise diagonal store, broadcasts) is bank-conflict free (LD 17: 2-4 way).
constexpr int BLK = 272;        // floats per block
// shared memory (bytes): X = k tile (phase 0-1), then NA [2][4] + AO [2][6] fp32 blocks; then g, beta
constexpr int X_BYTES = 20 * BLK * 4;
constexpr int OFF_G = X_BYTES;
constexpr int OFF_B = OFF_G + 2 * 64 * 4;
constexpr int SMEM = OFF_B + 2 * 64 * 4;
static_assert(X_BYTES >= 64 * KD * 2, "k tile must fit");

__host__ __device__ constexpr int swz(int r) {
    return ((r & 1) ? 1 : 0) ^ ((r & 2) ? 9 : 0) ^ ((r & 4) ? 13 : 0) ^ ((r & 8) ? 2 : 0);
}
__host__ __device__ constexpr int prow(int r) { return (r & ~1) | ((r ^ (r >> 1)) & 1); }
__host__ __device__ constexpr int sx(int r, int c) { return prow(r) * 16 + (c ^ swz(r)); }
DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
DEVI void cp16(uint32_t dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
DEVI void cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
DEVI void ldsm4(uint32_t addr, uint32_t* r) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
DEVI void mma_bf16(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI void mma_tf32(float* d, const uint32_t* a, const uint32_t* b) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
DEVI float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}
DEVI uint32_t pack_bf16(float lo, float hi) {
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
DEVI uint32_t fu(float x) { return __float_as_uint(x); }
DEVI float shf(float v, int src) { return __shfl_sync(0xffffffffu, v, src); }

// ------------------------------------------------------------------------------------------------ phase 1 + 2
// warp W's blocks (p, q): W0 (0,0) (1,0) (1,1); W1 (2,0) (2,1) (2,2); W2 (3,0) (3,1); W3 (3,2) (3,3)
__host__ __device__ constexpr int BN(int w) { return w < 2 ? 3 : 2; }
__host__ __device__ constexpr int BP(int w, int b) { return w == 0 ? (b == 0 ? 0 : 1) : (w == 1 ? 2 : 3); }
__host__ __device__ constexpr int BQ(int w, int b) { return w == 0 ? (b == 2 ? 1 : 0) : (w == 3 ? 2 + b : b); }
template <int W>
DEVI void phase12(unsigned char* smem, int lane, int Tc) {
    constexpr int N = BN(W);
    const uint32_t ks = smem_u32(smem);
    const int gq = lane >> 2, tq = lane & 3;
    float acc[N][2][4];
#pragma unroll
    for (int b = 0; b < N; ++b)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[b][nt][e] = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        uint32_t af[N][4], bf[N][4];
#pragma unroll
        for (int b = 0; b < N; ++b) {
            if (b == 0 || BP(W, b) != BP(W, b - 1)) {
                const int ra = 16 * BP(W, b) + (lane & 15), ca = 2 * s + (lane >> 4);
                ldsm4(ks + ra * 256 + ((ca ^ (ra & 7)) << 4), af[b]);
            } else {
#pragma unroll
                for (int x = 0; x < 4; ++x) af[b][x] = af[b - 1][x];
            }
            bool reuse = false;
#pragma unroll
            for (int c = 0; c < b; ++c) {
                if (!reuse && BQ(W, c) == BQ(W, b)) {
#pragma unroll
                    for (int x = 0; x < 4; ++x) bf[b][x] = bf[c][x];
                    reuse = true;
                }
            }
            if (!reuse) {
                const int rb = 16 * BQ(W, b) + (lane & 7) + ((lane >> 4) << 3), cb = 2 * s + ((lane >> 3) & 1);
                ldsm4(ks + rb * 256 + ((cb ^ (rb & 7)) << 4), bf[b]);
            }
        }
#pragma unroll
        for (int b = 0; b < N; ++b) {
            mma_bf16(acc[b][0], af[b], bf[b][0], bf[b][1]);
            mma_bf16(acc[b][1], af[b], bf[b][2], bf[b][3]);
        }
    }
    if (Tc < BT) {   // fla never computes blocks of sub-chunks at or past T: they stay +0
#pragma unroll
        for (int b = 0; b < N; ++b)
            if (16 * BP(W, b) >= Tc)
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[b][nt][e] = 0.f;
    }
    // gate operands to registers before anything is stored (the k tile region is overwritten below)
    const float* gs = reinterpret_cast<const float*>(smem + OFF_G);
    const float* bs = reinterpret_cast<const float*>(smem + OFF_B);
    float grow[N][2][2], gcol[N][2][2][2], brow[N][2][2];   // [block][head][row half] / [block][head][nt][e]
#pragma unroll
    for (int b = 0; b < N; ++b)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                grow[b][h][hr] = gs[64 * h + 16 * BP(W, b) + gq + 8 * hr];
                brow[b][h][hr] = bs[64 * h + 16 * BP(W, b) + gq + 8 * hr];
            }
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) gcol[b][h][nt][e] = gs[64 * h + 16 * BQ(W, b) + 8 * nt + 2 * tq + e];
        }
    __syncthreads();   // every warp is done with the k tile
    float* X = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int b = 0; b < N; ++b) {
        const int p = BP(W, b), q = BQ(W, b);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float* dst = p == q ? X + (h * 4 + p) * BLK : X + (8 + h * 6 + (p * (p - 1)) / 2 + q) * BLK;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int hr = e >> 1, row = gq + 8 * hr, col = 8 * nt + 2 * tq + (e & 1);
                    const int ri = 16 * p + row, ci = 16 * q + col;
                    const bool ok = (p != q || row > col) && ri < Tc && ci < Tc;
                    const float w = ok ? ex2(__fsub_rn(grow[b][h][hr], gcol[b][h][nt][e & 1])) : 0.f;
                    const float rw = __fmul_rn(acc[b][nt][e], w);
                    dst[sx(row, col)] = p == q ? __fmaf_rn(-rw, brow[b][h][hr], 0.f) : __fmul_rn(rw, brow[b][h][hr]);
                }
        }
    }
}

// ------------------------------------------------------------------------------------------------ phase 4 helpers
// a 16 x 16 fp32 matrix in mma C layout held by one warp: d[j][0..3] = (g, 8j+2t), (g, 8j+2t+1), (g+8, ..), (g+8, ..)
// operand fragments from a padded shared block
DEVI void smA_k2(uint32_t a[2][4], const float* X, int g, int t) {      // [s][slot]
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        a[s][0] = fu(X[sx(g, 2 * t + s)]);
        a[s][1] = fu(X[sx(g + 8, 2 * t + s)]);
        a[s][2] = fu(X[sx(g, 8 + 2 * t + s)]);
        a[s][3] = fu(X[sx(g + 8, 8 + 2 * t + s)]);
    }
}
DEVI void smA_k1(uint32_t a[2][4], const float* X, int g, int t) {
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        a[s][0] = fu(X[sx(g, 8 * s + t)]);
        a[s][1] = fu(X[sx(g + 8, 8 * s + t)]);
        a[s][2] = fu(X[sx(g, 8 * s + t + 4)]);
        a[s][3] = fu(X[sx(g + 8, 8 * s + t + 4)]);
    }
}
DEVI void smB_k2(uint32_t b[2][2][2], const float* Y, int g, int t) {   // [n tile j][s][slot]
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            b[j][s][0] = fu(Y[sx(2 * t + s, 8 * j + g)]);
            b[j][s][1] = fu(Y[sx(8 + 2 * t + s, 8 * j + g)]);
        }
}
DEVI void smB_k1(uint32_t b[2][2][2], const float* Y, int g, int t) {
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            b[j][s][0] = fu(Y[sx(8 * s + t, 8 * j + g)]);
            b[j][s][1] = fu(Y[sx(8 * s + t + 4, 8 * j + g)]);
        }
}
// operand fragments from a C-layout matrix in registers (warp shuffles)
DEVI void rgA_k1(uint32_t a[2][4], const float d[2][4], int lane) {
    const int t = lane & 3, base = lane & ~3, odd = t & 1;
    const int L0 = base | (t >> 1), L2 = base | (2 + (t >> 1));
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        const float x0 = shf(d[s][0], L0), x1 = shf(d[s][1], L0), x2 = shf(d[s][2], L0), x3 = shf(d[s][3], L0);
        const float y0 = shf(d[s][0], L2), y1 = shf(d[s][1], L2), y2 = shf(d[s][2], L2), y3 = shf(d[s][3], L2);
        a[s][0] = fu(odd ? x1 : x0);
        a[s][1] = fu(odd ? x3 : x2);
        a[s][2] = fu(odd ? y1 : y0);
        a[s][3] = fu(odd ? y3 : y2);
    }
}
DEVI void rgB_k2(uint32_t b[2][2][2], const float d[2][4], int lane) {
    const int g = lane >> 2, t = lane & 3, odd = g & 1;
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        const int L = (2 * t + s) * 4 + (g >> 1);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float x0 = shf(d[j][0], L), x1 = shf(d[j][1], L), x2 = shf(d[j][2], L), x3 = shf(d[j][3], L);
            b[j][s][0] = fu(odd ? x1 : x0);
            b[j][s][1] = fu(odd ? x3 : x2);
        }
    }
}
DEVI void rgB_k1(uint32_t b[2][2][2], const float d[2][4], int lane) {
    const int g = lane >> 2, t = lane & 3, odd = g & 1;
    const int L0 = t * 4 + (g >> 1), L1 = (t + 4) * 4 + (g >> 1);
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            const float x0 = shf(d[j][2 * s], L0), x1 = shf(d[j][2 * s + 1], L0);
            const float y0 = shf(d[j][2 * s], L1), y1 = shf(d[j][2 * s + 1], L1);
            b[j][s][0] = fu(odd ? x1 : x0);
            b[j][s][1] = fu(odd ? y1 : y0);
        }
}
// d[j] (+)= A @ B[:, 8j .. 8j + 7] over the two k-steps
DEVI void mm(float d[2][4], const uint32_t a[2][4], const uint32_t b[2][2][2]) {
#pragma unroll
    for (int s = 0; s < 2; ++s)
#pragma unroll
        for (int j = 0; j < 2; ++j) mma_tf32(d[j], a[s], b[j][s]);
}
DEVI void zero(float d[2][4]) {
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) d[j][e] = 0.f;
}
DEVI void neg0(float d[2][4]) {
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) d[j][e] = __fsub_rn(0.f, d[j][e]);
}
// bf16 store of an off-diagonal result block (p, q) from C layout
DEVI void st_out(__nv_bfloat16* Ah, const float d[2][4], int p, int q, int g, int t, int Tc) {
    const int r0 = 16 * p + g;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int c = 16 * q + 8 * j + 2 * t;
        if (r0 < Tc) *reinterpret_cast<uint32_t*>(Ah + (size_t)r0 * (HV * BT) + c) = pack_bf16(d[j][0], d[j][1]);
        if (r0 + 8 < Tc)
            *reinterpret_cast<uint32_t*>(Ah + (size_t)(r0 + 8) * (HV * BT) + c) = pack_bf16(d[j][2], d[j][3]);
    }
}

// a + b where either term may be absent (presence known at compile time after unrolling)
DEVI float add_opt(float a, bool pa, float b, bool pb, bool& p) {
    p = pa || pb;
    return pa && pb ? __fadd_rn(a, b) : (pa ? a : b);
}
// forward substitution of one column (thread) of one diagonal block; FULL: all 16 rows below T
template <bool FULL>
DEVI void subst(float* X, int j, int n) {
    float Ai[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) Ai[r] = X[sx(r, j)];
#pragma unroll
    for (int i = 2; i < 16; ++i) {
        if (FULL || i < n) {
            // row i of the block as aligned float4 groups (only the groups holding columns < i are used)
            float rw[16];
            const float4* Xi = reinterpret_cast<const float4*>(X + prow(i) * 16);
#pragma unroll
            for (int q4 = 0; q4 < 4; ++q4) {
                const float4 v = Xi[q4];
                rw[4 * q4] = v.x; rw[4 * q4 + 1] = v.y; rw[4 * q4 + 2] = v.z; rw[4 * q4 + 3] = v.w;
            }
            float bb[16];
#pragma unroll
            for (int kk = 0; kk < 16; ++kk) bb[kk] = kk < i ? __fadd_rn(rw[kk ^ swz(i)], 0.f) : 0.f;
            const float bj = j < i ? __fadd_rn(Ai[i], 0.f) : 0.f;
            // v7: terms whose b is the masked +0 (columns >= i, known at compile time) are left out of the products
            // and the butterfly. Exact for finite values: such a term is +-0 (Ai finite), x + (+-0) = x and
            // fma(a, b, +-0) = a * b for every non-zero result, and a zero w only reaches Ai[i] through bj + w with
            // bj never -0 (+0 or Ai[i] + 0), so its sign never shows.
            float s[8];
            bool ps[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                ps[r] = r < i;
                if (r + 8 < i) s[r] = __fmaf_rn(Ai[r], bb[r], __fmul_rn(Ai[r + 8], bb[r + 8]));
                else if (r < i) s[r] = __fmul_rn(Ai[r], bb[r]);
                else s[r] = 0.f;
            }
            bool p04, p26, p15, p37, pl, pr;
            const float a04 = add_opt(s[0], ps[0], s[4], ps[4], p04);
            const float a26 = add_opt(s[2], ps[2], s[6], ps[6], p26);
            const float a15 = add_opt(s[1], ps[1], s[5], ps[5], p15);
            const float a37 = add_opt(s[3], ps[3], s[7], ps[7], p37);
            const float l = add_opt(a04, p04, a26, p26, pl);
            const float rr = add_opt(a15, p15, a37, p37, pr);
            bool pw;
            const float w = add_opt(l, pl, rr, pr, pw);
            Ai[i] = __fadd_rn(bj, w);   // i >= 2: s[0] is always present
        }
    }
#pragma unroll
    for (int r = 0; r < 16; ++r) Ai[r] = __fadd_rn(Ai[r], r == j ? 1.f : 0.f);
    __syncthreads();   // every row of every block read
#pragma unroll
    for (int r = 0; r < 16; ++r) X[sx(r, j)] = Ai[r];
}
}  // namespace

#ifdef KTS
#define STAMP(i) do { if (threadIdx.x == 0) { long long c_; asm volatile("mov.u64 %0, %%clock64;" : "=l"(c_)); \
    kts[(blockIdx.y * gridDim.x + blockIdx.x) * 8 + (i)] = c_; } } while (0)
#define GSTAMP(i) do { if (threadIdx.x == 0) { long long c_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(c_)); \
    kts[(blockIdx.y * gridDim.x + blockIdx.x) * 8 + (i)] = c_; } } while (0)
#define SMSTAMP() do { if (threadIdx.x == 0) { unsigned s_; asm volatile("mov.u32 %0, %%smid;" : "=r"(s_)); \
    kts[(blockIdx.y * gridDim.x + blockIdx.x) * 8 + 5] = s_; } } while (0)
__device__ long long kts[1 << 16];
__device__ long long kts2[1 << 16];
#define MSTAMP(i) do { if (threadIdx.x == 0) { long long c_; asm volatile("mov.u64 %0, %%clock64;" : "=l"(c_)); \
    kts2[(blockIdx.y * gridDim.x + blockIdx.x) * 8 + (i)] = c_; } } while (0)
#else
#define MSTAMP(i) do {} while (0)
#define STAMP(i) do {} while (0)
#define GSTAMP(i) do {} while (0)
#define SMSTAMP() do {} while (0)
#endif

// grid (NT, B * H), 128 threads. k [B, T, H, 128] bf16, g [B, T, HV] fp32 (chunk-cumulated, x 1/ln2),
// beta [B, T, HV] bf16, A [B, T, HV, 64] bf16 (out), slen [B] int32 (used if use_slen: slots of length 0 exit).
extern "C" __global__ void __launch_bounds__(128, 4)
kkt2n(const __nv_bfloat16* __restrict__ k, const float* __restrict__ g, const __nv_bfloat16* __restrict__ beta,
      __nv_bfloat16* __restrict__ A, const int* __restrict__ slen, int T, int use_slen) {
    extern __shared__ __align__(128) unsigned char smem[];
    STAMP(0);
    GSTAMP(6);
    SMSTAMP();
    const int i_t = blockIdx.x, bh = blockIdx.y;
    const int b = bh / H, kh = bh % H;
    if (use_slen && slen[b] == 0) return;
    const int t0 = i_t * BT;
    if (t0 >= T) return;
    const int Tc = T - t0;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int gq = lane >> 2, tq = lane & 3;
    float* X = reinterpret_cast<float*>(smem);

    // ------------------------------------------------------------------ phase 0: loads
    {
        const __nv_bfloat16* kb = k + ((size_t)(b * T + t0) * H + kh) * KD;
        const uint32_t ks = smem_u32(smem);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int q = tid + 128 * i, r = q >> 4, c = q & 15;
            const bool ok = r < Tc;
            cp16(ks + r * 256 + ((c ^ (r & 7)) << 4), kb + (size_t)(ok ? r : 0) * (H * KD) + c * 8, ok ? 16 : 0);
        }
        cp_commit();
        float* gs = reinterpret_cast<float*>(smem + OFF_G);
        float* bs = reinterpret_cast<float*>(smem + OFF_B);
        if (tid < 64) {
            float2 gv = make_float2(0.f, 0.f);
            if (tid < Tc) gv = *reinterpret_cast<const float2*>(g + (size_t)(b * T + t0 + tid) * HV + 2 * kh);
            gs[tid] = gv.x;
            gs[64 + tid] = gv.y;
        } else {
            const int r = tid - 64;
            float b0 = 0.f, b1 = 0.f;
            if (r < Tc) {
                const __nv_bfloat162 bv =
                    *reinterpret_cast<const __nv_bfloat162*>(beta + (size_t)(b * T + t0 + r) * HV + 2 * kh);
                b0 = __bfloat162float(bv.x);
                b1 = __bfloat162float(bv.y);
            }
            bs[r] = b0;
            bs[64 + r] = b1;
        }
        cp_wait0();
        __syncthreads();
    }
    STAMP(1);
    // ------------------------------------------------------------------ phase 1 + 2
    switch (warp) {
        case 0: phase12<0>(smem, lane, Tc); break;
        case 1: phase12<1>(smem, lane, Tc); break;
        case 2: phase12<2>(smem, lane, Tc); break;
        default: phase12<3>(smem, lane, Tc); break;
    }
    __syncthreads();
    STAMP(2);
    // ------------------------------------------------------------------ phase 3: forward substitution (per column)
    {
        const int hh = tid >> 6, p = (tid >> 4) & 3, j = tid & 15;
        float* Xb = X + (hh * 4 + p) * BLK;
        if (Tc >= BT) subst<true>(Xb, j, 16);
        else subst<false>(Xb, j, min(16, Tc - 16 * p));
        __syncthreads();
    }
    STAMP(3);
    // diagonal blocks out: thread (head, p, row) stores 16 bf16 of row 16p + row, cols 16p ..
    {
        const int hh = tid >> 6, p = (tid >> 4) & 3, r = tid & 15;
        const float* Xb = X + (hh * 4 + p) * BLK;
        if (16 * p + r < Tc) {
            float Xr[16];
#pragma unroll
            for (int c = 0; c < 16; ++c) Xr[c] = Xb[sx(r, c)];
            uint4 v0, v1;
            v0.x = pack_bf16(Xr[0], Xr[1]);   v0.y = pack_bf16(Xr[2], Xr[3]);
            v0.z = pack_bf16(Xr[4], Xr[5]);   v0.w = pack_bf16(Xr[6], Xr[7]);
            v1.x = pack_bf16(Xr[8], Xr[9]);   v1.y = pack_bf16(Xr[10], Xr[11]);
            v1.z = pack_bf16(Xr[12], Xr[13]); v1.w = pack_bf16(Xr[14], Xr[15]);
            __nv_bfloat16* dst = A + ((size_t)(b * T + t0 + 16 * p + r) * HV + 2 * kh + hh) * BT + 16 * p;
            reinterpret_cast<uint4*>(dst)[0] = v0;
            reinterpret_cast<uint4*>(dst)[1] = v1;
        }
    }
    // ------------------------------------------------------------------ phase 4: block merge, two warps per head
    // split by dependency chain (no exchange between the warps): role 0 = T10 -> Ai10 -> U20 -> Ai20 -> U30 -> Ai30
    // (36 mma), role 1 = T21 -> Ai21 -> U31 -> Ai31 and T32 -> Ai32 (28 mma); each warp holds whole 16 x 16 tiles
    {
        const int hh = warp >> 1;
        const float* Ai00 = X + (hh * 4 + 0) * BLK;
        const float* Ai11 = X + (hh * 4 + 1) * BLK;
        const float* Ai22 = X + (hh * 4 + 2) * BLK;
        const float* Ai33 = X + (hh * 4 + 3) * BLK;
        const float* A10 = X + (8 + hh * 6 + 0) * BLK;
        const float* A20 = X + (8 + hh * 6 + 1) * BLK;
        const float* A21 = X + (8 + hh * 6 + 2) * BLK;
        const float* A30 = X + (8 + hh * 6 + 3) * BLK;
        const float* A31 = X + (8 + hh * 6 + 4) * BLK;
        const float* A32 = X + (8 + hh * 6 + 5) * BLK;
        __nv_bfloat16* Ah = A + ((size_t)(b * T + t0) * HV + 2 * kh + hh) * BT;
        uint32_t fa[2][4], fb[2][2][2];
        if ((warp & 1) == 0) {
            float t10[2][4], u20[2][4], u30[2][4];
            // L1 (kWidth 2): T10 = Ai11 A10, U20 = A20 Ai00, U30 = A30 Ai00
            zero(t10); smA_k2(fa, Ai11, gq, tq); smB_k2(fb, A10, gq, tq); mm(t10, fa, fb);
            smB_k2(fb, Ai00, gq, tq);
            zero(u20); smA_k2(fa, A20, gq, tq); mm(u20, fa, fb);
            zero(u30); smA_k2(fa, A30, gq, tq); mm(u30, fa, fb);
            MSTAMP(0);
            // L2 (kWidth 1): Ai10 = -(T10 Ai00)
            float a10[2][4];
            zero(a10); rgA_k1(fa, t10, lane); smB_k1(fb, Ai00, gq, tq); mm(a10, fa, fb); neg0(a10);
            MSTAMP(1);
            st_out(Ah, a10, 1, 0, gq, tq, Tc);
            // L3 (kWidth 2): U20 += A21 Ai10, U30 += A31 Ai10
            rgB_k2(fb, a10, lane);
            smA_k2(fa, A21, gq, tq); mm(u20, fa, fb);
            smA_k2(fa, A31, gq, tq); mm(u30, fa, fb);
            MSTAMP(2);
            // L4 (kWidth 1): Ai20 = -(Ai22 U20)
            float a20[2][4];
            zero(a20); smA_k1(fa, Ai22, gq, tq); rgB_k1(fb, u20, lane); mm(a20, fa, fb); neg0(a20);
            MSTAMP(3);
            // L5 (kWidth 2): U30 += A32 Ai20
            smA_k2(fa, A32, gq, tq); rgB_k2(fb, a20, lane); mm(u30, fa, fb);
            MSTAMP(4);
            st_out(Ah, a20, 2, 0, gq, tq, Tc);
            // L6 (kWidth 1): Ai30 = -(Ai33 U30)
            float a30[2][4];
            zero(a30); smA_k1(fa, Ai33, gq, tq); rgB_k1(fb, u30, lane); mm(a30, fa, fb); neg0(a30);
            st_out(Ah, a30, 3, 0, gq, tq, Tc);
        } else {
            float t21[2][4], t32[2][4], u31[2][4];
            // L1 (kWidth 2): T21 = Ai22 A21, T32 = Ai33 A32, U31 = A31 Ai11
            zero(t21); smA_k2(fa, Ai22, gq, tq); smB_k2(fb, A21, gq, tq); mm(t21, fa, fb);
            zero(t32); smA_k2(fa, Ai33, gq, tq); smB_k2(fb, A32, gq, tq); mm(t32, fa, fb);
            zero(u31); smA_k2(fa, A31, gq, tq); smB_k2(fb, Ai11, gq, tq); mm(u31, fa, fb);
            // L2 (kWidth 1): Ai21 = -(T21 Ai11), Ai32 = -(T32 Ai22)
            float a21[2][4], a32[2][4];
            zero(a21); rgA_k1(fa, t21, lane); smB_k1(fb, Ai11, gq, tq); mm(a21, fa, fb); neg0(a21);
            zero(a32); rgA_k1(fa, t32, lane); smB_k1(fb, Ai22, gq, tq); mm(a32, fa, fb); neg0(a32);
            // L3 (kWidth 2): U31 += A32 Ai21
            smA_k2(fa, A32, gq, tq); rgB_k2(fb, a21, lane); mm(u31, fa, fb);
            st_out(Ah, a21, 2, 1, gq, tq, Tc);
            st_out(Ah, a32, 3, 2, gq, tq, Tc);
            // L4 (kWidth 1): Ai31 = -(Ai33 U31)
            float a31[2][4];
            zero(a31); smA_k1(fa, Ai33, gq, tq); rgB_k1(fb, u31, lane); mm(a31, fa, fb); neg0(a31);
            st_out(Ah, a31, 3, 1, gq, tq, Tc);
        }
    }
    STAMP(4);
    GSTAMP(7);
}

extern "C" __global__ void kkt2n_smem_bytes(int* out) { *out = SMEM; }
