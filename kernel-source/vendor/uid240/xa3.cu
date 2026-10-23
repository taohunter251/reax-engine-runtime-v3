// XA3 (kai/rf-xa, BACKLOG2 2.4 XA-LAT): xa.cu's xa_fwd2q with two global-load schedules changed, nothing else.
// Derived by kai from the image's own xa.cu (rgh; sha256 231ea2f7...), which stays unchanged next to it with its xa.cubin.
//   LATE_GATE: the sigmoid-gate epilogue no longer issues 16 dependent global loads per thread (one bf16x2 per output
//     pair inside the okA / okB branches). The CTA's 64 x 256 bf16 gate tile (the last 256 of each q_proj head row,
//     32 KB) is copied by cp.async into the Q tile's shared memory (sQ, same swizzle) once sQ is dead: in the first
//     V-chunk step of the final key block (after that step's barrier every warp is past its last Q ldmatrix, which
//     happens in the final block's QK before the softmax barrier). Rows outside [plen, n) copy nothing (src-size 0,
//     source pointer qg). The copies join that step's cp.async commit group; cp.async.wait_group 0 + __syncthreads
//     before the epilogue, which reads the same bf16 gate values from shared memory.
//   BATCH_Q: the Q-tile build (q norm + RoPE, `#pragma unroll 1` over the warp's 4 rows) issues all 8 row loads
//     (ua / ub of the 4 rows = 4 heads at the one position P0 + w they share) and the position's cos / sin values
//     once, before any arithmetic; then the 4 rows are computed exactly as before.
// Every arithmetic operation, its operands and its order are xa.cu's (HMMA, softmax, epilogue, gate1 / gatex, the
// q norm's explicit __fmul_rn / __fadd_rn / bfr / rsqrt.approx); shared memory stays 100352 B. Kernels:
//   xa_fwd3q  = LATE_GATE + BATCH_Q (the lever), xa_fwd3qe = LATE_GATE only (attribution).
// Start-up gate (batchfwd.PackedForward.check_xa with pf.xa_want, fcl.xl_check): byte for byte against xa_fwd2q on
// every input the XA gate draws (and on outlier / order-witness inputs), per bucket; any miss keeps xa_fwd2q there.
//
// ---- xa.cu's header (unchanged below) ----
// XA (ATTN): native sm_120 attention for the packed attention layers, bit-identical to the reference's masked SDPA
// (torch 2.13 mem-efficient kernel fmha_cutlassF_bf16_aligned_32x128_gmem_sm80: AttentionKernel<bf16, Sm80, aligned,
// 32 queries, 128 keys, kMaxK 65536, dropout, bias>, called with the [L, L] additive causal mask, so every query row
// walks all ceil(L / 128) key blocks) followed by the sigmoid output gate (fused.attn_gate_*), for the computed rows
// only (positions plen .. n-1 of every decision), GQA heads folded (one CTA = 16 positions x the 4 query heads of one
// KV head, K/V tiles shared by the 4 heads).
//
// Per query row the arithmetic is the cutlass kernel's (headers in torch/include/.../mem_eff_attention, SASS of the
// sm_120 cubin in libtorch_cuda.so, 2026-10-21):
//   S = q . k as mma.sync m16n8k8 bf16 (HMMA.1688: DefaultGemmType<Sm80, bf16> has InstructionShape 16x8x8), fp32
//       accumulation over d in ascending k8 order from 0;
//   x = scale * S (FMUL; the bias add of +0 / -inf only turns -0 into +0 or masks), y = x * log2e (FMUL, separate);
//   per 128-key block: mi = max(mi, max y), if (m_prime < mi) { r = exp2f(m_prime - mi); s *= r } else r = 1;
//       e = exp2f(y - mi) (ex2.approx.f32 with the -126 range fix: non-fast-math exp2f), masked e = +0;
//       s += the block's row sum in cutlass order: per 32-key warp column, per lane of the quad the 8 columns
//       {8t + 2q, 8t + 2q + 1 : t = 0..3} in ascending order from 0.0, then (t0 + t1) + (t2 + t3), then the 4 warp
//       columns in ascending order;
//       P = bf16_rn(e); acc = P . V as m16n8k8 over the block's keys in ascending k8 order from 0;
//   epilogue (MemoryEfficientAttentionNormalize, fp32 output accumulator between blocks):
//       first block: buf = acc; a middle block (neither first nor last of the walk): buf = fma(r, buf, acc) (one
//       rounding: nvcc contracts the epilogue's 1 * acc + r * src into FFMA, SASS of the middle-block epilogue);
//       the LAST block of the reference's walk (block (L - 1) / 128, num_keys = L): alpha = 1 / (s == 0 ? 1 : s)
//       (IEEE division), out = bf16(fma(alpha, acc, (alpha * r) * buf)) (first & last: bf16(alpha * acc)).
//   A row at position p has keys only in blocks 0 .. p / 128; the reference's blocks after that are fully masked
//   (r = 1, e = +0, acc = +0), which the kernel replays in closed form: buf <- buf + 0 per non-last trailing block,
//   out = bf16(fma(alpha, +0, alpha * buf)) for the last one. Keys past the CTA's last position inside its diagonal
//   block are skipped (all-zero k8 steps leave the fp32 accumulator unchanged; masked columns add +0 to the sums).
//   Gate: out = bf16(o * bf16(1 / (1 + expf(-gate)))) with o the bf16 attention output (IEEE division, libdevice
//   expf without ftz), as fused._attn_gate_*_kernel.
// The engine's start-up check (batchfwd.PackedForward.check_xa) compares this path byte for byte with the reference
// chain on every bucket (and all 65536 gate values); any mismatch keeps the torch path.
//
// Layouts: q [R, 16, 256] (normed + RoPE'd rows, packed; qmode 1: [B, 16, L, 256] padded), k4 / v4 [B, 4, L, 256]
// (rep-1 K/V, zeros past n), qg [R, 16, 512] (q_proj output, gate = last 256 of each head), out [R, 4096].
// Grid (L / 16, 4, B + 1): blockIdx.z == B zero-fills the packed rows past the used ones (rows nobody writes).
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

namespace xa {

constexpr int HD = 256;     // head dim
constexpr int NH = 16;      // query heads
constexpr int NG = 4;       // KV heads
constexpr int REP = 4;      // query heads per KV head
constexpr int ROWS = 16;    // positions per CTA

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
DEVI void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
// D = A (16x8 bf16, row) * B (8x8 bf16, col) + D, fp32: HMMA.1688.F32.BF16, the cutlass kernel's instruction
DEVI void mma1688(float* d, uint32_t a0, uint32_t a1, uint32_t b0) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(b0));
}
DEVI uint32_t pack_bf2(float a, float b) {
    __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}
// swizzled byte offset of 16-byte chunk c of row r (rows of `rb` bytes; XOR of the low 3 chunk bits with r & 7)
DEVI int swz(int r, int c, int rb) { return r * rb + ((c ^ (r & 7)) << 4); }
DEVI float gate1(float o, float g) {
    // fused._sigmoid_f32 (libdevice.div_rn(1, 1 + libdevice.exp(-x))) rounded to bf16, times o, rounded to bf16
    float s = __bfloat162float(__float2bfloat16_rn(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g)))));
    return __fmul_rn(o, s);
}

// with gtab (65536 bf16 sigmoid values by bit pattern, computed by fused.sigmoid_table with the gate kernels' own
// _sigmoid_f32): the same value by lookup
DEVI float gatex(float o, __nv_bfloat16 g, const __nv_bfloat16* __restrict__ gtab) {
    if (gtab != nullptr) {
        const unsigned short bits = *reinterpret_cast<const unsigned short*>(&g);
        return __fmul_rn(o, __bfloat162float(gtab[bits]));
    }
    return gate1(o, __bfloat162float(g));
}

}  // namespace xa

using namespace xa;

// ------------------------------------------------------------------------------------------------------------------
// xa_fwd2*: the same per-element arithmetic as xa_fwd (see the header), re-tiled for throughput:
//   * NW warps (16: 512 threads at <= 128 registers; 8: 256 threads with up to 255): warp w = (head hh = w & 3,
//     part p = w >> 2 of NP = NW / 4); QK: 16 rows x the KW = 128 / NP keys KW p .. of the 128-key block (one or two
//     cutlass 32-key warp columns, so the lane's 8-value row sums are the reference's in-thread order); PV: 16 rows x
//     the CW = 256 / NP output columns CW p .. (O and the fp32 block buffer in registers);
//   * a 3-stage cp.async ring of 16 KB chunks (two chunks in flight while one is consumed, one barrier per chunk);
//   * 1-D grid, heaviest position tiles first: CTA i < T * 4 * B -> tile T - 1 - i / (4 B), KV head i % 4, slot
//     (i / 4) % B; CTAs past that zero-fill the packed rows nobody writes;
//   * FQ (xa_fwd2q*): the Q tile is built from the raw q_proj rows (qg, the first 256 of each head's 512) with
//     fused._norm_rope_rows' arithmetic (Triton, enable_fp_fusion off): per dim sq = x0^2 + x1^2 (x1 = dim + 128), per
//     lane i the dims 4i..4i+3 as ((s0 + s1) + s2) + s3, then the 32-lane shfl_down tree 16, 8, 4, 2, 1;
//     r = rsqrt.approx(eps + inv_d * s); n = bf16((x * r) * w); dims < 64: e = bf16(bf16(n * cos) + bf16(rh * sin)),
//     rh = -n[d + 32] (d < 32) / n[d - 32] (32 <= d < 64) - this replaces fused.q_norm_rope_rows.
// ------------------------------------------------------------------------------------------------------------------
namespace xb {
constexpr int SQ = 0;                    // [64 rows][512 B] Q
constexpr int SP = 32768;                // [64 rows][256 B] P of the current block
constexpr int SR = 49152;                // 3 x 16 KB ring: K chunk [128 keys][64 d] / V chunk [32 keys][256]
constexpr int NST = 3;
constexpr int SMX = SR + NST * 16384;    // [4 heads][4 32-key columns][16 rows] f32 (row max per warp column group)
constexpr int SSM = SMX + 1024;          // [4 heads][4 32-key columns][16 rows] f32 (row sums)
constexpr int SMEM = SSM + 1024;         // 100352 (xa.py KERNELS)
}  // namespace xb

DEVI float rsqrt_approx(float x) {   // libdevice rsqrt as Triton emits it (rsqrt.approx.f32, no ftz)
    float y;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
DEVI float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

template <bool FQ, int NW, bool LATE_GATE, bool BATCH_Q>
__device__ __forceinline__ void xa3_body(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k4,
                                         const __nv_bfloat16* __restrict__ v4, const __nv_bfloat16* __restrict__ qg,
                                         __nv_bfloat16* __restrict__ out, const int* __restrict__ start,
                                         const int* __restrict__ slen, const int* __restrict__ plen, int L, int B,
                                         int R, float scale, const __nv_bfloat16* __restrict__ gtab,
                                         const float* __restrict__ qw, const __nv_bfloat16* __restrict__ cosb,
                                         const __nv_bfloat16* __restrict__ sinb, float eps, float inv_d) {
    constexpr int NT_ = NW * 32;          // threads
    constexpr int NP = NW / 4;            // parts per head
    constexpr int KW = 128 / NP;          // keys per warp in QK (32 or 64)
    constexpr int CW = 256 / NP;          // output columns per warp in PV (64 or 128)
    constexpr int ST = KW / 8, OT = CW / 8;
    extern __shared__ __align__(128) unsigned char smem[];
    const int tid = threadIdx.x;
    const int NT = L >> 4;
    const int nmain = NT * NG * B;
    const int idx = blockIdx.x;
    if (idx >= nmain) {   // zero-fill packed rows [rused, R)
        int rused = 0;
        for (int i = 0; i < B; ++i) {
            int n = slen[i];
            if (n > 0) {
                int e = start[i] + n - (plen != nullptr ? plen[i] : 0);
                rused = e > rused ? e : rused;
            }
        }
        const int nct = gridDim.x - nmain, ct = idx - nmain;
        const int64_t c0 = (int64_t)rused * (NH * HD / 8), c1 = (int64_t)R * (NH * HD / 8);
        uint4 z = make_uint4(0, 0, 0, 0);
        for (int64_t c = c0 + (int64_t)ct * NT_ + tid; c < c1; c += (int64_t)nct * NT_)
            reinterpret_cast<uint4*>(out)[c] = z;
        return;
    }
    const int tr = idx / (NG * B);
    const int rem = idx - tr * (NG * B);
    const int g4 = rem & (NG - 1);
    const int b = rem >> 2;
    const int P0 = (NT - 1 - tr) * ROWS;
    const int n = slen[b];
    const int pl = plen != nullptr ? plen[b] : 0;
    if (P0 >= n || P0 + ROWS <= pl) return;
    const int s0 = start[b];
    const int w = tid >> 5, lane = tid & 31, gq = lane >> 2, qd = lane & 3;
    const int hh = w & 3, wp = w >> 2;
    const int jmax = P0 >> 7;
    const int jL = (L - 1) >> 7;
    const int kmax = min(n - 1, P0 + ROWS - 1);
    const int nkrd = ((kmax - 128 * jmax + 1) + 15) & ~15;   // keys of the diagonal block, rounded up to 16
    const int nvd = (nkrd + 31) >> 5;
    const int nch = 8 * jmax + 4 + nvd;
    const float LOG2E = 1.4426950408889634074f;
    const float NEG_INF = -__int_as_float(0x7f800000);

    unsigned char* sP = smem + xb::SP;
    float* sMax = reinterpret_cast<float*>(smem + xb::SMX);
    float* sSum = reinterpret_cast<float*>(smem + xb::SSM);
    const uint32_t uQ = smem_u32(smem + xb::SQ), uR = smem_u32(smem + xb::SR), uP = smem_u32(sP);
    const __nv_bfloat16* kb = k4 + (int64_t)(b * NG + g4) * L * HD;
    const __nv_bfloat16* vb = v4 + (int64_t)(b * NG + g4) * L * HD;

    if constexpr (!FQ) {
#pragma unroll
        for (int i = 0; i < 2048 / NT_; ++i) {
            int id = tid + NT_ * i;
            int row = id >> 5, c = id & 31;
            int head = g4 * REP + (row >> 4);
            int pos = P0 + (row & 15);
            bool ok = pos >= pl && pos < n;
            const __nv_bfloat16* src = ok ? q + ((int64_t)(s0 + pos - pl) * NH + head) * HD + c * 8 : q;
            cp_async16(uQ + swz(row, c, 512), src, ok);
        }
    }
    // issue cursor: block ij, chunk is (0..3 K, 4.. V), ring slot islot
    int ij = 0, is = 0, islot = 0;
    // per-thread cp.async coordinates: K chunk rows kr0 + (NT_ / 8) i (16-byte column kc), V chunk rows
    // vr0 + (NT_ / 32) i (column vcc); the swizzle term depends on row & 7 only, so later rows are linear offsets
    const int kr0 = tid >> 3, kc = tid & 7, vr0 = tid >> 5, vcc = tid & 31;
    const uint32_t kdst = swz(kr0, kc, 128), vdst = swz(vr0, vcc, 512);
    const int kgo = kr0 * HD + kc * 8, vgo = vr0 * HD + vcc * 8;
    auto issue = [&]() {
        const int nkr = (ij < jmax) ? 128 : nkrd;
        const uint32_t base = uR + islot * 16384;
        if (is < 4) {
            const __nv_bfloat16* src = kb + (128 * ij) * HD + is * 64 + kgo;
#pragma unroll
            for (int i = 0; i < 1024 / NT_; ++i)
                if (kr0 + (NT_ / 8) * i < nkr)
                    cp_async16(base + kdst + (NT_ / 8) * i * 128, src + (NT_ / 8) * i * HD, true);
        } else {
            const int vc = is - 4;
            const __nv_bfloat16* src = vb + (128 * ij + 32 * vc) * HD + vgo;
#pragma unroll
            for (int i = 0; i < 1024 / NT_; ++i)
                if (32 * vc + vr0 + (NT_ / 32) * i < nkr)
                    cp_async16(base + vdst + (NT_ / 32) * i * 512, src + (NT_ / 32) * i * HD, true);
        }
        if (++is == ((ij < jmax) ? 8 : 4 + nvd)) { ++ij; is = 0; }
        islot = islot == xb::NST - 1 ? 0 : islot + 1;
    };
    issue();
    cp_commit();
    if (nch > 1) issue();
    cp_commit();
    if constexpr (FQ) {
        // Q tile from the raw rows (K chunks 0 / 1 in flight meanwhile): warp w builds rows w, w + NW, ...
        const float4 w0 = *reinterpret_cast<const float4*>(qw + 4 * lane);
        const float4 w1 = *reinterpret_cast<const float4*>(qw + 128 + 4 * lane);
        const float wa[4] = {w0.x, w0.y, w0.z, w0.w}, wb[4] = {w1.x, w1.y, w1.z, w1.w};
        unsigned char* sQ = smem + xb::SQ;
        if constexpr (BATCH_Q) {
            // the warp's rows w, w + 16, w + 32, w + 48 = heads g4 * REP + 0..3 at the one position P0 + w: every
            // global load first (8 x 64-bit q_proj loads, 8 cos / sin values), then each row's arithmetic as below
            static_assert(NW == 16, "BATCH_Q: 16 warps, 4 rows per warp at one position");
            const int pos = P0 + w;
            const bool okq = pos >= pl && pos < n;   // warp-uniform, the same for the 4 rows
            uint2 ua[REP], ub[REP];
            __nv_bfloat16 cv[4], sv[4];
            if (okq) {
                const __nv_bfloat16* src0 = qg + ((int64_t)(s0 + pos - pl) * NH + g4 * REP) * 512;
#pragma unroll
                for (int i = 0; i < REP; ++i) {
                    ua[i] = *reinterpret_cast<const uint2*>(src0 + i * 512 + 4 * lane);
                    ub[i] = *reinterpret_cast<const uint2*>(src0 + i * 512 + 128 + 4 * lane);
                }
                if (lane < 16) {
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        cv[jj] = cosb[pos * 64 + 4 * lane + jj];
                        sv[jj] = sinb[pos * 64 + 4 * lane + jj];
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < REP; ++i) {
                const int row = w + NW * i;
                uint2 lo = make_uint2(0, 0), hi = make_uint2(0, 0);
                if (okq) {
                    const __nv_bfloat16* pa = reinterpret_cast<const __nv_bfloat16*>(&ua[i]);
                    const __nv_bfloat16* pb = reinterpret_cast<const __nv_bfloat16*>(&ub[i]);
                    float x0[4], x1[4], sq[4];
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        x0[jj] = __bfloat162float(pa[jj]);
                        x1[jj] = __bfloat162float(pb[jj]);
                        sq[jj] = __fadd_rn(__fmul_rn(x0[jj], x0[jj]), __fmul_rn(x1[jj], x1[jj]));
                    }
                    float t = __fadd_rn(__fadd_rn(__fadd_rn(sq[0], sq[1]), sq[2]), sq[3]);
                    t = __fadd_rn(t, __shfl_down_sync(0xffffffffu, t, 16));
                    t = __fadd_rn(t, __shfl_down_sync(0xffffffffu, t, 8));
                    t = __fadd_rn(t, __shfl_down_sync(0xffffffffu, t, 4));
                    t = __fadd_rn(t, __shfl_down_sync(0xffffffffu, t, 2));
                    t = __fadd_rn(t, __shfl_down_sync(0xffffffffu, t, 1));
                    const float ss = __shfl_sync(0xffffffffu, t, 0);
                    const float r = rsqrt_approx(__fadd_rn(eps, __fmul_rn(inv_d, ss)));
                    float n0[4], n1[4];
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        n0[jj] = bfr(__fmul_rn(__fmul_rn(x0[jj], r), wa[jj]));
                        n1[jj] = bfr(__fmul_rn(__fmul_rn(x1[jj], r), wb[jj]));
                    }
                    float e[4];
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const float np = __shfl_xor_sync(0xffffffffu, n0[jj], 8);
                        e[jj] = n0[jj];
                        if (lane < 16) {
                            const float c = __bfloat162float(cv[jj]);
                            const float sn = __bfloat162float(sv[jj]);
                            const float rh = lane < 8 ? -np : np;
                            e[jj] = bfr(__fadd_rn(bfr(__fmul_rn(n0[jj], c)), bfr(__fmul_rn(rh, sn))));
                        }
                    }
                    lo = make_uint2(pack_bf2(e[0], e[1]), pack_bf2(e[2], e[3]));
                    hi = make_uint2(pack_bf2(n1[0], n1[1]), pack_bf2(n1[2], n1[3]));
                }
                *reinterpret_cast<uint2*>(sQ + swz(row, lane >> 1, 512) + (lane & 1) * 8) = lo;
                *reinterpret_cast<uint2*>(sQ + swz(row, 16 + (lane >> 1), 512) + (lane & 1) * 8) = hi;
            }
        } else {
#pragma unroll 1
        for (int row = w; row < 64; row += NW) {
            const int head = g4 * REP + (row >> 4);
            const int pos = P0 + (row & 15);
            uint2 lo = make_uint2(0, 0), hi = make_uint2(0, 0);
            if (pos >= pl && pos < n) {   // warp-uniform
                const __nv_bfloat16* src = qg + ((int64_t)(s0 + pos - pl) * NH + head) * 512;
                const uint2 ua = *reinterpret_cast<const uint2*>(src + 4 * lane);
                const uint2 ub = *reinterpret_cast<const uint2*>(src + 128 + 4 * lane);
                const __nv_bfloat16* pa = reinterpret_cast<const __nv_bfloat16*>(&ua);
                const __nv_bfloat16* pb = reinterpret_cast<const __nv_bfloat16*>(&ub);
                float x0[4], x1[4], sq[4];
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    x0[jj] = __bfloat162float(pa[jj]);
                    x1[jj] = __bfloat162float(pb[jj]);
                    sq[jj] = __fadd_rn(__fmul_rn(x0[jj], x0[jj]), __fmul_rn(x1[jj], x1[jj]));
                }
                float t = __fadd_rn(__fadd_rn(__fadd_rn(sq[0], sq[1]), sq[2]), sq[3]);
                t = __fadd_rn(t, __shfl_down_sync(0xffffffffu, t, 16));
                t = __fadd_rn(t, __shfl_down_sync(0xffffffffu, t, 8));
                t = __fadd_rn(t, __shfl_down_sync(0xffffffffu, t, 4));
                t = __fadd_rn(t, __shfl_down_sync(0xffffffffu, t, 2));
                t = __fadd_rn(t, __shfl_down_sync(0xffffffffu, t, 1));
                const float ss = __shfl_sync(0xffffffffu, t, 0);
                const float r = rsqrt_approx(__fadd_rn(eps, __fmul_rn(inv_d, ss)));
                float n0[4], n1[4];
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    n0[jj] = bfr(__fmul_rn(__fmul_rn(x0[jj], r), wa[jj]));
                    n1[jj] = bfr(__fmul_rn(__fmul_rn(x1[jj], r), wb[jj]));
                }
                float e[4];
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const float np = __shfl_xor_sync(0xffffffffu, n0[jj], 8);
                    e[jj] = n0[jj];
                    if (lane < 16) {
                        const float c = __bfloat162float(cosb[pos * 64 + 4 * lane + jj]);
                        const float sn = __bfloat162float(sinb[pos * 64 + 4 * lane + jj]);
                        const float rh = lane < 8 ? -np : np;
                        e[jj] = bfr(__fadd_rn(bfr(__fmul_rn(n0[jj], c)), bfr(__fmul_rn(rh, sn))));
                    }
                }
                lo = make_uint2(pack_bf2(e[0], e[1]), pack_bf2(e[2], e[3]));
                hi = make_uint2(pack_bf2(n1[0], n1[1]), pack_bf2(n1[2], n1[3]));
            }
            *reinterpret_cast<uint2*>(sQ + swz(row, lane >> 1, 512) + (lane & 1) * 8) = lo;
            *reinterpret_cast<uint2*>(sQ + swz(row, 16 + (lane >> 1), 512) + (lane & 1) * 8) = hi;
        }
        }   // !BATCH_Q
    }

    float Bf[OT][4];
    float O[OT][4];
    float mpA = NEG_INF, mpB = NEG_INF;
    float sA = 0.f, sB = 0.f;
    float rA = 1.f, rB = 1.f;
    const int pA = P0 + gq, pB = P0 + gq + 8;

    int c = 0, cslot = 0;
    // one pipeline step: wait for chunk c, barrier, issue chunk c + 2 into the slot freed by chunk c - 1
    // LATE_GATE (gate_here: the first V-chunk step of the final block, sQ dead for every warp after the barrier): the
    // CTA's gate tile into sQ, in this step's commit group
    auto step = [&](bool gate_here) -> uint32_t {
        cp_wait<1>();
        __syncthreads();
        if (c + 2 < nch) issue();
        if constexpr (LATE_GATE) {
            if (gate_here) {
#pragma unroll
                for (int i = 0; i < 2048 / NT_; ++i) {
                    const int id = tid + NT_ * i;
                    const int row = id >> 5, cc = id & 31;
                    const int head = g4 * REP + (row >> 4);
                    const int pos = P0 + (row & 15);
                    const bool ok = pos >= pl && pos < n;
                    const __nv_bfloat16* src = ok ? qg + ((int64_t)(s0 + pos - pl) * NH + head) * 512 + 256 + cc * 8 : qg;
                    cp_async16(uQ + swz(row, cc, 512), src, ok);
                }
            }
        }
        cp_commit();
        const uint32_t slot = uR + cslot * 16384;
        ++c;
        cslot = cslot == xb::NST - 1 ? 0 : cslot + 1;
        return slot;
    };
#pragma unroll 1
    for (int j = 0; j <= jmax; ++j) {
        const int nkr = (j < jmax) ? 128 : nkrd;
        // ---------------- S = Q K^T for the warp's KW keys (d ascending in k8 steps, 4 chunks of 64 d)
        float S[ST][4];
#pragma unroll
        for (int t = 0; t < ST; ++t) S[t][0] = S[t][1] = S[t][2] = S[t][3] = 0.f;
#pragma unroll 1
        for (int sub = 0; sub < 4; ++sub) {
            const uint32_t slot = step(false);
#pragma unroll
            for (int s16 = 0; s16 < 4; ++s16) {
                uint32_t a0, a1, a2, a3;
                {
                    int row = hh * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    int cc = sub * 8 + s16 * 2 + (lane >> 4);
                    ldsm_x4(uQ + swz(row, cc, 512), a0, a1, a2, a3);
                }
#pragma unroll
                for (int tp = 0; tp < ST / 2; ++tp) {
                    const int k0 = KW * wp + 16 * tp;
                    if (k0 < nkr) {
                        uint32_t b0, b1, b2, b3;
                        int row = k0 + (lane & 7) + (lane >> 4) * 8;
                        int cc = s16 * 2 + ((lane >> 3) & 1);
                        ldsm_x4(slot + swz(row, cc, 128), b0, b1, b2, b3);
                        mma1688(S[2 * tp], a0, a1, b0);
                        mma1688(S[2 * tp], a2, a3, b1);
                        mma1688(S[2 * tp + 1], a0, a1, b2);
                        mma1688(S[2 * tp + 1], a2, a3, b3);
                    }
                }
            }
        }
        {
            // ---------------- softmax of block j (cutlass iterative_softmax order)
            const bool diag = (j == jmax);
            float mA = NEG_INF, mB = NEG_INF;
#pragma unroll
            for (int t = 0; t < ST; ++t) {
#pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    const int kin = KW * wp + 8 * t;   // first key of the tile within the block
                    const int key = 128 * j + kin + 2 * qd + cc;
                    float yA = __fmul_rn(__fmul_rn(scale, S[t][cc]), LOG2E);
                    float yB = __fmul_rn(__fmul_rn(scale, S[t][2 + cc]), LOG2E);
                    const bool act = kin < nkr;
                    const bool vA = act && (!diag || key <= pA);
                    const bool vB = act && (!diag || key <= pB);
                    yA = vA ? yA : NEG_INF;
                    yB = vB ? yB : NEG_INF;
                    S[t][cc] = yA;
                    S[t][2 + cc] = yB;
                    mA = fmaxf(mA, yA);
                    mB = fmaxf(mB, yB);
                }
            }
            mA = fmaxf(mA, __shfl_xor_sync(0xffffffffu, mA, 1));
            mA = fmaxf(mA, __shfl_xor_sync(0xffffffffu, mA, 2));
            mB = fmaxf(mB, __shfl_xor_sync(0xffffffffu, mB, 1));
            mB = fmaxf(mB, __shfl_xor_sync(0xffffffffu, mB, 2));
            if (qd == 0) {   // the same max for each of the warp's 32-key columns (max is order-free)
#pragma unroll
                for (int h2 = 0; h2 < KW / 32; ++h2) {
                    sMax[(hh * 4 + (KW / 32) * wp + h2) * 16 + gq] = mA;
                    sMax[(hh * 4 + (KW / 32) * wp + h2) * 16 + gq + 8] = mB;
                }
            }
            __syncthreads();
            const float* mx = sMax + hh * 64;
            const float miA = fmaxf(mpA, fmaxf(fmaxf(mx[gq], mx[16 + gq]), fmaxf(mx[32 + gq], mx[48 + gq])));
            const float miB = fmaxf(mpB, fmaxf(fmaxf(mx[gq + 8], mx[24 + gq]), fmaxf(mx[40 + gq], mx[56 + gq])));
            if (mpA < miA) { rA = exp2f(__fsub_rn(mpA, miA)); sA = __fmul_rn(sA, rA); } else { rA = 1.f; }
            if (mpB < miB) { rB = exp2f(__fsub_rn(mpB, miB)); sB = __fmul_rn(sB, rB); } else { rB = 1.f; }
            float tA[KW / 32], tB[KW / 32];
#pragma unroll
            for (int h2 = 0; h2 < KW / 32; ++h2) tA[h2] = tB[h2] = 0.f;
#pragma unroll
            for (int t = 0; t < ST; ++t) {
                const int kin = KW * wp + 8 * t;
                if (kin < nkr) {
                    float e0 = exp2f(__fsub_rn(S[t][0], miA));
                    float e1 = exp2f(__fsub_rn(S[t][1], miA));
                    float e2 = exp2f(__fsub_rn(S[t][2], miB));
                    float e3 = exp2f(__fsub_rn(S[t][3], miB));
                    tA[t >> 2] = __fadd_rn(__fadd_rn(tA[t >> 2], e0), e1);
                    tB[t >> 2] = __fadd_rn(__fadd_rn(tB[t >> 2], e2), e3);
                    const int col = kin + 2 * qd;
                    *reinterpret_cast<uint32_t*>(sP + swz(hh * 16 + gq, col >> 3, 256) + (col & 7) * 2) =
                        pack_bf2(e0, e1);
                    *reinterpret_cast<uint32_t*>(sP + swz(hh * 16 + gq + 8, col >> 3, 256) + (col & 7) * 2) =
                        pack_bf2(e2, e3);
                }
            }
#pragma unroll
            for (int h2 = 0; h2 < KW / 32; ++h2) {
                float a = tA[h2], cB = tB[h2];
                a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, 1));
                a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, 2));
                cB = __fadd_rn(cB, __shfl_xor_sync(0xffffffffu, cB, 1));
                cB = __fadd_rn(cB, __shfl_xor_sync(0xffffffffu, cB, 2));
                if (qd == 0) {
                    sSum[(hh * 4 + (KW / 32) * wp + h2) * 16 + gq] = a;
                    sSum[(hh * 4 + (KW / 32) * wp + h2) * 16 + gq + 8] = cB;
                }
            }
            mpA = miA;
            mpB = miB;
            // P and the warp-column sums are read after the next chunk's barrier
        }
        // ---------------- O = P V[:, CW p ..] over the block's keys (ascending in k8 steps, chunks of 32 keys)
#pragma unroll
        for (int t = 0; t < OT; ++t) O[t][0] = O[t][1] = O[t][2] = O[t][3] = 0.f;
        const int nvc = (j < jmax) ? 4 : nvd;
#pragma unroll 1
        for (int vc = 0; vc < nvc; ++vc) {
            const uint32_t slot = step(LATE_GATE && j == jmax && vc == 0);
            if (vc == 0) {
                const float* sm = sSum + hh * 64;
#pragma unroll
                for (int wc = 0; wc < 4; ++wc) {
                    sA = __fadd_rn(sA, sm[wc * 16 + gq]);
                    sB = __fadd_rn(sB, sm[wc * 16 + gq + 8]);
                }
            }
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                if (32 * vc + 16 * u < nkr) {
                    uint32_t a0, a1, a2, a3;
                    {
                        int row = hh * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                        int cc = 4 * vc + 2 * u + (lane >> 4);
                        ldsm_x4(uP + swz(row, cc, 256), a0, a1, a2, a3);
                    }
#pragma unroll
                    for (int nn = 0; nn < OT; nn += 2) {
                        uint32_t b0, b1, b2, b3;
                        int row = 16 * u + (lane & 7) + ((lane >> 3) & 1) * 8;
                        int cc = (CW / 8) * wp + nn + (lane >> 4);
                        ldsm_x4_t(slot + swz(row, cc, 512), b0, b1, b2, b3);
                        mma1688(O[nn], a0, a1, b0);
                        mma1688(O[nn], a2, a3, b1);
                        mma1688(O[nn + 1], a0, a1, b2);
                        mma1688(O[nn + 1], a2, a3, b3);
                    }
                }
            }
        }
        if (j < jmax) {   // epilogue of a non-final block: buf = acc (first) or fma(r, buf, acc) (middle)
#pragma unroll
            for (int t = 0; t < OT; ++t) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float r = e < 2 ? rA : rB;
                    Bf[t][e] = (j == 0) ? O[t][e] : __fmaf_rn(r, Bf[t][e], O[t][e]);
                }
            }
        }
    }
    // ---------------- epilogue of the final block jmax (MemoryEfficientAttentionNormalize replay) + gate
    if constexpr (LATE_GATE) {   // the gate tile is in sQ
        cp_wait<0>();
        __syncthreads();
    }
    {
        const float dA = sA == 0.f ? 1.f : sA, dB = sB == 0.f ? 1.f : sB;
        const float alA = __fdiv_rn(1.0f, dA), alB = __fdiv_rn(1.0f, dB);
        const bool last_is_ref_last = (jmax == jL);
        // the row / column bases are pinned here (empty asm), so their address arithmetic is not hoisted above the
        // block loop where it would hold registers through the whole walk
        int rbA = s0 + pA - pl, rbB = s0 + pB - pl, cb = (g4 * REP + hh) * HD + CW * wp + 2 * qd;
        asm volatile("" : "+r"(rbA), "+r"(rbB), "+r"(cb));
        const bool okA = pA >= pl && pA < n, okB = pB >= pl && pB < n;
        const __nv_bfloat16* gA = qg + (int64_t)rbA * (NH * 512) + cb + (cb & ~(HD - 1)) + 256;
        const __nv_bfloat16* gB = qg + (int64_t)rbB * (NH * 512) + cb + (cb & ~(HD - 1)) + 256;
        __nv_bfloat16* oA = out + (int64_t)rbA * (NH * HD) + cb;
        __nv_bfloat16* oB = out + (int64_t)rbB * (NH * HD) + cb;
#pragma unroll
        for (int t = 0; t < OT; ++t) {
            float D[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float r = e < 2 ? rA : rB;
                const float al = e < 2 ? alA : alB;
                const float acc = O[t][e];
                if (last_is_ref_last) {
                    D[e] = (jmax == 0) ? __fmul_rn(al, acc)
                                       : __fmaf_rn(al, acc, __fmul_rn(__fmul_rn(al, r), Bf[t][e]));
                } else {
                    float b2 = (jmax == 0) ? acc : __fmaf_rn(r, Bf[t][e], acc);
                    if (jmax + 1 < jL) b2 = __fadd_rn(b2, 0.0f);   // middle masked blocks: fma(1, buf, +0) (idempotent)
                    D[e] = __fmaf_rn(al, 0.0f, __fmul_rn(al, b2));
                }
            }
            if (okA) {
                const __nv_bfloat162 gg = LATE_GATE ? *reinterpret_cast<const __nv_bfloat162*>(
                                                          smem + xb::SQ + swz(hh * 16 + gq, CW / 8 * wp + t, 512) + 4 * qd)
                                                    : *reinterpret_cast<const __nv_bfloat162*>(gA + 8 * t);
                float o0 = __bfloat162float(__float2bfloat16_rn(D[0]));
                float o1 = __bfloat162float(__float2bfloat16_rn(D[1]));
                *reinterpret_cast<uint32_t*>(oA + 8 * t) = pack_bf2(gatex(o0, gg.x, gtab), gatex(o1, gg.y, gtab));
            }
            if (okB) {
                const __nv_bfloat162 gg = LATE_GATE ? *reinterpret_cast<const __nv_bfloat162*>(
                                                          smem + xb::SQ + swz(hh * 16 + gq + 8, CW / 8 * wp + t, 512) + 4 * qd)
                                                    : *reinterpret_cast<const __nv_bfloat162*>(gB + 8 * t);
                float o0 = __bfloat162float(__float2bfloat16_rn(D[2]));
                float o1 = __bfloat162float(__float2bfloat16_rn(D[3]));
                *reinterpret_cast<uint32_t*>(oB + 8 * t) = pack_bf2(gatex(o0, gg.x, gtab), gatex(o1, gg.y, gtab));
            }
        }
    }
}

#define XA2_PARAMS                                                                                                   \
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k4, const __nv_bfloat16 *__restrict__ v4, \
        const __nv_bfloat16 *__restrict__ qg, __nv_bfloat16 *__restrict__ out, const int *__restrict__ start,       \
        const int *__restrict__ slen, const int *__restrict__ plen, int L, int B, int R, float scale,               \
        const __nv_bfloat16 *__restrict__ gtab, const float *__restrict__ qw, const __nv_bfloat16 *__restrict__ cosb, \
        const __nv_bfloat16 *__restrict__ sinb, float eps, float inv_d
#define XA2_ARGS q, k4, v4, qg, out, start, slen, plen, L, B, R, scale, gtab, qw, cosb, sinb, eps, inv_d

// xa_fwd2q's signature (q unused), launched exactly as xa_fwd2q (xa.py): grid (L / 16) * 4 * B + 4, 512 threads,
// 100352 B dynamic shared memory
extern "C" __global__ void __launch_bounds__(512, 1) xa_fwd3q(XA2_PARAMS) { xa3_body<true, 16, true, true>(XA2_ARGS); }
extern "C" __global__ void __launch_bounds__(512, 1) xa_fwd3qe(XA2_PARAMS) { xa3_body<true, 16, true, false>(XA2_ARGS); }
