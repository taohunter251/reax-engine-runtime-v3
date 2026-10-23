"""FF (rgh): the gated-delta-rule forward with recompute_w_u + chunk_fwd_h + chunk_fwd_o fused into one Triton kernel.

fla 0.5.2's forward (use_qk_l2norm_in_kernel, no initial / final state, chunk 64) runs l2norm(q), l2norm(k),
chunk_local_cumsum(g), the fused kkt + solve_tril kernel (A, bf16), then three kernels that hand bf16 intermediates to
each other through memory: recompute_w_u (w = A (k beta 2^g), u = A (v beta)), fwd_h (per chunk: store h, v_new =
u - w h, h = h 2^g_last + k^T (v_new 2^(g_last - g))) and fwd_o (o = (q h) 2^g scale + (masked gated q k^T) v_new
scale). Here the first four steps are fla's own kernels (same pins); the fused kernel walks the chunks of one
(V-block, value head) and evaluates the same expressions in the same form (the same tl.dot / dot-accumulate
patterns, exp2 from fla.ops.utils.op, the bf16 rounds where fla stores w, u, h and v_new), so every output element
gets the same arithmetic; w, u, h and v_new never leave registers. Bit equality with fla is checked at start-up
(check()) and the engine falls back to fla otherwise.

FF-PEEL (kai, v4; S1ENG_FF_PEEL=1, default off: with it off gdn_forward_ps launches _ps_seq_kernel exactly as job 38
does, tests/test_ff_levers.py). For job 38's served FF path, ps mode (S1ENG_FF_MODE=ps), gdn_forward_ps(peel=True)
runs _ps_seq_x_kernel, which is _ps_seq_kernel plus constexpr branches (PEEL False = the job-38 kernel statement for
statement):
  chunk 0: h starts at +0 (no initial state), so its h-dependent dots - the two w h dots (v_new = u - w h) and the two
           q h dots of o - are skipped, with the w and q loads that only feed them: under IEEE each of those dots is
           exactly +0 (every product is +-0 and the accumulator starts at +0; spike/ff_bench.py probes the GPU's mma),
           so v_new = u - (+0) and o = (+0) 2^g scale + (A_q v_new) scale are kept as written with the +0 tile.
  last:    the h update after the last chunk is never read (no final state) and is skipped, with the k^T loads only it
           reads: the loop runs chunks 1 .. NT - 2 with the job-38 body, the last chunk is that body without them (no
           runtime guard inside the loop: in v3 one made ptxas spill ~180 more instructions per chunk on sm_120).
The pass P kernel (_ps_pre_kernel) is unchanged (it still stores chunk 0's w, which the peeled kernel does not read).
The one-kernel fused path (gdn_forward, job 38's fallback when ps mode fails its start-up check) is unchanged.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from fla.ops.utils.op import exp2


@triton.jit(do_not_specialize=['T'])
def _fused_wuho_kernel(q, k, v, beta, A, g, o, scale, slen, T, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                       V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, HAS_SLEN: tl.constexpr):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    if HAS_SLEN:
        # an unused slot of a packed batch (length 0): its outputs are never read (the gated norm gathers real rows)
        if tl.load(slen + i_n) == 0:
            return
    bos = i_n * T
    NT = tl.cdiv(T, BT)

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    b_h2 = tl.zeros([64, BV], dtype=tl.float32)

    qk = (bos * H + i_h // (HV // H)).to(tl.int64) * K
    vo = (bos * HV + i_h).to(tl.int64) * V
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_A = tl.arange(0, BT)

    for i_t in range(NT):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        # ---------------------------------------------------------------- recompute_w_u (this chunk)
        p_b = beta + bos * HV + i_h + o_t * HV
        b_b = tl.load(p_b, mask=m_t, other=0.0)
        p_A = A + (bos * HV + i_h) * BT + o_t[:, None] * (HV * BT) + o_A[None, :]
        # kkt_solve stores the lower block triangle (16 x 16 blocks, the diagonal blocks whole); fla zero-fills A
        # first, here the blocks above the diagonal are masked to zero instead (no fill launch)
        m_blk = (o_A[:, None] // 16) >= (o_A[None, :] // 16)
        b_A = tl.load(p_A, mask=m_t[:, None] & (o_A[None, :] < BT) & m_blk, other=0.0)
        p_v = v + vo + o_t[:, None] * (HV * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        b_u = b_u.to(v.dtype.element_ty)                       # u is stored in v's dtype
        p_g = g + (bos * HV + i_h) + o_t * HV
        b_gw = exp2(tl.load(p_g, mask=m_t, other=0.0))
        p_k = k + qk + o_t[:, None] * (H * K) + o_k1[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        b_kb = b_k * b_b[:, None]
        b_kb *= b_gw[:, None]
        b_w1 = tl.dot(b_A, b_kb.to(b_k.dtype)).to(k.dtype.element_ty)   # w is stored in k's dtype
        p_k = k + qk + o_t[:, None] * (H * K) + o_k2[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & m_k2[None, :], other=0.0)
        b_kb = b_k * b_b[:, None]
        b_kb *= b_gw[:, None]
        b_w2 = tl.dot(b_A, b_kb.to(b_k.dtype)).to(k.dtype.element_ty)

        # ---------------------------------------------------------------- fwd_h: h as stored, v_new
        b_hs1 = b_h1.to(k.dtype.element_ty)                     # the bf16 copy fwd_h stores (and fwd_o loads)
        b_hs2 = b_h2.to(k.dtype.element_ty)
        b_vn = tl.dot(b_w1, b_hs1.to(b_w1.dtype))
        b_vn += tl.dot(b_w2, b_hs2.to(b_w1.dtype))
        b_vn = b_u - b_vn
        b_vns = b_vn.to(v.dtype.element_ty)                     # v_new as stored

        # ---------------------------------------------------------------- fwd_o (this chunk, this V block)
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_Aq = tl.zeros([BT, BT], dtype=tl.float32)
        p_q = q + qk + o_t[:, None] * (H * K) + o_k1[None, :]
        p_kt1 = k + qk + o_k1[:, None] + o_t[None, :] * (H * K)
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        b_kt1 = tl.load(p_kt1, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        b_o += tl.dot(b_q, b_hs1)
        b_Aq += tl.dot(b_q, b_kt1)
        p_q = q + qk + o_t[:, None] * (H * K) + o_k2[None, :]
        p_kt2 = k + qk + o_k2[:, None] + o_t[None, :] * (H * K)
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k2[None, :], other=0.0)
        b_kt2 = tl.load(p_kt2, mask=m_k2[:, None] & m_t[None, :], other=0.0)
        b_o += tl.dot(b_q, b_hs2)
        b_Aq += tl.dot(b_q, b_kt2)
        b_go = tl.load(p_g, mask=m_t, other=0.0)
        b_o = b_o * exp2(b_go)[:, None]
        b_Aq = b_Aq * exp2(b_go[:, None] - b_go[None, :])
        m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
        b_Aq = tl.where(m_A, b_Aq, 0)
        b_o = b_o * scale + tl.dot(b_Aq.to(b_vns.dtype), b_vns) * scale
        p_o = o + vo + o_t[:, None] * (HV * V) + o_v[None, :]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])

        # ---------------------------------------------------------------- fwd_h: state update
        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
        b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
        b_vd = b_vn * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
        b_g_last = exp2(b_g_last)
        b_h1 *= b_g_last
        b_h2 *= b_g_last
        b_vd = b_vd.to(k.dtype.element_ty)
        b_h1 += tl.dot(b_kt1, b_vd)
        b_h2 += tl.dot(b_kt2, b_vd)


def gdn_forward(q, k, v, g, beta, BV: int = 32, num_warps: int = 4, num_stages: int = 2, slen=None):
    """o of chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=True) for [B, T, H, K] q / k, [B, T, HV, V] v, [B, T, HV] g / beta, K = 128, T % 64 == 0."""
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel
    from fla.ops.utils.constant import RCP_LN2
    from fla.ops.utils.cumsum import chunk_local_cumsum

    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    scale = k.shape[-1] ** -0.5
    if k.data_ptr() == q.data_ptr() + q.numel() * q.element_size() and k.shape == q.shape:
        # q and k are the halves of one buffer (conv_silu_split_packed): one l2norm launch (row-wise, same config)
        nq = q.shape[0]
        qk = torch.as_strided(q, (2 * nq,) + tuple(q.shape[1:]), q.stride())
        qk, _ = l2norm_fwd(qk)
        q, k = qk[:nq], qk[nq:]
    else:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
    g = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2)
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    BT = 64
    NT = triton.cdiv(T, BT)
    A = torch.empty(B, T, HV, BT, device=k.device, dtype=k.dtype)   # upper blocks masked in the fused kernel
    chunk_gated_delta_rule_fwd_kkt_solve_kernel[(NT, B * HV)](k=k, g=g, beta=beta, A=A, cu_seqlens=None,
                                                              chunk_indices=None, T=T, H=H, HV=HV, K=K, BT=BT, BC=16)
    o = torch.empty_like(v)
    _fused_wuho_kernel[(triton.cdiv(V, BV), B * HV)](q, k, v, beta, A, g, o, scale, slen if slen is not None else o, T,
                                                      H=H, HV=HV, K=K, V=V, BT=BT, BV=BV, HAS_SLEN=slen is not None,
                                                      num_warps=num_warps, num_stages=num_stages)
    return o


@triton.jit(do_not_specialize=['T'])
def _ps_pre_kernel(q, k, v, beta, A, g, w, u, Aq, slen, T, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                   V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, HAS_SLEN: tl.constexpr):
    """FF ps mode, pass P (parallel over chunks x value heads): the h-independent part of the fused kernel - u (per
    BV-wide V block, as the fused kernel tiles it), w and the masked gated q k^T of fwd_o - in the fused kernel's exact
    expressions, stored at the points where the fused kernel rounds them to bf16 (u, w as fla stores them; the q k^T
    block as it enters the dot with v_new)."""
    i_t, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    if HAS_SLEN:
        if tl.load(slen + i_n) == 0:
            return
    bos = i_n * T
    qk = (bos * H + i_h // (HV // H)).to(tl.int64) * K
    vo = (bos * HV + i_h).to(tl.int64) * V
    wo = (bos * HV + i_h).to(tl.int64) * K
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_A = tl.arange(0, BT)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p_b = beta + bos * HV + i_h + o_t * HV
    b_b = tl.load(p_b, mask=m_t, other=0.0)
    p_A = A + (bos * HV + i_h) * BT + o_t[:, None] * (HV * BT) + o_A[None, :]
    m_blk = (o_A[:, None] // 16) >= (o_A[None, :] // 16)
    b_A = tl.load(p_A, mask=m_t[:, None] & (o_A[None, :] < BT) & m_blk, other=0.0)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        p_v = v + vo + o_t[:, None] * (HV * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        p_u = u + vo + o_t[:, None] * (HV * V) + o_v[None, :]
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])
    p_g = g + (bos * HV + i_h) + o_t * HV
    b_gw = exp2(tl.load(p_g, mask=m_t, other=0.0))
    p_k = k + qk + o_t[:, None] * (H * K) + o_k1[None, :]
    b_k = tl.load(p_k, mask=m_t[:, None] & m_k1[None, :], other=0.0)
    b_kb = b_k * b_b[:, None]
    b_kb *= b_gw[:, None]
    b_w1 = tl.dot(b_A, b_kb.to(b_k.dtype)).to(k.dtype.element_ty)
    p_w = w + wo + o_t[:, None] * (HV * K) + o_k1[None, :]
    tl.store(p_w, b_w1, mask=m_t[:, None] & m_k1[None, :])
    p_k = k + qk + o_t[:, None] * (H * K) + o_k2[None, :]
    b_k = tl.load(p_k, mask=m_t[:, None] & m_k2[None, :], other=0.0)
    b_kb = b_k * b_b[:, None]
    b_kb *= b_gw[:, None]
    b_w2 = tl.dot(b_A, b_kb.to(b_k.dtype)).to(k.dtype.element_ty)
    p_w = w + wo + o_t[:, None] * (HV * K) + o_k2[None, :]
    tl.store(p_w, b_w2, mask=m_t[:, None] & m_k2[None, :])
    b_Aq = tl.zeros([BT, BT], dtype=tl.float32)
    p_q = q + qk + o_t[:, None] * (H * K) + o_k1[None, :]
    p_kt1 = k + qk + o_k1[:, None] + o_t[None, :] * (H * K)
    b_q = tl.load(p_q, mask=m_t[:, None] & m_k1[None, :], other=0.0)
    b_kt1 = tl.load(p_kt1, mask=m_k1[:, None] & m_t[None, :], other=0.0)
    b_Aq += tl.dot(b_q, b_kt1)
    p_q = q + qk + o_t[:, None] * (H * K) + o_k2[None, :]
    p_kt2 = k + qk + o_k2[:, None] + o_t[None, :] * (H * K)
    b_q = tl.load(p_q, mask=m_t[:, None] & m_k2[None, :], other=0.0)
    b_kt2 = tl.load(p_kt2, mask=m_k2[:, None] & m_t[None, :], other=0.0)
    b_Aq += tl.dot(b_q, b_kt2)
    b_go = tl.load(p_g, mask=m_t, other=0.0)
    b_Aq = b_Aq * exp2(b_go[:, None] - b_go[None, :])
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_Aq = tl.where(m_A, b_Aq, 0)
    p_Aq = Aq + (bos * HV + i_h) * BT + o_t[:, None] * (HV * BT) + o_A[None, :]
    tl.store(p_Aq, b_Aq.to(v.dtype.element_ty), mask=m_t[:, None] & (o_A[None, :] < BT))


@triton.jit(do_not_specialize=['T'])
def _ps_seq_kernel(q, k, w, u, Aq, g, o, scale, slen, T, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                   V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, HAS_SLEN: tl.constexpr):
    """FF ps mode, pass S (per V block and value head, over the chunks): the h-dependent chain of the fused kernel
    (v_new = u - w h, o, the state update) on P's stored u / w / q k^T, same expressions as the fused kernel."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    if HAS_SLEN:
        if tl.load(slen + i_n) == 0:
            return
    bos = i_n * T
    NT = tl.cdiv(T, BT)
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    qk = (bos * H + i_h // (HV // H)).to(tl.int64) * K
    vo = (bos * HV + i_h).to(tl.int64) * V
    wo = (bos * HV + i_h).to(tl.int64) * K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_A = tl.arange(0, BT)
    for i_t in range(NT):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        p_u = u + vo + o_t[:, None] * (HV * V) + o_v[None, :]
        b_u = tl.load(p_u, mask=m_t[:, None] & m_v[None, :], other=0.0)
        p_w = w + wo + o_t[:, None] * (HV * K) + o_k1[None, :]
        b_w1 = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        p_w = w + wo + o_t[:, None] * (HV * K) + o_k2[None, :]
        b_w2 = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
        p_g = g + (bos * HV + i_h) + o_t * HV
        b_hs1 = b_h1.to(k.dtype.element_ty)
        b_hs2 = b_h2.to(k.dtype.element_ty)
        b_vn = tl.dot(b_w1, b_hs1.to(b_w1.dtype))
        b_vn += tl.dot(b_w2, b_hs2.to(b_w1.dtype))
        b_vn = b_u - b_vn
        b_vns = b_vn.to(u.dtype.element_ty)
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        p_q = q + qk + o_t[:, None] * (H * K) + o_k1[None, :]
        p_kt1 = k + qk + o_k1[:, None] + o_t[None, :] * (H * K)
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        b_kt1 = tl.load(p_kt1, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        b_o += tl.dot(b_q, b_hs1)
        p_q = q + qk + o_t[:, None] * (H * K) + o_k2[None, :]
        p_kt2 = k + qk + o_k2[:, None] + o_t[None, :] * (H * K)
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k2[None, :], other=0.0)
        b_kt2 = tl.load(p_kt2, mask=m_k2[:, None] & m_t[None, :], other=0.0)
        b_o += tl.dot(b_q, b_hs2)
        b_go = tl.load(p_g, mask=m_t, other=0.0)
        b_o = b_o * exp2(b_go)[:, None]
        p_Aq = Aq + (bos * HV + i_h) * BT + o_t[:, None] * (HV * BT) + o_A[None, :]
        b_Aq = tl.load(p_Aq, mask=m_t[:, None] & (o_A[None, :] < BT), other=0.0)
        b_o = b_o * scale + tl.dot(b_Aq, b_vns) * scale
        p_o = o + vo + o_t[:, None] * (HV * V) + o_v[None, :]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])
        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
        b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
        b_vd = b_vn * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
        b_g_last = exp2(b_g_last)
        b_h1 *= b_g_last
        b_h2 *= b_g_last
        b_vd = b_vd.to(k.dtype.element_ty)
        b_h1 += tl.dot(b_kt1, b_vd)
        b_h2 += tl.dot(b_kt2, b_vd)


@triton.jit(do_not_specialize=['T'])
def _ps_seq_x_kernel(q, k, w, u, Aq, g, o, scale, slen, T, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                     V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, HAS_SLEN: tl.constexpr, PEEL: tl.constexpr):
    """_ps_seq_kernel with FF-PEEL (PEEL, module docstring): chunk 0 without its h-dependent dots, chunks 1 .. NT - 2
    as the job-38 body, the last chunk without the dead h update."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    if HAS_SLEN:
        if tl.load(slen + i_n) == 0:
            return
    bos = i_n * T
    NT = tl.cdiv(T, BT)
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    qk = (bos * H + i_h // (HV // H)).to(tl.int64) * K
    vo = (bos * HV + i_h).to(tl.int64) * V
    wo = (bos * HV + i_h).to(tl.int64) * K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_A = tl.arange(0, BT)
    if PEEL:
        # ================================================================ chunk 0 (h = +0): the statements of the
        # loop body below that do not involve h; the h-dependent ones are replaced by the +0 they evaluate to
        i_t = 0
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        p_u = u + vo + o_t[:, None] * (HV * V) + o_v[None, :]
        b_u = tl.load(p_u, mask=m_t[:, None] & m_v[None, :], other=0.0)
        p_g = g + (bos * HV + i_h) + o_t * HV
        # w is read only by v_new = u - w h: with h = +0, w h = +0 (no w loads, no w h dots)
        b_vn = b_u - tl.zeros([BT, BV], dtype=tl.float32)
        b_vns = b_vn.to(u.dtype.element_ty)
        b_o = tl.zeros([BT, BV], dtype=tl.float32)               # + q h = +0 (no q loads, no q h dots)
        b_go = tl.load(p_g, mask=m_t, other=0.0)
        b_o = b_o * exp2(b_go)[:, None]
        p_Aq = Aq + (bos * HV + i_h) * BT + o_t[:, None] * (HV * BT) + o_A[None, :]
        b_Aq = tl.load(p_Aq, mask=m_t[:, None] & (o_A[None, :] < BT), other=0.0)
        b_o = b_o * scale + tl.dot(b_Aq, b_vns) * scale
        p_o = o + vo + o_t[:, None] * (HV * V) + o_v[None, :]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])
        if NT > 1:
            # the h update (with the k^T tiles only it reads), as the loop body below
            p_kt1 = k + qk + o_k1[:, None] + o_t[None, :] * (H * K)
            b_kt1 = tl.load(p_kt1, mask=m_k1[:, None] & m_t[None, :], other=0.0)
            p_kt2 = k + qk + o_k2[:, None] + o_t[None, :] * (H * K)
            b_kt2 = tl.load(p_kt2, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            last_idx = min((i_t + 1) * BT, T) - 1
            b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
            b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
            b_vd = b_vn * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
            b_g_last = exp2(b_g_last)
            b_h1 *= b_g_last
            b_h2 *= b_g_last
            b_vd = b_vd.to(k.dtype.element_ty)
            b_h1 += tl.dot(b_kt1, b_vd)
            b_h2 += tl.dot(b_kt2, b_vd)
    # PEEL: chunks 1 .. NT - 2 here, the last chunk below without the h update
    for i_t in range(1 if PEEL else 0, NT - 1 if PEEL else NT):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        p_u = u + vo + o_t[:, None] * (HV * V) + o_v[None, :]
        b_u = tl.load(p_u, mask=m_t[:, None] & m_v[None, :], other=0.0)
        p_w = w + wo + o_t[:, None] * (HV * K) + o_k1[None, :]
        b_w1 = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        p_w = w + wo + o_t[:, None] * (HV * K) + o_k2[None, :]
        b_w2 = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
        p_g = g + (bos * HV + i_h) + o_t * HV
        b_hs1 = b_h1.to(k.dtype.element_ty)
        b_hs2 = b_h2.to(k.dtype.element_ty)
        b_vn = tl.dot(b_w1, b_hs1.to(b_w1.dtype))
        b_vn += tl.dot(b_w2, b_hs2.to(b_w1.dtype))
        b_vn = b_u - b_vn
        b_vns = b_vn.to(u.dtype.element_ty)
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        p_q = q + qk + o_t[:, None] * (H * K) + o_k1[None, :]
        p_kt1 = k + qk + o_k1[:, None] + o_t[None, :] * (H * K)
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        b_kt1 = tl.load(p_kt1, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        b_o += tl.dot(b_q, b_hs1)
        p_q = q + qk + o_t[:, None] * (H * K) + o_k2[None, :]
        p_kt2 = k + qk + o_k2[:, None] + o_t[None, :] * (H * K)
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k2[None, :], other=0.0)
        b_kt2 = tl.load(p_kt2, mask=m_k2[:, None] & m_t[None, :], other=0.0)
        b_o += tl.dot(b_q, b_hs2)
        b_go = tl.load(p_g, mask=m_t, other=0.0)
        b_o = b_o * exp2(b_go)[:, None]
        p_Aq = Aq + (bos * HV + i_h) * BT + o_t[:, None] * (HV * BT) + o_A[None, :]
        b_Aq = tl.load(p_Aq, mask=m_t[:, None] & (o_A[None, :] < BT), other=0.0)
        b_o = b_o * scale + tl.dot(b_Aq, b_vns) * scale
        p_o = o + vo + o_t[:, None] * (HV * V) + o_v[None, :]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])
        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
        b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
        b_vd = b_vn * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
        b_g_last = exp2(b_g_last)
        b_h1 *= b_g_last
        b_h2 *= b_g_last
        b_vd = b_vd.to(k.dtype.element_ty)
        b_h1 += tl.dot(b_kt1, b_vd)
        b_h2 += tl.dot(b_kt2, b_vd)
    if PEEL:
        # ================================================================ the last chunk (when it is not chunk 0):
        # the loop body without the h update and without the k^T loads only the update reads (h is never read
        # again: no final state)
        if NT > 1:
            i_t = NT - 1
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            p_u = u + vo + o_t[:, None] * (HV * V) + o_v[None, :]
            b_u = tl.load(p_u, mask=m_t[:, None] & m_v[None, :], other=0.0)
            p_w = w + wo + o_t[:, None] * (HV * K) + o_k1[None, :]
            b_w1 = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
            p_w = w + wo + o_t[:, None] * (HV * K) + o_k2[None, :]
            b_w2 = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            p_g = g + (bos * HV + i_h) + o_t * HV
            b_hs1 = b_h1.to(k.dtype.element_ty)
            b_hs2 = b_h2.to(k.dtype.element_ty)
            b_vn = tl.dot(b_w1, b_hs1.to(b_w1.dtype))
            b_vn += tl.dot(b_w2, b_hs2.to(b_w1.dtype))
            b_vn = b_u - b_vn
            b_vns = b_vn.to(u.dtype.element_ty)
            b_o = tl.zeros([BT, BV], dtype=tl.float32)
            p_q = q + qk + o_t[:, None] * (H * K) + o_k1[None, :]
            b_q = tl.load(p_q, mask=m_t[:, None] & m_k1[None, :], other=0.0)
            b_o += tl.dot(b_q, b_hs1)
            p_q = q + qk + o_t[:, None] * (H * K) + o_k2[None, :]
            b_q = tl.load(p_q, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            b_o += tl.dot(b_q, b_hs2)
            b_go = tl.load(p_g, mask=m_t, other=0.0)
            b_o = b_o * exp2(b_go)[:, None]
            p_Aq = Aq + (bos * HV + i_h) * BT + o_t[:, None] * (HV * BT) + o_A[None, :]
            b_Aq = tl.load(p_Aq, mask=m_t[:, None] & (o_A[None, :] < BT), other=0.0)
            b_o = b_o * scale + tl.dot(b_Aq, b_vns) * scale
            p_o = o + vo + o_t[:, None] * (HV * V) + o_v[None, :]
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])


def gdn_forward_ps(q, k, v, g, beta, BV: int = 32, num_warps: int = 4, num_stages: int = 1, slen=None,
                   pre_warps: int = 0, peel: bool = False, pre: bool = False, norm=None):
    """FF ps mode: fla l2norm / cumsum / kkt_solve, then pass P (u, w, gated q k^T for all chunks in parallel) and
    pass S (the sequential h chain). Same arithmetic as gdn_forward; S's chunk step no longer waits on the u / w /
    q k^T dots, and those are done once per value head instead of once per V block.
    peel: FF-PEEL (kai, module docstring): pass S runs _ps_seq_x_kernel (default off: job 38's _ps_seq_kernel).
    pre (E4 PRO): q and k are already l2-normalised and g already chunk-cumulated times 1/ln 2 by
    fused.gdn_prologue_packed (the same bits, checked at start-up): fla's l2norm_fwd and chunk_local_cumsum are skipped."""
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel
    from fla.ops.utils.constant import RCP_LN2
    from fla.ops.utils.cumsum import chunk_local_cumsum

    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    scale = k.shape[-1] ** -0.5
    if pre:
        pass
    elif k.data_ptr() == q.data_ptr() + q.numel() * q.element_size() and k.shape == q.shape:
        nq = q.shape[0]
        qk = torch.as_strided(q, (2 * nq,) + tuple(q.shape[1:]), q.stride())
        qk, _ = l2norm_fwd(qk)
        q, k = qk[:nq], qk[nq:]
    else:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
    if not pre:
        g = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2)
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    BT = 64
    NT = triton.cdiv(T, BT)
    A = torch.empty(B, T, HV, BT, device=k.device, dtype=k.dtype)
    nat = KKT2["nat"]
    if nat is not None and nat.takes(k, g, beta, A):
        # KKT2 (rgh): native kkt_solve, one CTA per key head (kkt2n.py; bit-equal to fla's kernel, kkt2_enable)
        nat(k, g, beta, A, slen=slen)
    else:
        chunk_gated_delta_rule_fwd_kkt_solve_kernel[(NT, B * HV)](k=k, g=g, beta=beta, A=A, cu_seqlens=None,
                                                                  chunk_indices=None, T=T, H=H, HV=HV, K=K, BT=BT,
                                                                  BC=16)
    if peel and slen is not None:
        # FFS (rgh, S1ENG_FFS=1): pass P + pass S PEEL in one native sm_120 kernel, same bits (ffs.py)
        from . import ffs

        if ffs.lib() is not None and ffs.supported(q, k, v, g, beta, A, slen):
            if norm is not None:
                # ONORM (rgh): + the gated RMSNorm as the kernel's epilogue -> packed out_proj rows [R * 32, 128]
                from . import onorm

                return onorm.fused_norm(q, k, v, g, beta, A, slen, scale, *norm)
            return ffs.fused(q, k, v, g, beta, A, slen, scale)
    w = k.new_empty(B, T, HV, K)
    u = torch.empty_like(v)
    Aq = torch.empty(B, T, HV, BT, device=k.device, dtype=v.dtype)
    sl = slen if slen is not None else v
    _ps_pre_kernel[(NT, B * HV)](q, k, v, beta, A, g, w, u, Aq, sl, T, H=H, HV=HV, K=K, V=V, BT=BT, BV=BV,
                                 HAS_SLEN=slen is not None, num_warps=pre_warps or num_warps, num_stages=1)
    o = torch.empty_like(v)
    if not peel:
        _ps_seq_kernel[(triton.cdiv(V, BV), B * HV)](q, k, w, u, Aq, g, o, scale, sl, T, H=H, HV=HV, K=K, V=V, BT=BT,
                                                      BV=BV, HAS_SLEN=slen is not None, num_warps=num_warps,
                                                      num_stages=num_stages)
        return o
    _ps_seq_x_kernel[(triton.cdiv(V, BV), B * HV)](q, k, w, u, Aq, g, o, scale, sl, T, H=H, HV=HV, K=K, V=V, BT=BT,
                                                    BV=BV, HAS_SLEN=slen is not None, num_warps=num_warps,
                                                    num_stages=num_stages, PEEL=True)
    return o


def gdn_forward_ps_peel(q, k, v, g, beta, BV: int = 32, num_warps: int = 4, num_stages: int = 1, slen=None,
                        pre_warps: int = 0, pre: bool = False, norm=None):
    """FF-PEEL (kai): gdn_forward_ps with peel=True, the forward PackedForward and check() run with S1ENG_FF_PEEL=1 in
    ps mode. norm (ONORM): (z, w, eps, start, plen) of the gated norm, applied in the FFS kernel when FFS runs (the
    result is then the packed [R * 32, 128] norm output instead of o)."""
    return gdn_forward_ps(q, k, v, g, beta, BV=BV, num_warps=num_warps, num_stages=num_stages, slen=slen,
                          pre_warps=pre_warps, peel=True, pre=pre, norm=norm)


# KKT2 (rgh 2026-10-20): the native kkt_solve gdn_forward_ps uses once kkt2_enable() found it bit-equal to fla's
# kernel on this GPU (None: fla's kernel)
KKT2 = {"nat": None}


@torch.no_grad()
def kkt2_check(nat, seed: int = 43) -> str:
    """A of the native kkt_solve vs fla's chunk_gated_delta_rule_fwd_kkt_solve_kernel (the pinned config) on random
    serving-shaped decisions (16 key heads, 32 value heads, K = 128; l2-normalised k, chunk-cumulated g x 1/ln 2 as
    the prologue gives them, zeros past each length, gate scales 0.05 .. 8, an empty packed slot): every entry fla
    stores (rows < T, lower block triangle of 16 x 16 blocks, diagonal blocks whole) of every used slot must be
    bit-equal. "" if so, else the reason."""
    import torch.nn.functional as F
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel
    from fla.ops.utils.constant import RCP_LN2
    from fla.ops.utils.cumsum import chunk_local_cumsum

    g0 = torch.Generator(device="cuda")
    g0.manual_seed(seed)
    H, HV, K, BT = 16, 32, 128, 64
    cases = ((1, 128, 1.0), (2, 192, 0.3), (3, 256, 8.0), (5, 192, 0.05), (8, 256, 0.3), (6, 320, 1.0),
             (12, 192, 0.3), (16, 128, 2.0), (4, 320, 0.3))
    for B, T, gs in cases:
        n = torch.randint(max(1, T - 63), T + 1, (B,), generator=g0, device="cuda")
        keep = (torch.arange(T, device="cuda")[None, :] < n[:, None]).float()
        k = (torch.randn(B, T, H, K, generator=g0, device="cuda") * keep[..., None, None]).to(torch.bfloat16)
        k, _ = l2norm_fwd(k)
        g = (-F.softplus(torch.randn(B, T, HV, generator=g0, device="cuda")) * gs * keep[..., None]).float()
        g = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2)
        beta = (torch.sigmoid(torch.randn(B, T, HV, generator=g0, device="cuda")) * keep[..., None]).to(torch.bfloat16)
        slen = n.to(torch.int32)
        if B > 1:
            slen[-1] = 0
        ref = torch.zeros(B, T, HV, BT, device="cuda", dtype=torch.bfloat16)
        chunk_gated_delta_rule_fwd_kkt_solve_kernel[(triton.cdiv(T, BT), B * HV)](
            k=k, g=g, beta=beta, A=ref, cu_seqlens=None, chunk_indices=None, T=T, H=H, HV=HV, K=K, BT=BT, BC=16)
        got = torch.full((B, T, HV, BT), float("nan"), device="cuda", dtype=torch.bfloat16)
        if not nat.takes(k, g, beta, got):
            return "the native kernel does not take serving shapes"
        nat(k, g, beta, got, slen=slen)
        torch.cuda.synchronize()
        r = torch.arange(T, device="cuda") % BT
        c = torch.arange(BT, device="cuda")
        m = ((c[None, :] // 16) <= (r[:, None] // 16))[None, :, None, :] & (slen > 0)[:, None, None, None]
        m = m.expand(B, T, HV, BT)
        if not torch.equal(got.view(torch.int16)[m], ref.view(torch.int16)[m]):
            bad = int((got.view(torch.int16)[m] != ref.view(torch.int16)[m]).sum())
            return f"B={B} T={T}: {bad} A entries differ from fla's kkt_solve"
    return ""


def kkt2_enable() -> str:
    """KKT2 start-up gate: loads the native kkt_solve and keeps it only if kkt2_check passes on this GPU. Returns ""
    when it is on, else why it is off (fla's kernel is then used, as before)."""
    import os

    KKT2["nat"] = None
    if os.environ.get("S1ENG_KKT2", "1") != "1":
        return "S1ENG_KKT2=0"
    try:
        from .kkt2n import Kkt2n

        nat = Kkt2n()
        why = kkt2_check(nat)
    except Exception as exc:  # noqa: BLE001 - the lever must never take the engine down with it
        return f"{type(exc).__name__}: {exc}"
    if not why:
        KKT2["nat"] = nat
    return why


MODES = {"": None, "fused": None, "split": "gdn_forward_split", "ps": "gdn_forward_ps"}


def forward_of(mode: str):
    name = MODES.get(mode or "")
    return globals()[name] if name else gdn_forward


def supported(q, k, v) -> bool:
    return (q.dim() == 4 and k.shape[-1] == 128 and q.shape[-1] == 128 and q.shape[1] % 64 == 0
            and v.shape[2] % k.shape[2] == 0)


@triton.jit(do_not_specialize=['T'])
def _fused_wuh_kernel(k, v, beta, A, g, h, v_new, slen, T, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                      V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, HAS_SLEN: tl.constexpr):
    """FF split mode: recompute_w_u fused into fwd_h (w, u never stored); h and v_new are stored exactly as fla's
    fwd_h stores them, and fla's chunk_fwd_o (parallel over chunks) computes o from them."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    if HAS_SLEN:
        if tl.load(slen + i_n) == 0:
            return
    bos = i_n * T
    NT = tl.cdiv(T, BT)
    boh = i_n * NT
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    qk = (bos * H + i_h // (HV // H)).to(tl.int64) * K
    vo = (bos * HV + i_h).to(tl.int64) * V
    hb = h + (boh * HV + i_h).to(tl.int64) * K * V
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_A = tl.arange(0, BT)
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        # store h (as fwd_h does, before the update)
        p_h1 = hb + i_t_int64 * HV * K * V + o_k1[:, None] * V + o_v[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_k1[:, None] & m_v[None, :])
        p_h2 = hb + i_t_int64 * HV * K * V + o_k2[:, None] * V + o_v[None, :]
        tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_k2[:, None] & m_v[None, :])
        # recompute_w_u (this chunk)
        p_b = beta + bos * HV + i_h + o_t * HV
        b_b = tl.load(p_b, mask=m_t, other=0.0)
        p_A = A + (bos * HV + i_h) * BT + o_t[:, None] * (HV * BT) + o_A[None, :]
        m_blk = (o_A[:, None] // 16) >= (o_A[None, :] // 16)
        b_A = tl.load(p_A, mask=m_t[:, None] & (o_A[None, :] < BT) & m_blk, other=0.0)
        p_v = v + vo + o_t[:, None] * (HV * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        b_u = b_u.to(v.dtype.element_ty)
        p_g = g + (bos * HV + i_h) + o_t * HV
        b_gw = exp2(tl.load(p_g, mask=m_t, other=0.0))
        p_k = k + qk + o_t[:, None] * (H * K) + o_k1[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        b_kb = b_k * b_b[:, None]
        b_kb *= b_gw[:, None]
        b_w1 = tl.dot(b_A, b_kb.to(b_k.dtype)).to(k.dtype.element_ty)
        p_k = k + qk + o_t[:, None] * (H * K) + o_k2[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & m_k2[None, :], other=0.0)
        b_kb = b_k * b_b[:, None]
        b_kb *= b_gw[:, None]
        b_w2 = tl.dot(b_A, b_kb.to(b_k.dtype)).to(k.dtype.element_ty)
        # fwd_h body
        b_vn = tl.dot(b_w1, b_h1.to(b_w1.dtype))
        b_vn += tl.dot(b_w2, b_h2.to(b_w1.dtype))
        b_vn = b_u - b_vn
        p_vn = v_new + vo + o_t[:, None] * (HV * V) + o_v[None, :]
        tl.store(p_vn, b_vn.to(p_vn.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])
        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
        b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
        b_vn = b_vn * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
        b_g_last = exp2(b_g_last)
        b_h1 *= b_g_last
        b_h2 *= b_g_last
        b_vn = b_vn.to(k.dtype.element_ty)
        p_kt = k + qk + o_k1[:, None] + o_t[None, :] * (H * K)
        b_kt = tl.load(p_kt, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        b_h1 += tl.dot(b_kt, b_vn)
        p_kt = k + qk + o_k2[:, None] + o_t[None, :] * (H * K)
        b_kt = tl.load(p_kt, mask=m_k2[:, None] & m_t[None, :], other=0.0)
        b_h2 += tl.dot(b_kt, b_vn)


def gdn_forward_split(q, k, v, g, beta, BV: int = 32, num_warps: int = 4, num_stages: int = 2, slen=None):
    """FF split mode: fla l2norm / cumsum / kkt_solve, the fused w/u + fwd_h kernel above (stores h, v_new), then
    fla's own chunk_fwd_o."""
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.common.chunk_o import chunk_fwd_o
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel
    from fla.ops.utils.constant import RCP_LN2
    from fla.ops.utils.cumsum import chunk_local_cumsum

    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    scale = k.shape[-1] ** -0.5
    if k.data_ptr() == q.data_ptr() + q.numel() * q.element_size() and k.shape == q.shape:
        nq = q.shape[0]
        qk = torch.as_strided(q, (2 * nq,) + tuple(q.shape[1:]), q.stride())
        qk, _ = l2norm_fwd(qk)
        q, k = qk[:nq], qk[nq:]
    else:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
    g = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2)
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    BT = 64
    NT = triton.cdiv(T, BT)
    A = torch.empty(B, T, HV, BT, device=k.device, dtype=k.dtype)
    chunk_gated_delta_rule_fwd_kkt_solve_kernel[(NT, B * HV)](k=k, g=g, beta=beta, A=A, cu_seqlens=None,
                                                              chunk_indices=None, T=T, H=H, HV=HV, K=K, BT=BT, BC=16)
    h = k.new_empty(B, NT, HV, K, V)
    v_new = torch.empty_like(v)
    _fused_wuh_kernel[(triton.cdiv(V, BV), B * HV)](k, v, beta, A, g, h, v_new, slen if slen is not None else v, T,
                                                     H=H, HV=HV, K=K, V=V, BT=BT, BV=BV, HAS_SLEN=slen is not None,
                                                     num_warps=num_warps, num_stages=num_stages)
    return chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, cu_seqlens=None, chunk_indices=None, chunk_size=64)


@torch.no_grad()
def check(cfg=(32, 4, 1), split: bool = False, seed: int = 41, mode: str = "", peel: bool = False) -> str:
    """Start-up gate on THIS GPU: gdn_forward (or the split mode) vs fla chunk_gated_delta_rule on random decisions of
    serving shapes (16 q/k heads, 32 value heads, K = V = 128; q / k as halves of one buffer; zeros past each length;
    packed-slot skip with empty slots). Returns "" when every output element is bit-equal, else the reason. Triton
    lowers tl.dot per architecture, so the engine runs this before using the fused path and falls back to fla.
    peel (FF-PEEL, kai): in ps mode the peeled pass S (gdn_forward_ps_peel) is the one checked."""
    import torch.nn.functional as F
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    fwd = gdn_forward_split if split else forward_of(mode)
    if peel and mode == "ps":
        fwd = gdn_forward_ps_peel
    g0 = torch.Generator(device="cuda")
    g0.manual_seed(seed)
    H, HV, K, V = 16, 32, 128, 128
    for B, T in ((1, 128), (2, 192), (3, 256), (5, 192), (8, 256), (6, 320)):
        n = torch.randint(max(1, T - 63), T + 1, (B,), generator=g0, device="cuda")
        keep = (torch.arange(T, device="cuda")[None, :] < n[:, None]).float()
        qk = (torch.randn(2, B, T, H, K, generator=g0, device="cuda") * keep[None, ..., None, None]).to(torch.bfloat16)
        q, k = qk[0], qk[1]
        v = (torch.randn(B, T, HV, V, generator=g0, device="cuda") * keep[..., None, None]).to(torch.bfloat16)
        g = (-F.softplus(torch.randn(B, T, HV, generator=g0, device="cuda")) * 0.3 * keep[..., None]).float()
        beta = (torch.sigmoid(torch.randn(B, T, HV, generator=g0, device="cuda")) * keep[..., None]).to(torch.bfloat16)
        ref = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
                                     use_qk_l2norm_in_kernel=True, cu_seqlens=None)[0]
        slen = n.to(torch.int32)
        if B > 1:
            slen[-1] = 0                      # an unused slot: skipped, its outputs are never read
        got = fwd(q, k, v, g, beta, BV=cfg[0], num_warps=cfg[1], num_stages=cfg[2], slen=slen)
        used = slen > 0
        if not torch.equal(got[used].view(torch.int16), ref[used].view(torch.int16)):
            bad = int((got[used].view(torch.int16) != ref[used].view(torch.int16)).sum())
            return f"B={B} T={T}: {bad} output elements differ from fla"
    return ""
