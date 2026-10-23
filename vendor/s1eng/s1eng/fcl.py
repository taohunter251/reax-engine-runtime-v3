"""FCL (ruj v14 fixedcost track): fewer kernel nodes per lane graph, every output byte-equal to the path it replaces.

BA (S1ENG_BA=1): the GDN b|a projection (in_proj_b | in_proj_a, N = 2 x 32, K = 2560) in ONE Triton launch instead of
cuBLASLt's split-K GEMM (cutlass Kernel2, gridZ = 40 slices) + splitKreduce. The pins of this shape (lt.Pinned,
verified against the reference's per-decision F.linear) are split-K 40 / 41 with reduction OUTPUT_TYPE for buckets
128 / 192 / 256: 40 K-slices of 64 columns (cuBLAS runs 40 slices for 41 too: gridZ 40 in the trace), every slice
an mma.sync m16n8k16 chain from +0 in ascending k (acc_s), its partial rounded to bf16 (P_s = bf16(acc_s)), and the
reduce kernel folds the slices in order in fp32: y = bf16(((f(P0) + f(P1)) + ...) + f(P39)). The BA kernel does
exactly that per output element: per slice one tl.dot([BM, 64] x [64, 64]) from zero (Triton lowers it to the same
4 chained m16n8k16 MMAs, ascending k), the bf16 round, the in-order fp32 fold, one bf16 round at the end.
Row-independent arithmetic, so any R and any row tiling give the same bytes. Bucket 320's pin (split 35, reduction
COMPUTE_TYPE: 20 fp32 slices of 128) is NOT reproduced by an in-order fold (unit test: ~0.15 % of outputs differ),
so the gate keeps that bucket on the pinned path.
Start-up gate (ba_check, per bucket): BA vs PackedForward.pair(d, "b_a", ...) (the lanes' own pinned / S6 / fused-
or-parts path at that R) on random inputs at three scales with the real b|a weights of 3 GDN layers, at EVERY packed
R of the bucket for the first layer and at a spread of R for the others; byte equality required, else the bucket
keeps pair(). KP classes the kernel "small" (high priority), so its few CTAs dispatch first beside the qkv GEMM.

S1ENG_BA: "1" on (default in this image), "0" off. S1ENG_BA_CFG="BM,warps,stages" (default 32,4,4).

QKR (S1ENG_QKR=1), LT (S1ENG_LT=1), EMB (S1ENG_EMB=1) and RO (S1ENG_RO=1): see their sections below. Every lever is
per bucket, behind its own start-up byte gate (a failed gate keeps the bucket on the launches it would replace), and
off while a prefix is recorded. fcab.py (S1ENG_FCAB) is the in-situ A/B timing harness for them.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# bucket -> (slice width, slices, fp32 partials); the gate decides whether the pinned path really is this
BA_MODES = {128: (64, 40, False), 192: (64, 40, False), 256: (64, 40, False), 320: (128, 20, True)}


@triton.jit
def _ba_kernel(x, w, y, R, XS, YS, K: tl.constexpr, N: tl.constexpr, CH: tl.constexpr, NS: tl.constexpr,
               BM: tl.constexpr, F32P: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    rm = rows < R
    cols = tl.arange(0, N)
    kk = tl.arange(0, CH)
    xr = x + rows[:, None].to(tl.int64) * XS
    a = tl.load(xr + kk[None, :], mask=rm[:, None], other=0.0)
    b = tl.load(w + cols[None, :] * K + kk[:, None])
    acc = tl.dot(a, b, out_dtype=tl.float32)            # slice 0 from +0, ascending k
    if F32P:
        fold = acc
    else:
        fold = acc.to(tl.bfloat16).to(tl.float32)       # P_0 = bf16(acc_0)
    for s in range(1, NS):
        a = tl.load(xr + (s * CH + kk)[None, :], mask=rm[:, None], other=0.0)
        b = tl.load(w + cols[None, :] * K + (s * CH + kk)[:, None])
        acc = tl.dot(a, b, out_dtype=tl.float32)
        if F32P:
            fold = fold + acc
        else:
            fold = fold + acc.to(tl.bfloat16).to(tl.float32)
    tl.store(y + rows[:, None].to(tl.int64) * YS + cols[None, :], fold.to(tl.bfloat16), mask=rm[:, None])


def ba_cfg_env():
    v = os.environ.get("S1ENG_BA_CFG", "32,4,4")
    bm, warps, stages = (int(t) for t in v.split(","))
    return bm, warps, stages


def ba_gemm(x: torch.Tensor, w: torch.Tensor, mode, kcfg=None) -> torch.Tensor:
    """x [R, K] (row stride may be larger, unit column stride), w [N, K] contiguous -> y [R, N] bf16."""
    R, K = x.shape
    N = w.shape[0]
    ch, ns, f32p = mode
    assert x.stride(1) == 1 and w.is_contiguous() and w.shape[1] == K and ch * ns == K
    bm, warps, stages = kcfg or ba_cfg_env()
    y = torch.empty((R, N), dtype=x.dtype, device=x.device)
    _ba_kernel[(triton.cdiv(R, bm),)](x, w, y, R, x.stride(0), y.stride(0), K=K, N=N, CH=ch, NS=ns, BM=bm,
                                      F32P=f32p, num_warps=warps, num_stages=stages)
    return y


def ba_weight(d) -> torch.Tensor | None:
    """The layer's [b; a] weight block (the fused block FUSE made, else built once and kept on the layer dict)."""
    blk = d.get("fused", {}).get("b_a")
    if blk is not None:
        return blk
    if "_fcl_ba_w" not in d:
        la = d["la"]
        d["_fcl_ba_w"] = torch.cat([la.in_proj_b.weight, la.in_proj_a.weight]).contiguous()
    return d["_fcl_ba_w"]


@torch.inference_mode()
def ba_check(pf, L: int, layers, rs, seed: int = 61) -> str:
    """BA start-up gate of bucket L: "" if BA's [R, 64] equals pair(d, "b_a")'s two column blocks byte for byte at
    every R of rs (first layer; a spread of R for the others), three input scales; else the first difference."""
    mode = BA_MODES.get(L)
    if mode is None:
        return "no mode for this bucket"
    g = torch.Generator(device="cuda")
    g.manual_seed(seed + L)
    rs = sorted(rs)
    spread = sorted({rs[0], rs[len(rs) // 4], rs[len(rs) // 2], rs[(3 * len(rs)) // 4], rs[-1]})
    for li, d in enumerate(layers):
        la = d["la"]
        w = ba_weight(d)
        n1 = la.in_proj_b.weight.shape[0]
        for R in (rs if li == 0 else spread):
            sc = (1.0, 6.0, 0.05)[(R // 32 + li) % 3]
            x = (torch.randn((R, 2560), generator=g, device="cuda") * sc).to(torch.bfloat16)
            b0, a0 = pf.pair(d, "b_a", la.in_proj_b, la.in_proj_a, x, L)
            y = ba_gemm(x, w, mode)
            if not (torch.equal(y[:, :n1].view(torch.int16), b0.view(torch.int16))
                    and torch.equal(y[:, n1:].view(torch.int16), a0.view(torch.int16))):
                bad = int((y[:, :n1].view(torch.int16) != b0.view(torch.int16)).sum()
                          + (y[:, n1:].view(torch.int16) != a0.view(torch.int16)).sum())
                return f"R={R} layer #{li}: {bad} of {R * 2 * n1} outputs differ from the pinned path"
    return ""


# ================================================================================================ QKR
# QKR (S1ENG_QKR=1): an attention layer's q norm + RoPE (fused._q_norm_rope_packed_kernel) and k norm + RoPE / v copy
# (fused._kv_norm_rope_packed_kernel) in ONE launch after the k|v join: programs [0, NQ) run the q kernel's statements
# with (pid_l, h, bb) taken from the 1-D program id, the rest the kv kernel's; the same jit helpers (_norm_rope_rows,
# _ld_rows), the same EXACT compile flags, so the same bytes (qkr_check compares both outputs at start-up).
from .fused import EXACT, INV_256, _ld_rows, _norm_rope_rows  # noqa: E402


@triton.jit
def _qkr_kernel(qg, w1p, cos, sin, out, start, slen, L, eps, inv_d, cq, CQS, plen, pvar,
                kr, vr, kw1p, kout, vout, KS, VS, ck, cv, CKS, CVS, NQ, NLB,
                H: tl.constexpr, HK: tl.constexpr, REP: tl.constexpr, BL: tl.constexpr, PFX: tl.constexpr,
                PMAX: tl.constexpr):
    pid = tl.program_id(0)
    if pid < NQ:
        # ---- fused._q_norm_rope_packed_kernel (ONLY_LAST False), program (pid_l, h, bb)
        pid_l = pid % NLB
        h = (pid // NLB) % H
        bb = pid // (NLB * H)
        s0 = tl.load(start + bb)
        n = tl.load(slen + bb)
        out = out + bb.to(tl.int64) * H * L * 256
        ls = pid_l * BL + tl.arange(0, BL)
        lsm = tl.minimum(ls, tl.maximum(n - 1, 0))
        if PFX > 0:
            pb = tl.load(plen + bb)
            cb0 = tl.load(pvar + bb).to(tl.int64) * PMAX
            rows_off = (s0 + lsm - pb).to(tl.int64) * (H * 512) + h * 512
            crows_off = (cb0 + lsm.to(tl.int64)) * CQS + h * 512
            f0, f1 = _norm_rope_rows(qg, rows_off, w1p, cos + lsm * 64, sin + lsm * 64, eps, inv_d, BL, cq, crows_off,
                                     lsm < pb, lsm >= pb, PFX)
        else:
            rows_off = (s0 + lsm).to(tl.int64) * (H * 512) + h * 512
            f0, f1 = _norm_rope_rows(qg, rows_off, w1p, cos + lsm * 64, sin + lsm * 64, eps, inv_d, BL, qg, rows_off,
                                     rows_off, rows_off, 0)
        lane = tl.arange(0, 32)[:, None] * 4 + tl.arange(0, 4)[None, :]
        dst = out + (h * L + ls[:, None, None]).to(tl.int64) * 256 + lane[None, :, :]
        m = (ls < L)[:, None, None]
        keep = (ls < n)[:, None, None]
        tl.store(dst, tl.where(keep, f0, 0.0), mask=m)
        tl.store(dst + 128, tl.where(keep, f1, 0.0), mask=m)
    else:
        # ---- fused._kv_norm_rope_packed_kernel, program (pid_l, h, bb)
        j = pid - NQ
        pid_l = j % NLB
        h = (j // NLB) % HK
        bb = j // (NLB * HK)
        s0 = tl.load(start + bb)
        n = tl.load(slen + bb)
        kout = kout + bb.to(tl.int64) * (HK * REP) * L * 256
        vout = vout + bb.to(tl.int64) * (HK * REP) * L * 256
        ls = pid_l * BL + tl.arange(0, BL)
        lsm = tl.minimum(ls, tl.maximum(n - 1, 0))
        if PFX > 0:
            lane = tl.arange(0, 32)[:, None] * 4 + tl.arange(0, 4)[None, :]
            pb = tl.load(plen + bb)
            cb0 = tl.load(pvar + bb).to(tl.int64) * PMAX
            rows_off = (s0 + lsm - pb).to(tl.int64) * KS + h * 256
            vrows_off = (s0 + lsm - pb).to(tl.int64) * VS + h * 256
            crows_off = (cb0 + lsm.to(tl.int64)) * CKS + h * 256
            cvrows_off = (cb0 + lsm.to(tl.int64)) * CVS + h * 256
            use_c = lsm < pb
            use_x = lsm >= pb
            f0, f1 = _norm_rope_rows(kr, rows_off, kw1p, cos + lsm * 64, sin + lsm * 64, eps, inv_d, BL, ck,
                                     crows_off, use_c, use_x, PFX)
            uc = use_c[:, None, None]
            ux = use_x[:, None, None]
            v0 = _ld_rows(vr, vrows_off[:, None, None] + lane[None, :, :], cv,
                          cvrows_off[:, None, None] + lane[None, :, :], uc, ux, PFX)
            v1 = _ld_rows(vr, vrows_off[:, None, None] + 128 + lane[None, :, :], cv,
                          cvrows_off[:, None, None] + 128 + lane[None, :, :], uc, ux, PFX)
        else:
            rows_off = (s0 + lsm).to(tl.int64) * KS + h * 256
            vrows_off = (s0 + lsm).to(tl.int64) * VS + h * 256
            f0, f1 = _norm_rope_rows(kr, rows_off, kw1p, cos + lsm * 64, sin + lsm * 64, eps, inv_d, BL, kr, rows_off,
                                     rows_off, rows_off, 0)
            lane = tl.arange(0, 32)[:, None] * 4 + tl.arange(0, 4)[None, :]
            v0 = tl.load(vr + vrows_off[:, None, None] + lane[None, :, :])
            v1 = tl.load(vr + vrows_off[:, None, None] + 128 + lane[None, :, :])
        m = (ls < L)[:, None, None]
        keep = (ls < n)[:, None, None]
        f0 = tl.where(keep, f0, 0.0)
        f1 = tl.where(keep, f1, 0.0)
        v0 = tl.where(keep, v0, 0.0)
        v1 = tl.where(keep, v1, 0.0)
        for r in tl.static_range(REP):
            hh = h * REP + r
            kd = (hh * L + ls[:, None, None]).to(tl.int64) * 256 + lane[None, :, :]
            tl.store(kout + kd, f0, mask=m)
            tl.store(kout + kd + 128, f1, mask=m)
            tl.store(vout + kd, v0, mask=m)
            tl.store(vout + kd + 128, v1, mask=m)


def qkr(qg, w1p, kr, vr, kw1p, cos, sin, eps, rep, start, slen, B, L, cache=None, ck=None, cv=None, pfx: int = 0,
        plen=None, pvar=None, pmax: int = 0):
    """(fused.q_norm_rope_packed(qg, w1p, ...), fused.kv_norm_rope_rep_packed(kr, vr, kw1p, ..., rep, ...)) in one
    launch -> (q, k, v) with the same shapes, strides and bytes."""
    R, H, D2 = qg.shape
    assert D2 == 512 and qg.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()
    if kr.dim() == 3:
        _, HK, D = kr.shape
        assert D == 256 and kr.is_contiguous() and vr.is_contiguous() and vr.shape == kr.shape
        ks = vs = HK * 256
    else:
        _, HD = kr.shape
        HK = HD // 256
        assert HD == HK * 256 and kr.stride(1) == 1 and vr.stride(1) == 1 and vr.shape == kr.shape
        ks, vs = kr.stride(0), vr.stride(0)
    if pfx:
        pmax = pmax or pfx
        assert cache.shape[1] == H * 512 and cache.shape[0] % pmax == 0 and cache.stride(1) == 1
        assert ck.shape[1] == HK * 256 and cv.shape == ck.shape and ck.shape[0] % pmax == 0
        assert ck.stride(1) == 1 and cv.stride(1) == 1 and plen is not None and pvar is not None
    else:
        cache, ck, cv, plen, pvar = qg, kr, vr, start, start
    q = torch.empty((B, H, L, 256), dtype=qg.dtype, device=qg.device)
    kout = torch.empty((B, HK * rep, L, 256), dtype=kr.dtype, device=kr.device)
    vout = torch.empty_like(kout)
    BL = 8
    nlb = triton.cdiv(L, BL)
    nq = nlb * H * B
    _qkr_kernel[(nq + nlb * HK * B,)](qg, w1p, cos, sin, q, start, slen, L, eps, INV_256, cache, cache.stride(0), plen,
                                      pvar, kr, vr, kw1p, kout, vout, ks, vs, ck, cv, ck.stride(0), cv.stride(0), nq,
                                      nlb, H=H, HK=HK, REP=rep, BL=BL, PFX=pfx, PMAX=pmax, num_warps=4, **EXACT)
    return q, kout, vout


@torch.inference_mode()
def qkr_check(fz, d, cos, sin, eps, L: int, bmax: int, pfx: int = 0, pmax: int = 0, rep: int = 1,
              seed: int = 67) -> str:
    """QKR start-up gate of bucket L: qkr(...) vs the two fused kernels on random packed layouts (B up to bmax, with
    and without the template prefix cache): q, k, v byte-equal. "" if so."""
    import numpy as np

    g = torch.Generator(device="cuda")
    g.manual_seed(seed + L)
    rng = np.random.default_rng(seed + L)
    sa = d["sa"]
    hd = sa.head_dim
    H = sa.q_proj.weight.shape[0] // (2 * hd)
    HK = sa.k_proj.weight.shape[0] // hd
    lprev = max([x for x in (0, 128, 192, 256, 320) if x < L])
    pmax = max(pmax, pfx)
    for B in sorted({1, 2, 3, 5, bmax}):
        for trial in range(2):
            lo = max(lprev + 1, pfx + 1)
            n = rng.integers(lo, L + 1, B)
            pl = np.minimum(rng.integers(pfx, pmax + 1, B), n - 1) if pfx else np.zeros(B, dtype=np.int64)
            pv = rng.integers(0, 3, B)
            rows = (n - pl).astype(np.int64)
            st = np.concatenate([[0], np.cumsum(rows)[:-1]]).astype(np.int32)
            R = int(rows.sum()) + 16
            sc = (1.0, 5.0)[trial]
            qg = (torch.randn((R, H, 512), generator=g, device="cuda") * sc).to(torch.bfloat16)
            kv = (torch.randn((R, 2 * HK * 256), generator=g, device="cuda") * sc).to(torch.bfloat16)
            kr, vr = kv[:, :HK * 256], kv[:, HK * 256:]
            start = torch.from_numpy(st).cuda()
            slen = torch.from_numpy(n.astype(np.int32)).cuda()
            kw = {}
            if pfx:
                cq = (torch.randn((3 * pmax, H * 512), generator=g, device="cuda") * sc).to(torch.bfloat16)
                ckv = (torch.randn((3 * pmax, 2 * HK * 256), generator=g, device="cuda") * sc).to(torch.bfloat16)
                kw = dict(pfx=pfx, plen=torch.from_numpy(pl.astype(np.int32)).cuda(),
                          pvar=torch.from_numpy(pv.astype(np.int32)).cuda(), pmax=pmax)
                ref_q = fz.q_norm_rope_packed(qg, d["qn_w"], cos, sin, eps, start, slen, B, L, cache=cq, **kw)
                ref_k, ref_v = fz.kv_norm_rope_rep_packed(kr, vr, d["kn_w"], cos, sin, eps, rep, start, slen, B, L,
                                                          ck=ckv[:, :HK * 256], cv=ckv[:, HK * 256:], **kw)
                q1, k1, v1 = qkr(qg, d["qn_w"], kr, vr, d["kn_w"], cos, sin, eps, rep, start, slen, B, L, cache=cq,
                                 ck=ckv[:, :HK * 256], cv=ckv[:, HK * 256:], **kw)
            else:
                ref_q = fz.q_norm_rope_packed(qg, d["qn_w"], cos, sin, eps, start, slen, B, L)
                ref_k, ref_v = fz.kv_norm_rope_rep_packed(kr, vr, d["kn_w"], cos, sin, eps, rep, start, slen, B, L)
                q1, k1, v1 = qkr(qg, d["qn_w"], kr, vr, d["kn_w"], cos, sin, eps, rep, start, slen, B, L)
            for name, a, b in (("q", ref_q, q1), ("k", ref_k, k1), ("v", ref_v, v1)):
                if a.shape != b.shape or a.stride() != b.stride() or not torch.equal(a.view(torch.int16),
                                                                                    b.view(torch.int16)):
                    return f"{name} differs (B={B}, pfx={pfx}, trial {trial})"
    return ""


# ================================================================================================ LT
# LT (S1ENG_LT=1): the pruned last layer's gathers in ONE launch: xl = x[last], hl = h[last], pos_last =
# row_pos[last] (the three index_selects) and, with LR1 + GQA, the one-row attention bias of every (slot, KV group):
# mask[0, 0, max(slen - 1, 0), :] repeated G times (today: slen.long(), - 1, clamp_, index_select, repeat_interleave
# = five more launches). Pure copies: the same bytes, shapes and strides (lt_check).
@triton.jit
def _lt_kernel(x, h, row_pos, last, slen, mask, xl, hl, pl, bias, plo, L, G, D: tl.constexpr, BD: tl.constexpr,
               BLM: tl.constexpr, HAS_BIAS: tl.constexpr):
    # grid (B, 2 * D / BD + 1): programs c < D / BD copy x row chunks, the next D / BD the h row chunks, the last one
    # pos_last, plen_last = max(slen - 1, 0) (XL's first computed position) and the bias rows
    b = tl.program_id(0)
    c = tl.program_id(1)
    NC: tl.constexpr = D // BD
    r = tl.load(last + b).to(tl.int64)
    offs = tl.arange(0, BD)
    if c < NC:
        ox = c * BD + offs
        tl.store(xl + b * D + ox, tl.load(x + r * D + ox))
    elif c < 2 * NC:
        oh = (c - NC) * BD + offs
        tl.store(hl + b * D + oh, tl.load(h + r * D + oh))
    else:
        tl.store(pl + b, tl.load(row_pos + r))
        n = tl.load(slen + b)
        tl.store(plo + b, tl.maximum(n - 1, 0))
        if HAS_BIAS:
            row = tl.maximum(n - 1, 0).to(tl.int64)
            lo = tl.arange(0, BLM)
            lm = lo < L
            mv = tl.load(mask + row * L + lo, mask=lm)
            for gg in range(G):
                tl.store(bias + (b * G + gg).to(tl.int64) * L + lo, mv, mask=lm)


def lt_gather(x, h, row_pos, last, slen, mask, B: int, L: int, G: int, with_plo: bool = False):
    """-> (xl [B, D], hl [B, D], pos_last [B] int32, bias [B * G, 1, 1, L] or None (G = 0)) and, with_plo, plen_last
    [B] int32 = max(slen - 1, 0)."""
    R, D = x.shape
    assert x.is_contiguous() and h.is_contiguous() and h.shape == x.shape and D % 512 == 0
    assert row_pos.is_contiguous() and last.is_contiguous() and last.shape[0] == B
    xl = torch.empty((B, D), dtype=x.dtype, device=x.device)
    hl = torch.empty((B, D), dtype=h.dtype, device=h.device)
    pl = torch.empty((B,), dtype=row_pos.dtype, device=row_pos.device)
    plo = torch.empty((B,), dtype=torch.int32, device=x.device)
    if G:
        m2 = mask[0, 0]
        assert m2.is_contiguous() and m2.shape == (L, L)
        bias = torch.empty((B * G, 1, 1, L), dtype=mask.dtype, device=mask.device)
    else:
        m2, bias = x, x
    _lt_kernel[(B, 2 * (D // 512) + 1)](x, h, row_pos, last, slen, m2, xl, hl, pl, bias, plo, L, G, D=D, BD=512,
                                         BLM=triton.next_power_of_2(L), HAS_BIAS=bool(G), num_warps=4)
    out = (xl, hl, pl, (bias if G else None))
    return out + (plo,) if with_plo else out


@torch.inference_mode()
def lt_check(mask, L: int, bmax: int, G: int, seed: int = 71) -> str:
    """LT start-up gate of bucket L: lt_gather vs index_select / the LR1 bias chain + repeat_interleave, byte for byte,
    shapes and strides, at B = 1 .. bmax (an unused slot included). "" if so."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed + L)
    for B in sorted({1, 2, 3, 5, bmax}):
        R = B * L
        x = torch.randn((R, 2560), generator=g, device="cuda").to(torch.bfloat16)
        h = torch.randn((R, 2560), generator=g, device="cuda").to(torch.bfloat16)
        row_pos = torch.randint(0, R, (R,), generator=g, device="cuda", dtype=torch.int32)
        last = torch.randint(0, R, (B,), generator=g, device="cuda", dtype=torch.int64)
        slen = torch.randint(0, L + 1, (B,), generator=g, device="cuda", dtype=torch.int32)
        slen[0] = 0
        xl0 = x.index_select(0, last).contiguous()
        hl0 = h.index_select(0, last).contiguous()
        pl0 = row_pos.index_select(0, last).contiguous()
        xl, hl, pl, bias, plo = lt_gather(x, h, row_pos, last, slen, mask, B, L, G, with_plo=True)
        outs = [("xl", xl0, xl), ("hl", hl0, hl), ("pos_last", pl0, pl),
                ("plen_last", (slen.long() - 1).clamp_(min=0).to(torch.int32), plo)]
        if G:
            b0 = mask[0, 0].index_select(0, (slen.long() - 1).clamp_(min=0)).view(B, 1, 1, L)
            outs.append(("bias", b0.repeat_interleave(G, dim=0), bias))
        for name, a, b in outs:
            if a.dtype != b.dtype or a.shape != b.shape or a.stride() != b.stride():
                return f"{name}: layout differs (B={B})"
            ai = a.view(torch.int16) if a.dtype in (torch.bfloat16, torch.float16) else a
            bi = b.view(torch.int16) if b.dtype in (torch.bfloat16, torch.float16) else b
            if not torch.equal(ai, bi):
                return f"{name} differs (B={B})"
    return ""


# ================================================================================================ EMB
# EMB (S1ENG_EMB=1, needs L0T): the packed graph's first gathers in ONE launch: h = embed_tokens(ids) (F.embedding =
# index_select of the [V, 2560] table) and layer 0's L0T table rows (l0t.Table.rows: qkv|z as one [R, NQ + NZ] gather
# or the two contiguous parts [R, NQ] + [R, NZ], b|a as one [R, NB + NA] gather or [R, NB] + [R, NA]: pair()'s layout
# at this (bucket, R)) - today 3 to 5 launches. Pure copies: the same bytes, shapes and strides (emb_check).
@triton.jit
def _emb_kernel(ids, slots, ew, h, qt, q1, q2, bt, b1, b2, D: tl.constexpr, NQ: tl.constexpr, NZ: tl.constexpr,
                NB: tl.constexpr, NA: tl.constexpr, FQ: tl.constexpr, FB: tl.constexpr, BLK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    offs = tl.arange(0, BLK)
    NE: tl.constexpr = D // BLK
    NQC: tl.constexpr = NQ // BLK
    NZC: tl.constexpr = NZ // BLK
    if c < NE:                                   # embedding row ids[r]
        tok = tl.load(ids + r).to(tl.int64)
        eo = c * BLK + offs
        tl.store(h + r * D + eo, tl.load(ew + tok * D + eo))
    elif c < NE + NQC + NZC:                     # qkv|z table row slots[r]
        sq = tl.load(slots + r).to(tl.int64)
        qo = (c - NE) * BLK + offs
        qv = tl.load(qt + sq * (NQ + NZ) + qo)
        if FQ:
            tl.store(q1 + r * (NQ + NZ) + qo, qv)
        else:
            if c < NE + NQC:
                tl.store(q1 + r * NQ + qo, qv)
            else:
                tl.store(q2 + r * NZ + (qo - NQ), qv)
    else:                                        # b|a table row slots[r]
        sb = tl.load(slots + r).to(tl.int64)
        bo = offs
        bm = bo < NB + NA
        bv = tl.load(bt + sb * (NB + NA) + bo, mask=bm)
        if FB:
            tl.store(b1 + r * (NB + NA) + bo, bv, mask=bm)
        else:
            tl.store(b1 + r * NB + bo, bv, mask=bo < NB)
            tl.store(b2 + r * NA + (bo - NB), bv, mask=(bo >= NB) & bm)


EMB_BLK = 512


def emb_gather(ew: torch.Tensor, ids: torch.Tensor, slots: torch.Tensor, qt: torch.Tensor, nq: int, bt: torch.Tensor,
               nb: int, fq: bool, fb: bool):
    """-> (h [R, D], ((qkv, z), (b, a))) with embed_tokens' and l0t.Table.rows' layouts (fq / fb: the fused views)."""
    R = ids.shape[0]
    V, D = ew.shape
    NQZ, NBA = qt.shape[1], bt.shape[1]
    nz, na = NQZ - nq, NBA - nb
    assert ew.is_contiguous() and qt.is_contiguous() and bt.is_contiguous() and ids.is_contiguous()
    assert slots.is_contiguous() and slots.shape[0] == R and ids.dim() == 1
    assert D % EMB_BLK == 0 and nq % EMB_BLK == 0 and nz % EMB_BLK == 0 and NBA <= EMB_BLK
    dev = ew.device
    h = torch.empty((R, D), dtype=ew.dtype, device=dev)
    if fq:
        yq = torch.empty((R, NQZ), dtype=qt.dtype, device=dev)
        q1, q2 = yq, yq
        qkvz = (yq[:, :nq], yq[:, nq:])
    else:
        q1 = torch.empty((R, nq), dtype=qt.dtype, device=dev)
        q2 = torch.empty((R, nz), dtype=qt.dtype, device=dev)
        qkvz = (q1, q2)
    if fb:
        yb = torch.empty((R, NBA), dtype=bt.dtype, device=dev)
        b1, b2 = yb, yb
        ba = (yb[:, :nb], yb[:, nb:])
    else:
        b1 = torch.empty((R, nb), dtype=bt.dtype, device=dev)
        b2 = torch.empty((R, na), dtype=bt.dtype, device=dev)
        ba = (b1, b2)
    nch = D // EMB_BLK + NQZ // EMB_BLK + 1
    _emb_kernel[(R, nch)](ids, slots, ew, h, qt, q1, q2, bt, b1, b2, D=D, NQ=nq, NZ=nz, NB=nb, NA=na, FQ=fq, FB=fb,
                          BLK=EMB_BLK, num_warps=2)
    return h, (qkvz, ba)


@torch.inference_mode()
def emb_check(embed, l0t, L: int, rs, fused_of, seed: int = 73) -> str:
    """EMB start-up gate of bucket L: emb_gather vs embed(ids) and l0t.Table.rows' gathers (on random tables of the
    bucket's shapes, every slot incl. the dump row reachable), at every R of rs with that R's fused-or-parts choice
    (fused_of(R) -> (fq, fb)) and with all four layouts at the largest R: bytes, shapes and strides. "" if equal."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed + L)
    bk = l0t.books[L]
    ew = embed.weight
    V = ew.shape[0]
    qt = torch.randn(bk.qkvz.shape, generator=g, device="cuda").to(bk.qkvz.dtype)
    bt = torch.randn(bk.ba.shape, generator=g, device="cuda").to(bk.ba.dtype)
    rs = sorted(rs)
    cases = [(R,) + tuple(bool(f) for f in fused_of(R)) for R in rs]
    cases += [(rs[-1], fq, fb) for fq in (False, True) for fb in (False, True)]
    for R, fq, fb in cases:
        ids = torch.randint(0, V, (R,), generator=g, device="cuda", dtype=torch.long)
        ids[0] = 0
        ids[-1] = V - 1
        slots = torch.randint(0, bk.cap + 1, (R,), generator=g, device="cuda", dtype=torch.long)
        slots[0] = bk.cap
        h0 = embed(ids)
        outs0 = []
        for t, n1, fused in ((qt, bk.nq, fq), (bt, bk.nb, fb)):   # l0t.Table.rows on these tables
            if fused:
                y = t.index_select(0, slots)
                outs0.append((y[:, :n1], y[:, n1:]))
            else:
                outs0.append((t[:, :n1].index_select(0, slots), t[:, n1:].index_select(0, slots)))
        h1, outs1 = emb_gather(ew, ids, slots, qt, bk.nq, bt, bk.nb, fq, fb)
        pairs = [("h", h0, h1), ("qkv", outs0[0][0], outs1[0][0]), ("z", outs0[0][1], outs1[0][1]),
                 ("b", outs0[1][0], outs1[1][0]), ("a", outs0[1][1], outs1[1][1])]
        for name, a, b in pairs:
            if a.dtype != b.dtype or a.shape != b.shape or a.stride() != b.stride():
                return f"{name}: layout differs (R={R}, fused {fq}/{fb}: {tuple(a.shape)} {a.stride()} vs " \
                       f"{tuple(b.shape)} {b.stride()})"
            if not torch.equal(a.view(torch.int16), b.view(torch.int16)):
                return f"{name} differs (R={R}, fused {fq}/{fb})"
    return ""


# ================================================================================================ RO
# RO (S1ENG_RO=1): the readout's B separate [1, 2560] @ W^T products (the reference op: one cuBLAS GEMV-class launch
# per slot) + torch.cat in ONE call: "bmm" = torch.bmm of [B, 1, 2560] by the weight expanded over the batch (stride
# 0; cuBLAS strided-batched, the batch in the grid) or "mm" = one [B, 2560] @ W^T. Used only where ro_check finds the
# result byte-equal to the B products for every packed B of the bucket on random rows at several scales.
RO_MODES = ("bmm", "mm")


def readout(hn: torch.Tensor, sw_t: torch.Tensor, mode: str) -> torch.Tensor:
    B = hn.shape[0]
    if mode == "bmm":
        return torch.bmm(hn.view(B, 1, hn.shape[1]), sw_t.unsqueeze(0).expand(B, -1, -1)).view(B, sw_t.shape[1])
    return hn @ sw_t


@torch.inference_mode()
def ro_check(slot_weight: torch.Tensor, bs, seed: int = 79):
    """RO start-up gate: the first mode of RO_MODES whose output equals torch.cat([hn[b:b+1] @ W^T ...]) byte for byte
    (dtype, shape, strides) for every B in bs, 4 inputs each (scales 1, 8, 0.02 and a near-tie row block) ->
    (mode, "") or (None, why)."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    sw_t = slot_weight.T
    K = slot_weight.shape[1]
    notes = []
    for mode in RO_MODES:
        why = ""
        for B in sorted(set(bs)):
            for trial, sc in enumerate((1.0, 8.0, 0.02, 1.0)):
                hn = (torch.randn((B, K), generator=g, device="cuda") * sc).to(slot_weight.dtype)
                if trial == 3:
                    hn[:] = hn[:1]
                ref = torch.cat([hn[b:b + 1] @ sw_t for b in range(B)])
                try:
                    got = readout(hn, sw_t, mode)
                except Exception as exc:  # noqa: BLE001
                    why = f"{type(exc).__name__}: {exc}"
                    break
                if got.dtype != ref.dtype or got.shape != ref.shape or got.stride() != ref.stride():
                    why = f"layout differs (B={B})"
                    break
                if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                    why = f"{int((got.view(torch.int16) != ref.view(torch.int16)).sum())} of {ref.numel()} " \
                          f"logits differ (B={B}, trial {trial})"
                    break
            if why:
                break
        if not why:
            return mode, ""
        notes.append(f"{mode}: {why}")
    return None, "; ".join(notes)


# ================================================================================================ ROP
# ROP (S1ENG_RO_PAD=1; pjs e10, on ruj's RO): RO's bmm only at row counts where cuBLAS runs the reference's GEMV kernel.
# cuBLAS picks the instance of its strided-batched GEMV (internal::gemvx::kernel<int, int, bf16, bf16, bf16, float,
# false, true, true, false, T, false, cublasGemvParamsEx<...>>) by batch count. On the RTX PRO 6000 Blackwell Server
# Edition (torch 2.13 / CUDA 13.0) RO's bmm runs T = 6 at B = 11-12 and T = 7 at every other B in 1-16, and the
# reference's own [1, 2560] @ W^T (per-decision graphs, BatchForward, the reference engine) runs T = 7. The two
# instances add the 2560 products in a different fp32 order, so at 11-12 rows about 1.4e-4 of the logits (those within
# ~1e-6 of a bf16 rounding midpoint) round to the other bf16 neighbour (e10 ROOTCAUSE.md: item 1c951e864bd1b9b1, logit 0
# = 26.0625016 in fp64, 26.0 by T = 7 and 26.125 by T = 6, so it is answered off the reference whenever it rides in a
# 12-slot graph, packed or mixed). ro_check's 768 logits at B 12 pass such a kernel about 90 % of the time.
# ro_plan picks, per graph batch count B, the row count P >= B for bmm (P - B zero rows appended, the first B output
# rows returned; the batch is in the GEMV's grid, so a row's sum never sees the other rows) or 0 = the per-row products
# + cat (the reference op itself). A count is taken only when (1) its GEMV kernel node, captured in a CUDA graph, is the
# per-row product's (same function, block and shared memory; the grid differs only by the batch) and (2) every logit
# is byte-equal to the per-row products on random rows AND on near-tie rows (rows with an exact fp64 logit within
# `tie` of a bf16 rounding midpoint: the only logits a different fp32 summation order can flip; a 131072-row draw gives
# ~300 of them, as many flips as ~2e6 random logits would show). (1) is skipped, and the log says so, where the driver
# gives no kernel names; (2) always runs.
_ROP_ZERO = {}


def _zero_rows(n: int, like: torch.Tensor) -> torch.Tensor:
    """ROP: n zero rows shaped like `like`'s rows, cached; made by ro_plan before any capture (a block first made inside
    a capture would hold the memset's bytes only once the graph replays)"""
    key = (int(n), int(like.shape[1]), like.dtype, like.device)
    z = _ROP_ZERO.get(key)
    if z is None:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"ROP: no zero block of {n} rows made before this capture")
        z = _ROP_ZERO[key] = torch.zeros((int(n), like.shape[1]), dtype=like.dtype, device=like.device)
    return z


def readout_rows(hn: torch.Tensor, sw_t: torch.Tensor, rows: int) -> torch.Tensor:
    """ROP: the readout of hn's B rows as RO's bmm at `rows` >= B batch entries (rows - B zero rows appended; the first
    B output rows, a contiguous [B, N] view) or, for rows = 0, the per-row products + cat (the reference op)."""
    B = hn.shape[0]
    if rows <= 0:
        return torch.cat([hn[b:b + 1] @ sw_t for b in range(B)])
    if rows == B:
        return readout(hn, sw_t, "bmm")
    if rows < B:
        raise ValueError(f"ROP: {rows} rows for a {B}-row readout")
    return readout(torch.cat([hn, _zero_rows(rows - B, hn)]), sw_t, "bmm")[:B]


def _kshort(name: str) -> str:
    """a GEMV kernel's template ints for the log (gemvx<...,7,...> -> "gemvx<7>"); other kernels by a name prefix"""
    import re

    if "gemvx" in name:
        return "gemvx<" + ",".join(re.findall(r"Li(\d+)E", name)) + ">"
    return name[:48]


def _graph_kernels(fn) -> list:
    """[(name, grid, block, smem)] of the kernel nodes of fn() captured in a CUDA graph (never instantiated) in node
    order, read as KP reads them (kprio: cuGraphKernelNodeGetParams_v2 + cuFuncGetName); None where the driver gives no
    kernel name. fn runs once before the capture (cuBLAS handle and heuristics as in the lane captures)."""
    import ctypes as C

    from . import kprio

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    s.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.stream(s):
        g.capture_begin()
        try:
            fn()
        finally:
            g.capture_end()
    try:
        cu = kprio._cu()
        gr = C.c_void_p(g.raw_cuda_graph())
        n = C.c_size_t(0)
        cu.cuGraphGetNodes(gr, None, C.byref(n))
        nodes = (C.c_void_p * n.value)()
        cu.cuGraphGetNodes(gr, nodes, C.byref(n))
        out = []
        for i in range(n.value):
            t = C.c_int()
            cu.cuGraphNodeGetType(C.c_void_p(nodes[i]), C.byref(t))
            if t.value != kprio.CU_GRAPH_NODE_TYPE_KERNEL:
                continue
            p = kprio._KParams()
            if cu.cuGraphKernelNodeGetParams_v2(C.c_void_p(nodes[i]), C.byref(p)) != 0:
                return None
            name = kprio._name(cu, nodes[i])
            if not name:
                return None
            out.append((name, (p.gridDimX, p.gridDimY, p.gridDimZ), (p.blockDimX, p.blockDimY, p.blockDimZ),
                        int(p.sharedMemBytes)))
        return out
    finally:
        g.reset()


def _gemv_of(kernels):
    """the one GEMV node of a readout's kernels (the pad copy aside), else None"""
    gv = [k for k in kernels or () if "gemv" in k[0].lower()]
    return gv[0] if len(gv) == 1 else None


def _same_gemv(ref, got, P: int) -> bool:
    """the per-row product's GEMV (batch 1) and RO's at P batch entries: the same function, block and shared memory,
    the grid equal but for one dimension holding the batch (ref's extent x P)"""
    if ref is None or got is None or ref[0] != got[0] or ref[2] != got[2] or ref[3] != got[3]:
        return False
    diff = [i for i in range(3) if ref[1][i] != got[1][i]]
    return (not diff and P == 1) or (len(diff) == 1 and got[1][diff[0]] == ref[1][diff[0]] * P)


@torch.inference_mode()
def _rop_rows(slot_weight: torch.Tensor, seed: int, nrand: int, pool: int, tie: float, chunk: int = 8192):
    """ROP gate rows: (random rows at the scales 1, 8, 0.02 and 3; the near-tie rows of `pool` random rows, i.e. rows
    with an exact (fp64) logit within `tie` (relative) of a bf16 rounding midpoint; 16 copies of one row)"""
    dev = slot_weight.device
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    K = slot_weight.shape[1]
    q = max(1, nrand // 4)
    rnd = torch.cat([(torch.randn((q, K), generator=g, device=dev) * sc).to(slot_weight.dtype)
                     for sc in (1.0, 8.0, 0.02, 3.0)])
    wd = slot_weight.double().T
    ties, done, i = [], 0, 0
    while done < pool:
        n = min(chunk, pool - done)
        x = (torch.randn((n, K), generator=g, device=dev) * (1.0, 3.0)[i % 2]).to(slot_weight.dtype)
        e = x.double() @ wd                              # exact: bf16 x bf16 products, 2560 terms, fp64 sum
        _m, ex = torch.frexp(e)
        t = e * torch.pow(2.0, (8 - ex).to(torch.float64))   # |t| in [128, 256): bf16 neighbours 1 apart, midpoints at .5
        near = ((t - torch.floor(t) - 0.5).abs() < tie * t.abs()) & (e != 0)
        ties.append(x[near.any(1)])
        done += n
        i += 1
    return rnd, torch.cat(ties), rnd[:1].expand(16, K).contiguous()


def _ref_rows(H: torch.Tensor, sw_t: torch.Tensor) -> torch.Tensor:
    """the reference op on every row: [1, K] @ W^T per row, as the per-decision graphs and the reference engine run it"""
    return torch.cat([H[i:i + 1] @ sw_t for i in range(H.shape[0])])


def _rop_gate(H: torch.Tensor, REF: torch.Tensor, sw_t: torch.Tensor, B: int, P: int):
    """readout_rows(., P) on H's rows in B-row blocks (wrapping around) vs the reference op's rows ->
    (logits that differ, layout ok)"""
    N = H.shape[0]
    nb = -(-N // B)
    idx = torch.arange(nb * B, device=H.device) % N
    Hb = H.index_select(0, idx).view(nb, B, H.shape[1])
    outs = [readout_rows(Hb[c], sw_t, P) for c in range(nb)]
    want0 = torch.cat([Hb[0][b:b + 1] @ sw_t for b in range(B)])
    o0 = outs[0]
    lay = o0.dtype == want0.dtype and o0.shape == want0.shape and o0.stride() == want0.stride()
    got = torch.cat(outs)
    want = REF.index_select(0, idx)
    return int((got.view(torch.int16) != want.view(torch.int16)).sum()), lay


@torch.inference_mode()
def ro_plan(slot_weight: torch.Tensor, bs, seed: int = 79, nrand: int = 8192, pool: int = 131072, tie: float = 4e-7,
            kernels: bool = True):
    """ROP start-up gate -> ({B: row count P >= B, or 0 = the per-row products + cat}, summary for the log). For each B
    in bs the smallest P in B..max(bs) whose GEMV is the per-row product's (when kernel names are available) and whose
    every logit equals the per-row products byte for byte (dtype, shape, strides too) on the random, near-tie (each at
    two slot positions) and equal-row sets; none -> 0. Every candidate gets both checks; a rejected one is logged with
    what each found."""
    import time

    t0 = time.time()
    sw_t = slot_weight.T
    bs = sorted({int(b) for b in bs if int(b) >= 1})
    pmax = max(bs)
    like = slot_weight[:1]
    for n in range(1, pmax):           # every zero block a padded readout can use, before any capture
        _zero_rows(n, like)
    rnd, tie_rows, eq = _rop_rows(slot_weight, seed, nrand, pool, tie)
    sets = [("random", rnd), ("near-tie", torch.cat([tie_rows, tie_rows.roll(1, 0)])), ("equal", eq)]
    refs = [(nm, H, _ref_rows(H, sw_t)) for nm, H in sets if H.shape[0]]
    kref, knote = None, "kernel identity n/a (not checked)"
    if kernels and slot_weight.is_cuda:
        try:
            ks = _graph_kernels(lambda: rnd[:1] @ sw_t)
            kref = _gemv_of(ks)
            if kref is None:
                knote = "kernel identity n/a (" + ("no kernel names" if ks is None else
                                                   f"per-row product runs {[_kshort(k[0]) for k in ks]}") + ")"
        except Exception as exc:  # noqa: BLE001 - the byte gate still decides
            knote = f"kernel identity n/a ({type(exc).__name__}: {exc})"
    plan, rej = {}, []
    for B in bs:
        plan[B] = 0
        for P in range(B, max(pmax, B) + 1):
            why = []
            if kref is not None:
                try:
                    got = _gemv_of(_graph_kernels(lambda: readout_rows(rnd[:B], sw_t, P)))
                except Exception as exc:  # noqa: BLE001
                    got = None
                    why.append(f"kernel capture: {type(exc).__name__}: {exc}")
                if not _same_gemv(kref, got, P):
                    why.append(f"{_kshort(got[0])} grid {got[1]} block {got[2]}" if got else "no single GEMV node")
            bad, lay_ok = [], True
            for nm, H, REF in refs:
                nd, lay = _rop_gate(H, REF, sw_t, B, P)
                lay_ok = lay_ok and lay
                if nd:
                    bad.append(f"{nd} of {REF.numel()} {nm}")
            if not lay_ok:
                why.append("layout differs")
            if bad:
                why.append(" + ".join(bad) + " logits differ")
            if why:
                rej.append(f"bmm@{P} for B {B}: " + ", ".join(why))
                continue
            plan[B] = P
            break
    own = [B for B in bs if plan[B] == B]
    pad = {B: plan[B] for B in bs if plan[B] > B}
    cat = [B for B in bs if plan[B] == 0]
    kdesc = (f"each count's GEMV = the per-row product's ({_kshort(kref[0])}, block {kref[2][0]}x{kref[2][1]}x"
             f"{kref[2][2]}, grid x P)" if kref is not None else knote)
    summ = (f"RO readout row counts by graph B: padded {pad or '-'}, bmm at its own count for B {own}, per-row products "
            f"for B {cat}; {kdesc}; byte-equal to the per-row products on {rnd.shape[0]} random + {tie_rows.shape[0]} "
            f"near-tie (x2 slot positions; exact logit within {tie:g} of a bf16 midpoint, from {pool} rows) + "
            f"{eq.shape[0]} equal rows" + (f"; rejected: {'; '.join(rej)}" if rej else "") +
            f" ({time.time() - t0:.1f}s)")
    return plan, summ


# ================================================================================================ TG
# TG (S1ENG_TG=1): the pruned last layer's M = B GEMMs (q_proj, gate / up, o_proj, down_proj on B <= 16 rows) as one
# Triton kernel with 16-row tiles instead of the bucket's pinned cuBLASLt algo run at M = B (128-row tiles: 8x the MMA
# work on padding rows; 64-96 CTAs, each streaming a 128-column weight slab at ~10 GB/s). The same per-element
# arithmetic as the pin: a non-split pin (reduction 0) = one mma.sync m16n8k16 chain from +0 over all K in ascending
# k16 order, rounded to bf16 once; a split-K pin with reduction OUTPUT_TYPE (4) = per K slice a chain from +0, the
# partial rounded to bf16, the slices folded in order in fp32 (fold = P0, then + P1, ...), bf16 at the end (BA's
# scheme). The slice length of a split pin is found by the gate among roundup(ceil(K / S), g) for g = 64, 128, 32, 16,
# 256: the first candidate byte-equal to the pinned path at every B is kept. Other reductions (COMPUTE_TYPE, as BA's
# bucket 320) are not reproduced: those shapes keep the pin. Row-independent arithmetic: any B, any tile, same bytes.
@triton.jit
def _tg_kernel(x, w, y, B, XS, YS, K: tl.constexpr, SL: tl.constexpr, NS: tl.constexpr, SPLIT: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, BM)
    rm = rows < B
    xr = x + rows[:, None].to(tl.int64) * XS
    wc = w + cols[None, :].to(tl.int64) * K
    kk = tl.arange(0, BK)
    # slice 0: its chain from +0 (P0 kept as the fold's start: no +0 + P0, which would turn -0 into +0)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, SL, BK):
        a = tl.load(xr + (k + kk)[None, :], mask=rm[:, None], other=0.0)
        b = tl.load(wc + (k + kk)[:, None])
        acc = tl.dot(a, b, acc)
    if SPLIT:
        fold = acc.to(tl.bfloat16).to(tl.float32)
        for s in range(1, NS):
            k0 = s * SL
            k1 = tl.minimum(k0 + SL, K)
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for k in range(k0, k1, BK):
                a = tl.load(xr + (k + kk)[None, :], mask=rm[:, None], other=0.0)
                b = tl.load(wc + (k + kk)[:, None])
                acc = tl.dot(a, b, acc)
            fold = fold + acc.to(tl.bfloat16).to(tl.float32)
    else:
        fold = acc
    tl.store(y + rows[:, None].to(tl.int64) * YS + cols[None, :], fold.to(tl.bfloat16), mask=rm[:, None])


TG_BM = 16


def tg_cfg_env():
    v = os.environ.get("S1ENG_TG_CFG", "64,4,4")
    bn, warps, stages = (int(t) for t in v.split(","))
    return bn, warps, stages


def tg_gemm(x: torch.Tensor, w: torch.Tensor, mode, kcfg=None) -> torch.Tensor:
    """x [B, K] (B <= 16, unit column stride), w [N, K] contiguous -> y [B, N] bf16 = the pinned GEMM's bytes.
    mode = (slice length SL, slices NS, split, BK)."""
    B, K = x.shape
    N = w.shape[0]
    sl, ns, split, bk = mode
    bn, warps, stages = kcfg or tg_cfg_env()
    assert B <= TG_BM and x.stride(1) == 1 and w.is_contiguous() and w.shape[1] == K and N % bn == 0
    y = torch.empty((B, N), dtype=x.dtype, device=x.device)
    _tg_kernel[(N // bn,)](x, w, y, B, x.stride(0), y.stride(0), K=K, SL=sl, NS=ns, SPLIT=split, BM=TG_BM, BN=bn,
                           BK=bk, num_warps=warps, num_stages=stages)
    return y


def tg_modes(cfg: dict, K: int) -> list:
    """candidate modes for a pin's config (cfg_of): [] if its reduction is not reproduced"""
    S, red = int(cfg.get("splitK", 1)), int(cfg.get("reduction", 0))
    if S <= 1:
        return [(K, 1, False, 64 if K % 64 == 0 else 32 if K % 32 == 0 else 16)]
    if red != 4:
        return []
    out = []
    for g in (64, 128, 32, 16, 256):
        sl = -(-(-(-K // S)) // g) * g
        ns = -(-K // sl)
        last = K - (ns - 1) * sl
        for bk in (64, 32, 16):
            if sl % bk == 0 and last % bk == 0:
                m = (sl, ns, ns > 1, bk)
                if m not in out:
                    out.append(m)
                break
    return out


@torch.inference_mode()
def tg_check(gemm, W: torch.Tensor, modes, bs, seed: int = 83):
    """TG start-up gate of one (N, K, bucket): the first mode whose tg_gemm equals gemm(W, x) (the lanes' pinned path
    at M = B) byte for byte at every B in bs, three input scales each and a repeated-row block -> (mode, "") or
    (None, why)."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed + W.shape[0] + W.shape[1])
    K = W.shape[1]
    notes = []
    for mode in modes:
        why = ""
        for B in sorted(set(bs)):
            for trial, sc in enumerate((1.0, 6.0, 0.05, 1.0)):
                x = (torch.randn((B, K), generator=g, device="cuda") * sc).to(W.dtype)
                if trial == 3:
                    x[:] = x[:1]
                ref = gemm(W, x)
                got = tg_gemm(x, W, mode)
                if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                    why = f"{int((got.view(torch.int16) != ref.view(torch.int16)).sum())} of {ref.numel()} differ " \
                          f"(B={B}, trial {trial})"
                    break
            if why:
                break
        if not why:
            return mode, ""
        notes.append(f"{mode[:2]}: {why}")
    return None, "; ".join(notes) or "no reproducible mode for this pin"


# TGS (S1ENG_TGS=1, with the TG kernel family): the last layer's MLP at M = B - gate_proj and up_proj (non-split
# pins: one chain each) and SwiGLU (fused.silu_mul: s = bf16(silu_f32(bf16 g)), out = bf16(s * bf16 u)) - in ONE
# launch: each CTA streams the gate and up slabs of its columns and applies the SwiGLU to its own outputs (row- and
# element-independent, the same expressions under the same EXACT flags), instead of 2 GEMM launches on 2 streams +
# silu_mul. Gate tgs_check: byte-equal to silu_mul(gemm(Wg, x), gemm(Wu, x)) (the PAR path) at every packed B.
from .fused import _silu_f32  # noqa: E402


@triton.jit
def _tgs_kernel(x, wg, wu, y, B, XS, YS, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, BM)
    rm = rows < B
    xr = x + rows[:, None].to(tl.int64) * XS
    wgc = wg + cols[None, :].to(tl.int64) * K
    wuc = wu + cols[None, :].to(tl.int64) * K
    kk = tl.arange(0, BK)
    ag = tl.zeros((BM, BN), dtype=tl.float32)
    au = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(xr + (k + kk)[None, :], mask=rm[:, None], other=0.0)
        ag = tl.dot(a, tl.load(wgc + (k + kk)[:, None]), ag)
        au = tl.dot(a, tl.load(wuc + (k + kk)[:, None]), au)
    gf = ag.to(tl.bfloat16).to(tl.float32)
    uf = au.to(tl.bfloat16).to(tl.float32)
    s = _silu_f32(gf).to(tl.bfloat16).to(tl.float32)
    tl.store(y + rows[:, None].to(tl.int64) * YS + cols[None, :], (s * uf).to(tl.bfloat16), mask=rm[:, None])


def tgs_cfg_env():
    v = os.environ.get("S1ENG_TGS_CFG", "64,4,4")
    bn, warps, stages = (int(t) for t in v.split(","))
    return bn, warps, stages


def tgs_mlp(x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor, bk: int = 64, kcfg=None) -> torch.Tensor:
    """x [B, K] (B <= 16) -> silu_mul(x @ wg^T, x @ wu^T) [B, N] bf16 with the non-split pins' bytes"""
    from .fused import EXACT

    B, K = x.shape
    N = wg.shape[0]
    bn, warps, stages = kcfg or tgs_cfg_env()
    assert B <= TG_BM and x.stride(1) == 1 and wg.is_contiguous() and wu.is_contiguous() and wu.shape == wg.shape
    assert wg.shape[1] == K and N % bn == 0 and K % bk == 0
    y = torch.empty((B, N), dtype=x.dtype, device=x.device)
    _tgs_kernel[(N // bn,)](x, wg, wu, y, B, x.stride(0), y.stride(0), K=K, BM=TG_BM, BN=bn, BK=bk, num_warps=warps,
                            num_stages=stages, **EXACT)
    return y


@torch.inference_mode()
def tgs_check(gemm, silu_mul, wg: torch.Tensor, wu: torch.Tensor, bs, seed: int = 89) -> str:
    """TGS start-up gate of one bucket: tgs_mlp == silu_mul(gemm(wg, x), gemm(wu, x)) byte for byte at every B in bs,
    four inputs each (scales 1, 6, 0.05 and a repeated-row block). "" if equal."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    K = wg.shape[1]
    for B in sorted(set(bs)):
        for trial, sc in enumerate((1.0, 6.0, 0.05, 1.0)):
            x = (torch.randn((B, K), generator=g, device="cuda") * sc).to(wg.dtype)
            if trial == 3:
                x[:] = x[:1]
            ref = silu_mul(gemm(wg, x), gemm(wu, x))
            got = tgs_mlp(x, wg, wu)
            if ref.shape != got.shape or ref.stride() != got.stride():
                return f"layout differs (B={B})"
            if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                return f"{int((got.view(torch.int16) != ref.view(torch.int16)).sum())} of {ref.numel()} differ " \
                       f"(B={B}, trial {trial})"
    return ""


# ================================================================================================ XL
# XL (S1ENG_XL=1, needs v15's XA kernel, LR1 + GQA): the pruned last layer's one-row-per-slot attention through XA's
# computed-rows kernel (xa.attn_q: q norm + RoPE, the masked-SDPA arithmetic and the sigmoid gate in one launch) with
# exactly one computed row per slot (start = slot, plen = slen - 1) instead of q_norm_rope_last + the bias rows + the
# efficient SDPA with the bias + attn_gate_grouped. Both replay the reference's masked SDPA row n - 1 (LR1 verified
# it for the efficient call, check_xa for XA); the gate xl_check compares XL with the LR1 + GQA chain byte for byte.
@torch.inference_mode()
def xl_check(pf, d, cos, sin, eps: float, mask, L: int, bs, seed: int = 97) -> str:
    """XL start-up gate of bucket L: xa.attn_q on one computed row per slot == attn_gate_grouped(sdpa_grouped(
    q_norm_rope_last(...), bias rows)) (the LR1 + GQA path) at every B in bs, random K / V (zeros past n), an unused
    slot, scales 0.3 / 1 / 4. "" if every byte agrees."""
    from . import xa

    fz = pf.fz
    sa = d["sa"]
    hd = sa.head_dim
    H = sa.q_proj.weight.shape[0] // (2 * hd)
    HK = sa.k_proj.weight.shape[0] // hd
    if not xa.supported(H, HK, hd, L):
        return "XA kernel unavailable or unsupported shape"
    g = torch.Generator(device="cuda")
    g.manual_seed(seed + L)
    lprev = max([x for x in (0, 128, 192, 256, 320) if x < L])
    for B in sorted(set(bs)):
        ar = torch.arange(B, dtype=torch.int32, device="cuda")
        for trial, sc in enumerate((1.0, 0.3, 4.0)):
            n = torch.randint(lprev + 1, L + 1, (B,), generator=g, device="cuda")
            if trial == 2 and B > 1:
                n[-1] = 0
            slen = n.to(torch.int32)
            qg = (torch.randn((B, H, 2 * hd), generator=g, device="cuda") * sc * 2).to(torch.bfloat16)
            keep = (torch.arange(L, device="cuda")[None, :] < n[:, None]).to(torch.bfloat16)[:, None, :, None]
            k4 = ((torch.randn((B, HK, L, hd), generator=g, device="cuda") * sc).to(torch.bfloat16) * keep).contiguous()
            v4 = ((torch.randn((B, HK, L, hd), generator=g, device="cuda") * 2).to(torch.bfloat16) * keep).contiguous()
            q1 = fz.q_norm_rope_last(qg, d["qn_w"], cos, sin, eps, slen, B)
            bias = mask[0, 0].index_select(0, (n.long() - 1).clamp_(min=0)).view(B, 1, 1, L)
            o_g = pf.sdpa_grouped(q1, k4, v4, mask, sa.scaling, L, bias=bias)
            ref = fz.attn_gate_grouped(o_g, qg, ar, 1, HK)
            plo = (n - 1).clamp(min=0).to(torch.int32)
            got = xa.attn_q(qg, d["qn_w"], cos, sin, eps, k4, v4, ar, slen, plo, L, sa.scaling, gtab=pf.xa_gtab)
            if ref.shape != got.shape or ref.stride() != got.stride():
                return f"layout differs (B={B})"
            if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                nd = int((got.view(torch.int16) != ref.view(torch.int16)).sum())
                return f"{nd} of {ref.numel()} differ (B={B}, trial {trial})"
            # XA3 (kai/rf-xa): the variant being gated (pf.xa_want) on the same inputs, byte for byte against
            # xa_fwd2q's output; a miss turns only the variant off for XL at this bucket
            xa.variant_check(pf, "xl", L, got, lambda v, vo: xa.attn_q(qg, d["qn_w"], cos, sin, eps, k4, v4, ar, slen,
                                                                    plo, L, sa.scaling, gtab=pf.xa_gtab, ver=v, out=vo))
    if getattr(pf, "xa_want", None) and ("xl", L) not in pf.xa_vbad:
        # XA3: outlier and order-witness inputs (after the checks above: their random draws are unchanged)
        for B in sorted(set(bs)):
            ar = torch.arange(B, dtype=torch.int32, device="cuda")
            for kind in ("outlier", "witness"):
                n = torch.randint(lprev + 1, L + 1, (B,), generator=g, device="cuda")
                if B > 1:
                    n[-1] = 0
                slen = n.to(torch.int32)
                qg, k4, v4 = xa.witness_inputs(kind, g, B, B, L, H, HK, hd, slen, device="cuda")
                plo = (n - 1).clamp(min=0).to(torch.int32)
                base = xa.attn_q(qg, d["qn_w"], cos, sin, eps, k4, v4, ar, slen, plo, L, sa.scaling, gtab=pf.xa_gtab)
                xa.variant_check(pf, "xl", L, base, lambda v, vo: xa.attn_q(qg, d["qn_w"], cos, sin, eps, k4, v4, ar,
                                                                         slen, plo, L, sa.scaling, gtab=pf.xa_gtab,
                                                                         ver=v, out=vo))
    return ""
