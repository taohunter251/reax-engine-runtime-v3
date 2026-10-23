"""Fused Triton kernels that reproduce the reference's PyTorch op chains bit for bit.

Rules every kernel follows:
  * every rounding point of the reference stays (bf16 casts where PyTorch writes a bf16 tensor, f32 elsewhere);
  * no FMA contraction (launch option enable_fp_fusion=False), no ftz in libdevice (enable_reflect_ftz=False);
  * transcendental functions through libdevice (expf, log1pf), divisions with div_rn, as nvcc compiles PyTorch;
  * reductions replay the exact addition order of PyTorch's reduce_kernel for that shape (documented per kernel).
Each kernel has a bitwise unit test against the torch op chain (tests/unit_fused.py).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

EXACT = {"enable_fp_fusion": False, "enable_reflect_ftz": False}
D_MODEL = 2560
INV_D = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(float(D_MODEL), dtype=torch.float32))


# --------------------------------------------------------------------------------------------- RMSNorm (D=2560)
# Reference: h = x.float(); h * rsqrt(h.pow(2).mean(-1, keepdim=True) + eps); (h * (1+w)).type_as(x)
# mean over 2560 f32 = reduce_kernel<512,1> with input vectorisation 4: one warp per row; lane t keeps 4 running sums
# s_i(t) = sum_{j=0..19} sq[128 j + 4 t + i] (j ascending), lane value ((s0+s1)+s2)+s3, then a shfl_down butterfly
# with offsets 16,8,4,2,1 (verified by tools/reduce_order.py); mean = sum * fl(1/2560).
@triton.jit
def _halve(v, H: tl.constexpr):
    # v[0:H] + v[H:2H]  (one shfl_down step with offset H)
    p, q = tl.split(tl.permute(tl.reshape(v, (2, H)), (1, 0)))
    return p + q


@triton.jit
def _warp_tree32(v):
    # reduce_kernel's warp reduction: shfl_down with offsets 16, 8, 4, 2, 1 (lane 0 keeps the result)
    v = _halve(v, 16)
    v = _halve(v, 8)
    v = _halve(v, 4)
    v = _halve(v, 2)
    return tl.sum(v, axis=0)  # two elements: one addition


@triton.jit
def _cols4_tree(acc):
    # acc [32, 4] -> per lane ((c0 + c1) + c2) + c3
    a, b = tl.split(tl.reshape(acc, (32, 2, 2)))       # a: cols 0,2  b: cols 1,3
    a0, a2 = tl.split(a)
    b1, b3 = tl.split(b)
    return ((a0 + b1) + a2) + b3


@triton.jit
def _rmsnorm_kernel(x, y, hout, out, w, eps, inv_d, HAS_Y: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    xr = x + row * 2560
    yr = y + row * 2560
    # pass 1: loads only (no stores in between, so every load can be in flight at once), exact summation order
    offs = tl.arange(0, 32)[:, None] * 4 + tl.arange(0, 4)[None, :]     # [32 lanes, 4 vector elements]
    acc = tl.zeros((32, 4), tl.float32)
    for j in tl.static_range(20):
        v = tl.load(xr + j * 128 + offs).to(tl.float32)
        if HAS_Y:
            v = (v + tl.load(yr + j * 128 + offs).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        acc = acc + v * v
    r = libdevice.rsqrt(_warp_tree32(_cols4_tree(acc)) * inv_d + eps)
    # pass 2: one 2-D tile over the whole row (rows j >= 20 masked)
    jj = tl.arange(0, 32)[:, None]
    cc = tl.arange(0, 128)[None, :]
    o2 = jj * 128 + cc
    m2 = jj < 20
    v = tl.load(xr + o2, mask=m2, other=0.0).to(tl.float32)
    if HAS_Y:
        hb = (v + tl.load(yr + o2, mask=m2, other=0.0).to(tl.float32)).to(tl.bfloat16)
        tl.store(hout + row * 2560 + o2, hb, mask=m2)
        v = hb.to(tl.float32)
    wv = tl.load(w + o2, mask=m2, other=0.0)
    tl.store(out + row * 2560 + o2, ((v * r) * wv).to(tl.bfloat16), mask=m2)


RMS_WARPS = 4


def rmsnorm(x: torch.Tensor, w1p: torch.Tensor, eps: float) -> torch.Tensor:
    assert x.is_contiguous() and x.shape[-1] == D_MODEL
    rows = x.numel() // D_MODEL
    out = torch.empty_like(x)
    _rmsnorm_kernel[(rows,)](x, x, x, out, w1p, eps, INV_D, HAS_Y=False, num_warps=RMS_WARPS, **EXACT)
    return out


def add_rmsnorm(h: torch.Tensor, y: torch.Tensor, w1p: torch.Tensor, eps: float):
    """(h + y in bf16, rmsnorm of it)."""
    assert h.is_contiguous() and y.is_contiguous() and h.shape == y.shape
    rows = h.numel() // D_MODEL
    hn = torch.empty_like(h)
    out = torch.empty_like(h)
    _rmsnorm_kernel[(rows,)](h, y, hn, out, w1p, eps, INV_D, HAS_Y=True, num_warps=RMS_WARPS, **EXACT)
    return hn, out


# --------------------------------------------------------------------------------------------- batched reduction trees
@triton.jit
def _halve_b(v, R: tl.constexpr, H: tl.constexpr):
    # v [R, 2H] -> v[:, :H] + v[:, H:]
    p, q = tl.split(tl.permute(tl.reshape(v, (R, 2, H)), (0, 2, 1)))
    return p + q


@triton.jit
def _warp_tree32_b(v, R: tl.constexpr):
    # [R, 32] -> [R]; shfl_down offsets 16, 8, 4, 2, 1
    v = _halve_b(v, R, 16)
    v = _halve_b(v, R, 8)
    v = _halve_b(v, R, 4)
    v = _halve_b(v, R, 2)
    p, q = tl.split(v)
    return p + q


@triton.jit
def _cols4_tree_b(acc, R: tl.constexpr):
    # acc [R, 32, 4] -> [R, 32]: ((c0 + c1) + c2) + c3
    a, b = tl.split(tl.reshape(acc, (R, 32, 2, 2)))
    a0, a2 = tl.split(a)
    b1, b3 = tl.split(b)
    return ((a0 + b1) + a2) + b3


@triton.jit
def _silu_f32(x):
    # PyTorch silu_kernel (opmath f32): x / (1 + exp(-x)), expf from libdevice, IEEE division
    return libdevice.div_rn(x, 1.0 + libdevice.exp(-x))


@triton.jit
def _sigmoid_f32(x):
    # PyTorch sigmoid_kernel_cuda: 1 / (1 + exp(-x))
    return libdevice.div_rn(1.0, 1.0 + libdevice.exp(-x))


# --------------------------------------------------------------------------------------------- SwiGLU: silu(g) * u
# Reference: F.silu(gate_proj(x)) -> bf16; * up_proj(x) -> bf16.
@triton.jit
def _silu_mul_kernel(g, u, out, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    gf = tl.load(g + offs, mask=m, other=0.0).to(tl.float32)
    uf = tl.load(u + offs, mask=m, other=0.0).to(tl.float32)
    s = _silu_f32(gf).to(tl.bfloat16).to(tl.float32)
    tl.store(out + offs, (s * uf).to(tl.bfloat16), mask=m)


def silu_mul(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    if not (g.is_contiguous() and u.is_contiguous()):
        return silu_mul_strided(g, u)
    assert g.shape == u.shape
    out = torch.empty_like(g)
    n = g.numel()
    BLOCK = 2048
    _silu_mul_kernel[(triton.cdiv(n, BLOCK),)](g, u, out, n, BLOCK=BLOCK, num_warps=4, **EXACT)
    return out


# Same op on row-strided 2-D inputs (the gate / up column blocks of one fused [R, 2N] GEMM output); out contiguous.
@triton.jit
def _silu_mul_strided_kernel(g, u, out, N, NC: tl.constexpr, GS, US, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    row = offs // NC
    col = offs % NC
    gf = tl.load(g + row * GS + col, mask=m, other=0.0).to(tl.float32)
    uf = tl.load(u + row * US + col, mask=m, other=0.0).to(tl.float32)
    s = _silu_f32(gf).to(tl.bfloat16).to(tl.float32)
    tl.store(out + offs, (s * uf).to(tl.bfloat16), mask=m)


def silu_mul_strided(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    assert g.dim() == 2 and g.shape == u.shape and g.stride(1) == 1 and u.stride(1) == 1
    R, NC = g.shape
    out = torch.empty((R, NC), dtype=g.dtype, device=g.device)
    n = R * NC
    BLOCK = 2048
    _silu_mul_strided_kernel[(triton.cdiv(n, BLOCK),)](g, u, out, n, NC=NC, GS=g.stride(0), US=u.stride(0), BLOCK=BLOCK,
                                                       num_warps=4, **EXACT)
    return out


# --------------------------------------------------------------------------------------------- attention output gate
# Reference: gate = chunk(q_proj(x).view(B,L,H,2*hd), 2)[1].reshape(B,L,H*hd) (copy); o = sdpa_out.transpose(1,2)
# .contiguous().reshape(B,L,H*hd); o * torch.sigmoid(gate) (sigmoid -> bf16, product -> bf16).
# Here: o_bhld = SDPA output [B, H, L, hd] (any strides), qg = q_proj output [B, L, H, 2*hd]; out [B, L, H*hd].
@triton.jit
def _attn_gate_kernel(o, qg, out, L, H: tl.constexpr, HD: tl.constexpr,
                      so_h, so_l, so_b, BL: tl.constexpr):
    pid_l = tl.program_id(0)
    h = tl.program_id(1)
    bb = tl.program_id(2).to(tl.int64)
    o = o + bb * so_b
    qg = qg + bb * L * (H * 2 * HD)
    out = out + bb * L * (H * HD)
    ls = pid_l * BL + tl.arange(0, BL)
    d = tl.arange(0, HD)
    m = (ls < L)[:, None]
    ov = tl.load(o + h * so_h + ls[:, None] * so_l + d[None, :], mask=m, other=0.0).to(tl.float32)
    gv = tl.load(qg + ls[:, None] * (H * 2 * HD) + h * (2 * HD) + HD + d[None, :], mask=m, other=0.0).to(tl.float32)
    s = _sigmoid_f32(gv).to(tl.bfloat16).to(tl.float32)
    tl.store(out + ls[:, None] * (H * HD) + h * HD + d[None, :], (ov * s).to(tl.bfloat16), mask=m)


def attn_gate(o_bhld: torch.Tensor, qg: torch.Tensor) -> torch.Tensor:
    B, H, L, HD = o_bhld.shape
    assert qg.is_contiguous() and qg.shape == (B, L, H, 2 * HD) and o_bhld.stride(3) == 1
    out = torch.empty((B, L, H * HD), dtype=o_bhld.dtype, device=o_bhld.device)
    BL = 16
    _attn_gate_kernel[(triton.cdiv(L, BL), H, B)](o_bhld, qg, out, L, H=H, HD=HD, so_h=o_bhld.stride(1),
                                                   so_l=o_bhld.stride(2), so_b=o_bhld.stride(0), BL=BL, num_warps=4,
                                                   **EXACT)
    return out


# --------------------------------------------------------------------------------------------- GDN gate glue
# Reference: beta = b.sigmoid() (bf16); g = -A_log.float().exp() * F.softplus(a.float() + dt_bias) (f32)
# softplus (beta=1, threshold=20): t > 20 ? t : log1p(exp(t)) / 1
@triton.jit
def _gdn_gate_kernel(b, a, negexpA, dt_bias, beta_out, g_out, N, NH: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    hh = offs % NH
    bf = tl.load(b + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(beta_out + offs, _sigmoid_f32(bf).to(tl.bfloat16), mask=m)
    af = tl.load(a + offs, mask=m, other=0.0).to(tl.float32)
    t = af + tl.load(dt_bias + hh, mask=m, other=0.0).to(tl.float32)
    sp = tl.where(t > 20.0, t, libdevice.div_rn(libdevice.log1p(libdevice.exp(t)), 1.0))
    tl.store(g_out + offs, tl.load(negexpA + hh, mask=m, other=0.0) * sp, mask=m)


def gdn_gate(b: torch.Tensor, a: torch.Tensor, negexpA: torch.Tensor, dt_bias: torch.Tensor):
    assert b.is_contiguous() and a.is_contiguous() and b.shape == a.shape
    beta = torch.empty_like(b)
    g = torch.empty(a.shape, dtype=torch.float32, device=a.device)
    n = b.numel()
    NH = b.shape[-1]
    BLOCK = 1024
    _gdn_gate_kernel[(triton.cdiv(n, BLOCK),)](b, a, negexpA, dt_bias, beta, g, n, NH=NH, BLOCK=BLOCK, num_warps=4,
                                               **EXACT)
    return beta, g


# --------------------------------------------------------------------------------------------- gated RMSNorm (D=128)
# Reference (Qwen3_5RMSNormGated): h = o.float(); h * rsqrt(h.pow(2).mean(-1) + eps) -> bf16; w * that (bf16);
# * silu(z.float()) (f32) -> bf16.  mean over 128: lane t value ((s0+s1)+s2)+s3 of sq[4t..4t+3], warp tree.
@triton.jit
def _gated_norm_kernel(o, z, w, out, N, eps, inv_d, R: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    m = (rows < N)[:, None, None]
    offs = rows[:, None, None].to(tl.int64) * 128 + (tl.arange(0, 32)[None, :, None] * 4 + tl.arange(0, 4)[None, None, :])
    of = tl.load(o + offs, mask=m, other=0.0).to(tl.float32)
    s = _warp_tree32_b(_cols4_tree_b(of * of, R), R)
    r = libdevice.rsqrt(s * inv_d + eps)
    hn = (of * r[:, None, None]).to(tl.bfloat16).to(tl.float32)
    wv = tl.load(w + (tl.arange(0, 32)[:, None] * 4 + tl.arange(0, 4)[None, :])).to(tl.float32)
    hw = (wv[None, :, :] * hn).to(tl.bfloat16).to(tl.float32)
    zf = tl.load(z + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(out + offs, (hw * _silu_f32(zf)).to(tl.bfloat16), mask=m)


INV_128 = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(128.0, dtype=torch.float32))


def gated_rmsnorm(o: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    assert o.is_contiguous() and z.is_contiguous() and o.shape == z.shape and o.shape[-1] == 128
    out = torch.empty_like(o)
    n = o.numel() // 128
    R = 8
    _gated_norm_kernel[(triton.cdiv(n, R),)](o, z, w, out, n, eps, INV_128, R=R, num_warps=4, **EXACT)
    return out


# --------------------------------------------------------------------------------------------- causal conv + SiLU
# Reference: F.conv1d(x[B,C,L] (transposed view of in_proj_qkv output), w[C,1,4], padding=3, groups=C)[..., :L]
# (conv_depthwise2d kernel: acc=0; for k in 0..3 if in range: acc += w[c,k]*x[t-3+k] in f32; bf16 out) -> F.silu (bf16)
# -> split q/k/v, q/k repeat_interleave(2, dim=heads), v made contiguous by fla's input_guard.
# Here: x = in_proj_qkv output [L, C] (no transposes), outputs q, k [L, 32, 128] (heads repeated) and v [L, 32, 128].
@triton.jit
def _conv_silu_kernel(x, w, q, k, v, L, C: tl.constexpr, KD: tl.constexpr, BL: tl.constexpr, BC: tl.constexpr):
    pid_l = tl.program_id(0)
    pid_c = tl.program_id(1)
    bb = tl.program_id(2).to(tl.int64)       # decision (sequence) index: each sequence is its own causal conv
    x = x + bb * L * C
    q = q + bb * L * (KD * 2)
    k = k + bb * L * (KD * 2)
    v = v + bb * L * (C - 2 * KD)
    ls = pid_l * BL + tl.arange(0, BL)
    cs = pid_c * BC + tl.arange(0, BC)
    lm = ls < L
    acc = tl.zeros((BL, BC), tl.float32)
    for kk in tl.static_range(4):
        src = ls - 3 + kk
        mk = (src >= 0) & lm
        xv = tl.load(x + src[:, None].to(tl.int64) * C + cs[None, :], mask=mk[:, None], other=0.0).to(tl.float32)
        wv = tl.load(w + cs * 4 + kk).to(tl.float32)
        # taps with src < 0 are skipped by the reference; adding +0.0 (from the masked load) leaves acc unchanged
        # except that a -0.0 accumulator would become +0.0: acc starts at +0.0 and stays +0.0 until a real tap.
        acc = acc + wv[None, :] * xv
    cb = acc.to(tl.bfloat16).to(tl.float32)
    y = _silu_f32(cb).to(tl.bfloat16)
    c0 = pid_c * BC
    lrow = ls[:, None].to(tl.int64)
    m2 = lm[:, None]
    if c0 < KD:                      # q channels: head h = c // 128 -> heads 2h, 2h+1 of [L, 32, 128]
        hc = cs[None, :] // 128
        dc = cs[None, :] % 128
        base = q + lrow * (KD * 2) + (2 * hc) * 128 + dc
        tl.store(base, y, mask=m2)
        tl.store(base + 128, y, mask=m2)
    elif c0 < 2 * KD:
        c2 = cs[None, :] - KD
        hc = c2 // 128
        dc = c2 % 128
        base = k + lrow * (KD * 2) + (2 * hc) * 128 + dc
        tl.store(base, y, mask=m2)
        tl.store(base + 128, y, mask=m2)
    else:
        tl.store(v + lrow * (C - 2 * KD) + (cs[None, :] - 2 * KD), y, mask=m2)


def conv_silu_split(mixed: torch.Tensor, w: torch.Tensor, key_dim: int, value_dim: int, rep: int, hk: int, hv: int):
    B, L, C = mixed.shape
    assert mixed.is_contiguous() and C == 2 * key_dim + value_dim and rep == 2 and hk == 128 and hv == 128
    assert w.shape == (C, 4) and w.is_contiguous()
    q = torch.empty((B, L, 2 * key_dim // hk, hk), dtype=mixed.dtype, device=mixed.device)
    k = torch.empty_like(q)
    v = torch.empty((B, L, value_dim // hv, hv), dtype=mixed.dtype, device=mixed.device)
    BL, BC = 16, 256
    _conv_silu_kernel[(triton.cdiv(L, BL), C // BC, B)](mixed, w, q, k, v, L, C=C, KD=key_dim, BL=BL, BC=BC,
                                                      num_warps=4, **EXACT)
    return q, k, v


# --------------------------------------------------------------------------------------------- q/k norm + RoPE (+ GQA repeat)
# Reference (attention layers): q = chunk(q_proj(x).view(B,L,H,2*hd), 2)[0]; q_norm (zero-centred RMSNorm over hd=256,
# f32, mean = vec4_32 tree with two 128-blocks) -> bf16 -> transpose(1,2); same for k (k_proj view [B,L,Hk,hd]).
# RoPE on the first 64 dims: bf16(bf16(x*cos) + bf16(rotate_half(x)*sin)), rotate_half = cat(-x[32:64], x[0:32]);
# torch.cat with the pass-through dims -> contiguous [B,H,L,hd]; k and v then repeat_kv'd to H heads (contiguous).
INV_256 = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(256.0, dtype=torch.float32))


@triton.jit
def _norm_rope_rows(src, rows_off, w1p, cosr, sinr, eps, inv_d, BL: tl.constexpr, csrc, crows_off, use_c, use_x,
                    PFX: tl.constexpr):
    # src rows (hd=256) at element offsets rows_off [BL]; returns (first 128 dims after RoPE, last 128 dims) as bf16.
    # PFX > 0 (E1): rows where use_c [BL] come from csrc at crows_off instead (prefix cache).
    lane = tl.arange(0, 32)[:, None] * 4 + tl.arange(0, 4)[None, :]           # [32, 4] -> dim index 0..127
    o0 = rows_off[:, None, None] + lane[None, :, :]
    if PFX > 0:
        uc = use_c[:, None, None]
        ux = use_x[:, None, None]
        c0 = crows_off[:, None, None] + lane[None, :, :]
        x0 = _ld_rows(src, o0, csrc, c0, uc, ux, PFX).to(tl.float32)
        x1 = _ld_rows(src, o0 + 128, csrc, c0 + 128, uc, ux, PFX).to(tl.float32)
    else:
        x0 = tl.load(src + o0).to(tl.float32)
        x1 = tl.load(src + o0 + 128).to(tl.float32)
    s = _warp_tree32_b(_cols4_tree_b(x0 * x0 + x1 * x1, BL), BL)
    r = libdevice.rsqrt(s * inv_d + eps)[:, None, None]
    w0 = tl.load(w1p + lane)[None, :, :]
    w1 = tl.load(w1p + 128 + lane)[None, :, :]
    n0 = ((x0 * r) * w0).to(tl.bfloat16)
    n1 = ((x1 * r) * w1).to(tl.bfloat16)
    # partner dims for rotate_half on d < 64
    part = tl.where(lane < 32, lane + 32, tl.where(lane < 64, lane - 32, lane))
    if PFX > 0:
        xp = _ld_rows(src, rows_off[:, None, None] + part[None, :, :], csrc, crows_off[:, None, None] + part[None, :, :],
                      uc, ux, PFX).to(tl.float32)
    else:
        xp = tl.load(src + rows_off[:, None, None] + part[None, :, :]).to(tl.float32)
    wp = tl.load(w1p + part)[None, :, :]
    npf = ((xp * r) * wp).to(tl.bfloat16).to(tl.float32)
    rh = tl.where((lane < 32)[None, :, :], -npf, npf)
    rmask = (lane < 64)[None, :, :]
    c = tl.load(cosr[:, None, None] + lane[None, :, :], mask=rmask, other=0.0).to(tl.float32)
    sn = tl.load(sinr[:, None, None] + lane[None, :, :], mask=rmask, other=0.0).to(tl.float32)
    a = (n0.to(tl.float32) * c).to(tl.bfloat16).to(tl.float32)
    b = (rh * sn).to(tl.bfloat16).to(tl.float32)
    e = (a + b).to(tl.bfloat16)
    return tl.where(rmask, e, n0), n1


@triton.jit
def _q_norm_rope_kernel(qg, w1p, cos, sin, out, L, eps, inv_d, H: tl.constexpr, BL: tl.constexpr):
    pid_l = tl.program_id(0)
    h = tl.program_id(1)
    bb = tl.program_id(2).to(tl.int64)
    qg = qg + bb * L * (H * 512)
    out = out + bb * H * L * 256
    ls = pid_l * BL + tl.arange(0, BL)
    lsm = tl.minimum(ls, L - 1)
    rows_off = lsm.to(tl.int64) * (H * 512) + h * 512
    f0, f1 = _norm_rope_rows(qg, rows_off, w1p, cos + lsm * 64, sin + lsm * 64, eps, inv_d, BL, qg, rows_off,
                                 rows_off, rows_off, 0)
    lane = tl.arange(0, 32)[:, None] * 4 + tl.arange(0, 4)[None, :]
    dst = out + (h * L + ls[:, None, None]).to(tl.int64) * 256 + lane[None, :, :]
    m = (ls < L)[:, None, None]
    tl.store(dst, f0, mask=m)
    tl.store(dst + 128, f1, mask=m)


@triton.jit
def _kv_norm_rope_kernel(kr, vr, w1p, cos, sin, kout, vout, L, eps, inv_d, HK: tl.constexpr, REP: tl.constexpr,
                         BL: tl.constexpr):
    pid_l = tl.program_id(0)
    h = tl.program_id(1)
    bb = tl.program_id(2).to(tl.int64)
    kr = kr + bb * L * (HK * 256)
    vr = vr + bb * L * (HK * 256)
    kout = kout + bb * (HK * REP) * L * 256
    vout = vout + bb * (HK * REP) * L * 256
    ls = pid_l * BL + tl.arange(0, BL)
    lsm = tl.minimum(ls, L - 1)
    rows_off = lsm.to(tl.int64) * (HK * 256) + h * 256
    f0, f1 = _norm_rope_rows(kr, rows_off, w1p, cos + lsm * 64, sin + lsm * 64, eps, inv_d, BL, kr, rows_off,
                                 rows_off, rows_off, 0)
    lane = tl.arange(0, 32)[:, None] * 4 + tl.arange(0, 4)[None, :]
    v0 = tl.load(vr + rows_off[:, None, None] + lane[None, :, :])
    v1 = tl.load(vr + rows_off[:, None, None] + 128 + lane[None, :, :])
    m = (ls < L)[:, None, None]
    for r in tl.static_range(REP):
        hh = h * REP + r
        dst = (hh * L + ls[:, None, None]).to(tl.int64) * 256 + lane[None, :, :]
        tl.store(kout + dst, f0, mask=m)
        tl.store(kout + dst + 128, f1, mask=m)
        tl.store(vout + dst, v0, mask=m)
        tl.store(vout + dst + 128, v1, mask=m)


def q_norm_rope(qg: torch.Tensor, w1p: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float):
    B, L, H, D2 = qg.shape
    assert D2 == 512 and qg.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()
    out = torch.empty((B, H, L, 256), dtype=qg.dtype, device=qg.device)
    BL = 8
    _q_norm_rope_kernel[(triton.cdiv(L, BL), H, B)](qg, w1p, cos, sin, out, L, eps, INV_256, H=H, BL=BL, num_warps=4,
                                                  **EXACT)
    return out


def kv_norm_rope_rep(kr: torch.Tensor, vr: torch.Tensor, w1p: torch.Tensor, cos, sin, eps: float, rep: int):
    B, L, HK, D = kr.shape
    assert D == 256 and kr.is_contiguous() and vr.is_contiguous() and vr.shape == kr.shape
    kout = torch.empty((B, HK * rep, L, 256), dtype=kr.dtype, device=kr.device)
    vout = torch.empty_like(kout)
    BL = 8
    _kv_norm_rope_kernel[(triton.cdiv(L, BL), HK, B)](kr, vr, w1p, cos, sin, kout, vout, L, eps, INV_256, HK=HK, REP=rep,
                                                    BL=BL, num_warps=4, **EXACT)
    return kout, vout


# ============================================================================================== S5: packed rows
# Packed layout: the R GEMM rows hold the real tokens of B decisions back to back (decision b at rows
# start[b] .. start[b] + len[b] - 1; rows >= sum(len) are spare). fla and SDPA run on the padded layout [B, L] (the
# reference's right padding), with ZERO in every padding position; real positions never read padding positions
# (causal conv window, causal attention mask, fla's causal chunk recurrences), so the real rows keep their bits.
# row_pos[r] = b * L + p maps a packed row to its padded position (spare rows map to position 0).
#
# E1 template prefix (PFX > 0, a compile-time constant; PFX = 0 compiles exactly the S5 kernels): every decision
# starts with the same PFX template tokens, so the packed rows hold only positions PFX .. len-1 (decision b at rows
# start[b] .. start[b] + len[b] - PFX - 1) and the GEMM outputs of positions 0 .. PFX-1 come from a cache recorded
# once per bucket (cx etc.: [PFX, N] rows, row stride CXS). The padded layouts fla and SDPA read are the same bits.


@triton.jit
def _ld_pfx(x, off, cx, coff, use_c, use_x, m, PFX: tl.constexpr):
    """x[off] (packed row) where use_x, cx[coff] (prefix cache row) where use_c (use_x = not use_c); masked lanes
    read 0.0."""
    if PFX > 0:
        a = tl.load(x + off, mask=m & use_x, other=0.0)
        c = tl.load(cx + coff, mask=m & use_c, other=0.0)
        return tl.where(use_c, c, a)
    else:
        return tl.load(x + off, mask=m, other=0.0)


@triton.jit
def _ld_rows(x, off, cx, coff, use_c, use_x, PFX: tl.constexpr):
    """As _ld_pfx for row loads that need no mask in the S5 kernels (rows clamped to real positions)."""
    if PFX > 0:
        a = tl.load(x + off, mask=use_x, other=0.0)
        c = tl.load(cx + coff, mask=use_c, other=0.0)
        return tl.where(use_c, c, a)
    else:
        return tl.load(x + off)


@triton.jit
def _conv_silu_packed_body(pid_l, pid_c, bb, x, w, q, k, v, start, slen, L, XS, cx, CXS, plen, pvar, C: tl.constexpr,
                           KD: tl.constexpr, BL: tl.constexpr, BC: tl.constexpr, REP: tl.constexpr, PFX: tl.constexpr,
                           PMAX: tl.constexpr):
    # PV (a4): slot bb's prefix is plen[bb] tokens long, its cached rows start at row pvar[bb] * PMAX of cx
    s0 = tl.load(start + bb).to(tl.int64)
    n = tl.load(slen + bb)
    b64 = bb.to(tl.int64)
    q = q + b64 * L * (KD * REP)
    k = k + b64 * L * (KD * REP)
    v = v + b64 * L * (C - 2 * KD)
    ls = pid_l * BL + tl.arange(0, BL)
    cs = pid_c * BC + tl.arange(0, BC)
    lm = ls < L
    valid = ls < n
    acc = tl.zeros((BL, BC), tl.float32)
    for kk in tl.static_range(4):
        src = ls - 3 + kk
        mk = (src >= 0) & valid
        if PFX > 0:
            s64 = src[:, None].to(tl.int64)
            pb = tl.load(plen + bb)
            cb0 = tl.load(pvar + bb).to(tl.int64) * PMAX
            xv = _ld_pfx(x, (s0 + s64 - pb) * XS + cs[None, :], cx, (cb0 + s64) * CXS + cs[None, :],
                         (src < pb)[:, None], (src >= pb)[:, None], mk[:, None], PFX).to(tl.float32)
        else:
            xv = tl.load(x + (s0 + src[:, None].to(tl.int64)) * XS + cs[None, :], mask=mk[:, None],
                         other=0.0).to(tl.float32)
        wv = tl.load(w + cs * 4 + kk).to(tl.float32)
        acc = acc + wv[None, :] * xv
    cb = acc.to(tl.bfloat16).to(tl.float32)
    y = _silu_f32(cb).to(tl.bfloat16)          # padding positions: acc = 0 -> y = 0
    c0 = pid_c * BC
    lrow = ls[:, None].to(tl.int64)
    m2 = lm[:, None]
    if c0 < KD:
        hc = cs[None, :] // 128
        dc = cs[None, :] % 128
        base = q + lrow * (KD * REP) + (REP * hc) * 128 + dc
        tl.store(base, y, mask=m2)
        if REP == 2:
            tl.store(base + 128, y, mask=m2)
    elif c0 < 2 * KD:
        c2 = cs[None, :] - KD
        hc = c2 // 128
        dc = c2 % 128
        base = k + lrow * (KD * REP) + (REP * hc) * 128 + dc
        tl.store(base, y, mask=m2)
        if REP == 2:
            tl.store(base + 128, y, mask=m2)
    else:
        tl.store(v + lrow * (C - 2 * KD) + (cs[None, :] - 2 * KD), y, mask=m2)


@triton.jit
def _conv_silu_packed_kernel(x, w, q, k, v, start, slen, L, XS, cx, CXS, plen, pvar, C: tl.constexpr, KD: tl.constexpr,
                             BL: tl.constexpr, BC: tl.constexpr, REP: tl.constexpr, PFX: tl.constexpr,
                             PMAX: tl.constexpr):
    _conv_silu_packed_body(tl.program_id(0), tl.program_id(1), tl.program_id(2), x, w, q, k, v, start, slen, L, XS,
                           cx, CXS, plen, pvar, C, KD, BL, BC, REP, PFX, PMAX)


def conv_silu_split_packed(mixed, w, key_dim, value_dim, start, slen, B, L, gva: bool = False, cache=None, pfx: int = 0,
                           plen=None, pvar=None, pmax: int = 0):
    """mixed: in_proj_qkv output [R, C] (row stride may be larger: a column block of the fused qkv|z output).
    gva=False: q/k heads repeated to the value heads (as the reference passes them to fla); gva=True: q/k keep their
    16 heads (fla's native grouped-value path reads key head i // 2 for value head i).
    pfx > 0: mixed holds positions pfx.. only; cache [pfx, C] holds positions 0..pfx-1 (E1)."""
    R, C = mixed.shape
    assert mixed.stride(1) == 1 and C == 2 * key_dim + value_dim and w.shape == (C, 4) and w.is_contiguous()
    if pfx:
        pmax = pmax or pfx
        assert cache is not None and cache.shape[1] == C and cache.shape[0] % pmax == 0 and cache.stride(1) == 1
        assert plen is not None and pvar is not None
    else:
        cache, plen, pvar = mixed, start, start
    rep = 1 if gva else 2
    # q and k are the two halves of one buffer (FF: one l2norm launch covers both; each is contiguous on its own)
    qk = torch.empty((2, B, L, rep * key_dim // 128, 128), dtype=mixed.dtype, device=mixed.device)
    q, k = qk[0], qk[1]
    v = torch.empty((B, L, value_dim // 128, 128), dtype=mixed.dtype, device=mixed.device)
    BL, BC = 16, 256
    _conv_silu_packed_kernel[(triton.cdiv(L, BL), C // BC, B)](mixed, w, q, k, v, start, slen, L, mixed.stride(0), cache,
                                                                cache.stride(0), plen, pvar, C=C, KD=key_dim, BL=BL,
                                                                BC=BC, REP=rep, PFX=pfx, PMAX=pmax, num_warps=4,
                                                                **EXACT)
    return q, k, v


@triton.jit
def _gdn_gate_packed_body(pid, b, a, negexpA, dt_bias, beta_out, g_out, start, slen, L, N, BS, AS, cb, ca, CBS, CAS,
                          plen, pvar, NH: tl.constexpr, BLOCK: tl.constexpr, PFX: tl.constexpr, PMAX: tl.constexpr):
    offs = pid * BLOCK + tl.arange(0, BLOCK)       # padded element index (b, p, h)
    m = offs < N
    hh = offs % NH
    bl = offs // NH
    bq = bl // L
    p = bl % L
    s0 = tl.load(start + bq, mask=m, other=0)
    n = tl.load(slen + bq, mask=m, other=0)
    valid = m & (p < n)
    if PFX > 0:
        pb = tl.load(plen + bq, mask=m, other=PFX)
        cb0 = tl.load(pvar + bq, mask=m, other=0).to(tl.int64) * PMAX
        row = (s0 + p - pb).to(tl.int64)
        crow = cb0 + p.to(tl.int64)
        use_c = p < pb
        use_x = p >= pb
        bf = _ld_pfx(b, row * BS + hh, cb, crow * CBS + hh, use_c, use_x, valid, PFX).to(tl.float32)
        af = _ld_pfx(a, row * AS + hh, ca, crow * CAS + hh, use_c, use_x, valid, PFX).to(tl.float32)
    else:
        row = (s0 + p).to(tl.int64)
        bf = tl.load(b + row * BS + hh, mask=valid, other=0.0).to(tl.float32)
    tl.store(beta_out + offs, tl.where(valid, _sigmoid_f32(bf), 0.0).to(tl.bfloat16), mask=m)
    if PFX == 0:
        af = tl.load(a + row * AS + hh, mask=valid, other=0.0).to(tl.float32)
    t = af + tl.load(dt_bias + hh, mask=m, other=0.0).to(tl.float32)
    sp = tl.where(t > 20.0, t, libdevice.div_rn(libdevice.log1p(libdevice.exp(t)), 1.0))
    tl.store(g_out + offs, tl.where(valid, tl.load(negexpA + hh, mask=m, other=0.0) * sp, 0.0), mask=m)


@triton.jit
def _gdn_gate_packed_kernel(b, a, negexpA, dt_bias, beta_out, g_out, start, slen, L, N, BS, AS, cb, ca, CBS, CAS,
                            plen, pvar, NH: tl.constexpr, BLOCK: tl.constexpr, PFX: tl.constexpr, PMAX: tl.constexpr):
    _gdn_gate_packed_body(tl.program_id(0), b, a, negexpA, dt_bias, beta_out, g_out, start, slen, L, N, BS, AS, cb,
                          ca, CBS, CAS, plen, pvar, NH, BLOCK, PFX, PMAX)


# HF (rgh): conv_silu_split_packed and gdn_gate_packed in ONE launch (independent inputs and outputs): programs
# [0, NCONV) run the conv body (pid -> l block, c block, slot), the rest the gate body; same bodies, same arithmetic
@triton.jit
def _conv_gate_packed_kernel(x, w, q, k, v, start, slen, L, XS, cx, CXS, plen, pvar,
                             b, a, negexpA, dt_bias, beta_out, g_out, N, BS, AS, cb, ca, CBS, CAS,
                             NCONV, NL, NCB, C: tl.constexpr, KD: tl.constexpr, BL: tl.constexpr, BC: tl.constexpr,
                             REP: tl.constexpr, NH: tl.constexpr, BLOCK: tl.constexpr, PFX: tl.constexpr,
                             PMAX: tl.constexpr, PMAXG: tl.constexpr):
    pid = tl.program_id(0)
    if pid < NCONV:
        pid_l = pid % NL
        pid_c = (pid // NL) % NCB
        bb = pid // (NL * NCB)
        _conv_silu_packed_body(pid_l, pid_c, bb, x, w, q, k, v, start, slen, L, XS, cx, CXS, plen, pvar, C, KD, BL,
                               BC, REP, PFX, PMAX)
    else:
        _gdn_gate_packed_body(pid - NCONV, b, a, negexpA, dt_bias, beta_out, g_out, start, slen, L, N, BS, AS, cb, ca,
                              CBS, CAS, plen, pvar, NH, BLOCK, PFX, PMAXG)


def gdn_gate_packed(b, a, negexpA, dt_bias, start, slen, B, L, cb=None, ca=None, pfx: int = 0, plen=None, pvar=None,
                    pmax: int = 0):
    """b, a: [R, NH] (row strides may be larger: the column blocks of the fused b|a output).
    pfx > 0: b, a hold positions pfx.. only; cb, ca [pfx, NH] hold positions 0..pfx-1 (E1)."""
    R, NH = b.shape
    assert b.stride(1) == 1 and a.stride(1) == 1 and b.shape == a.shape
    if pfx:
        pmax = pmax or pfx
        assert cb.shape[1] == NH and cb.shape == ca.shape and cb.shape[0] % pmax == 0
        assert cb.stride(1) == 1 and ca.stride(1) == 1 and plen is not None and pvar is not None
    else:
        cb, ca, plen, pvar = b, a, start, start
    beta = torch.empty((B, L, NH), dtype=b.dtype, device=b.device)
    g = torch.empty((B, L, NH), dtype=torch.float32, device=b.device)
    n = B * L * NH
    BLOCK = 1024
    _gdn_gate_packed_kernel[(triton.cdiv(n, BLOCK),)](b, a, negexpA, dt_bias, beta, g, start, slen, L, n, b.stride(0),
                                                      a.stride(0), cb, ca, cb.stride(0), ca.stride(0), plen, pvar,
                                                      NH=NH, BLOCK=BLOCK, PFX=pfx, PMAX=pmax, num_warps=4, **EXACT)
    return beta, g


def conv_gate_packed(mixed, w, key_dim, value_dim, b, a, negexpA, dt_bias, start, slen, B, L, gva: bool = False,
                     cache=None, cb=None, ca=None, pfx: int = 0, plen=None, pvar=None, pmax: int = 0):
    """HF (rgh): conv_silu_split_packed(...) and gdn_gate_packed(...) in one launch -> (q, k, v, beta, g), the same
    outputs bit for bit (the same bodies)."""
    R, C = mixed.shape
    assert mixed.stride(1) == 1 and C == 2 * key_dim + value_dim and w.shape == (C, 4) and w.is_contiguous()
    NH = b.shape[1]
    assert b.stride(1) == 1 and a.stride(1) == 1 and b.shape == a.shape
    if pfx:
        pmax = pmax or pfx
        assert cache is not None and cache.shape[1] == C and cache.shape[0] % pmax == 0 and cache.stride(1) == 1
        assert cb.shape[1] == NH and cb.shape == ca.shape and cb.shape[0] % pmax == 0
        assert cb.stride(1) == 1 and ca.stride(1) == 1 and plen is not None and pvar is not None
    else:
        cache, cb, ca, plen, pvar = mixed, b, a, start, start
    rep = 1 if gva else 2
    qk = torch.empty((2, B, L, rep * key_dim // 128, 128), dtype=mixed.dtype, device=mixed.device)
    q, k = qk[0], qk[1]
    v = torch.empty((B, L, value_dim // 128, 128), dtype=mixed.dtype, device=mixed.device)
    beta = torch.empty((B, L, NH), dtype=b.dtype, device=b.device)
    g = torch.empty((B, L, NH), dtype=torch.float32, device=b.device)
    BL, BC, BLOCK = 16, 256, 1024
    NL, NCB = triton.cdiv(L, BL), C // BC
    nconv = NL * NCB * B
    n = B * L * NH
    _conv_gate_packed_kernel[(nconv + triton.cdiv(n, BLOCK),)](
        mixed, w, q, k, v, start, slen, L, mixed.stride(0), cache, cache.stride(0), plen, pvar,
        b, a, negexpA, dt_bias, beta, g, n, b.stride(0), a.stride(0), cb, ca, cb.stride(0), ca.stride(0),
        nconv, NL, NCB, C=C, KD=key_dim, BL=BL, BC=BC, REP=rep, NH=NH, BLOCK=BLOCK, PFX=pfx, PMAX=pmax, PMAXG=pmax,
        num_warps=4, **EXACT)
    return q, k, v, beta, g


# E4 PRO: the GDN prologue in ONE launch per layer, for what kai v4 runs as four kernels (conv_silu_split_packed and
# gdn_gate_packed, then inside FF ps mode fla's l2norm_fwd on q|k and chunk_local_cumsum on g):
#   programs [0, NCONV): _conv_silu_packed_body's statements with one 128-column head per program (BC 128: the conv is
#     elementwise, so the block shape changes no bit); on q / k heads the [BL = 16, 128] tile then runs fla 0.5.2's
#     l2norm_fwd_kernel statements at its pin (BT 16, 4 warps): b_x = y.to(f32); rstd = 1 / sqrt(sum(b_x * b_x, 1)
#     + eps); (b_x * rstd).to(bf16). EXACT compiles the squares as mul.rn + add.rn where fla contracts them to FMAs:
#     the same bits, since a bf16 value squared is exact in f32. Same tile, warps and tl.sum, so the same reduction
#     layout and order (tests/e4_cpu.py pro_aot compares the TTGIR layout and the shuffles with fla's kernel).
#   programs [NCONV, NCONV + NT * B * NH): one 64-position chunk of one (slot, head): _gdn_gate_packed_body's beta and
#     g expressions, then fla's chunk_local_cumsum_scalar_kernel statements on the [64] f32 vector (tl.cumsum, then
#     * scale); fla's cumsum gives the same bits with every num_warps (fla_pin), this uses 4 (its PRO 6000 pin).
# PackedForward.check_prologue compares q, k, v, beta and g with the four-kernel path bit for bit at start-up; FF ps
# then skips its own l2norm and cumsum (gdn_forward_ps pre=True).
@triton.jit
def _conv_l2_body(pid_l, pid_c, bb, x, w, q, k, v, start, slen, L, XS, cx, CXS, plen, pvar, eps, C: tl.constexpr,
                  KD: tl.constexpr, BL: tl.constexpr, BC: tl.constexpr, REP: tl.constexpr, PFX: tl.constexpr,
                  PMAX: tl.constexpr):
    s0 = tl.load(start + bb).to(tl.int64)
    n = tl.load(slen + bb)
    b64 = bb.to(tl.int64)
    q = q + b64 * L * (KD * REP)
    k = k + b64 * L * (KD * REP)
    v = v + b64 * L * (C - 2 * KD)
    ls = pid_l * BL + tl.arange(0, BL)
    cs = pid_c * BC + tl.arange(0, BC)
    lm = ls < L
    valid = ls < n
    acc = tl.zeros((BL, BC), tl.float32)
    for kk in tl.static_range(4):
        src = ls - 3 + kk
        mk = (src >= 0) & valid
        if PFX > 0:
            s64 = src[:, None].to(tl.int64)
            pb = tl.load(plen + bb)
            cb0 = tl.load(pvar + bb).to(tl.int64) * PMAX
            xv = _ld_pfx(x, (s0 + s64 - pb) * XS + cs[None, :], cx, (cb0 + s64) * CXS + cs[None, :],
                         (src < pb)[:, None], (src >= pb)[:, None], mk[:, None], PFX).to(tl.float32)
        else:
            xv = tl.load(x + (s0 + src[:, None].to(tl.int64)) * XS + cs[None, :], mask=mk[:, None],
                         other=0.0).to(tl.float32)
        wv = tl.load(w + cs * 4 + kk).to(tl.float32)
        acc = acc + wv[None, :] * xv
    cb = acc.to(tl.bfloat16).to(tl.float32)
    y = _silu_f32(cb).to(tl.bfloat16)          # padding positions: acc = 0 -> y = 0 (-> 0 after the l2norm too)
    c0 = pid_c * BC
    lrow = ls[:, None].to(tl.int64)
    m2 = lm[:, None]
    if c0 < 2 * KD:
        b_x = y.to(tl.float32)
        b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
        yn = (b_x * b_rstd[:, None]).to(tl.bfloat16)
        if c0 < KD:
            hc = cs[None, :] // 128
            dc = cs[None, :] % 128
            base = q + lrow * (KD * REP) + (REP * hc) * 128 + dc
        else:
            c2 = cs[None, :] - KD
            hc = c2 // 128
            dc = c2 % 128
            base = k + lrow * (KD * REP) + (REP * hc) * 128 + dc
        tl.store(base, yn, mask=m2)
        if REP == 2:
            tl.store(base + 128, yn, mask=m2)
    else:
        tl.store(v + lrow * (C - 2 * KD) + (cs[None, :] - 2 * KD), y, mask=m2)


@triton.jit
def _gate_cs_body(i_t, i_bh, b, a, negexpA, dt_bias, beta_out, gc_out, start, slen, L, BS, AS, cb, ca, CBS, CAS, plen,
                  pvar, scale, NH: tl.constexpr, BT: tl.constexpr, PFX: tl.constexpr, PMAX: tl.constexpr):
    bq = i_bh // NH
    hh = i_bh % NH
    p = i_t * BT + tl.arange(0, BT)
    s0 = tl.load(start + bq)
    n = tl.load(slen + bq)
    valid = p < n
    if PFX > 0:
        pb = tl.load(plen + bq)
        cb0 = tl.load(pvar + bq).to(tl.int64) * PMAX
        row = (s0 + p - pb).to(tl.int64)
        crow = cb0 + p.to(tl.int64)
        use_c = p < pb
        use_x = p >= pb
        bf = _ld_pfx(b, row * BS + hh, cb, crow * CBS + hh, use_c, use_x, valid, PFX).to(tl.float32)
        af = _ld_pfx(a, row * AS + hh, ca, crow * CAS + hh, use_c, use_x, valid, PFX).to(tl.float32)
    else:
        row = (s0 + p).to(tl.int64)
        bf = tl.load(b + row * BS + hh, mask=valid, other=0.0).to(tl.float32)
        af = tl.load(a + row * AS + hh, mask=valid, other=0.0).to(tl.float32)
    offs = (bq * L + p).to(tl.int64) * NH + hh
    tl.store(beta_out + offs, tl.where(valid, _sigmoid_f32(bf), 0.0).to(tl.bfloat16))
    t = af + tl.load(dt_bias + hh).to(tl.float32)
    sp = tl.where(t > 20.0, t, libdevice.div_rn(libdevice.log1p(libdevice.exp(t)), 1.0))
    b_s = tl.where(valid, tl.load(negexpA + hh) * sp, 0.0)
    b_o = tl.cumsum(b_s, axis=0)
    b_o *= scale
    tl.store(gc_out + offs, b_o)


@triton.jit
def _gdn_prologue_kernel(x, w, q, k, v, start, slen, L, XS, cx, CXS, plen, pvar, eps,
                         b, a, negexpA, dt_bias, beta_out, gc_out, BS, AS, cb, ca, CBS, CAS, scale, NCONV, NL, NCB,
                         NTB, C: tl.constexpr, KD: tl.constexpr, BL: tl.constexpr, BC: tl.constexpr, REP: tl.constexpr,
                         NH: tl.constexpr, BT: tl.constexpr, PFX: tl.constexpr, PMAX: tl.constexpr,
                         PMAXG: tl.constexpr):
    pid = tl.program_id(0)
    if pid < NCONV:
        _conv_l2_body(pid % NL, (pid // NL) % NCB, pid // (NL * NCB), x, w, q, k, v, start, slen, L, XS, cx, CXS,
                      plen, pvar, eps, C, KD, BL, BC, REP, PFX, PMAX)
    else:
        j = pid - NCONV
        _gate_cs_body(j % NTB, j // NTB, b, a, negexpA, dt_bias, beta_out, gc_out, start, slen, L, BS, AS, cb, ca,
                      CBS, CAS, plen, pvar, scale, NH, BT, PFX, PMAXG)


def gdn_prologue_packed(mixed, w, key_dim, value_dim, b, a, negexpA, dt_bias, start, slen, B, L, eps: float,
                        scale: float, gva: bool = False, cache=None, cb=None, ca=None, pfx: int = 0, plen=None,
                        pvar=None, pmax: int = 0):
    """E4 PRO: conv_gate_packed(...)'s outputs with q and k l2-normalised (fla's l2norm_fwd with eps) and g cumulated
    per 64-position chunk times scale (fla's chunk_local_cumsum(g, 64, scale)) -> (q, k, v, beta, g), in one launch;
    each output equals the four-kernel path bit for bit (PackedForward.check_prologue). L must be a multiple of 64."""
    R, C = mixed.shape
    assert mixed.stride(1) == 1 and C == 2 * key_dim + value_dim and w.shape == (C, 4) and w.is_contiguous()
    assert L % 64 == 0 and key_dim % 128 == 0 and value_dim % 128 == 0
    NH = b.shape[1]
    assert b.stride(1) == 1 and a.stride(1) == 1 and b.shape == a.shape
    if pfx:
        pmax = pmax or pfx
        assert cache is not None and cache.shape[1] == C and cache.shape[0] % pmax == 0 and cache.stride(1) == 1
        assert cb.shape[1] == NH and cb.shape == ca.shape and cb.shape[0] % pmax == 0
        assert cb.stride(1) == 1 and ca.stride(1) == 1 and plen is not None and pvar is not None
    else:
        cache, cb, ca, plen, pvar = mixed, b, a, start, start
    rep = 1 if gva else 2
    qk = torch.empty((2, B, L, rep * key_dim // 128, 128), dtype=mixed.dtype, device=mixed.device)
    q, k = qk[0], qk[1]
    v = torch.empty((B, L, value_dim // 128, 128), dtype=mixed.dtype, device=mixed.device)
    beta = torch.empty((B, L, NH), dtype=b.dtype, device=b.device)
    g = torch.empty((B, L, NH), dtype=torch.float32, device=b.device)
    BL, BC, BT = 16, 128, 64
    NL, NCB, NTB = triton.cdiv(L, BL), C // BC, L // BT
    nconv = NL * NCB * B
    _gdn_prologue_kernel[(nconv + NTB * B * NH,)](
        mixed, w, q, k, v, start, slen, L, mixed.stride(0), cache, cache.stride(0), plen, pvar, eps,
        b, a, negexpA, dt_bias, beta, g, b.stride(0), a.stride(0), cb, ca, cb.stride(0), ca.stride(0), scale,
        nconv, NL, NCB, NTB, C=C, KD=key_dim, BL=BL, BC=BC, REP=rep, NH=NH, BT=BT, PFX=pfx, PMAX=pmax, PMAXG=pmax,
        num_warps=4, **EXACT)
    return q, k, v, beta, g


@triton.jit
def _gated_norm_packed_kernel(o, z, w, out, row_pos, N, eps, inv_d, ZS, NH: tl.constexpr, R: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)                # packed head-rows n = r * NH + h
    rm = rows < N
    m = rm[:, None, None]
    pr = tl.load(row_pos + rows // NH, mask=rm, other=0).to(tl.int64)
    orow = pr * NH + rows % NH
    lane = tl.arange(0, 32)[None, :, None] * 4 + tl.arange(0, 4)[None, None, :]
    of = tl.load(o + orow[:, None, None] * 128 + lane, mask=m, other=0.0).to(tl.float32)
    s = _warp_tree32_b(_cols4_tree_b(of * of, R), R)
    r = libdevice.rsqrt(s * inv_d + eps)
    hn = (of * r[:, None, None]).to(tl.bfloat16).to(tl.float32)
    wv = tl.load(w + (tl.arange(0, 32)[:, None] * 4 + tl.arange(0, 4)[None, :])).to(tl.float32)
    hw = (wv[None, :, :] * hn).to(tl.bfloat16).to(tl.float32)
    r64 = rows.to(tl.int64)
    zaddr = ((r64 // NH) * ZS + (r64 % NH) * 128)[:, None, None] + lane
    zf = tl.load(z + zaddr, mask=m, other=0.0).to(tl.float32)
    tl.store(out + r64[:, None, None] * 128 + lane, (hw * _silu_f32(zf)).to(tl.bfloat16), mask=m)


def gated_rmsnorm_packed(o_pad, z, w, eps, row_pos, nh):
    """o_pad: fla output [B, L, NH, 128] (padded layout); z: [R, NH * 128] packed (row stride may be larger: the z
    column block of the fused qkv|z output) -> out [R * NH, 128]."""
    assert o_pad.is_contiguous() and z.dim() == 2 and z.stride(1) == 1 and z.shape[1] == nh * 128
    n = z.shape[0] * nh
    out = torch.empty((n, 128), dtype=z.dtype, device=z.device)
    R = 8
    _gated_norm_packed_kernel[(triton.cdiv(n, R),)](o_pad, z, w, out, row_pos, n, eps, INV_128, z.stride(0), NH=nh, R=R,
                                                    num_warps=4, **EXACT)
    return out


@triton.jit
def _q_norm_rope_packed_kernel(qg, w1p, cos, sin, out, start, slen, L, eps, inv_d, cq, CQS, plen, pvar, H: tl.constexpr,
                               BL: tl.constexpr, ONLY_LAST: tl.constexpr, PFX: tl.constexpr, PMAX: tl.constexpr):
    pid_l = tl.program_id(0)
    h = tl.program_id(1)
    bb = tl.program_id(2)
    s0 = tl.load(start + bb)
    n = tl.load(slen + bb)
    out = out + bb.to(tl.int64) * H * L * 256
    ls = pid_l * BL + tl.arange(0, BL)
    lsm = tl.minimum(ls, tl.maximum(n - 1, 0))
    if ONLY_LAST:
        # qg holds one row per decision (its last token); every other query position gets zeros
        rows_off = bb.to(tl.int64) * (H * 512) + h * 512 + 0 * lsm
        f0, f1 = _norm_rope_rows(qg, rows_off, w1p, cos + lsm * 64, sin + lsm * 64, eps, inv_d, BL, qg, rows_off,
                                 rows_off, rows_off, 0)
    elif PFX > 0:
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
    if ONLY_LAST:
        keep = (ls == n - 1)[:, None, None]
    else:
        keep = (ls < n)[:, None, None]
    tl.store(dst, tl.where(keep, f0, 0.0), mask=m)
    tl.store(dst + 128, tl.where(keep, f1, 0.0), mask=m)


@triton.jit
def _q_norm_rope_last_kernel(qg, w1p, cos, sin, out, slen, eps, inv_d, H: tl.constexpr, BL: tl.constexpr):
    # E4 LR1: _q_norm_rope_packed_kernel's ONLY_LAST row (position n - 1: the same _norm_rope_rows call on a BL-row
    # block, here with every row at that position) stored once per (slot, head) into out [B, H, 1, 256]; an unused
    # slot (n = 0) gets zeros, as there
    h = tl.program_id(1)
    bb = tl.program_id(2)
    n = tl.load(slen + bb)
    ls = tl.arange(0, BL)
    lsm = tl.maximum(n - 1, 0) + 0 * ls
    rows_off = bb.to(tl.int64) * (H * 512) + h * 512 + 0 * lsm
    f0, f1 = _norm_rope_rows(qg, rows_off, w1p, cos + lsm * 64, sin + lsm * 64, eps, inv_d, BL, qg, rows_off,
                             rows_off, rows_off, 0)
    lane = tl.arange(0, 32)[:, None] * 4 + tl.arange(0, 4)[None, :]
    dst = out + (bb * H + h).to(tl.int64) * 256 + lane[None, :, :] + 0 * ls[:, None, None]
    first = (ls == 0)[:, None, None]
    tl.store(dst, tl.where(n > 0, f0, 0.0), mask=first)
    tl.store(dst + 128, tl.where(n > 0, f1, 0.0), mask=first)


@triton.jit
def _kv_norm_rope_packed_kernel(kr, vr, w1p, cos, sin, kout, vout, start, slen, L, eps, inv_d, KS, VS, ck, cv, CKS,
                                CVS, plen, pvar, HK: tl.constexpr, REP: tl.constexpr, BL: tl.constexpr,
                                PFX: tl.constexpr, PMAX: tl.constexpr):
    pid_l = tl.program_id(0)
    h = tl.program_id(1)
    bb = tl.program_id(2)
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
        f0, f1 = _norm_rope_rows(kr, rows_off, w1p, cos + lsm * 64, sin + lsm * 64, eps, inv_d, BL, ck, crows_off,
                                 use_c, use_x, PFX)
        uc = use_c[:, None, None]
        ux = use_x[:, None, None]
        v0 = _ld_rows(vr, vrows_off[:, None, None] + lane[None, :, :], cv, cvrows_off[:, None, None] + lane[None, :, :],
                      uc, ux, PFX)
        v1 = _ld_rows(vr, vrows_off[:, None, None] + 128 + lane[None, :, :], cv,
                      cvrows_off[:, None, None] + 128 + lane[None, :, :], uc, ux, PFX)
    else:
        rows_off = (s0 + lsm).to(tl.int64) * KS + h * 256
        vrows_off = (s0 + lsm).to(tl.int64) * VS + h * 256
        f0, f1 = _norm_rope_rows(kr, rows_off, w1p, cos + lsm * 64, sin + lsm * 64, eps, inv_d, BL, kr, rows_off,
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
        dst = (hh * L + ls[:, None, None]).to(tl.int64) * 256 + lane[None, :, :]
        tl.store(kout + dst, f0, mask=m)
        tl.store(kout + dst + 128, f1, mask=m)
        tl.store(vout + dst, v0, mask=m)
        tl.store(vout + dst + 128, v1, mask=m)


def q_norm_rope_packed(qg, w1p, cos, sin, eps, start, slen, B, L, only_last=False, cache=None, pfx: int = 0,
                       plen=None, pvar=None, pmax: int = 0):
    """pfx > 0 (E1): qg holds positions pfx.. only; cache [pfx, H * 512] (q_proj output rows) positions 0..pfx-1."""
    R, H, D2 = qg.shape
    assert D2 == 512 and qg.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()
    assert not only_last or (R == B and not pfx)
    if pfx:
        pmax = pmax or pfx
        assert cache.shape[1] == H * 512 and cache.shape[0] % pmax == 0 and cache.stride(1) == 1
        assert plen is not None and pvar is not None
    else:
        cache, plen, pvar = qg, start, start
    out = torch.empty((B, H, L, 256), dtype=qg.dtype, device=qg.device)
    BL = 8
    _q_norm_rope_packed_kernel[(triton.cdiv(L, BL), H, B)](qg, w1p, cos, sin, out, start, slen, L, eps, INV_256, cache,
                                                            cache.stride(0), plen, pvar, H=H, BL=BL, ONLY_LAST=only_last,
                                                            PFX=pfx, PMAX=pmax, num_warps=4, **EXACT)
    return out


def q_norm_rope_last(qg, w1p, cos, sin, eps, slen, B):
    """E4 LR1: [B, H, 1, 256], each slot's query row at its last position n - 1, bit for bit the row that
    q_norm_rope_packed(..., only_last=True) writes there (qg: one row per slot)."""
    R, H, D2 = qg.shape
    assert D2 == 512 and R == B and qg.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()
    out = torch.empty((B, H, 1, 256), dtype=qg.dtype, device=qg.device)
    _q_norm_rope_last_kernel[(1, H, B)](qg, w1p, cos, sin, out, slen, eps, INV_256, H=H, BL=8, num_warps=4, **EXACT)
    return out


@triton.jit
def _q_norm_rope_rows_kernel(qg, w1p, cos, sin, out, row_pos, Rn, L, eps, inv_d, H: tl.constexpr, BL: tl.constexpr):
    # XA (ATTN): _q_norm_rope_packed_kernel's rows for the packed (computed) rows only, [R, H, 256] (the same
    # _norm_rope_rows call on BL-row blocks, each row at its own position row_pos % L); rows past Rn are not stored
    pid_r = tl.program_id(0)
    h = tl.program_id(1)
    rs = pid_r * BL + tl.arange(0, BL)
    rsm = tl.minimum(rs, Rn - 1)
    lsm = tl.load(row_pos + rsm) % L
    rows_off = rsm.to(tl.int64) * (H * 512) + h * 512
    f0, f1 = _norm_rope_rows(qg, rows_off, w1p, cos + lsm * 64, sin + lsm * 64, eps, inv_d, BL, qg, rows_off,
                             rows_off, rows_off, 0)
    lane = tl.arange(0, 32)[:, None] * 4 + tl.arange(0, 4)[None, :]
    dst = out + (rs[:, None, None].to(tl.int64) * H + h) * 256 + lane[None, :, :]
    m = (rs < Rn)[:, None, None]
    tl.store(dst, f0, mask=m)
    tl.store(dst + 128, f1, mask=m)


def q_norm_rope_rows(qg, w1p, cos, sin, eps, row_pos, L):
    """XA (ATTN): [R, H, 256] q rows of the packed rows (row r at position row_pos[r] % L), bit for bit the rows
    q_norm_rope_packed writes at those positions."""
    R, H, D2 = qg.shape
    assert D2 == 512 and qg.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()
    out = torch.empty((R, H, 256), dtype=qg.dtype, device=qg.device)
    BL = 8
    _q_norm_rope_rows_kernel[(triton.cdiv(R, BL), H)](qg, w1p, cos, sin, out, row_pos, R, L, eps, INV_256, H=H, BL=BL,
                                                       num_warps=4, **EXACT)
    return out


@triton.jit
def _sigmoid_table_kernel(out, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = i.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    tl.store(out + i, _sigmoid_f32(x).to(tl.bfloat16))


def sigmoid_table(device="cuda") -> torch.Tensor:
    """XA (ATTN): [65536] bf16, entry b = the attention gate's bf16(_sigmoid_f32(x)) for the bf16 x of bit pattern b
    (the value _attn_gate_*_kernel multiplies by)."""
    out = torch.empty((65536,), dtype=torch.bfloat16, device=device)
    _sigmoid_table_kernel[(65536 // 1024,)](out, BLOCK=1024, num_warps=4, **EXACT)
    return out


def kv_norm_rope_rep_packed(kr, vr, w1p, cos, sin, eps, rep, start, slen, B, L, ck=None, cv=None, pfx: int = 0,
                            plen=None, pvar=None, pmax: int = 0):
    """kr, vr: [R, HK, 256] contiguous, or [R, HK * 256] row-strided column blocks of the fused k|v output.
    pfx > 0 (E1): kr, vr hold positions pfx.. only; ck, cv [pfx, HK * 256] positions 0..pfx-1."""
    if kr.dim() == 3:
        R, HK, D = kr.shape
        assert D == 256 and kr.is_contiguous() and vr.is_contiguous() and vr.shape == kr.shape
        ks = vs = HK * 256
    else:
        R, HD = kr.shape
        HK = HD // 256
        assert HD == HK * 256 and kr.stride(1) == 1 and vr.stride(1) == 1 and vr.shape == kr.shape
        ks, vs = kr.stride(0), vr.stride(0)
    if pfx:
        pmax = pmax or pfx
        assert ck.shape[1] == HK * 256 and cv.shape == ck.shape and ck.shape[0] % pmax == 0
        assert ck.stride(1) == 1 and cv.stride(1) == 1 and plen is not None and pvar is not None
    else:
        ck, cv, plen, pvar = kr, vr, start, start
    kout = torch.empty((B, HK * rep, L, 256), dtype=kr.dtype, device=kr.device)
    vout = torch.empty_like(kout)
    BL = 8
    _kv_norm_rope_packed_kernel[(triton.cdiv(L, BL), HK, B)](kr, vr, w1p, cos, sin, kout, vout, start, slen, L, eps,
                                                              INV_256, ks, vs, ck, cv, ck.stride(0), cv.stride(0), plen,
                                                              pvar, HK=HK, REP=rep, BL=BL, PFX=pfx, PMAX=pmax,
                                                              num_warps=4, **EXACT)
    return kout, vout


@triton.jit
def _attn_gate_packed_kernel(o, qg, out, row_pos, Rn, L, H: tl.constexpr, HD: tl.constexpr, so_b, so_h, so_l,
                             BL: tl.constexpr):
    pid_r = tl.program_id(0)
    h = tl.program_id(1)
    rs = pid_r * BL + tl.arange(0, BL)
    rm = rs < Rn
    pr = tl.load(row_pos + rs, mask=rm, other=0).to(tl.int64)
    bq = pr // L
    p = pr % L
    d = tl.arange(0, HD)
    m = rm[:, None]
    ov = tl.load(o + (bq * so_b + h * so_h + p * so_l)[:, None] + d[None, :], mask=m, other=0.0).to(tl.float32)
    gv = tl.load(qg + rs[:, None].to(tl.int64) * (H * 2 * HD) + h * (2 * HD) + HD + d[None, :], mask=m,
                 other=0.0).to(tl.float32)
    s = _sigmoid_f32(gv).to(tl.bfloat16).to(tl.float32)
    tl.store(out + rs[:, None].to(tl.int64) * (H * HD) + h * HD + d[None, :], (ov * s).to(tl.bfloat16), mask=m)


@triton.jit
def _attn_gate_grouped_kernel(o, qg, out, row_pos, Rn, L, H: tl.constexpr, HD: tl.constexpr, G: tl.constexpr,
                              so_b, so_r, so_l, BL: tl.constexpr):
    # E4 GQA: _attn_gate_packed_kernel for an attention output grouped by KV head, o [B * G, H / G, L, HD] (head h of
    # slot b at batch b * G + h // (H / G), head h % (H / G)); the same gate arithmetic
    pid_r = tl.program_id(0)
    h = tl.program_id(1)
    RR: tl.constexpr = H // G
    rs = pid_r * BL + tl.arange(0, BL)
    rm = rs < Rn
    pr = tl.load(row_pos + rs, mask=rm, other=0).to(tl.int64)
    bq = pr // L
    p = pr % L
    d = tl.arange(0, HD)
    m = rm[:, None]
    ov = tl.load(o + ((bq * G + h // RR) * so_b + (h % RR) * so_r + p * so_l)[:, None] + d[None, :], mask=m,
                 other=0.0).to(tl.float32)
    gv = tl.load(qg + rs[:, None].to(tl.int64) * (H * 2 * HD) + h * (2 * HD) + HD + d[None, :], mask=m,
                 other=0.0).to(tl.float32)
    s = _sigmoid_f32(gv).to(tl.bfloat16).to(tl.float32)
    tl.store(out + rs[:, None].to(tl.int64) * (H * HD) + h * HD + d[None, :], (ov * s).to(tl.bfloat16), mask=m)


def attn_gate_grouped(o_g, qg, row_pos, L, G):
    """E4 GQA: attn_gate_packed for o_g [B * G, H / G, L, HD] (SDPA over KV-head groups), the same output."""
    BG, RR, LL, HD = o_g.shape
    H = RR * G
    Rn = qg.shape[0]
    assert qg.is_contiguous() and qg.shape == (Rn, H, 2 * HD) and o_g.stride(3) == 1 and LL == L and BG % G == 0
    out = torch.empty((Rn, H * HD), dtype=o_g.dtype, device=o_g.device)
    BL = 16
    _attn_gate_grouped_kernel[(triton.cdiv(Rn, BL), H)](o_g, qg, out, row_pos, Rn, L, H=H, HD=HD, G=G,
                                                        so_b=o_g.stride(0), so_r=o_g.stride(1), so_l=o_g.stride(2),
                                                        BL=BL, num_warps=4, **EXACT)
    return out


def attn_gate_packed(o_bhld, qg, row_pos, L):
    B, H, LL, HD = o_bhld.shape
    Rn = qg.shape[0]
    assert qg.is_contiguous() and qg.shape == (Rn, H, 2 * HD) and o_bhld.stride(3) == 1 and LL == L
    out = torch.empty((Rn, H * HD), dtype=o_bhld.dtype, device=o_bhld.device)
    BL = 16
    _attn_gate_packed_kernel[(triton.cdiv(Rn, BL), H)](o_bhld, qg, out, row_pos, Rn, L, H=H, HD=HD,
                                                       so_b=o_bhld.stride(0), so_h=o_bhld.stride(1),
                                                       so_l=o_bhld.stride(2), BL=BL, num_warps=4, **EXACT)
    return out
