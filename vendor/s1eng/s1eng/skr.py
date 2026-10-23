"""SKR (ruj r2 fixedcost2): fold cuBLASLt's own split-K slice partials into the add_rmsnorm that follows, and drop the
splitKreduce node from the lane graphs.

Where: the N = 2560 projections whose output only feeds add_rmsnorm (GDN out_proj, attention o_proj, MLP down_proj:
PackedForward.lin_res) at the (K, bucket, R) where the S6 / plan pick is a cuBLASLt split-K algo with the OUTPUT_TYPE
reduction (cutlass Kernel2, gridZ = S slices) and rgh's RARN is not used. Today that is three launches: Kernel2 (each
slice's bf16 partial into the cuBLASLt workspace), splitKreduce (y = bf16(((f(P0) + f(P1)) + ...) in f32), alpha 1,
beta 0), then fused.add_rmsnorm(h, y).
SKR keeps the Kernel2 launch exactly as today (same algo, same workspace, same bytes), then runs ONE kernel that reads
the S partials straight from the workspace (cuBLASLt keeps them at offset 0 as [S][R][2560] bf16: probe
test/skr_probe.py, 10-21) and does fold + residual add + RMSNorm with RARN's arithmetic (rarn._rarn_kernel: the
OUTPUT_TYPE fold in slice order, then _rmsnorm_kernel(HAS_Y)'s add / reduction order); after the lane graph is
captured, prune() deletes the splitKreduce node in front of every SKR fold (its parents -> its children). Net per site:
one launch and one y round trip less (the fold reads the partials the reduce would have read).

Start-up gate select(): per (K, bucket, R), with the real weights, two workspaces (zeroed before every check, so stale
partials cannot fake a pass) and random rows (plain and x64 outliers): SKR's (hn, normed) must equal the pick +
fused.add_rmsnorm byte for byte, else that (K, bucket, R) stays on today's path. prune() only removes a splitKreduce
node whose single child is an SKR fold; anything else (no keep_graph, other node layout) keeps the reduce node, which
is harmless (the fold reads the partials either way). The engine's self_check (batch vs per-decision graphs) runs on
the pruned graphs.

S1ENG_SKR=1 on, "0" / unset off (= the base's graphs). FCAB arm toggle: noskr.
S1+KF (ruj): select() skips every (K, bucket, R) whose pick is a KF algo (kf.KfAlgo, is_kf): KF computes the pin's
bits in one launch and writes y itself, so that key keeps KF + add_rmsnorm (KF's table measured it vs today's path).
"""
from __future__ import annotations

import ctypes as C
import functools
import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .fused import EXACT, INV_D, _cols4_tree, _warp_tree32

N_OUT = 2560
FOLD_NAME = "_skr_kernel"


def enabled() -> bool:
    return os.environ.get("S1ENG_SKR", "0") == "1"


@triton.jit
def _skr_kernel(part, x, hout, out, w, eps, inv_d, pstride, S: tl.constexpr):
    """rarn._rarn_kernel's arithmetic (byte-identical body) under its own name, so prune() can tell an SKR fold node
    from a RARN one: hb = bf16(x + bf16(((p0 + p1) + ...) in f32)) -> hout; out = bf16((hb * rsqrt(mean(hb^2) + eps))
    * w) with _rmsnorm_kernel's reduction order. part = the cuBLASLt workspace, slices [S][R][2560]."""
    row = tl.program_id(0).to(tl.int64)
    xr = x + row * 2560
    pr = part + row * 2560
    offs = tl.arange(0, 32)[:, None] * 4 + tl.arange(0, 4)[None, :]
    acc = tl.zeros((32, 4), tl.float32)
    for j in tl.static_range(20):
        d = tl.load(pr + j * 128 + offs).to(tl.float32)
        for s in tl.static_range(1, S):
            d = d + tl.load(pr + s * pstride + j * 128 + offs).to(tl.float32)
        hb = (tl.load(xr + j * 128 + offs).to(tl.float32) + d.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        v = hb.to(tl.float32)
        acc = acc + v * v
    r = libdevice.rsqrt(_warp_tree32(_cols4_tree(acc)) * inv_d + eps)
    jj = tl.arange(0, 32)[:, None]
    cc = tl.arange(0, 128)[None, :]
    o2 = jj * 128 + cc
    m2 = jj < 20
    d = tl.load(pr + o2, mask=m2, other=0.0).to(tl.float32)
    for s in tl.static_range(1, S):
        d = d + tl.load(pr + s * pstride + o2, mask=m2, other=0.0).to(tl.float32)
    hb = (tl.load(xr + o2, mask=m2, other=0.0).to(tl.float32) + d.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    tl.store(hout + row * 2560 + o2, hb, mask=m2)
    wv = tl.load(w + o2, mask=m2, other=0.0)
    tl.store(out + row * 2560 + o2, ((hb.to(tl.float32) * r) * wv).to(tl.bfloat16), mask=m2)


class Deferred:
    """An N = 2560 cuBLASLt split-K GEMM whose output only feeds add_rmsnorm (PackedForward.add_norm folds it)."""
    __slots__ = ("W", "x", "algo", "S", "L")

    def __init__(self, W, x, algo, S, L):
        self.W, self.x, self.algo, self.S, self.L = W, x, algo, S, L


def apply(ctx, algo, W: torch.Tensor, x: torch.Tensor, h: torch.Tensor, w1p: torch.Tensor, eps: float, S: int,
          ws: torch.Tensor):
    """(hn, normed) = fused.add_rmsnorm(h, x @ W^T at `algo`, w1p, eps): the pick's GEMM as today (cuBLASLt writes the
    slice partials to ws[0:] and, unless prune() removed it, splitKreduce writes y), then the fold from ws."""
    R = x.shape[0]
    assert h.is_contiguous() and h.shape == (R, N_OUT) and S * R * N_OUT * 2 <= ws.numel()
    y = torch.empty((R, N_OUT), dtype=x.dtype, device=x.device)   # cuBLASLt's D (written by the reduce, if kept)
    ctx.run(algo, W, x, y, ws)
    part = ws.view(torch.bfloat16)
    hn = torch.empty_like(h)
    out = torch.empty_like(h)
    _skr_kernel[(R,)](part, h, hn, out, w1p, eps, INV_D, R * N_OUT, S=S, num_warps=4, **EXACT)
    return hn, out


def _inputs(g, R, K, outliers):
    x = torch.randn((R, K), generator=g, device="cuda")
    h = torch.randn((R, N_OUT), generator=g, device="cuda") * 3
    if outliers:
        x[torch.randint(0, R, (max(1, R // 50),), generator=g, device="cuda")] *= 64
        x[:, torch.randint(0, K, (32,), generator=g, device="cuda")] *= 64
        h[torch.randint(0, R, (max(1, R // 40),), generator=g, device="cuda")] *= 64
    return x.to(torch.bfloat16), h.to(torch.bfloat16)


@torch.inference_mode()
def select(pinned, ctx, weights: dict, rows_of: dict, w1p: torch.Tensor, eps: float, skip=(), log=print,
           seed: int = 31) -> dict:
    """{(K, L, R): (algo, S)} for the N = 2560 (K, bucket) pins that are cuBLASLt split-K with the OUTPUT_TYPE
    reduction, at every packed R in rows_of[L] whose pick is a cuBLASLt algo of that class and that is not in `skip`
    (RARN's sites): SKR's (hn, normed) must equal the pick + fused.add_rmsnorm byte for byte (plain and x64-outlier
    rows, workspaces zeroed before every check, three runs on two workspaces)."""
    from . import fused as fz
    from . import ng
    from .lt import cfg_of

    out, fails = {}, []
    wss = [torch.empty(max(pinned.ws_bytes, 1 << 20), dtype=torch.uint8, device="cuda") for _ in range(2)]
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    for (N, K, L), pin in sorted(pinned.algo.items(), key=lambda kv: kv[0]):
        if N != N_OUT or pin is None or isinstance(pin, ng.NgAlgo) or L not in rows_of or (N, K) not in weights:
            continue
        pc = cfg_of(pin)
        S, red = pc.get("splitK", 1), pc.get("reduction", 0)
        if S < 2 or red != 4:
            continue
        wl = weights[(N, K)]
        W = wl[0] if isinstance(wl, (list, tuple)) else wl
        for R in sorted(rows_of[L]):
            if (K, L, R) in skip:
                continue
            algo = pinned.algo_r.get((N, K, L, R), pin)
            if isinstance(algo, ng.NgAlgo):
                continue
            if getattr(algo, "is_kf", False):
                # S1+KF: a (K, bucket, R) KF took over keeps KF's GEMM + add_rmsnorm (KF writes y itself: there are
                # no cuBLASLt partials to fold); as own-final keeps SKR out of MX's two-class shapes
                continue
            rc = cfg_of(algo)
            if rc.get("splitK", 1) != S or rc.get("reduction", 0) != 4 or S * R * N * 2 > wss[0].numel():
                continue
            ok = True
            try:
                for outl in (False, True):
                    x, h = _inputs(g, R, K, outl)
                    y = torch.empty((R, N), dtype=torch.bfloat16, device="cuda")
                    wss[0].zero_()
                    ctx.run(algo, W, x, y, wss[0])
                    he, xe = fz.add_rmsnorm(h, y, w1p, eps)
                    for wi in (0, 0, 1):
                        wss[wi].zero_()
                        hn, xn = apply(ctx, algo, W, x, h, w1p, eps, S, wss[wi])
                        if not (torch.equal(hn.view(torch.int16), he.view(torch.int16))
                                and torch.equal(xn.view(torch.int16), xe.view(torch.int16))):
                            ok = False
            except RuntimeError as exc:
                ok = False
                fails.append(f"{K}@{L}/R{R}: {exc}")
            if ok:
                out[(K, L, R)] = (algo, S)
            elif len(fails) < 8:
                fails.append(f"{K}@{L}/R{R}")
    del wss
    if fails:
        log(f"SKR not used at {len(fails)} (K, bucket, R): {fails[:8]}")
    return out


# ------------------------------------------------------------------------------------------------ graph surgery
CU_GRAPH_NODE_TYPE_KERNEL = 0


class _KParams(C.Structure):   # CUDA_KERNEL_NODE_PARAMS_v2
    _fields_ = [("func", C.c_void_p), ("gridDimX", C.c_uint), ("gridDimY", C.c_uint), ("gridDimZ", C.c_uint),
                ("blockDimX", C.c_uint), ("blockDimY", C.c_uint), ("blockDimZ", C.c_uint),
                ("sharedMemBytes", C.c_uint), ("kernelParams", C.c_void_p), ("extra", C.c_void_p),
                ("kern", C.c_void_p), ("ctx", C.c_void_p)]


@functools.lru_cache(maxsize=1)
def _cu():
    return C.CDLL("libcuda.so.1")


def _kname(cu, node) -> str:
    t = C.c_int()
    if cu.cuGraphNodeGetType(C.c_void_p(node), C.byref(t)) != 0 or t.value != CU_GRAPH_NODE_TYPE_KERNEL:
        return ""
    p = _KParams()
    if cu.cuGraphKernelNodeGetParams_v2(C.c_void_p(node), C.byref(p)) != 0:
        return ""
    nm = C.c_char_p()
    if p.func and cu.cuFuncGetName(C.byref(nm), C.c_void_p(p.func)) == 0 and nm.value:
        return nm.value.decode(errors="replace")
    if p.kern and cu.cuKernelGetName(C.byref(nm), C.c_void_p(p.kern)) == 0 and nm.value:
        return nm.value.decode(errors="replace")
    return ""


def _edges(cu, node, fn: str) -> list:
    n = C.c_size_t(0)
    getattr(cu, fn)(C.c_void_p(node), None, C.byref(n))
    if not n.value:
        return []
    arr = (C.c_void_p * n.value)()
    if getattr(cu, fn)(C.c_void_p(node), arr, C.byref(n)) != 0:
        raise RuntimeError(f"{fn} failed")
    return [arr[i] for i in range(n.value)]


def prune(graph) -> dict:
    """graph: a torch CUDAGraph captured with keep_graph=True, not yet instantiated. Deletes every splitKreduce node
    whose only child is an SKR fold (_skr_kernel), wiring its parents to its children first. Returns counts."""
    cu = _cu()
    g = C.c_void_p(graph.raw_cuda_graph())
    n = C.c_size_t(0)
    cu.cuGraphGetNodes(g, None, C.byref(n))
    nodes = (C.c_void_p * n.value)()
    cu.cuGraphGetNodes(g, nodes, C.byref(n))
    st = {"folds": 0, "pruned": 0, "kept": 0}
    folds = [nodes[i] for i in range(n.value) if FOLD_NAME in _kname(cu, nodes[i])]   # before any node is destroyed
    for fold in folds:
        st["folds"] += 1
        par = _edges(cu, fold, "cuGraphNodeGetDependencies")
        if len(par) != 1 or "splitKreduce" not in _kname(cu, par[0]):
            st["kept"] += 1
            continue
        red = par[0]
        kids = _edges(cu, red, "cuGraphNodeGetDependentNodes")
        if kids != [fold]:
            st["kept"] += 1
            continue
        ups = _edges(cu, red, "cuGraphNodeGetDependencies")
        if not ups:
            st["kept"] += 1
            continue
        frm = (C.c_void_p * len(ups))(*ups)
        to = (C.c_void_p * len(ups))(*([fold] * len(ups)))
        if cu.cuGraphAddDependencies(g, frm, to, C.c_size_t(len(ups))) != 0:
            raise RuntimeError("cuGraphAddDependencies failed")
        if cu.cuGraphDestroyNode(C.c_void_p(red)) != 0:
            raise RuntimeError("cuGraphDestroyNode failed")
        st["pruned"] += 1
    return st
