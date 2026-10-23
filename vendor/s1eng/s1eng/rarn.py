"""RARN (rgh 2026-10-20; idea from on-chain UID 198's split-K reduce-add-rmsnorm): the down_proj / out_proj / o_proj
GEMMs (N = 2560; their pins are split-K with the OUTPUT_TYPE reduction) hand their K-slice partials straight to the
add_rmsnorm that follows: no cuBLASLt splitKreduce launch, no y round trip through memory.

Today (S6): y = GEMM at the S6 pick (a cuBLASLt parallel split-K algo = S slice partials + a splitKreduce launch, or an
in-CTA split NG kernel ns9_*); then fused.add_rmsnorm(h, y) = (hn = bf16(h + y), rmsnorm(hn)).
RARN, through ngr.cubin (ngr.cu, nvcc 13.4 -arch=sm_120a; mainloop = ngemm9b's ns9_*: fp32 accumulation over each ceil64
K slice in ascending k16 order through mma.sync m16n8k16 from 0, the OUTPUT_TYPE fold tot = bf16(acc_0); tot +=
bf16(acc_s) in slice order; one RN round):
  * cuBLASLt split pick -> nr9p_* / nr9h_*: one CTA per (output tile, K slice), each stores its bf16 slice partial;
    then _rarn_kernel folds the S partials in slice order, adds the residual and normalises in one pass (the
    _rmsnorm_kernel(HAS_Y) arithmetic and reduction order with y = the fold);
  * in-CTA split pick -> nsq9_*: ns9_*'s kernel with the residual add in its epilogue, then fused.rmsnorm on hn (pass 1
    squares float(bf16(h + y)) either way, so the bits equal the HAS_Y path).
Used per (K, bucket, R) only where select()'s start-up byte check against today's path (the S6 pick at R +
add_rmsnorm, random rows incl. x64 outliers) passes; anything missing (cubin, driver call, sm_120) = today's path.
S1ENG_RARN=0 disables; S1ENG_RARN_RULE picks the kernels (rule_from_env: "plan" (default) / "plan:<file>" / "auto" /
explicit).
"""
from __future__ import annotations

import ctypes as C
import functools
import json
import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .fused import EXACT, INV_D, _cols4_tree, _warp_tree32

_DIR = os.path.dirname(os.path.abspath(__file__))
CUBIN = os.path.join(_DIR, "ngr.cubin")
N_OUT = 2560
# name: (kind, BM, BN, BK, stages, threads, small tile (BM, BN, BK, stages) or None); kind p = per-slice CTAs storing
# bf16 slice partials (fold + residual + RMSNorm in _rarn_kernel), i = in-CTA split + residual epilogue. nr9h_*: rows
# [0, mb) in the big tiles, the last row block in the small tiles (no MMA work on the empty rows of a 128-row tile
# past R); mb rides in the kernel's ldh argument.
KERNELS = {
    "nr9p_128x256k64": ("p", 128, 256, 64, 2, 256, None),
    "nr9p_128x128k64": ("p", 128, 128, 64, 3, 256, None),
    "nr9h_b128x256_s32x256k64": ("p", 128, 256, 64, 2, 256, (32, 256, 64, 2)),
    "nr9h_b128x256_s64x256k64": ("p", 128, 256, 64, 2, 256, (64, 256, 64, 2)),
    "nr9h_b128x128_s32x128k64": ("p", 128, 128, 64, 3, 256, (32, 128, 64, 3)),
    "nr9h_b128x128_s64x128k64": ("p", 128, 128, 64, 3, 256, (64, 128, 64, 3)),
    "nsq9_4_128x128k64": ("i", 128, 128, 64, 3, 256, None),
}


def _smem(bm, bn, bk, st):
    return max(st * (bm + bn) * bk * 2, bm * (bn + 8) * 2)


class _Lib:
    def __init__(self):
        self.cu = C.CDLL("libcuda.so.1")
        torch.zeros(1, device="cuda")
        data = open(CUBIN, "rb").read()
        self._img = C.create_string_buffer(data, len(data))
        self.mod = C.c_void_p()
        self._chk(self.cu.cuModuleLoadData(C.byref(self.mod), self._img), "cuModuleLoadData")
        self.fn = {}
        for name, (kind, bm, bn, bk, st, th, small) in KERNELS.items():
            f = C.c_void_p()
            self._chk(self.cu.cuModuleGetFunction(C.byref(f), self.mod, name.encode()), name)
            sm = max(_smem(bm, bn, bk, st), _smem(*small) if small else 0)
            self._chk(self.cu.cuFuncSetAttribute(f, 8, sm), "smem attribute")
            self.fn[name] = (f, sm)
        self.bufs: dict = {}   # workspace data_ptr -> bf16 partials; kept alive (graphs bake the pointers)
        self.keep: list = []

    @staticmethod
    def _chk(r, what):
        if r != 0:
            raise RuntimeError(f"RARN {what}: CUresult {r}")

    def part_for(self, ws: torch.Tensor, need: int) -> torch.Tensor:
        """partials buffer per lane workspace (lanes replay concurrently; one GEMM at a time per lane stream)"""
        key = ws.data_ptr()
        cur = self.bufs.get(key)
        if cur is None or cur.numel() < need:
            cur = torch.empty(max(need, 1 << 20), dtype=torch.bfloat16, device="cuda")
            self.bufs[key] = cur
            self.keep.append((cur, key))
        return cur


@functools.lru_cache(maxsize=1)
def lib():
    if os.environ.get("S1ENG_RARN", "1") != "1" or not os.path.exists(CUBIN):
        return None
    try:
        if tuple(torch.cuda.get_device_capability(0)) != (12, 0):
            return None
        return _Lib()
    except Exception as exc:  # noqa: BLE001 - no RARN on any loading problem
        print(f"[s1eng] RARN kernels unavailable: {exc!r}", flush=True)
        return None


def chunk_of(K: int, S: int) -> int:
    return -(-(K // 64) // S) * 64


def supported(K: int, S: int, red: int) -> bool:
    return red == 4 and 2 <= S <= 8 and K % 64 == 0 and (S - 1) * chunk_of(K, S) < K


def run(name: str, W: torch.Tensor, x: torch.Tensor, h, hn, S: int, ws: torch.Tensor):
    """kind p: the [S, M, N] bf16 slice partials of x @ W^T (per lane buffer); kind i: hn = bf16(h + x @ W^T) with
    the split pin's arithmetic (S slices, OUTPUT_TYPE fold)."""
    L = lib()
    M, K = x.shape
    N = W.shape[0]
    kind, bm, bn, bk, st, th, small = KERNELS[name]
    if kind == "p":
        h = hn = None
    if (L is None or N % bn or (small and N % small[1]) or x.stride(1) != 1 or W.stride(0) != K or W.stride(1) != 1
            or x.data_ptr() % 16 or (x.stride(0) * 2) % 16
            or (kind == "i" and (h.stride(1) != 1 or hn.stride(1) != 1 or h.data_ptr() % 16 or hn.data_ptr() % 16
                                 or (h.stride(0) * 2) % 16 or (hn.stride(0) * 2) % 16 or hn.shape != h.shape))):
        raise RuntimeError("RARN GEMM: unavailable or unsupported layout")
    f, sm = L.fn[name]
    chunk = chunk_of(K, S)
    mb = 0
    if small:
        mb = (M // bm) * bm
        tiles = (mb // bm) * (N // bn) + (-(-(M - mb) // small[0])) * (N // small[1])
    else:
        tiles = -(-M // bm) * (N // bn)
    part = None
    if kind == "p":
        part = L.part_for(ws, S * M * N)
        grid = tiles * S
    else:
        grid = tiles
    args = [C.c_void_p(x.data_ptr()), C.c_void_p(W.data_ptr()), C.c_void_p(hn.data_ptr() if hn is not None else 0),
            C.c_int(M), C.c_int(N), C.c_int(K), C.c_int(x.stride(0)), C.c_int(hn.stride(0) if hn is not None else N),
            C.c_int(chunk), C.c_int(S), C.c_void_p(part.data_ptr() if part is not None else 0), C.c_void_p(0),
            C.c_void_p(h.data_ptr() if h is not None else 0),
            C.c_int(mb if small else (h.stride(0) if h is not None else 0))]
    params = (C.c_void_p * len(args))(*[C.cast(C.pointer(a), C.c_void_p) for a in args])
    L._chk(L.cu.cuLaunchKernel(f, grid, 1, 1, th, 1, 1, sm, C.c_void_p(torch.cuda.current_stream().cuda_stream),
                               params, None), "launch")
    return part[:S * M * N].view(S, M, N) if kind == "p" else hn


@triton.jit
def _rarn_kernel(part, x, hout, out, w, eps, inv_d, pstride, S: tl.constexpr):
    """fused._rmsnorm_kernel(HAS_Y) with y = the split-K fold of part [S][R][2560] (bf16 slice partials, OUTPUT_TYPE
    scheme: y = bf16(((p0 + p1) + p2) + ...) in f32): hb = bf16(x + y) -> hout; out = bf16((hb * rsqrt(mean(hb^2) +
    eps)) * w) with _rmsnorm_kernel's reduction order; pass 2 recomputes hb from the partials (no store in pass 1)."""
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


def apply(name: str, W: torch.Tensor, x: torch.Tensor, h: torch.Tensor, w1p: torch.Tensor, eps: float, S: int,
          ws: torch.Tensor):
    """fused.add_rmsnorm(h, x @ W^T at the split pin's arithmetic, w1p, eps) through kernel `name` -> (hn, normed)."""
    from . import fused as fz

    assert h.is_contiguous() and h.shape[-1] == N_OUT
    hn = torch.empty_like(h)
    if KERNELS[name][0] == "p":
        part = run(name, W, x, None, None, S, ws)
        out = torch.empty_like(h)
        R = h.shape[0]
        _rarn_kernel[(R,)](part, h, hn, out, w1p, eps, INV_D, R * N_OUT, S=S, num_warps=4, **EXACT)
        return hn, out
    run(name, W, x, h, hn, S, ws)
    return hn, fz.rmsnorm(hn, w1p, eps)


class Deferred:
    """A GEMM whose output only feeds add_rmsnorm (PackedForward.add_norm folds it there)."""
    __slots__ = ("W", "x", "name", "S", "L")

    def __init__(self, W, x, name, S, L):
        self.W, self.x, self.name, self.S, self.L = W, x, name, S, L


def _inputs(g, R, K, outliers):
    x = torch.randn((R, K), generator=g, device="cuda")
    h = torch.randn((R, N_OUT), generator=g, device="cuda") * 3
    if outliers:   # x64 rows / columns of x and rows of h (activation outliers)
        x[torch.randint(0, R, (max(1, R // 50),), generator=g, device="cuda")] *= 64
        x[:, torch.randint(0, K, (32,), generator=g, device="cuda")] *= 64
        h[torch.randint(0, R, (max(1, R // 40),), generator=g, device="cuda")] *= 64
    return x.to(torch.bfloat16), h.to(torch.bfloat16)


@torch.inference_mode()
def select(pinned, ctx, weights: dict, rows_of: dict, w1p: torch.Tensor, eps: float, rule, log=print,
           seed: int = 29) -> dict:
    """{(K, L, R): (kernel, S)} for the N=2560 (K, bucket) pins that are split-K with the OUTPUT_TYPE fold, at every
    packed row count R in rows_of[L]: the first kernel of rule(K, L, R, S, today's algo) whose (hn, normed) equals
    today's (the S6 pick at R, then fused.add_rmsnorm) byte for byte on random rows (plain and x64 outliers; each
    kernel launched three times on two workspaces). weights[(N, K)]: a layer weight (or a list; the first is used)."""
    from . import fused as fz
    from .lt import cfg_of

    if lib() is None:
        return {}
    out, fails = {}, []
    wss = [torch.empty(max(pinned.ws_bytes, 1 << 20), dtype=torch.uint8, device="cuda") for _ in range(2)]
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    try:
        for (N, K, L), pin in sorted(pinned.algo.items(), key=lambda kv: kv[0]):
            if N != N_OUT or pin is None or L not in rows_of or (N, K) not in weights:
                continue
            rc = cfg_of(pin)
            S, red = rc.get("splitK", 1), rc.get("reduction", 0)
            if not supported(K, S, red):
                continue
            wl = weights[(N, K)]
            W = wl[0] if isinstance(wl, (list, tuple)) else wl
            for R in sorted(rows_of[L]):
                algo = pinned.algo_r.get((N, K, L, R), pin)
                for name in rule(K, L, R, S, algo):
                    ok = True
                    try:
                        for outl in (False, True):
                            x, h = _inputs(g, R, K, outl)
                            y = torch.empty((R, N), dtype=torch.bfloat16, device="cuda")
                            ctx.run(algo, W, x, y, wss[0])
                            he, xe = fz.add_rmsnorm(h, y, w1p, eps)
                            for wi in (0, 0, 1):
                                hn, xn = apply(name, W, x, h, w1p, eps, S, wss[wi])
                                if not (torch.equal(hn.view(torch.int16), he.view(torch.int16))
                                        and torch.equal(xn.view(torch.int16), xe.view(torch.int16))):
                                    ok = False
                    except RuntimeError as exc:   # e.g. a layout / size the kernel does not take: next / today's
                        ok = False
                        name = f"{name}({exc})"
                    if ok:
                        out[(K, L, R)] = (name, S)
                        break
                    fails.append(f"{K}@{L}/R{R}:{name}")
        torch.cuda.synchronize()
    finally:
        for w in wss:   # the check's buffers are not baked into any graph
            lib().bufs.pop(w.data_ptr(), None)
            lib().keep = [k for k in lib().keep if k[1] != w.data_ptr()]
    if fails:
        log(f"RARN byte check failed at {len(fails)} (K, bucket, R, kernel) (next candidate / today's path): "
            f"{' '.join(fails[:12])}")
    return out


# Policy "plan" (default): the per-(K, bucket, R) kernel of the shipped plan for this GPU model; keys the plan does not
# list (or lists as "-") keep today's path; no plan for this GPU model -> policy "auto". Every planned kernel still has to
# pass select()'s byte check at start-up. Plan rarn_plan_pro6000.json (Verda, idle RTX PRO 6000, 2026-10-21): (1)
# tools/rplan.py times today's segment (S6 pick + add_rmsnorm) and every byte-exact candidate in CUDA graphs on the real
# layer weights, 1 stream and 2 concurrent streams; (2) tools/rpair2.py times whole lane-graph forwards in the engine
# (1 lane, and both lanes concurrently = the engine's regime at concurrency 16) at the frequent (bucket, R) of a
# validator-style run for the plans "mean" / "pair" / "single" / "w73" (micro policies) and today's path; per measured
# (bucket, R) the arm with the best 0.2 x 1-lane + 0.8 x 2-lane gain is kept (today's path unless >= 0.2 % better), the
# other keys use the 2-stream micro policy ("pair": best 2-stream ratio, >= 2 % better). Plans picked on 1-stream
# timing alone lose in the 2-lane regime (wave-tail kernels look slow alone but overlap with the other lane).
# v15 (10-21): the plan RE-DECIDED on the v13b picks (COV9 nlg9 / ns9 / cuBLASLt): per (bucket, R) both lanes' full packed
# forwards with PAR + XA on, timed phase-averaged in pairs (dev_v15/rarnd.py), v13b pick vs the v14 kernel per K (out/o_proj
# 4096, down_proj 9216) and both; the kernel stays only where it wins by > 0.2 % (close calls re-timed with 24 rounds).
PLANS ={"NVIDIA RTX PRO 6000 Blackwell Server Edition": "rarn_plan_pro6000.json"}


def _plan_table(log=print, path: str | None = None):
    try:
        name = path or PLANS.get(torch.cuda.get_device_name())
        if name is None or not os.path.exists(os.path.join(_DIR, name)):
            return None
        with open(os.path.join(_DIR, name)) as f:
            plan = json.load(f)
        return {(int(K), int(L), int(R)): pick for K, L, R, pick, *_ in plan["plan"]}
    except Exception as exc:  # noqa: BLE001 - a broken plan file: the auto policy
        log(f"RARN plan not used ({type(exc).__name__}: {exc})")
        return None


# Which kernel replaces which of today's picks. Policy "auto" (Lium graph-timed micro + in-engine A/B
# 2026-10-20, 1 stream and 2 streams): an in-CTA split pick (ns9_* / ns4_* / ns1_*) -> nsq9; a cuBLASLt split pick ->
# partial kernel + _rarn_kernel with the tile family by R's last row block (R % 128): 32 rows -> hetero 128x128 / 32-row
# tiles at S >= 4 or R < 512, else hetero 128x256 / 32; 64 rows -> hetero 128x128 / 64 below R = 384, else 128x256 /
# 64; 0 or 96 rows -> nr9p_128x256k64 (also the fallback candidate). Or an explicit spec "split=k1|k2,ins=k3" (kernel
# names of KERNELS in order of preference; "-" or nothing = today's path). S1ENG_RARN_RULE overrides.
DEFAULT_RULE = "plan"


def auto_kernels(R: int, S: int, kind: str) -> list:
    if kind == "ins":
        return ["nsq9_4_128x128k64"]
    t = R % 128
    if t == 32:
        n = "nr9h_b128x128_s32x128k64" if (S >= 4 or R < 512) else "nr9h_b128x256_s32x256k64"
    elif t == 64:
        n = "nr9h_b128x128_s64x128k64" if R < 384 else "nr9h_b128x256_s64x256k64"
    else:
        n = "nr9p_128x256k64"
    return [n] + (["nr9p_128x256k64"] if n != "nr9p_128x256k64" else [])


def rule_from_env(spec: str | None = None, log=print):
    spec = os.environ.get("S1ENG_RARN_RULE", DEFAULT_RULE) if spec is None else spec
    path = spec.split(":", 1)[1] if spec.startswith("plan:") else None   # "plan:<file>": that plan file
    table = _plan_table(log, path) if spec == "plan" or path else None
    if (spec == "plan" or path) and table is None:
        spec = "auto"
    want = {k: [n for n in v.split("|") if n in KERNELS] for k, v in
            (kv.split("=", 1) for kv in spec.split(",") if "=" in kv)}

    def rule(K, L, R, S, algo):
        from . import ng

        if getattr(algo, "is_kf", False):   # KF (ruj gemm2/k2): its table key keeps today's path = KF + add_rmsnorm
            return []
        if isinstance(algo, ng.NgAlgo):
            k0 = ng._kspec(algo.name)[0]
            # v15: a COV9 grouped lastfold pick (nlg9_*, kind "LG") is a split-K scheme-4 pick like cuBLASLt's (same
            # slices, same OUTPUT_TYPE fold): the partial kernels replace it where the plan says so (byte-checked)
            kind = "ins" if k0 == "s" else ("split" if k0 == "LG" else None)
        else:
            kind = "split"
        if kind is None:
            return []
        if table is not None:
            pick = table.get((K, L, R), "-")
            return [pick] if pick in KERNELS else []
        if spec == "auto":
            return auto_kernels(R, S, kind)
        return list(want.get(kind, []))
    return rule
