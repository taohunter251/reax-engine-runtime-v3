"""RN (ruj r2 hotspot, 2026-10-21): exact single-pass row-norm kernels (rn.cubin, rn.cu) for the packed forward.

The nsys profile of v14a (v16 + fc8, 08:16 UTC) put the row-normalisation kernels at 145 us per decision (exclusive,
2143 MHz): fused._rmsnorm_kernel (add + RMSNorm, 38.8 launches per lane graph) 57, rarn._rarn_kernel (split-K fold +
residual + RMSNorm, 25.1 per graph) 53, fused._gated_norm_packed_kernel (GDN gated norm, 24 per graph) 35. UID 198 runs
one fused reduce + residual + RMSNorm kernel (~100 us per decision) and no gated-norm kernel at all; the gated norm is
the one op where ours still costs more than 198's (+36 us per decision).

The Triton kernels read every input twice (pass 1 sums the squares, pass 2 re-reads the row to normalise it) with
2-byte loads per thread. rn.cu keeps the row in registers (one warp per row / per (row, head), 8-byte loads, each input
read once) and does the same arithmetic in the same order (rn.cu header). Every kernel is used only where check()
finds it byte-equal to the Triton kernel it replaces (random rows at every packed R incl. outliers, zero rows and tiny
values, every RARN slice count in use, both z strides, the model's own norm weights); anything else = today's path.

Env: S1ENG_RN=1 turns it on (start-up gate in model.capture_batch); S1ENG_RN_PARTS = comma list of "norm" (add /
fold / plain RMSNorm) and "gnorm" (gated norm), default both; S1ENG_RN_KIND = "u" (fully unrolled kernels rn_*, the
default before h2) or "c" (compact kernels rnc_*: 5x unrolled loops, the row staged in shared memory, 3-4x less code
- the engine's GEMMs evict the instruction cache between layers) or "w" (the default since h2: rnw_*, one row per CTA
of 128 threads, thread k owning column 128 j + k as in the Triton tile - rn_* / rnc_* run one warp per row and are
latency-bound at the served R: 1.4-2.3x slower than Triton in the u1 microbench); S1ENG_RN_GHPW = head-rows per warp
of the gated-norm kernel (1 / 2 / 4 / 8; default 4 for kind u, 1 otherwise: hpw 1 is 7-11 % faster than Triton). Both kinds are gated at start-up; a kind that fails is
never used (its forward keeps the Triton kernels). FCAB levers: rn (the RMSNorm kernels), rng (the gated norm): arms
norn / norng; toggles rnk=<u|c>, rngh=<hpw>.
"""
from __future__ import annotations

import ctypes as C
import functools
import os

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
CUBIN = os.path.join(_DIR, "rn.cubin")
D = 2560
ROWS = 4          # rows per CTA of the RMSNorm kernels (rn.cu RN_ROWS)
INV_D = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(float(D), dtype=torch.float32))
INV_128 = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(128.0, dtype=torch.float32))


def parts_env() -> set:
    return {p for p in os.environ.get("S1ENG_RN_PARTS", "norm,gnorm").replace(" ", "").split(",") if p}


KINDS = ("w", "u", "c")


def kind() -> str:
    k = os.environ.get("S1ENG_RN_KIND", "w")
    return k if k in KINDS else "w"


def ghpw() -> int:
    v = int(os.environ.get("S1ENG_RN_GHPW", "4" if kind() == "u" else "1"))
    return v if v in (1, 2, 4, 8) else 1


OK = {}    # kind -> True where check_norm passed for that kind (set by the start-up gate)
GOK = {}   # hpw -> True where check_gnorm passed for that hpw


def usable(what: str) -> bool:
    """the current kind (norm) / hpw (gnorm) passed its start-up gate"""
    return bool(OK.get(kind())) if what == "norm" else bool(GOK.get(ghpw()))


class _Lib:
    def __init__(self):
        self.cu = C.CDLL("libcuda.so.1")
        torch.zeros(1, device="cuda")
        data = open(CUBIN, "rb").read()
        self._img = C.create_string_buffer(data, len(data))
        self.mod = C.c_void_p()
        self._chk(self.cu.cuModuleLoadData(C.byref(self.mod), self._img), "cuModuleLoadData")
        self.fn = {}
        names = (["rn_plain", "rn_add"] + [f"rn_fold{s}" for s in range(2, 9)] + [f"rn_gnorm{h}" for h in (1, 2, 4, 8)]
                 + ["rnc_plain", "rnc_add", "rnc_fold", "rnw_plain", "rnw_add"] + [f"rnw_fold{s}" for s in range(2, 9)])
        for name in names:
            f = C.c_void_p()
            self._chk(self.cu.cuModuleGetFunction(C.byref(f), self.mod, name.encode()), name)
            self.fn[name] = f

    @staticmethod
    def _chk(r, what):
        if r != 0:
            raise RuntimeError(f"RN {what}: CUresult {r}")

    def launch(self, name, grid, block, args):
        params = (C.c_void_p * len(args))(*[C.cast(C.pointer(a), C.c_void_p) for a in args])
        self._chk(self.cu.cuLaunchKernel(self.fn[name], grid, 1, 1, block, 1, 1, 0,
                                         C.c_void_p(torch.cuda.current_stream().cuda_stream), params, None),
                  f"launch {name}")


@functools.lru_cache(maxsize=1)
def lib():
    if not os.path.exists(CUBIN):
        return None
    try:
        if tuple(torch.cuda.get_device_capability(0)) != (12, 0):
            return None
        return _Lib()
    except Exception as exc:  # noqa: BLE001 - no RN on any loading problem
        print(f"[s1eng] RN kernels unavailable: {exc!r}", flush=True)
        return None


def _ok_rows(*ts) -> bool:
    return all(t is not None and t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous() and t.shape[-1] == D
               and t.data_ptr() % 16 == 0 for t in ts)


def _w_ok(w) -> bool:
    return w.is_cuda and w.dtype == torch.float32 and w.is_contiguous() and w.numel() == D and w.data_ptr() % 16 == 0


def _p(t):
    return C.c_void_p(t.data_ptr())


def rows_ok(h, y, w) -> bool:
    """the RMSNorm kernels take these tensors (else the caller keeps the Triton path)"""
    if lib() is None or not _w_ok(w) or not usable("norm"):
        return False
    if y is None:
        return _ok_rows(h)
    return _ok_rows(h, y) and h.shape == y.shape


def rmsnorm(x: torch.Tensor, w1p: torch.Tensor, eps: float) -> torch.Tensor:
    """fused.rmsnorm (HAS_Y False) through rn_plain"""
    L = lib()
    if L is None or not _ok_rows(x) or not _w_ok(w1p):
        raise RuntimeError("RN rmsnorm: unavailable or unsupported layout")
    R = x.numel() // D
    out = torch.empty_like(x)
    if R:
        k = kind()
        L.launch({"u": "rn_plain", "c": "rnc_plain", "w": "rnw_plain"}[k], R if k == "w" else -(-R // ROWS),
                 128 if k == "w" else 32 * ROWS, [_p(x), _p(out), _p(w1p), C.c_float(eps), C.c_float(INV_D), C.c_int(R)])
    return out


def add_rmsnorm(h: torch.Tensor, y: torch.Tensor, w1p: torch.Tensor, eps: float):
    """fused.add_rmsnorm (HAS_Y) through rn_add -> (hn, out)"""
    L = lib()
    if L is None or not _ok_rows(h, y) or h.shape != y.shape or not _w_ok(w1p):
        raise RuntimeError("RN add_rmsnorm: unavailable or unsupported layout")
    R = h.numel() // D
    hn = torch.empty_like(h)
    out = torch.empty_like(h)
    if R:
        k = kind()
        L.launch({"u": "rn_add", "c": "rnc_add", "w": "rnw_add"}[k], R if k == "w" else -(-R // ROWS),
                 128 if k == "w" else 32 * ROWS,
                 [_p(h), _p(y), _p(hn), _p(out), _p(w1p), C.c_float(eps), C.c_float(INV_D), C.c_int(R)])
    return hn, out


def fold_rmsnorm(part: torch.Tensor, h: torch.Tensor, w1p: torch.Tensor, eps: float, S: int):
    """rarn._rarn_kernel through rn_fold<S>: part = [S][R][2560] bf16 slice partials (slice stride R * 2560)"""
    L = lib()
    R = h.shape[0]
    if (L is None or not _ok_rows(h) or not _w_ok(w1p) or not (2 <= S <= 8) or part.dtype != torch.bfloat16
            or not part.is_contiguous() or part.numel() < S * R * D or part.data_ptr() % 16):
        raise RuntimeError("RN fold: unavailable or unsupported layout")
    hn = torch.empty_like(h)
    out = torch.empty_like(h)
    if R and kind() == "w":
        L.launch(f"rnw_fold{S}", R, 128,
                 [_p(part), C.c_longlong(R * D), _p(h), _p(hn), _p(out), _p(w1p), C.c_float(eps), C.c_float(INV_D),
                  C.c_int(R)])
    elif R and kind() == "u":
        L.launch(f"rn_fold{S}", -(-R // ROWS), 32 * ROWS,
                 [_p(part), C.c_longlong(R * D), _p(h), _p(hn), _p(out), _p(w1p), C.c_float(eps), C.c_float(INV_D),
                  C.c_int(R)])
    elif R:
        L.launch("rnc_fold", -(-R // ROWS), 32 * ROWS,
                 [_p(part), C.c_longlong(R * D), C.c_int(S), _p(h), _p(hn), _p(out), _p(w1p), C.c_float(eps),
                  C.c_float(INV_D), C.c_int(R)])
    return hn, out


def rarn_apply(name: str, W: torch.Tensor, x: torch.Tensor, h: torch.Tensor, w1p: torch.Tensor, eps: float, S: int,
               ws: torch.Tensor):
    """rarn.apply with the RN norm kernels: kind p -> the ngr GEMM's slice partials + rn_fold<S>; kind i -> the ngr
    GEMM's hn = bf16(h + x @ W^T) + rn_plain."""
    from . import rarn

    assert h.is_contiguous() and h.shape[-1] == rarn.N_OUT
    if rarn.KERNELS[name][0] == "p":
        part = rarn.run(name, W, x, None, None, S, ws)
        return fold_rmsnorm(part.view(-1), h, w1p, eps, S)
    hn = torch.empty_like(h)
    rarn.run(name, W, x, h, hn, S, ws)
    return hn, rmsnorm(hn, w1p, eps)


def gated_rmsnorm_packed(o_pad: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float, row_pos: torch.Tensor,
                         nh: int, hpw: int | None = None) -> torch.Tensor:
    """fused.gated_rmsnorm_packed through rn_gnorm<hpw>: o_pad [B, L, 32, 128] (padded layout), z [R, 4096] (row
    stride may be larger), w bf16 [128], row_pos int32 [R] -> out [R * 32, 128]"""
    L = lib()
    hpw = hpw or ghpw()
    if (L is None or nh != 32 or not o_pad.is_contiguous() or o_pad.dtype != torch.bfloat16 or z.dim() != 2
            or z.stride(1) != 1 or z.shape[1] != nh * 128 or z.dtype != torch.bfloat16 or z.data_ptr() % 8
            or (z.stride(0) * 2) % 8 or o_pad.data_ptr() % 8 or w.dtype != torch.bfloat16 or w.numel() != 128
            or not w.is_contiguous() or w.data_ptr() % 8 or row_pos.dtype != torch.int32 or not row_pos.is_contiguous()):
        raise RuntimeError("RN gated norm: unavailable or unsupported layout")
    n = z.shape[0] * nh
    out = torch.empty((n, 128), dtype=z.dtype, device=z.device)
    if n:
        warps = 4
        L.launch(f"rn_gnorm{hpw}", -(-n // (warps * hpw)), 32 * warps,
                 [_p(o_pad), _p(z), _p(w), _p(out), _p(row_pos), C.c_int(n), C.c_float(eps), C.c_float(INV_128),
                  C.c_int(z.stride(0))])
    return out


# --------------------------------------------------------------------------------------------- start-up gate
def _rows(R: int, g: torch.Generator, kind: int) -> torch.Tensor:
    """random bf16 rows [R, 2560] with the value ranges the model produces and a few hostile ones"""
    x = torch.randn((R, D), generator=g, device="cuda", dtype=torch.float32)
    sc = (1.0, 8.0, 0.02, 3e-3)[kind % 4]
    x = x * sc
    if R:
        x[torch.randint(0, R, (max(1, R // 8),), generator=g, device="cuda"),
          torch.randint(0, D, (max(1, R // 8),), generator=g, device="cuda")] *= 64.0
        if kind % 5 == 1:
            x[0] = 0.0                       # an all-zero row: rsqrt(eps)
        if kind % 7 == 2 and R > 1:
            x[1] *= 1e-30                    # tiny squares (denormal sums)
    return x.to(torch.bfloat16)


def _eq(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


@torch.inference_mode()
def check_norm(fz, rarn_mod, weights, eps: float, rs, ss, seed: int = 2560, kinds=KINDS) -> str:
    """both kernel kinds (OK[kind] = passed): "" if the env's kind passed, else its first difference"""
    if lib() is None:
        return "rn.cubin not loadable"
    old = os.environ.get("S1ENG_RN_KIND")
    res = {}
    try:
        for k in kinds:
            os.environ["S1ENG_RN_KIND"] = k
            OK[k] = True   # rows_ok() inside the check needs it; reset below on failure
            res[k] = _check_norm1(fz, rarn_mod, weights, eps, rs, ss, seed)
            OK[k] = not res[k]
    finally:
        if old is None:
            os.environ.pop("S1ENG_RN_KIND", None)
        else:
            os.environ["S1ENG_RN_KIND"] = old
    bad = {k: v for k, v in res.items() if v}
    return (res.get(kind()) or "") + (f" (other kinds: {bad})" if bad and not res.get(kind()) else "")


def _check_norm1(fz, rarn_mod, weights, eps: float, rs, ss, seed: int) -> str:
    """rn_plain / rn_add / rn_fold<S> (or the rnc_* kind) vs fused.rmsnorm / fused.add_rmsnorm / rarn._rarn_kernel,
    every R in rs (and S in ss for the fold), several input kinds and the given w1p weights: "" if every byte agrees."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    for n, R in enumerate(rs):
        w = weights[n % len(weights)]
        h = _rows(R, g, n)
        y = _rows(R, g, n + 1)
        ref = fz.rmsnorm(h, w, eps)
        got = rmsnorm(h, w, eps)
        if not _eq(ref, got):
            return f"plain RMSNorm differs (R={R})"
        r_hn, r_out = fz.add_rmsnorm(h, y, w, eps)
        g_hn, g_out = add_rmsnorm(h, y, w, eps)
        if not (_eq(r_hn, g_hn) and _eq(r_out, g_out)):
            return f"add + RMSNorm differs (R={R})"
        for S in ss:
            part = torch.stack([_rows(R, g, n + s + 2) for s in range(S)]).contiguous()
            r_hn = torch.empty_like(h)
            r_out = torch.empty_like(h)
            if R:
                rarn_mod._rarn_kernel[(R,)](part, h, r_hn, r_out, w, eps, rarn_mod.INV_D, R * D, S=S, num_warps=4,
                                            **rarn_mod.EXACT)
            g_hn, g_out = fold_rmsnorm(part.view(-1), h, w, eps, S)
            if not (_eq(r_hn, g_hn) and _eq(r_out, g_out)):
                return f"RARN fold + add + RMSNorm differs (R={R}, S={S})"
    torch.cuda.synchronize()
    return ""


@torch.inference_mode()
def check_gnorm(fz, weights, eps: float, cases, seed: int = 128, hpws=(1, 2, 4, 8)) -> str:
    """rn_gnorm<hpw> vs fused.gated_rmsnorm_packed on random packed layouts: cases = [(B, L, R)], z with row stride
    12288 (the z block of the fused qkv|z output) and 4096 (separate z), the given weights, inputs at several scales
    (incl. |z| up to 100: expf over- / underflow), every hpw in hpws (GOK[hpw] = passed): "" if the env's hpw
    passed, else its first difference."""
    if lib() is None:
        return "rn.cubin not loadable"
    res = {hp: "" for hp in hpws}
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    for n, (B, L, R) in enumerate(cases):
        if R <= 0:
            continue
        # packed row -> padded row (b * L + position): random valid rows (the arithmetic per head-row does not depend
        # on the mapping; the gather is exercised)
        rp = torch.randint(0, B * L, (R,), generator=g, device="cuda").to(torch.int32).contiguous()
        sc = (1.0, 8.0, 0.05, 30.0)[n % 4]
        o_pad = (torch.randn((B, L, 32, 128), generator=g, device="cuda") * sc).to(torch.bfloat16)
        if n % 3 == 1:
            o_pad[0, 0, :4] = 0.0
        full = (torch.randn((R, 12288), generator=g, device="cuda") * (4.0 if n % 2 else 1.0)).to(torch.bfloat16)
        full[:, 8192 + 7] = torch.linspace(-100.0, 100.0, R, device="cuda").to(torch.bfloat16)
        for z in (full[:, 8192:], full[:, 8192:].contiguous()):
            for w in weights:
                ref = fz.gated_rmsnorm_packed(o_pad, z, w, eps, rp, 32)
                for hp in hpws:
                    if res[hp]:
                        continue
                    got = gated_rmsnorm_packed(o_pad, z, w, eps, rp, 32, hpw=hp)
                    if not _eq(ref, got):
                        res[hp] = f"gated norm differs (B={B}, L={L}, R={R}, z stride {z.stride(0)}, hpw {hp})"
    torch.cuda.synchronize()
    for hp in hpws:
        GOK[hp] = not res[hp]
    bad = {k: v for k, v in res.items() if v}
    return (res.get(ghpw()) or "") + (f" (other hpw: {bad})" if bad and not res.get(ghpw()) else "")
