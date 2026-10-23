"""PROFUSE-v (ruj r3, 2026-10-23; the idea and the v staging are rgh's dev_profuse ffsp.cu): ONORM's FFS + gated-norm
kernel computing the GDN value rows v itself from the packed qkv GEMM rows (onormpv.cubin, built from onormpv.cu =
onorm.cu included unchanged + the PV section; nvcc 13.4 -O3 -std=c++17 -arch=sm_120a -DFFS_SHIP -DONORM_V=5 -DONORM_PF2
-DONORM_SILU_FIN -DONORM_FINROLL, the same flags as onorm.cubin), so that the GDN prologue runs PRON with
qk_only=True (pron.cu's q / k head blocks and gate units only: no v rows computed, written or read back).

v is PRON's conv_tile4 arithmetic op for op (FHFMA 4-tap causal conv of the raw rows - prefix-variant cache rows below
plen, packed rows above, +0 outside [0, slen) - then PRON's F9 SiLU; rows >= slen are +0), staged per 64-row chunk in
the kernel's shared memory (raw rows + a 3-row halo, converted at the top of the chunk); every other operation is
ffs_fused5n's (onorm.cubin): the same P / A / B / C stages, the same gated RMSNorm epilogue, the same packed output.

Dispatch (batchfwd._gdn): a packed graph (bucket L, B slots) runs the PV chain (pron qk_only -> KKT2 -> ffs_pv5n) iff
S1ENG_PROFUSE=1 (default off), L passed this module's start-up gate (ON), ONORM itself would run for (L, B) and PRON is
on for L; otherwise BASE's chain (pron full -> KKT2 -> ffs_fused5n) exactly as before. check() is the start-up gate: per
bucket, at every B ONORM runs at, random packed layouts (every given GDN layer's conv weights, input scales 1 / 8 / 0.02,
prefix layouts pfx = 0 and the bucket's pfx .. pmax over 3 variant blocks, chunk-edge prefixes, unused slots, z with
extreme values) through BOTH chains: q, k, beta, g, A of the qk_only prologue == the full prologue's, v of the gate
kernel ffs_pvx (the same staging + conversion code) == the full prologue's v on every row of every slot, and the packed
gated-norm output == BASE's on every real row (int16 views). A bucket with any difference (or any error) stays on
BASE's chain; nothing else changes."""
from __future__ import annotations

import ctypes as C
import functools
import os

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
CUBIN = os.path.join(_DIR, "onormpv.cubin")
KERNEL = "ffs_pv5n"
HV, VD, BT = 32, 128, 64
ON: set = set()   # buckets L whose start-up gate passed


class PVNArgs(C.Structure):   # onormpv.cu struct PVNArgs
    _fields_ = [("q", C.c_void_p), ("k", C.c_void_p), ("beta", C.c_void_p), ("A", C.c_void_p), ("g", C.c_void_p),
                ("slen", C.c_void_p), ("start", C.c_void_p), ("plen", C.c_void_p), ("pvar", C.c_void_p),
                ("z", C.c_void_p), ("nw", C.c_void_p), ("out", C.c_void_p), ("x", C.c_void_p), ("cx", C.c_void_p),
                ("cw", C.c_void_p), ("T", C.c_int), ("hq", C.c_int), ("zs", C.c_int), ("XS", C.c_int),
                ("CXS", C.c_int), ("pfx", C.c_int), ("pmax", C.c_int), ("scale", C.c_float), ("eps", C.c_float)]


def _chk(r, what):
    if r != 0:
        raise RuntimeError(f"PROFUSE {what}: CUresult {r}")


class _Lib:
    def __init__(self, path: str):
        self.cu = C.CDLL("libcuda.so.1")
        torch.zeros(1, device="cuda")
        data = open(path, "rb").read()
        self._img = C.create_string_buffer(data, len(data))
        self.mod = C.c_void_p()
        _chk(self.cu.cuModuleLoadData(C.byref(self.mod), self._img), "cuModuleLoadData")
        self.fns = {}
        f = self._fn("ffs_pv_smem")
        o = torch.zeros(1, dtype=torch.int32, device="cuda")
        a = [C.c_void_p(o.data_ptr())]
        prm = (C.c_void_p * 1)(*[C.cast(C.pointer(x), C.c_void_p) for x in a])
        _chk(self.cu.cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, None, prm, None), "smem query")
        torch.cuda.synchronize()
        self.smem = int(o.item())
        for name in (KERNEL, "ffs_pvx"):
            _chk(self.cu.cuFuncSetAttribute(self._fn(name), 8, self.smem), f"smem attribute {name}")

    def _fn(self, name):
        f = self.fns.get(name)
        if f is None:
            f = C.c_void_p()
            _chk(self.cu.cuModuleGetFunction(C.byref(f), self.mod, name.encode()), name)
            self.fns[name] = f
        return f


def enabled() -> bool:
    return os.environ.get("S1ENG_PROFUSE", "0") == "1"


@functools.lru_cache(maxsize=1)
def lib():
    if not enabled():
        return None
    try:
        if tuple(torch.cuda.get_device_capability(0)) != (12, 0) or not os.path.exists(CUBIN):
            return None
        return _Lib(CUBIN)
    except Exception as exc:  # noqa: BLE001 - the lever must never take the engine down with it
        print(f"[s1eng] PROFUSE kernel unavailable: {exc!r}", flush=True)
        return None


def use(L: int, B: int) -> bool:
    """the PV chain runs for this packed graph (bucket L, B slots); the caller has checked ONORM's own conditions"""
    if L not in ON:
        return False
    from . import onorm

    return onorm.use(L, B) and lib() is not None


def takes(mixed, conv_w, cache, pfx: int) -> bool:
    x = cache if pfx else mixed
    return (mixed.dim() == 2 and mixed.dtype == torch.bfloat16 and mixed.stride(1) == 1
            and mixed.shape[1] == 2 * 2048 + 4096 and mixed.data_ptr() % 16 == 0 and (mixed.stride(0) * 2) % 16 == 0
            and conv_w.dtype == torch.bfloat16 and conv_w.is_contiguous() and tuple(conv_w.shape) == (8192, 4)
            and conv_w.data_ptr() % 16 == 0 and x is not None and x.dim() == 2 and x.stride(1) == 1
            and x.dtype == torch.bfloat16 and x.data_ptr() % 16 == 0 and (x.stride(0) * 2) % 16 == 0
            and mixed.stride(0) < (1 << 31) and x.stride(0) < (1 << 31))


def _args(q, k, g, beta, A, slen, scale, z, w, eps, start, plen, mixed, conv_w, cache, pfx, pvar, pmax, out):
    B, T, Hq, _ = k.shape
    if not pfx:   # pron.prologue's substitutions (cache rows never read: plen 0)
        cache, pvar, pmax = mixed, start, 0
    return PVNArgs(q.data_ptr(), k.data_ptr(), beta.data_ptr(), A.data_ptr(), g.data_ptr(), slen.data_ptr(),
                   start.data_ptr(), plen.data_ptr(), pvar.data_ptr(), z.data_ptr(), w.data_ptr(), out.data_ptr(),
                   mixed.data_ptr(), cache.data_ptr(), conv_w.data_ptr(), T, Hq, z.stride(0), mixed.stride(0),
                   cache.stride(0), int(pfx), int(pmax), float(scale), float(eps))


def _launch(name, P, B, extra=None):
    lb = lib()
    args = [P] + ([C.c_void_p(extra.data_ptr())] if extra is not None else [])
    params = (C.c_void_p * len(args))(*[C.cast(C.pointer(x), C.c_void_p) for x in args])
    _chk(lb.cu.cuLaunchKernel(lb._fn(name), B * HV, 1, 1, 256, 1, 1, lb.smem,
                              C.c_void_p(torch.cuda.current_stream().cuda_stream), params, None), f"launch {name}")


def forward(q, k, g, beta, slen, z, w, eps, start, plen, mixed, conv_w, cache, pfx, pvar, pmax):
    """the PV chain after pron.prologue(qk_only=True): KKT2's A, then ffs_pv5n -> packed out [R * 32, 128]"""
    from . import fla_fused

    B, T, Hq, K = k.shape
    A = torch.empty(B, T, HV, BT, device=k.device, dtype=k.dtype)
    fla_fused.KKT2["nat"](k, g, beta, A, slen=slen)
    out = torch.empty((z.shape[0] * HV, VD), dtype=z.dtype, device=z.device)
    P = _args(q, k, g, beta, A, slen, K ** -0.5, z, w, eps, start, plen, mixed, conv_w, cache, pfx, pvar, pmax, out)
    _launch(KERNEL, P, B)
    return out


def vrows(q, k, g, beta, slen, start, plen, mixed, conv_w, cache, pfx, pvar, pmax, z, w):
    """gate helper: v [B, T, 32, 128] through ffs_pvx (the kernel's raw-row staging + conversion code)"""
    B, T, Hq, K = k.shape
    v = torch.full((B, T, HV, VD), float("nan"), dtype=torch.bfloat16, device=k.device)
    P = _args(q, k, g, beta, beta, slen, 1.0, z, w, 1e-6, start, plen, mixed, conv_w, cache, pfx, pvar, pmax, z)
    _launch("ffs_pvx", P, B, extra=v)
    return v


def _eq(a, b) -> bool:
    vw = torch.int32 if a.dtype == torch.float32 else torch.int16
    return a.shape == b.shape and torch.equal(a.contiguous().view(vw), b.contiguous().view(vw))


@torch.inference_mode()
def check(pf, L: int, bs, layers, pfx: int = 0, pmax: int = 0, seed: int = 71) -> str:
    """PROFUSE-v start-up gate of bucket L on THIS GPU: BASE's chain (pron.prologue full -> KKT2 -> onorm.fused_norm)
    against the PV chain (pron.prologue qk_only -> KKT2 -> forward) on random packed layouts as PackedForward.check_pron
    builds them (every given GDN layer's conv / gate / norm weights, each B in bs, two trials per B with input scales
    1 / 8 / 0.02, an unused slot, prefix lengths pfx .. pmax over 3 variant blocks when pfx > 0, else none) plus, per B,
    one chunk-edge prefix layout (plen 1..3 / 62..65 / 126..129 where L allows, pmax 130) when pfx > 0; z either beside
    the qkv rows (the qkv_z pair's row stride) or in its own tensor, with exp-overflow / zero / tiny / -0 values on
    the second trial. Equal bit for bit (int16 / int32 views): q, k, beta, g, A; the gate kernel's v (ffs_pvx) vs the
    full prologue's v on every row of every slot; the packed gated-norm output on every real row. "" if all agree."""
    import numpy as np
    from fla.ops.utils.constant import RCP_LN2

    from . import fla_fused, onorm, pron

    if lib() is None:
        return "onormpv.cubin unavailable"
    nat = fla_fused.KKT2["nat"]
    if nat is None:
        return "needs KKT2 (the native kkt_solve)"
    if onorm.lib() is None:
        return "needs ONORM (onorm.cubin)"
    dev = "cuda"
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed + L)
    rng = np.random.default_rng(seed + L)
    lprev = max([x for x in (0, 128, 192, 256, 320) if x < L])
    nvar = 3
    pmax = max(pmax, pfx)
    scale = 128 ** -0.5
    for li, d in enumerate(layers):
        la = d["la"]
        KD, VD = la.key_dim, la.value_dim
        Cc = 2 * KD + VD
        NH = la.num_v_heads
        nw, eps = la.norm.weight, la.norm.variance_epsilon
        for B in bs:
            for trial in range(3 if pfx else 2):
                edge = trial == 2   # chunk-edge prefixes (the halo across the cache / packed-row boundary)
                pm = 130 if edge else pmax
                lo = max(lprev + 1, pfx + 1)
                if edge:
                    pl = rng.choice([p for p in (1, 2, 3, 62, 63, 64, 65, 126, 127, 128, 129) if p < L - 1], B)
                    n = rng.integers(pl + 1, L + 1)
                else:
                    n = rng.integers(lo, L + 1, B)
                    pl = rng.integers(pfx, pmax + 1, B) if pfx else np.zeros(B, dtype=np.int64)
                    pl = np.minimum(pl, n - 1) if pfx else pl
                n, pl = np.asarray(n, dtype=np.int64), np.asarray(pl, dtype=np.int64)
                if B > 2 and trial == 1:
                    n[-1], pl[-1] = 0, 0     # unused packed slot
                pv = rng.integers(0, nvar, B)
                rows = np.maximum(n - pl, 0).astype(np.int64)
                st = np.concatenate([[0], np.cumsum(rows)[:-1]]).astype(np.int32)
                r = int(rows.sum())
                R = r + 32
                sc = (1.0, 8.0, 0.02)[(li + B + trial) % 3]
                if trial == 0:   # z beside the qkv rows (the qkv_z pair's output layout)
                    full = (torch.randn((R, Cc + 4096), generator=gen, device=dev) * sc).to(torch.bfloat16)
                    mixed, z = full[:, :Cc], full[:, Cc:]
                else:            # qkv and z from separate GEMMs (PAR "z")
                    mixed = (torch.randn((R, Cc), generator=gen, device=dev) * sc).to(torch.bfloat16)
                    zf = torch.randn((R, 4096), generator=gen, device=dev) * 2.0
                    if trial == 1:   # exp overflow / underflow, zeros, tiny values, -0 (ONORM's slow-division path)
                        m = torch.rand((R, 4096), generator=gen, device=dev)
                        zf = torch.where(m < 0.02, zf * 60.0, zf)
                        zf = torch.where((m > 0.02) & (m < 0.03), torch.zeros_like(zf), zf)
                        zf = torch.where((m > 0.03) & (m < 0.04), zf * 1e-30, zf)
                        zf = torch.where((m > 0.04) & (m < 0.045), -torch.zeros_like(zf), zf)
                    z = zf.to(torch.bfloat16)
                ba = (torch.randn((R, 2 * NH), generator=gen, device=dev) * 4).to(torch.bfloat16)
                ba[:, NH:NH + 4] += 24.0
                b, a = ba[:, :NH], ba[:, NH:]
                start = torch.from_numpy(st).to(dev)
                slen = torch.from_numpy(n.astype(np.int32)).to(dev)
                plen = torch.from_numpy(pl.astype(np.int32)).to(dev)
                pk = {}
                cache, pvar = None, None
                if pfx:
                    cache = (torch.randn((nvar * pm, Cc), generator=gen, device=dev) * sc).to(torch.bfloat16)
                    cba = (torch.randn((nvar * pm, 2 * NH), generator=gen, device=dev) * 4).to(torch.bfloat16)
                    pvar = torch.from_numpy(pv.astype(np.int32)).to(dev)
                    pk = dict(cache=cache, cb=cba[:, :NH].contiguous(), ca=cba[:, NH:].contiguous(), pfx=pfx if not edge
                              else 1, plen=plen, pvar=pvar, pmax=pm)
                pf_eff = (pfx if not edge else 1) if pfx else 0
                if not (pron.takes(mixed, d["conv_w"], b, a, pf.gva, KD, VD, L) and takes(mixed, d["conv_w"], cache,
                                                                                         pf_eff)
                        and onorm.supported(z, nw, start, plen)):
                    return "PROFUSE does not take these shapes"
                q0, k0, v0, b0, g0 = pron.prologue(mixed, d["conv_w"], KD, VD, b, a, d["negexpA"], la.dt_bias, start,
                                                   slen, B, L, pf.PRO_EPS, RCP_LN2, gva=pf.gva, **pk)
                A0 = torch.empty(B, L, HV, BT, device=dev, dtype=torch.bfloat16)
                nat(k0, g0, b0, A0, slen=slen)
                out0 = onorm.fused_norm(q0, k0, v0, g0, b0, A0, slen, scale, z, nw, eps, start, plen)
                q1, k1, v1, b1, g1 = pron.prologue(mixed, d["conv_w"], KD, VD, b, a, d["negexpA"], la.dt_bias, start,
                                                   slen, B, L, pf.PRO_EPS, RCP_LN2, gva=pf.gva, qk_only=True, **pk)
                A1 = torch.empty(B, L, HV, BT, device=dev, dtype=torch.bfloat16)
                nat(k1, g1, b1, A1, slen=slen)
                out1 = forward(q1, k1, g1, b1, slen, z, nw, eps, start, plen, mixed, d["conv_w"], cache, pf_eff, pvar,
                               pm if pfx else 0)
                vx = vrows(q1, k1, g1, b1, slen, start, plen, mixed, d["conv_w"], cache, pf_eff, pvar, pm if pfx else 0,
                           z, nw)
                torch.cuda.synchronize()
                where = f"(layer #{li}, B={B}, pfx={pf_eff}, scale {sc}{', edge prefixes' if edge else ''})"
                if v1 is not None:
                    return f"the qk_only prologue returned v {where}"
                for name, x0, x1 in (("q", q0, q1), ("k", k0, k1), ("beta", b0, b1), ("g", g0, g1)):
                    if not _eq(x0, x1):
                        return f"{name} of the qk_only prologue differs {where}"
                rr = torch.arange(L, device=dev) % BT   # A: every entry the native kkt_solve stores (used slots, rows
                cc = torch.arange(BT, device=dev)       # < T, the lower block triangle with whole diagonal blocks)
                msk = ((cc[None, :] // 16) <= (rr[:, None] // 16))[None, :, None, :] & (slen > 0)[:, None, None, None]
                msk = msk.expand(B, L, HV, BT)
                if not torch.equal(A0.view(torch.int16)[msk], A1.view(torch.int16)[msk]):
                    return f"A differs {where}"
                if not _eq(v0, vx):
                    nb = int((v0.contiguous().view(torch.int16) != vx.view(torch.int16)).sum())
                    return f"{nb} v values differ {where}"
                if not _eq(out0[:r * HV], out1[:r * HV]):
                    nb = int((out0[:r * HV].view(torch.int16) != out1[:r * HV].view(torch.int16)).sum())
                    return f"{nb} output values differ {where}"
        from .gov import GOV

        GOV.tick()
    return ""
