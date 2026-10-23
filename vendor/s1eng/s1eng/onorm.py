"""ONORM (rgh, after u198's howu3p): FFS (ffs.py, native pass P + S) with the GDN gated RMSNorm
(fused.gated_rmsnorm_packed, the Qwen3_5RMSNormGated chain) as its epilogue, in one sm_120 kernel (onorm.cubin, built
from onorm.cu with nvcc 13.4 -O3 -std=c++17 -arch=sm_120a -DFFS_SHIP -DONORM_V=5 + ONORM_BUILD). Each CTA (sequence n,
value head h) stages chunk i's o tile in shared memory and normalises its rows there: one warp per row, lane t owning
columns 4t .. 4t + 3 as in the Triton kernel: ((sq0 + sq1) + sq2) + sq3, the shfl tree 16, 8, 4, 2, 1,
rsqrt.approx(eps + s / 128), bf16(o r), mul.rn.bf16 by w, x silu(z) through the dumped PTX of libdevice expf and div.rn
(an exhaustively checked fast division for z in (-87, inf), an out-of-line div.rn for a warp with any other z) - every
op an explicit-rounding PTX instruction. z rows of chunk i are prefetched at the top of chunk i (ONORM_PF2) and loaded
before C(i); the norm runs after the o tile is complete, 4 rows at a time in a rolled loop (ONORM_FINROLL: one small
code body - in the engine other kernels evict the instruction cache between GDN layers, and an unrolled 8-row epilogue
cost ~8000 cold-fetch cycles per chunk in-engine, dev_onorm2 eprof5 d4 vs d7). Rows go straight to the packed out_proj
input [R * 32, 128] (positions plen[n] .. slen[n] - 1 at rows start[n] ..); prefix and padding positions are never
stored and no padded o is written. Unused packed rows (>= the real row count) are left unwritten (their consumers are
row-independent: out_proj, the residual and the next layer's GEMMs; no row of a decision reads them).

Dispatch: per bucket L for packed batches of B >= bmin(L) slots (S1ENG_ONORM_BMIN, default "2": every packed batch but
a lone decision; "<all buckets>[,L:B]..." per bucket). In-engine rounds with this build (engprof.py, p4 / p5, on-off-on):
1 lane -0.1..-1.8 %, 2 lanes -0.1..-2.0 % at B 2..16 in buckets 128..320 (256 x 2 one lane +0.2 %). Per bucket only where
check() found it bit-equal (int16 view) to FFS + the Triton gated norm on this GPU at every B it will run at; else
FFS + gated_rmsnorm_packed. S1ENG_ONORM=0 disables it."""
from __future__ import annotations

import ctypes as C
import functools
import os

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
CUBINS = {(12, 0): os.path.join(_DIR, "onorm.cubin")}
KERNEL = os.environ.get("S1ENG_ONORM_KERNEL", "ffs_fused5n")
_BM = os.environ.get("S1ENG_ONORM_BMIN", "2")   # "<all buckets>[,L:B]..."


def _parse_bmin(spec: str):
    dflt, per = 1 << 30, {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            L, b = part.split(":")
            per[int(L)] = int(b)
        else:
            dflt = int(part)
    return dflt, per


BMIN, BMIN_OF = _parse_bmin(_BM)


def bmin(L: int) -> int:
    return BMIN_OF.get(L, BMIN)


ONORM_BUILD = "-DONORM_PF2 -DONORM_SILU_FIN -DONORM_FINROLL"   # onorm.cubin's extra flags
SMEM = int(os.environ.get("S1ENG_ONORM_SMEM", "99328"))   # dynamic smem of the build (ONORM_WARM: 101376)
HV, VD = 32, 128
ON: set = set()   # buckets L whose start-up check passed


class _Lib:
    def __init__(self, path: str):
        self.cu = C.CDLL("libcuda.so.1")
        torch.zeros(1, device="cuda")
        data = open(path, "rb").read()
        self._img = C.create_string_buffer(data, len(data))
        self.mod = C.c_void_p()
        self._chk(self.cu.cuModuleLoadData(C.byref(self.mod), self._img), "cuModuleLoadData")
        self.fn = C.c_void_p()
        self._chk(self.cu.cuModuleGetFunction(C.byref(self.fn), self.mod, KERNEL.encode()), KERNEL)
        self._chk(self.cu.cuFuncSetAttribute(self.fn, 8, SMEM), "smem attribute")

    @staticmethod
    def _chk(r, what):
        if r != 0:
            raise RuntimeError(f"ONORM {what}: CUresult {r}")


def enabled() -> bool:
    return os.environ.get("S1ENG_ONORM", "1") == "1"


@functools.lru_cache(maxsize=1)
def lib():
    if not enabled():
        return None
    from . import ffs

    if ffs.lib() is None:
        return None
    try:
        path = CUBINS.get(tuple(torch.cuda.get_device_capability(0)))
        if path is None or not os.path.exists(path):
            return None
        return _Lib(path)
    except Exception as exc:  # noqa: BLE001 - no fused norm on any loading problem
        print(f"[s1eng] ONORM kernel unavailable: {exc!r}", flush=True)
        return None


def use(L: int, B: int) -> bool:
    """the fused norm runs for this packed graph (bucket L, B slots)"""
    if B < bmin(L) or L not in ON:
        return False
    from . import ffs

    return ffs.lib() is not None and lib() is not None


def supported(z, w, start, plen) -> bool:
    return (z.dim() == 2 and z.dtype == torch.bfloat16 and z.stride(1) == 1 and z.shape[1] == HV * VD
            and z.data_ptr() % 16 == 0 and (z.stride(0) * 2) % 16 == 0 and w.dtype == torch.bfloat16
            and w.is_contiguous() and w.numel() == VD and w.data_ptr() % 16 == 0 and start is not None
            and plen is not None and start.dtype == torch.int32 and plen.dtype == torch.int32
            and start.is_contiguous() and plen.is_contiguous())


def fused_norm(q, k, v, g, beta, A, slen, scale: float, z, w, eps: float, start, plen) -> torch.Tensor:
    """ffs.fused + fused.gated_rmsnorm_packed in one launch: out [R * 32, 128] (R = z.shape[0])."""
    Lb = lib()
    B, T, Hq, _ = k.shape
    out = torch.empty((z.shape[0] * HV, VD), dtype=z.dtype, device=z.device)
    args = [C.c_void_p(q.data_ptr()), C.c_void_p(k.data_ptr()), C.c_void_p(v.data_ptr()),
            C.c_void_p(beta.data_ptr()), C.c_void_p(A.data_ptr()), C.c_void_p(g.data_ptr()),
            C.c_void_p(slen.data_ptr()), C.c_int(T), C.c_float(scale), C.c_int(Hq), C.c_void_p(z.data_ptr()),
            C.c_int(z.stride(0)), C.c_void_p(w.data_ptr()), C.c_float(eps), C.c_void_p(start.data_ptr()),
            C.c_void_p(plen.data_ptr()), C.c_void_p(out.data_ptr())]
    params = (C.c_void_p * len(args))(*[C.cast(C.pointer(a), C.c_void_p) for a in args])
    Lb._chk(Lb.cu.cuLaunchKernel(Lb.fn, B * HV, 1, 1, 256, 1, 1, SMEM,
                                 C.c_void_p(torch.cuda.current_stream().cuda_stream), params, None), "launch")
    return out


def _case(L, B, Hq, gen, mode, extreme, eps):
    """One random packed layout of bucket L (B slots): (FFS inputs, z, w, start, plen, row_pos, real rows)."""
    import numpy as np
    import torch.nn.functional as F
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel
    from fla.ops.utils.constant import RCP_LN2
    from fla.ops.utils.cumsum import chunk_local_cumsum

    dev = "cuda"
    rnd = lambda *s: torch.randn(*s, generator=gen, device=dev)  # noqa: E731
    ri = lambda lo, hi: int(torch.randint(lo, hi, (1,), generator=gen, device=dev).item())  # noqa: E731
    if mode == "pv":
        pls = [[40, 42, 44][ri(0, 3)] for _ in range(B)]
    elif mode == "edge":
        pls = [[0, 1, 63, 64, 65][ri(0, 5)] for _ in range(B)]
    else:
        pls = [0] * B
    pls = [min(p, L - 1) for p in pls]
    lens = [ri(max(pls[b] + 1, L - 128), L + 1) if ri(0, 4) else [pls[b] + 1, L][ri(0, 2)] for b in range(B)]
    if B > 1 and ri(0, 2):
        lens[ri(0, B)] = 0    # an unused slot
    P = min(pls)
    n = torch.tensor(lens, device=dev)
    keep = (torch.arange(L, device=dev)[None, :] < n[:, None]).float()
    q = (rnd(B, L, Hq, 128) * keep[..., None, None]).to(torch.bfloat16)
    k = (rnd(B, L, Hq, 128) * keep[..., None, None]).to(torch.bfloat16)
    v = (rnd(B, L, HV, 128) * keep[..., None, None]).to(torch.bfloat16)
    g = (-F.softplus(rnd(B, L, HV)) * 0.3 * keep[..., None]).float()
    beta = (torch.sigmoid(rnd(B, L, HV)) * keep[..., None]).to(torch.bfloat16)
    qk, _ = l2norm_fwd(torch.cat([q, k]))
    q, k = qk[:B].contiguous(), qk[B:].contiguous()
    g = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2)
    A = torch.empty(B, L, HV, 64, device=dev, dtype=torch.bfloat16)
    chunk_gated_delta_rule_fwd_kkt_solve_kernel[(L // 64, B * HV)](k=k, g=g, beta=beta, A=A, cu_seqlens=None,
                                                                  chunk_indices=None, T=L, H=Hq, HV=HV, K=128, BT=64,
                                                                  BC=16)
    start = np.zeros(B, dtype=np.int32)
    plen = np.full(B, P, dtype=np.int32)
    rows, r = [], 0
    for b in range(B):
        if lens[b] == 0:
            continue
        p = pls[b]
        start[b], plen[b] = r, p
        rows += [b * L + x for x in range(p, lens[b])]
        r += lens[b] - p
    Rr = r + ri(0, 33)
    row_pos = np.zeros(Rr, dtype=np.int32)
    row_pos[:r] = rows
    zf = rnd(Rr, 3 * HV * VD) * 2.0
    if extreme:   # exp overflow / underflow, zeros, tiny values
        m = torch.rand(Rr, 3 * HV * VD, generator=gen, device=dev)
        zf = torch.where(m < 0.02, zf * 60.0, zf)
        zf = torch.where((m > 0.02) & (m < 0.03), torch.zeros_like(zf), zf)
        zf = torch.where((m > 0.03) & (m < 0.04), zf * 1e-30, zf)
        zf = torch.where((m > 0.04) & (m < 0.045), -torch.zeros_like(zf), zf)
    z = zf.to(torch.bfloat16)[:, 2 * HV * VD:]
    w = (rnd(VD) * 0.3 + 1.0).to(torch.bfloat16)
    slen = n.to(torch.int32)
    t = lambda a: torch.from_numpy(a).to(dev)  # noqa: E731
    return (q, k, v, g, beta, A, slen), z, w, t(start), t(plen), t(row_pos), r


def check(buckets, bs_of, eps: float = 1e-6, heads=(16,), seed: int = 7) -> dict:
    """Start-up gate on THIS GPU: per bucket L, the fused kernel vs ffs.fused + fused.gated_rmsnorm_packed on random
    packed layouts at every B in bs_of[L] (the B values it will run at), PV prefixes 40/42/44, chunk-edge prefixes and
    none, random lengths (one row .. the whole bucket), unused slots and rows, z at the fused qkv|z row stride, extreme
    z; int16-view equality of every real row. Returns {L: "" (pass) or reason}; passing buckets join ON."""
    from . import ffs
    from . import fused as fz

    res = {}
    if lib() is None:
        return {L: "ONORM kernel unavailable" for L in buckets}
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    scale = 128 ** -0.5
    for L in buckets:
        why = ""
        bs = sorted(b for b in bs_of.get(L, ()) if b >= bmin(L))
        if not bs:
            res[L] = f"no packed batch of >= {bmin(L)} slots"
            continue
        for i, B in enumerate(bs):
            for Hq in heads:
                for mode, ext in (("pv", i % 2 == 1), ("edge", True), ("none", i % 2 == 0)):
                    (q, k, v, g, beta, A, slen), z, w, start, plen, row_pos, r = _case(L, B, Hq, gen, mode, ext, eps)
                    if not (ffs.supported(q, k, v, g, beta, A, slen) and supported(z, w, start, plen)):
                        why = f"unsupported layout (B {B}, hq {Hq})"
                        break
                    ref = fz.gated_rmsnorm_packed(ffs.fused(q, k, v, g, beta, A, slen, scale), z, w, eps, row_pos, HV)
                    got = fused_norm(q, k, v, g, beta, A, slen, scale, z, w, eps, start, plen)
                    nb = int((ref[:r * HV].view(torch.int16) != got[:r * HV].view(torch.int16)).sum())
                    if nb:
                        why = f"{nb} bf16 values differ (B {B}, hq {Hq}, {mode})"
                        break
                if why:
                    break
            if why:
                break
        res[L] = why
        if not why:
            ON.add(L)
    torch.cuda.synchronize()
    return res
