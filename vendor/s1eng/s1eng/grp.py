"""GRP (rgh 2026-10-21): grouped projection GEMMs - the GEMMs of one site that read the same activation rows x run in
ONE launch (ngemm9g.cubin, built from ngemm9g.cu with nvcc 13.4 -arch=sm_120a; sm_120 only, S1ENG_GRP=0 disables).

Sites (PackedForward):
  * GDN: b|a (fused block [W_b; W_a], N 64, the pins' split-K: 40 ceil64 slices of 64, OUTPUT_TYPE fold at buckets
    128 / 192 / 256; 20 slices of 128, COMPUTE_TYPE fold at 320) as short X CTAs + the unsplit main GEMM, either qkv
    (8192; z then runs on the PAR side stream as before) or qkv|z (the fused 12288 block). Replaces the cuBLASLt
    split-K GEMM + splitKreduce on the side stream (two launches that wait for SMs behind qkv) and the fork / join.
  * ATT: k|v (fused block, N 2048, split-K 6 / 8 slices, OUTPUT_TYPE) as X tiles + q_proj (8192) as the main GEMM.
Bits: the main product is the unsplit pins' class (ascending-k16 fp32 mma.sync chain, one round), the X product the
split-K pin's (ns9_*'s in-CTA fold, slice order) - the same arithmetic as the kernels S6 already runs there, only the
CTAs share a launch. Used at a (site, bucket, R) only where the frozen GRP plan (grpd.py: two-lane graph-paired
full-forward timing) chose it AND PackedForward.check_grp found every output byte-equal to the selected path and to the
per-decision reference F.linear at start-up; anything else keeps the v13b path.
"""
from __future__ import annotations

import ctypes as C
import functools
import json
import os

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
CUBIN = os.path.join(_DIR, "ngemm9g.cubin")
PLAN = os.path.join(_DIR, "plan_grp.json")

# main tile plans (as nh9_ / nh9t_): tag -> (big (BM, BN, BK, stages), small (BM, BN, BK, stages))
MAINS = {
    "b128x256k64_s64x256k64": ((128, 256, 64, 2), (64, 256, 64, 2)),
    "b128x256k64_s32x256k64": ((128, 256, 64, 2), (32, 256, 64, 2)),
    "b128x256k64_s64x256": ((128, 256, 64, 2), (64, 256, 32, 4)),
    "b128x256k64_s32x256": ((128, 256, 64, 2), (32, 256, 32, 4)),
    "b128x128k64_s64x128k64": ((128, 128, 64, 3), (64, 128, 64, 3)),
    "b128x128k64_s32x128": ((128, 128, 64, 3), (32, 128, 32, 4)),
    "b128x256k64_t160x128k64": ((128, 256, 64, 2), (160, 128, 64, 2)),
    "b128x256k64_t192x128k64": ((128, 256, 64, 2), (192, 128, 64, 2)),
    "b128x256k64_t224x128k64": ((128, 256, 64, 2), (224, 128, 64, 2)),
    "b128x256k64_s96x256k64": ((128, 256, 64, 2), (96, 256, 64, 2)),
}
# X (split-K) tiles: tag -> ((BM, BN, BK, stages), fold MODE)
XS = {"x64m4": ((64, 64, 64, 3), 4), "x32m4": ((32, 64, 64, 4), 4), "x64m2": ((64, 64, 64, 3), 2),
      "x32m2": ((32, 64, 64, 4), 2), "k128m4": ((128, 128, 64, 3), 4), "k64m4": ((64, 128, 64, 3), 4)}
KERNELS = {f"ngg9_{mt}_{xt}": (mt, xt) for mt in MAINS for xt in XS}
TAIL_DROPS = {"b128x256k64_t160x128k64": (1,), "b128x256k64_t192x128k64": (1,), "b128x256k64_t224x128k64": (1,),
              "b128x256k64_s96x256k64": (0,)}


def enabled() -> bool:
    return os.environ.get("S1ENG_GRP", "1") == "1"


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
        for name, (mt, xt) in KERNELS.items():
            f = C.c_void_p()
            self._chk(self.cu.cuModuleGetFunction(C.byref(f), self.mod, name.encode()), name)
            big, small = MAINS[mt]
            smem = max(_smem(*big), _smem(*small), _smem(*XS[xt][0]))
            self._chk(self.cu.cuFuncSetAttribute(f, 8, smem), "smem attribute")
            self.fn[name] = (f, smem)

    @staticmethod
    def _chk(r, what):
        if r != 0:
            raise RuntimeError(f"GRP {what}: CUresult {r}")


@functools.lru_cache(maxsize=1)
def lib():
    if not enabled() or not os.path.exists(CUBIN):
        return None
    try:
        if tuple(torch.cuda.get_device_capability(0)) != (12, 0):
            return None
        return _Lib()
    except Exception as exc:  # noqa: BLE001 - no GRP on any loading problem
        print(f"[s1eng] GRP unavailable: {exc!r}", flush=True)
        return None


class Pick:
    """One GRP decision at (site, bucket, R): kernel = main tile plan + X tile, drop (big-tile rows moved to the
    small / tall tiles, as NgAlgo.drop), main = 'qkv' | 'qkvz' (GDN) | 'q' (ATT)."""

    __slots__ = ("name", "drop", "main", "chunk2")

    def __init__(self, name: str, drop: int, main: str, chunk2: int = 0):
        self.name, self.drop, self.main, self.chunk2 = name, int(drop), main, int(chunk2)

    def enc(self):
        return {"k": self.name, "drop": self.drop, "main": self.main}

    def __repr__(self):
        return f"GrpPick({self.name}, drop={self.drop}, main={self.main}, chunk2={self.chunk2})"


def chunk_of(split: int, red: int, K: int) -> tuple | None:
    """(MODE, chunk) of a split-K pin (ceil64 slices), None if no X tile implements its reduction scheme."""
    if split <= 1 or red not in (2, 4) or K % 64:
        return None
    return red, -(-(K // 64) // split) * 64


def supported_main(name: str, N: int, K: int) -> bool:
    mt, _ = KERNELS[name]
    big, small = MAINS[mt]
    return N % big[1] == 0 and N % small[1] == 0 and K % big[2] == 0 and K % small[2] == 0


def x_mode(name: str) -> int:
    return XS[KERNELS[name][1]][1]


def run(p: Pick, W: torch.Tensor, x: torch.Tensor, out: torch.Tensor, W2: torch.Tensor, out2: torch.Tensor) -> None:
    """out[R, N] = x @ W^T (unsplit class) and out2[R, N2] = x @ W2^T (split class: p.chunk2 slices, the kernel's fold
    MODE) in one launch on the current stream."""
    L = lib()
    M, K = x.shape
    N, N2 = W.shape[0], W2.shape[0]
    mt, xt = KERNELS[p.name]
    big, small = MAINS[mt]
    xc = XS[xt][0]
    if (L is None or x.dtype != torch.bfloat16 or x.stride(1) != 1 or out.stride(1) != 1 or out2.stride(1) != 1
            or W.stride() != (K, 1) or W2.stride() != (K, 1) or W.shape[1] != K or W2.shape[1] != K
            or x.data_ptr() % 16 or out.data_ptr() % 16 or out2.data_ptr() % 16 or W.data_ptr() % 16
            or W2.data_ptr() % 16 or (x.stride(0) * 2) % 16 or (out.stride(0) * 2) % 16 or (out2.stride(0) * 2) % 16
            or N % big[1] or N % small[1] or N2 % xc[1] or K % 64 or p.chunk2 <= 0 or p.chunk2 % 64
            or out.shape[0] < M or out2.shape[0] < M):
        raise RuntimeError("GRP: unavailable or unsupported layout")
    f, smem = L.fn[p.name]
    mb = max(0, M // big[0] - p.drop) * big[0]
    nx = -(-M // xc[0]) * (N2 // xc[1])
    grid = nx + (mb // big[0]) * (N // big[1]) + (-(-(M - mb) // small[0])) * (N // small[1])
    args = [C.c_void_p(x.data_ptr()), C.c_void_p(W.data_ptr()), C.c_void_p(out.data_ptr()), C.c_int(M), C.c_int(N),
            C.c_int(K), C.c_int(x.stride(0)), C.c_int(out.stride(0)), C.c_int(mb), C.c_void_p(W2.data_ptr()),
            C.c_void_p(out2.data_ptr()), C.c_int(N2), C.c_int(out2.stride(0)), C.c_int(p.chunk2)]
    params = (C.c_void_p * len(args))(*[C.cast(C.pointer(a), C.c_void_p) for a in args])
    L._chk(L.cu.cuLaunchKernel(f, grid, 1, 1, 256, 1, 1, smem, C.c_void_p(torch.cuda.current_stream().cuda_stream),
                               params, None), "launch")


def load_plan(path: str = "", log=print) -> dict:
    """GRP plan file -> {(site, L, R): Pick, ("late", L, R): set of PAR sites} (chunk2 filled in later from the
    bucket's pins). S1ENG_GRP_PLAN=<file>
    (default: plan_grp.json next to this file). Meta gate: GPU model."""
    path = path or os.environ.get("S1ENG_GRP_PLAN", "") or PLAN
    try:
        d = json.load(open(path))
    except Exception as exc:  # noqa: BLE001
        log(f"lever not used: grp (no plan: {type(exc).__name__}: {exc})")
        return {}
    if d.get("meta", {}).get("gpu") != torch.cuda.get_device_name():
        log(f"lever not used: grp (plan for {d.get('meta', {}).get('gpu')!r}, not this GPU)")
        return {}
    out = {}
    for site, L, R, e in d.get("picks", []):
        if site == "late":   # LATE sites of (L, R): a set of PAR site names (scheduling only, same kernels)
            out[(site, int(L), int(R))] = {x for x in e.get("sites", []) if x in ("z", "ba", "q", "last", "gu")}
        elif e.get("k") in KERNELS:
            out[(site, int(L), int(R))] = Pick(e["k"], e.get("drop", 0), e["main"])
    return out
