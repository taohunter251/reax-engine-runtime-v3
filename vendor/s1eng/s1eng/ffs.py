"""FFS (rgh): native sm_120 kernel for FF ps mode's pass P + pass S (ffs.cubin, built from ffs.cu with nvcc 13.4
-O3 -arch=sm_120a), used by fla_fused.gdn_forward_ps(peel=True) in place of _ps_pre_kernel + _ps_seq_x_kernel when
S1ENG_FFS=1.

One CTA per (sequence, value head), 8 warps, each warp owning 16 value columns for all 128 key rows of the state and
all 64 rows of a chunk. Same arithmetic as the two Triton kernels (dumped PTX/SASS, 2026-10-20): u, w, the gated
masked q k^T and v_new rounded to bf16 where pass P / pass S round them, every tl.dot as mma.sync m16n8k16 bf16 with
fp32 accumulation in ascending k16 order (the products evaluated transposed - v as the M dimension - which keeps h,
v_new and the gated v_new in registers between the dots; same products, same order, same instruction), exp2 as
ex2.approx.ftz.f32, the o epilogue as fma(scale, Aq v_new, scale * ((q h) 2^g)), chunk 0 / last chunk peeled as
_ps_seq_x_kernel PEEL. A's upper 16 x 16 blocks (masked to +0 by pass P) and Aq's tiles above the diagonal (where()
-> +0) are skipped: trailing all-zero k16 steps leave the fp32 accumulator unchanged. Bit-equal to the Triton ps chain
and to fla chunk_gated_delta_rule on every tested shape and input class (dev_ffs/test_stress.py); the engine's FF
start-up check (fla_fused.check, ps + peel) runs through this path, so any mismatch on the serving GPU falls back.
w, u and Aq are never written to memory (pass P's 3 bf16 intermediates per value head).
"""
from __future__ import annotations

import ctypes as C
import functools
import os

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
CUBINS = {(12, 0): os.path.join(_DIR, "ffs.cubin")}
KERNEL = os.environ.get("S1ENG_FFS_KERNEL", "ffs_fused5")   # ffs_fused3 / 4: earlier schedules, same arguments
SMEM = 99328
HV, KD, VD = 32, 128, 128


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
            raise RuntimeError(f"FFS {what}: CUresult {r}")


_OFF = []   # disable(): the start-up check found a mismatch on this GPU


def disable(why: str = "") -> None:
    _OFF.append(why)


@functools.lru_cache(maxsize=1)
def _lib():
    if os.environ.get("S1ENG_FFS", "0") != "1":
        return None
    try:
        path = CUBINS.get(tuple(torch.cuda.get_device_capability(0)))
        if path is None or not os.path.exists(path):
            return None
        return _Lib(path)
    except Exception as exc:  # noqa: BLE001 - no native path on any loading problem
        print(f"[s1eng] FFS kernel unavailable: {exc!r}", flush=True)
        return None


def lib():
    return None if _OFF else _lib()


def supported(q, k, v, g, beta, A, slen) -> bool:
    """[B, T, Hq, 128] q / k, [B, T, 32, 128] v, [B, T, 32] g (fp32) / beta (bf16), [B, T, 32, 64] A, int32 slen;
    all contiguous, T % 64 == 0, 32 % Hq == 0."""
    if slen is None or slen.dtype != torch.int32 or q.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        return False
    if g.dtype != torch.float32 or beta.dtype != torch.bfloat16 or A.dtype != torch.bfloat16:
        return False
    B, T, Hq, K = k.shape
    if K != KD or q.shape != k.shape or tuple(v.shape) != (B, T, HV, VD) or T % 64 or HV % Hq:
        return False
    if tuple(A.shape) != (B, T, HV, 64) or tuple(g.shape) != (B, T, HV) or tuple(beta.shape) != (B, T, HV):
        return False
    if B == 1 and T >= 256 and os.environ.get("S1ENG_FFS_B1_LONG", "0") != "1":
        # verify_ffs: a lone long request launches only 32 CTAs, slower than the Triton pair on one stream
        return False
    ts = (q, k, v, g, beta, A, slen)
    return all(t.is_contiguous() for t in ts) and all(t.data_ptr() % 16 == 0 for t in ts[:6])


def fused(q, k, v, g, beta, A, slen, scale: float) -> torch.Tensor:
    L = lib()
    B, T, Hq, _ = k.shape
    o = torch.empty_like(v)
    args = [C.c_void_p(q.data_ptr()), C.c_void_p(k.data_ptr()), C.c_void_p(v.data_ptr()),
            C.c_void_p(beta.data_ptr()), C.c_void_p(A.data_ptr()), C.c_void_p(g.data_ptr()),
            C.c_void_p(o.data_ptr()), C.c_void_p(slen.data_ptr()), C.c_int(T), C.c_float(scale), C.c_int(Hq)]
    params = (C.c_void_p * len(args))(*[C.cast(C.pointer(a), C.c_void_p) for a in args])
    L._chk(L.cu.cuLaunchKernel(L.fn, B * HV, 1, 1, 256, 1, 1, SMEM,
                               C.c_void_p(torch.cuda.current_stream().cuda_stream), params, None), "launch")
    return o
