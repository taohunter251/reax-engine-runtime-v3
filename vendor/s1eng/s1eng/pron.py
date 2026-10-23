"""PRON (rgh 2026-10-21): native sm_120 GDN prologue (pron.cubin, built from pron.cu with nvcc 13.4 -O3
-arch=sm_120a), a drop-in for fused.gdn_prologue_packed (E4 PRO) with the same outputs bit for bit: conv + SiLU of the
packed qkv rows (prefix-variant cache rows below plen), l2norm of q / k, beta and the chunk-cumulated g, in one launch
(pron.cu header: the Triton kernel's PTX op for op). Used only after PackedForward.check_pron found it bit-equal to the
Triton kernel on this GPU (S1ENG_PRON=1)."""
from __future__ import annotations

import ctypes as C
import functools
import os

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
CUBIN = os.path.join(_DIR, "pron.cubin")
KD, VD, NH, BT = 2048, 4096, 32, 64
KERNEL = os.environ.get("S1ENG_PRON_KERNEL", "pron4_r4")
RPT = {"pron": 8, "pron_r4": 4, "pron_r4f": 4, "pron_r2f": 2, "pron_r8f": 8, "pron3_r2": 2, "pron3_r4": 4,
       "pron3_r4x": 4, "pron3_r8": 8, "pron4_r2": 2, "pron4_r4": 4, "pron4_r8": 8}


class _Args(C.Structure):
    _fields_ = [("x", C.c_void_p), ("w", C.c_void_p), ("cx", C.c_void_p), ("b", C.c_void_p), ("a", C.c_void_p),
                ("cb", C.c_void_p), ("ca", C.c_void_p), ("dt_bias", C.c_void_p), ("negexpA", C.c_void_p),
                ("start", C.c_void_p), ("slen", C.c_void_p), ("plen", C.c_void_p), ("pvar", C.c_void_p),
                ("q", C.c_void_p), ("k", C.c_void_p), ("v", C.c_void_p), ("beta", C.c_void_p), ("g", C.c_void_p),
                ("L", C.c_int), ("XS", C.c_int), ("CXS", C.c_int), ("BS", C.c_int), ("AS", C.c_int),
                ("CBS", C.c_int), ("CAS", C.c_int), ("pfx", C.c_int), ("pmax", C.c_int), ("B", C.c_int),
                ("nconv", C.c_int), ("nhb", C.c_int), ("eps", C.c_float), ("scale", C.c_float)]


def _chk(r, what):
    if r != 0:
        raise RuntimeError(f"PRON {what}: CUresult {r}")


class _Lib:
    def __init__(self):
        self.cu = C.CDLL("libcuda.so.1")
        torch.zeros(1, device="cuda")
        data = open(CUBIN, "rb").read()
        self._img = C.create_string_buffer(data, len(data))
        self.mod = C.c_void_p()
        _chk(self.cu.cuModuleLoadData(C.byref(self.mod), self._img), "cuModuleLoadData")
        self.fns = {}

    def fn(self, name):
        f = self.fns.get(name)
        if f is None:
            f = C.c_void_p()
            _chk(self.cu.cuModuleGetFunction(C.byref(f), self.mod, name.encode()), name)
            self.fns[name] = f
        return f


@functools.lru_cache(maxsize=1)
def lib():
    if tuple(torch.cuda.get_device_capability(0)) != (12, 0) or not os.path.exists(CUBIN):
        return None
    return _Lib()


def takes(mixed, w, b, a, gva: bool, key_dim: int, value_dim: int, L: int) -> bool:
    R, Cc = mixed.shape
    return (gva and key_dim == KD and value_dim == VD and Cc == 2 * KD + VD and L % BT == 0
            and mixed.stride(1) == 1 and mixed.stride(0) % 8 == 0 and mixed.data_ptr() % 16 == 0
            and mixed.dtype == torch.bfloat16 and w.is_contiguous() and w.data_ptr() % 16 == 0
            and tuple(w.shape) == (Cc, 4) and w.dtype == torch.bfloat16 and b.shape[1] == NH
            and b.stride(1) == 1 and a.stride(1) == 1 and b.dtype == torch.bfloat16 and a.dtype == torch.bfloat16)


def prologue(mixed, w, key_dim, value_dim, b, a, negexpA, dt_bias, start, slen, B, L, eps: float, scale: float,
             gva: bool = True, cache=None, cb=None, ca=None, pfx: int = 0, plen=None, pvar=None, pmax: int = 0,
             kernel: str = "", qk_only: bool = False):
    """fused.gdn_prologue_packed(...) -> (q, k, v, beta, g), the same bits (pron.cu)."""
    kernel = kernel or KERNEL
    lb = lib()
    R, Cc = mixed.shape
    if pfx:
        pmax = pmax or pfx
        assert cache is not None and cache.stride(1) == 1 and cache.data_ptr() % 16 == 0 and cache.stride(0) % 8 == 0
        assert cb.stride(1) == 1 and ca.stride(1) == 1 and plen is not None and pvar is not None
    else:
        cache, cb, ca, plen, pvar = mixed, b, a, start, start
    dev = mixed.device
    qk = torch.empty((2, B, L, key_dim // 128, 128), dtype=mixed.dtype, device=dev)
    q, k = qk[0], qk[1]
    v = None if qk_only else torch.empty((B, L, value_dim // 128, 128), dtype=mixed.dtype, device=dev)
    beta = torch.empty((B, L, NH), dtype=b.dtype, device=dev)
    g = torch.empty((B, L, NH), dtype=torch.float32, device=dev)
    nch = L // BT
    nhb = (2 * key_dim if qk_only else Cc) // 128
    assert not qk_only or kernel.startswith("pron4")
    nconv = (L // (8 * RPT[kernel])) * nhb * B
    ngate = (nch * NH * B + 3) // 4
    args = _Args(mixed.data_ptr(), w.data_ptr(), cache.data_ptr(), b.data_ptr(), a.data_ptr(), cb.data_ptr(),
                 ca.data_ptr(), dt_bias.data_ptr(), negexpA.data_ptr(), start.data_ptr(), slen.data_ptr(),
                 plen.data_ptr(), pvar.data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr() if v is not None else 0, beta.data_ptr(),
                 g.data_ptr(), L, mixed.stride(0), cache.stride(0), b.stride(0), a.stride(0), cb.stride(0),
                 ca.stride(0), int(pfx), int(pmax), B, nconv, nhb, float(eps), float(scale))
    params = (C.c_void_p * 1)(C.cast(C.pointer(args), C.c_void_p))
    _chk(lb.cu.cuLaunchKernel(lb.fn(kernel), nconv + ngate, 1, 1, 128, 1, 1, 0,
                              C.c_void_p(torch.cuda.current_stream().cuda_stream), params, None), "launch")
    return q, k, v, beta, g
