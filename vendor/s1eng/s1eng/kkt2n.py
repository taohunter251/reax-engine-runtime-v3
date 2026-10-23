"""KKT2 (rgh 2026-10-20): native sm_120 kkt_solve (kkt2n.cubin, built from kkt2n.cu with nvcc 13.4 -arch=sm_120a) for
grouped values (HV = 2 H, S1ENG_GVA=1): one CTA per (64-row chunk, batch row, KEY head) computes the raw K K^T blocks
once for both value heads of that key head (fla runs one CTA per value head and computes them twice), then per value
head the gate / beta scaling, the forward substitution and the block merge in the Triton kernel's exact operations
(kkt2n.cu header), and stores A = fla chunk_gated_delta_rule_fwd_kkt_solve_kernel's A on every entry fla stores (rows
< T, the lower block triangle of 16 x 16 blocks, diagonal blocks whole). Unused packed slots (slen = 0) exit at once.
Used only after fla_fused.kkt2_enable() found it bit-equal to fla's kernel on this GPU (S1ENG_KKT2=0 disables).
Kernel v7 (10-21): bank-conflict-free swizzled fp32 blocks + masked-zero terms left out of the substitution, same bits
(Verda: -6..-12 % kernel time vs v2, 1-stream and paired)."""
from __future__ import annotations

import ctypes as C
import os

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
CUBIN = os.path.join(_DIR, "kkt2n.cubin")


def _chk(cu, r, what):
    if r != 0:
        raise RuntimeError(f"{what}: CUresult {r}")


class Kkt2n:
    """kkt2n launcher on the current torch stream (graph-capturable)."""

    def __init__(self):
        if tuple(torch.cuda.get_device_capability(0)) != (12, 0):
            raise RuntimeError("kkt2n.cubin is sm_120 only")
        self.cu = C.CDLL("libcuda.so.1")
        torch.zeros(1, device="cuda")
        data = open(CUBIN, "rb").read()
        self._img = C.create_string_buffer(data, len(data))
        self.mod = C.c_void_p()
        _chk(self.cu, self.cu.cuModuleLoadData(C.byref(self.mod), self._img), "cuModuleLoadData")
        self.fn = C.c_void_p()
        _chk(self.cu, self.cu.cuModuleGetFunction(C.byref(self.fn), self.mod, b"kkt2n"), "kkt2n")
        f2 = C.c_void_p()
        _chk(self.cu, self.cu.cuModuleGetFunction(C.byref(f2), self.mod, b"kkt2n_smem_bytes"), "kkt2n_smem_bytes")
        out = torch.zeros(1, dtype=torch.int32, device="cuda")
        a = [C.c_void_p(out.data_ptr())]
        prm = (C.c_void_p * 1)(*[C.cast(C.pointer(x), C.c_void_p) for x in a])
        _chk(self.cu, self.cu.cuLaunchKernel(f2, 1, 1, 1, 1, 1, 1, 0, None, prm, None), "smem query")
        torch.cuda.synchronize()
        self.smem = int(out.item())
        _chk(self.cu, self.cu.cuFuncSetAttribute(self.fn, 8, self.smem), "smem attr")

    @staticmethod
    def takes(k, g, beta, A) -> bool:
        if k.dim() != 4 or beta.dim() != 3:
            return False
        B, T, H, K = k.shape
        return (H == 16 and K == 128 and beta.shape[2] == 2 * H and k.dtype == torch.bfloat16
                and beta.dtype == torch.bfloat16 and g.dtype == torch.float32
                and k.stride() == (T * H * K, H * K, K, 1) and g.is_contiguous() and beta.is_contiguous()
                and A.is_contiguous() and tuple(A.shape) == (B, T, 2 * H, 64) and A.dtype == torch.bfloat16
                and k.data_ptr() % 16 == 0 and g.data_ptr() % 8 == 0 and beta.data_ptr() % 4 == 0
                and A.data_ptr() % 16 == 0)

    def __call__(self, k, g, beta, A, slen=None):
        B, T = k.shape[0], k.shape[1]
        a = [C.c_void_p(k.data_ptr()), C.c_void_p(g.data_ptr()), C.c_void_p(beta.data_ptr()), C.c_void_p(A.data_ptr()),
             C.c_void_p(slen.data_ptr() if slen is not None else 0), C.c_int(T), C.c_int(1 if slen is not None else 0)]
        prm = (C.c_void_p * len(a))(*[C.cast(C.pointer(x), C.c_void_p) for x in a])
        _chk(self.cu, self.cu.cuLaunchKernel(self.fn, (T + 63) // 64, B * 16, 1, 128, 1, 1, self.smem,
                                             C.c_void_p(torch.cuda.current_stream().cuda_stream), prm, None),
             "launch kkt2n")
        return A
