"""KP (rgh): per-kernel CUDA graph node priorities. Two lanes replay graphs at the same time; when both have CTAs
pending, the block scheduler dispatches the higher-priority kernel's CTAs first. KP sets the priority of every kernel
node of a captured lane graph by kernel class (GEMMs vs the small latency-bound kernels around them) before the graph
is instantiated. Scheduling only: the same kernels with the same arguments, so the same bits.

S1ENG_KPRIO="small" (small kernels high), "gemm" (GEMMs high); "" / "0" = off.
"""
from __future__ import annotations

import ctypes as C
import functools
import os

GEMM_MARKS = ("gemm", "cutlass", "splitkreduce", "gemv", "nh_", "nh4_", "nsw_", "ns4_", "ns1_", "nb_", "ngs", "nf4_",
              "nf1_", "nf2_", "kernel2", "nvjet",   # nvjet: cuBLAS kernels SEL2b table configs bring (kai)
              "nh9_", "nsw9_", "ns9_",   # NG9B kernel names (rgh 10-20: were classed "small" -> high priority)
              "nlg9_",   # COV9b grouped lastfold split-K kernels
              "nr9p_", "nr9h_", "nsq9_",   # RARN GEMMs (rarn.py)
              "xa_fwd",   # XA attention kernels: the class of the SDPA kernel they replace (fmha_cutlassF)
              "ngg9_",   # GRP grouped projection GEMMs (rgh 10-21)
              "nsw7l_")   # NG7 gate|up + SwiGLU kernels (pjs nsw7l; matches no other mark -> would be classed "small")
CU_GRAPH_NODE_TYPE_KERNEL = 0
CU_LAUNCH_ATTRIBUTE_PRIORITY = 8


class _KParams(C.Structure):   # CUDA_KERNEL_NODE_PARAMS_v2
    _fields_ = [("func", C.c_void_p), ("gridDimX", C.c_uint), ("gridDimY", C.c_uint), ("gridDimZ", C.c_uint),
                ("blockDimX", C.c_uint), ("blockDimY", C.c_uint), ("blockDimZ", C.c_uint),
                ("sharedMemBytes", C.c_uint), ("kernelParams", C.c_void_p), ("extra", C.c_void_p),
                ("kern", C.c_void_p), ("ctx", C.c_void_p)]


@functools.lru_cache(maxsize=1)
def _cu():
    return C.CDLL("libcuda.so.1")


def mode() -> str:
    m = os.environ.get("S1ENG_KPRIO", "")
    return "" if m in ("", "0") else m


def _name(cu, node) -> str:
    p = _KParams()
    if cu.cuGraphKernelNodeGetParams_v2(C.c_void_p(node), C.byref(p)) != 0:
        return ""
    nm = C.c_char_p()
    if p.func and cu.cuFuncGetName(C.byref(nm), C.c_void_p(p.func)) == 0 and nm.value:
        return nm.value.decode(errors="replace")
    if p.kern and cu.cuKernelGetName(C.byref(nm), C.c_void_p(p.kern)) == 0 and nm.value:
        return nm.value.decode(errors="replace")
    return ""


def apply(graph, how: str, high: int = -3) -> dict:
    """graph: a torch CUDAGraph captured with keep_graph=True (not yet instantiated). Returns counts per class."""
    cu = _cu()
    g = C.c_void_p(graph.raw_cuda_graph())
    n = C.c_size_t(0)
    cu.cuGraphGetNodes(g, None, C.byref(n))
    nodes = (C.c_void_p * n.value)()
    cu.cuGraphGetNodes(g, nodes, C.byref(n))
    counts = {"gemm": 0, "small": 0, "unknown": 0, "set_failed": 0}
    for i in range(n.value):
        t = C.c_int()
        cu.cuGraphNodeGetType(C.c_void_p(nodes[i]), C.byref(t))
        if t.value != CU_GRAPH_NODE_TYPE_KERNEL:
            continue
        name = _name(cu, nodes[i]).lower()
        if not name:
            counts["unknown"] += 1
            continue
        cls = "gemm" if any(m in name for m in GEMM_MARKS) else "small"
        counts[cls] += 1
        val = (C.c_int * 16)()   # CUlaunchAttributeValue (64 bytes); priority = int at offset 0
        val[0] = high if cls == how else 0
        if cu.cuGraphKernelNodeSetAttribute(C.c_void_p(nodes[i]), CU_LAUNCH_ATTRIBUTE_PRIORITY, val) != 0:
            counts["set_failed"] += 1
    return counts
