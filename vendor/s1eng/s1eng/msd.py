"""MSD (ruj r2 fixedcost2): every kernel node of lane 1's graphs in the REMOTE memory synchronization domain, lane 0's in
the default one, so a memory fence / kernel-end flush in one lane need not wait for the other lane's memory traffic.

Two lanes replay independent graphs at the same time (separate pools, workspaces, inputs and outputs; no data flows
between them). CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN (sm_90+) scopes fences to a domain; stream / graph dependencies are
still honoured across domains by the driver. Scheduling only: the same kernels with the same arguments, so the same
bits; the engine's self_check runs on the lane graphs as captured. Applied before instantiation (keep_graph graphs,
like KP's priorities). Devices with fewer than 2 domains: nothing is set.

S1ENG_MSD=1 on, "0" / unset off. FCAB arm toggle: nomsd.
"""
from __future__ import annotations

import ctypes as C
import functools
import os

import torch

CU_GRAPH_NODE_TYPE_KERNEL = 0
CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN = 10
CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE = 1
CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126


def enabled() -> bool:
    return os.environ.get("S1ENG_MSD", "0") == "1"


@functools.lru_cache(maxsize=1)
def _cu():
    return C.CDLL("libcuda.so.1")


@functools.lru_cache(maxsize=1)
def domains() -> int:
    cu = _cu()
    dev = C.c_int()
    if cu.cuDeviceGet(C.byref(dev), C.c_int(torch.cuda.current_device())) != 0:
        return 1
    v = C.c_int(0)
    if cu.cuDeviceGetAttribute(C.byref(v), C.c_int(CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT), dev) != 0:
        return 1
    return int(v.value)


def apply(graph, lane_idx: int) -> dict:
    """graph: a torch CUDAGraph captured with keep_graph=True (not yet instantiated); odd lanes -> REMOTE domain."""
    st = {"set": 0, "failed": 0, "domains": domains()}
    if st["domains"] < 2 or lane_idx % 2 == 0:
        return st
    cu = _cu()
    g = C.c_void_p(graph.raw_cuda_graph())
    n = C.c_size_t(0)
    cu.cuGraphGetNodes(g, None, C.byref(n))
    nodes = (C.c_void_p * n.value)()
    cu.cuGraphGetNodes(g, nodes, C.byref(n))
    for i in range(n.value):
        t = C.c_int()
        cu.cuGraphNodeGetType(C.c_void_p(nodes[i]), C.byref(t))
        if t.value != CU_GRAPH_NODE_TYPE_KERNEL:
            continue
        val = (C.c_int * 16)()   # CUlaunchAttributeValue (64 bytes); memSyncDomain = int at offset 0
        val[0] = CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE
        if cu.cuGraphKernelNodeSetAttribute(C.c_void_p(nodes[i]), CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN, val) != 0:
            st["failed"] += 1
        else:
            st["set"] += 1
    return st
