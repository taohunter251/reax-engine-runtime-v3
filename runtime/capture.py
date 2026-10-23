"""Capture the private kernels without replacing native mixed-class graphs."""
import copy
import ctypes as C
import gc
import json
from pathlib import Path
import threading
import time
import torch

def dedicated(ctx):
    pointer = C.c_void_p()
    assert C.CDLL('libcuda.so.1').cuStreamCreateWithPriority(C.byref(pointer), 1, 0) == 0
    stream = torch.cuda.ExternalStream(pointer.value, device=0)
    ctx.retained.setdefault('external_streams', []).append((pointer, stream))
    return stream

def finalize(graph, model, lane_idx):
    from s1eng import kprio, msd
    pruned = model.skr_module.prune(graph)
    kprio.apply(graph, kprio.mode())
    domains = msd.apply(graph, lane_idx)
    assert not domains['failed'], domains
    graph.instantiate()
    return dict(pruned=pruned, domains=domains)

def capture_fast(ctx, model):
    packed, checkpoints, reports = ([], {}, [])
    for lane in ctx.gpu.lanes:
        pool, current = (torch.cuda.graph_pool_handle(), {})
        entries = [(length, batch, entry) for (length, batch), values in lane.packed.items() for entry in values]
        for index, (length, batch, (rows, _, meta, _)) in enumerate(sorted(entries, key=lambda x: -x[2][0])):
            with torch.cuda.stream(lane.cap_stream):
                model(meta, True)
            lane.cap_stream.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, pool=pool, stream=lane.cap_stream):
                value, state = model(meta, True)
            report = finalize(graph, model, lane.idx)
            checkpoints[id(graph)] = state
            current.setdefault((length, batch), []).append((rows, graph, meta, value))
            reports.append(dict(lane=lane.idx, length=length, batch=batch, rows=rows, **report))
            if index % 64 == 0:
                print('UID240_PRIVATE_CAPTURE ' + str((lane.idx, index, len(entries))), flush=True)
        for values in current.values():
            values.sort(key=lambda e: e[0])
        packed.append(current)
    return (packed, checkpoints, reports)

def capture_repair(ctx, model):
    from s1eng.model import Lane
    from s1eng.batchfwd import PackedMeta, packed_layout
    stream, cap = (dedicated(ctx), dedicated(ctx))
    ctx.full.core._sides[cap.cuda_stream] = dedicated(ctx)
    pool = torch.cuda.graph_pool_handle()
    packed, inputs = ({}, {})
    for length in (128, 192, 256, 320):
        prefix = ctx.gpu.lanes[0].pfx.get(length, 0)
        for batch in (1, 2, 4):
            limit = batch * (length - prefix)
            for rows in [*range(32, limit, 32), limit]:
                meta = PackedMeta(batch, length, rows, pfx=prefix)
                used_length = min(length, prefix + rows // batch)
                start, slen, positions, last = packed_layout([used_length] * batch, batch, length, rows, prefix)
                for name, array in (('start', start), ('slen', slen), ('row_pos', positions), ('last', last)):
                    getattr(meta, name).copy_(torch.from_numpy(array))
                states = tuple((torch.zeros((rows, 2560), device='cuda', dtype=torch.bfloat16) for _ in range(2)))
                with torch.cuda.stream(cap):
                    model.suffix(meta, *states, False)
                cap.synchronize()
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph, pool=pool, stream=cap):
                    value = model.suffix(meta, *states, False)
                finalize(graph, model, 2)
                inputs[id(graph)] = states
                packed.setdefault((length, batch), []).append((rows, graph, meta, value))
            print('UID240_REPAIR_CAPTURE ' + str((length, batch)), flush=True)
    workspace = ctx.full.workspaces[cap.cuda_stream]
    return (Lane(2, stream, {}, {}, packed, ws=[workspace], cap_stream=cap, pfx=ctx.gpu.lanes[0].pfx), inputs)
