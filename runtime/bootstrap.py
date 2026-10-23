"""Construct and serve the frozen v3 candidate without research commands."""
import gc
import inspect
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

def install(engine, encoder, gpu, front):
    import torch
    from reaxv2.assets import Assets
    from reaxv2.projection import MergedCheckpoint
    from native_core import NativeFull, NativeCheckpoint
    from restore_model import main as restore
    from capture import dedicated, capture_fast, capture_repair
    from guard_loop import GuardLoop
    from s1eng import kprio, skr
    import copy

    assert front is not None, 'The frozen v3 frontend requires MPF'
    assert len(gpu.m.prefix_variants) >= 384, 'Insufficient prefix cache capacity'
    assert len(gpu.lanes) == 2, 'The frozen v3 configuration requires two lanes'
    gpu.m.pf.bf.ws = gpu.lanes[0].ws[0]
    assets = Assets(ROOT/'parameters')
    full = NativeFull(gpu, assets)
    assets.bind(full.core)
    model = NativeCheckpoint(full, ('down', 'gdn_input', 'gdn_output'), first=12, start=12)
    ctx = SimpleNamespace(engine=engine, encoder=encoder, gpu=gpu, front=front,
                          assets=assets, full=full, model=model, retained={})
    model = restore(ctx, None)
    model.skr_module = skr
    for lane in gpu.lanes:
        lane.cap_stream = dedicated(ctx)
        full.core._sides[lane.cap_stream.cuda_stream] = dedicated(ctx)
    marks = kprio.GEMM_MARKS
    try:
        kprio.GEMM_MARKS = (*marks, *model.gemm_marks)
        packed, checkpoints, _ = capture_fast(ctx, model)
        repair_lane, inputs = capture_repair(ctx, model)
    finally:
        kprio.GEMM_MARKS = marks
    torch.cuda.synchronize()
    lanes = [copy.copy(lane) for lane in gpu.lanes]
    loop_type = front.loop_class(GuardLoop)
    loop = loop_type(gpu.m, lanes, repair_lane, checkpoints, inputs,
                     front, float(assets.payload['exact_recheck_margin']), gpu.depth)
    assert loop.zc_ok and loop.zc, 'The frozen zero-copy path failed its startup check'
    loop.native_control = gpu
    loop.mode = 'combined_nomix'
    loop.bb_join = []
    # Both fast lanes use one precision policy per request length bucket.
    for lane, entries in zip(lanes, packed):
        lane.packed = entries
    front.tag = lambda jobs: None
    import ctypes
    cuda = ctypes.CDLL('libcuda.so.1')
    for lane in [*lanes, repair_lane]:
        for entries in lane.packed.values():
            for _, graph, _, _ in entries:
                assert cuda.cuGraphUpload(ctypes.c_void_p(graph.raw_cuda_graph_exec()),
                                          ctypes.c_void_p(lane.stream.cuda_stream)) == 0
    torch.cuda.synchronize()
    why = front.serve(loop, os.environ.get('RUNTIME_UDS'), time.time())
    assert not why, why
    gpu._mpf = front
    # Retain all graph, parameter and stream owners for the serving lifetime.
    loop.release_context = ctx
    loop.release_model = model
    gc.collect()
    gc.freeze()
    print('REAX_V3_READY', flush=True)
    loop.run()

def main():
    # MPF must fork its HTTP workers before importing torch or creating CUDA state.
    import s1eng.server as server
    original = server.Engine.__init__
    parent = os.getpid()
    def initialize(self, encoder, gpu, *args, **kwargs):
        original(self, encoder, gpu, *args, **kwargs)
        caller = inspect.currentframe().f_back
        if (os.getpid() == parent and caller.f_globals.get('__name__') == 's1eng.main'
                and caller.f_code.co_name == 'main'):
            import torch
            with torch.inference_mode():
                install(self, encoder, gpu, caller.f_locals.get('front'))
            raise SystemExit(0)
    server.Engine.__init__ = initialize
    from s1eng.main import main as inherited_main
    inherited_main()

if __name__ == '__main__':
    main()
