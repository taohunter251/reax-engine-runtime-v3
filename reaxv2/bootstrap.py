"""Construct the frozen model and serve the unchanged decision API."""
import gc
import os
from pathlib import Path
import sys
import torch


@torch.inference_mode()
def install(encoder, gpu, assets_dir=None):
    from .assets import Assets
    from .core import FullCore
    from .projection import MergedCheckpoint
    from .capture import capture
    from .serving import ArrivalLoop, ArrivalEngine, exact_lane
    from s1eng import kprio

    assets = Assets(assets_dir or Path(__file__).resolve().parents[1]/'parameters')
    full = FullCore(gpu,assets)
    assets.bind(full.core)
    model = MergedCheckpoint(full,('down','gdn_input','gdn_output'),
                             first=assets.payload['gate_up_first'],
                             start=assets.payload['checkpoint_start'])
    marks = kprio.GEMM_MARKS
    try:
        kprio.GEMM_MARKS = (*marks,*model.gemm_marks)
        packed, checkpoints = capture(gpu,model,None)
        lane, inputs = exact_lane(gpu,model,None)
    finally:
        kprio.GEMM_MARKS = marks
    for target, entries in zip(gpu.lanes,packed):
        target.packed = entries
    repair = ArrivalLoop(gpu.m,gpu.lanes,lane,checkpoints,inputs,gpu.depth)
    repair.enabled = True
    repair.margin = assets.payload['exact_recheck_margin']
    repair.guard_kind = 'fixed'
    repair.release_model = model
    torch.cuda.synchronize()
    return repair, ArrivalEngine


def main():
    # The inherited base initializes the same canonical model, tokenization,
    # prefix states and packed layouts used by the retained comparison.
    from s1eng import main as base_main, server
    from .serving import ArrivalEngine
    original = server.Engine
    class ReleaseEngine:
        def __new__(cls, encoder, gpu, temperature, token, max_tokens, revision):
            repair, engine_type = install(encoder,gpu)
            engine = engine_type(encoder,repair,temperature,token,max_tokens,revision)
            # base_main owns the foreground GPU loop; route it to the final loop.
            gpu.run = repair.run
            gc.collect()
            gc.freeze()
            return engine
    server.Engine = ReleaseEngine
    try:
        base_main.main()
    finally:
        server.Engine = original


if __name__ == '__main__':
    main()
