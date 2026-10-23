"""Pin every timing-autotuned fla Triton kernel used by the GDN forward to one fixed config.

The reference lets Triton time all configs at start-up. Only l2norm_fwd_kernel has configs that change the output
bits (two bit classes besides the main one, see NOTES.md); the other kernels give identical bits for every config.
With a single config Triton's Autotuner launches it directly: no benchmarking, no start-up timing noise.
"""
from __future__ import annotations

import os


def _autotuner(k):
    while k is not None and not hasattr(k, "cache") and hasattr(k, "fn"):
        k = k.fn
    return k


def _match(cfg, want: dict) -> bool:
    for key, val in want.items():
        if key in ("num_warps", "num_stages"):
            if getattr(cfg, key) != val:
                return False
        elif cfg.kwargs.get(key) != val:
            return False
    return True


# kernel -> config (5090 reference start-up picks for the bit-neutral kernels). l2norm: the config the owner engine
# (reaxlabs/reax-engine ffa8398, plumb_engine.pin_fla_autotune: BT=16, 4 warps; the expected next reference) pins, so
# our l2norm bits equal its bits on every GPU; on the 5090 it is in the main bit class (class A, with BT8/nw1 that
# round 2 pinned), which the official reference picks in ~80 % of its start-ups.
DEFAULT_PINS = {
    "cumsum_scalar": {"num_warps": 1},
    "kkt_solve": {"BK": 64, "num_warps": 2},
    "recompute_w_u": {"num_warps": 8, "num_stages": 3},
    "fwd_h": {"BV": 32, "num_warps": 2, "num_stages": 2},
    "fwd_o": {"BK": 64, "BV": 64, "num_warps": 4, "num_stages": 3},
    "l2norm": {"BT": 16, "num_warps": 4},
}


# T1 (a4): per-GPU pins of the bit-neutral kernels, chosen by timing every config at serving shapes (B = 4..8 decisions
# of L = 192 / 256, 16 q/k heads, 32 value heads); filled from spike fla_tune.py on that GPU.
# RTX PRO 6000 (2026-10-19): kkt_solve 57.3 -> 38.6 us, recompute_w_u 61.7 -> 51.5 us; fwd_h / fwd_o already best;
# the whole chain 11-14 % faster at 6-8 x 256 / 8 x 192, output bits equal (fla_tune same_bits).
GPU_PINS: dict = {
    "RTX PRO 6000": {"cumsum_scalar": {"num_warps": 4}, "kkt_solve": {"BK": 64, "num_warps": 1},
                     "recompute_w_u": {"num_warps": 4, "num_stages": 2}},
}


def kernels():
    import fla.modules.l2norm as l2
    import fla.ops.common.chunk_delta_h as dh
    import fla.ops.common.chunk_o as co
    import fla.ops.gated_delta_rule.chunk_fwd as cf
    import fla.ops.gated_delta_rule.wy_fast as wy
    import fla.ops.utils.cumsum as cs

    return {"cumsum_scalar": cs.chunk_local_cumsum_scalar_kernel,
            "kkt_solve": cf.chunk_gated_delta_rule_fwd_kkt_solve_kernel,
            "recompute_w_u": wy.recompute_w_u_fwd_kernel,
            "fwd_h": dh.chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
            "fwd_o": co.chunk_fwd_kernel_o,
            "l2norm": l2.l2norm_fwd_kernel}


ALL_CONFIGS: dict = {}


def pin(overrides: dict | None = None, log=None) -> dict:
    """Pins every kernel. Fail-safe on other GPUs (config lists depend on the architecture): l2norm gets its config
    set explicitly (as the owner engine does); a bit-neutral kernel whose pin matches no config stays autotuned."""
    pins = dict(DEFAULT_PINS)
    # T1 (a4): the bit-neutral kernels' configs timed on serving-sized batches of this GPU (DEFAULT_PINS are the
    # reference's batch-1 picks); S1ENG_FLA_GPU_PINS=0 keeps DEFAULT_PINS. l2norm is never changed here.
    if os.environ.get("S1ENG_FLA_GPU_PINS", "1") == "1":
        try:
            import torch

            name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            name = ""
        for needle, table in GPU_PINS.items():
            if needle in name:
                pins.update({k: v for k, v in table.items() if k != "l2norm"})
                break
    raw = os.environ.get("S1ENG_FLA_PINS")  # JSON {kernel: {param: value}} (experiments only; l2norm excluded)
    if raw:
        import json

        pins.update({k: v for k, v in json.loads(raw).items() if k != "l2norm"})
    env = os.environ.get("S1ENG_L2NORM")  # "BT,num_warps" (experiments only)
    if env:
        bt, nw = (int(v) for v in env.split(","))
        pins["l2norm"] = {"BT": bt, "num_warps": nw}
    pins.update(overrides or {})
    chosen = {}
    for name, k in kernels().items():
        at = _autotuner(k)
        ALL_CONFIGS.setdefault(name, list(at.configs))
        cands = [c for c in ALL_CONFIGS[name] if _match(c, pins[name])]
        if len(cands) != 1 and name == "l2norm":
            import triton

            want = pins[name]
            cands = [triton.Config({"BT": want["BT"]}, num_warps=want["num_warps"])]
        if len(cands) != 1:
            msg = f"fla pin {name}: {len(cands)} configs match {pins[name]} on this GPU: left autotuned"
            if log is None:
                import sys

                print(f"[s1eng] {msg}", file=sys.stderr, flush=True)
            else:
                log(msg)
            chosen[name] = "autotuned"
            continue
        at.configs = cands
        at.cache.clear()
        chosen[name] = str(cands[0])
    return chosen
