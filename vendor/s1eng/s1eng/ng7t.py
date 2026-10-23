"""NG7T (rgh 10-21): the nsw7l SwiGLU kernels (pjs NG7, ng.SW_KERNELS7) at the (bucket, R) keys of a frozen table that
an in-situ screen (co-run window timing in a mixed served stream, both lanes) found faster than the plan's SG pick.

Switches: S1ENG_NG7=1 (loads the kernels; ng.check_ng7 checks the SiLU-table epilogue over every bf16 gate value) and
S1ENG_NG7_TABLE=1 (the built-in table for this GPU model, ng7_table_pro6000.json) or =<path>. Unset / 0: nothing here
runs and the plan's SG picks stay exactly.

apply() runs after FP's apply_sg (or select_swiglu) and before the graphs are captured. Table rows [L, R, name, drop]:
each served key gets the nsw7l kernel ONLY if its output equals the per-decision reference (F.linear of the fused
gate|up block on M = bucket blocks via pinned.ref_linear, then silu_mul = apply_sg's per-R check; also equal to
F.silu(g) * u) byte for byte at that R on two input sets (apply_sg's random blocks; x64 outlier rows / columns), each
launched twice. A key that fails, or any exception, keeps the pick it had (the plan's nsw9 kernel or GEMM + silu_mul).
"""
from __future__ import annotations

import json
import os

import torch

ENV = "S1ENG_NG7_TABLE"
_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES = {"NVIDIA RTX PRO 6000 Blackwell Server Edition": "ng7_table_pro6000.json"}


def spec(env=os.environ) -> str:
    return env.get(ENV, "0")


def load_table(log=print) -> dict:
    """{(L, R): (name, drop)} from the table the switch names ("1": the built-in one for this GPU model)."""
    s = spec()
    if s in ("", "0"):
        return {}
    try:
        if s == "1":
            name = TABLES.get(torch.cuda.get_device_name())
            if name is None:
                log("lever not used: NG7T (no built-in table for this GPU model)")
                return {}
            path = os.path.join(_DIR, name)
        else:
            path = s
        with open(path) as f:
            t = json.load(f)
        rows = t["table"] if isinstance(t, dict) else t
        gpu = t.get("meta", {}).get("gpu") if isinstance(t, dict) else None
        if gpu not in (None, torch.cuda.get_device_name()):
            log(f"lever not used: NG7T (table for {gpu})")
            return {}
        return {(int(L), int(R)): (str(nm), int(d)) for L, R, nm, d, *_ in rows}
    except Exception as exc:  # noqa: BLE001
        log(f"lever not used: NG7T (table: {type(exc).__name__}: {exc})")
        return {}


def _outliers(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """x with ~2 % of its rows and 32 of its columns scaled by 64 (as kf._outliers)."""
    x = x.clone().float()
    R, K = x.shape
    x[torch.randint(0, R, (max(1, R // 50),), generator=g, device="cuda")] *= 64
    x[:, torch.randint(0, K, (32,), generator=g, device="cuda")] *= 64
    return x.to(torch.bfloat16)


def _eq(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


@torch.inference_mode()
def apply(pinned, fz, finfo, groups_of, buckets, log=print) -> int:
    """Byte-gated overlay of the table's nsw7l picks onto pinned.swiglu_r (call after SG, before capture). Returns the
    number of (bucket, R) now running an nsw7l kernel."""
    from . import ng

    if not ng.ng7_on():
        return 0
    table = load_table(log)
    if not table:
        return 0
    ent = [((Nf, K), e) for (Nf, K), e in finfo.items() if e[0] == "gate_up"]
    if not ent:
        log("lever not used: NG7T (no fused gate|up block)")
        return 0
    (Nf, K), (gname, n1, n2, blks) = ent[0]
    H = n1
    cands = {(c.name, c.drop): c for c in (ng.swiglu_candidates7(H, K) if n1 == n2 else [])}
    if not cands:
        L_ = ng.lib()
        log(f"lever not used: NG7T (no nsw7l candidates: "
            f"{getattr(L_, 'ng7_sw_why', '') or getattr(L_, 'ng7_why', '') or 'NG unavailable'})")
        return 0
    W = [b for _, b in blks][0]
    g = torch.Generator(device="cuda")
    g.manual_seed(7707)
    kept = rej = repl = add = 0
    why: dict = {}
    for L in buckets:
        groups = groups_of(L)
        served = sorted({r for rs in (groups or {}).values() for r in rs if (L, r) in table})
        if not served:
            continue
        if pinned.algo.get((Nf, K, L)) is None or pinned.algo.get((H, K, L)) is None:
            why["no pin"] = why.get("no pin", 0) + len(served)
            continue
        nb = -(-max(served) // L)
        xs = torch.cat([(torch.randn((L, K), generator=g, device="cuda") * (1 + b % 3)).to(torch.bfloat16)
                        for b in range(nb)])
        xo = _outliers(xs, g)
        refs = []
        for xx in (xs, xo):
            y = torch.cat([pinned.ref_linear(Nf, K, W, xx[b * L:(b + 1) * L]) for b in range(nb)])
            gt, up = y[:, :H].contiguous(), y[:, H:].contiguous()
            r1 = fz.silu_mul(gt, up)
            r2 = torch.nn.functional.silu(gt) * up
            refs.append((r1, r2))
            del y, gt, up
        for r in served:
            c = cands.get(table[(L, r)])
            if c is None:
                rej += 1
                why["kernel"] = why.get("kernel", 0) + 1
                continue
            ok = True
            try:
                for xx, (r1, r2) in zip((xs, xo), refs):
                    xr = xx[:r].contiguous()
                    for _ in range(2):
                        o = torch.full((r, H), float("nan"), dtype=torch.bfloat16, device="cuda")
                        ng.run_swiglu(c, W, xr, o)
                        if not (_eq(o, r1[:r]) and _eq(o, r2[:r])):
                            ok = False
            except RuntimeError as exc:
                ok = False
                why[str(exc)[:40]] = why.get(str(exc)[:40], 0) + 1
            if not ok:
                rej += 1
                why["bits"] = why.get("bits", 0) + 1
                continue
            if (L, r) in pinned.swiglu_r:
                repl += 1
            else:
                add += 1
            pinned.swiglu_r[(L, r)] = c
            kept += 1
        del xs, xo, refs
    torch.cuda.synchronize()
    if kept:
        log(f"lever used: NG7T ({ENV}) nsw7l at {kept} (bucket, R) ({repl} replace a plan SG pick, {add} replace GEMM + "
            f"silu_mul), bit-equal to the per-decision reference; {rej} rejected {why or ''} (they keep their pick)")
    else:
        log(f"lever not used: NG7T (0 of the table's keys passed: {rej} rejected {why or ''})")
    return kept
