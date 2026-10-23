"""FP (rgh 2026-10-20, after u198's shipped plan caches): frozen start-up GEMM selection.

S6 (lt.tune) and SG (model.select_swiglu) choose, per (shape, bucket, packed row count R), the fastest exact algorithm
by TIMING the candidates at start-up. That costs minutes, and the timings are noisy when the GPU is shared (the
validator's loaded condition: other engines and serving traffic run while ours starts), so picks made under load can
be slower than the picks of an idle start-up. FP ships the picks of an idle start-up (S1ENG_PLAN_DUMP=<file> writes
them) and at start-up (S1ENG_PLAN=1: the built-in plan for this GPU model; S1ENG_PLAN=<file>) restores each pick and
keeps it only after the byte checks S6 / SG apply to a winner (the per-decision reference output at every R it
serves). Nothing is timed. Bits cannot differ from a start-up without FP: every kept pick is an algorithm S6 / SG
would have accepted (same checks); only the choice among exact candidates comes from the file.

cuBLASLt algos are stored as the raw cublasLtMatmulAlgo_t bytes (NVIDIA: "can be trivially serialized and later
restored for use with the same version of cuBLAS library"), NG algos by kernel name and parameters. A plan is used only
on the GPU model, libcublasLt version, torch version and NG kernel set it was written with (else: normal timing); a
pick that fails its check keeps the pin (S6) or the current path (SG).
"""
from __future__ import annotations

import ctypes as C
import json
import os

import torch

SCHEMA = 1
_DIR = os.path.dirname(os.path.abspath(__file__))
BUILTIN = {"NVIDIA RTX PRO 6000 Blackwell Server Edition": "plan_pro6000.json"}


def meta() -> dict:
    from . import gtable, ng
    from . import lt as ltmod

    lib = ng.lib()
    kern = sorted(lib.fn) if lib is not None else []
    sw = sorted(lib.sw) if lib is not None and lib.sw else []
    return {"schema": SCHEMA, "gpu": torch.cuda.get_device_name(), "lt": gtable.loaded_lt_version(ltmod.lt),
            "torch": torch.__version__, "ng": kern, "nsw": sw}


def _algo_hex(a) -> str:
    return C.string_at(C.byref(a), C.sizeof(a)).hex()


def _algo_of(ltmod, hx: str):
    raw = bytes.fromhex(hx)
    a = ltmod.Algo()
    if len(raw) != C.sizeof(a):
        return None
    C.memmove(C.byref(a), raw, len(raw))
    return a


def _enc(a) -> dict:
    from . import ng
    from . import lt as ltmod

    if isinstance(a, ng.NgAlgo):
        return {"ng": a.name, "drop": a.drop, "chunk": a.chunk, "split": a.split, "red": a.red}
    return {"lt": _algo_hex(a), "key": list(ltmod.cfg_key(a))}


def _dec(e: dict):
    from . import ng
    from . import lt as ltmod

    if "ng" in e:
        return ng.NgAlgo(e["ng"], int(e["drop"]), int(e["chunk"]), int(e["split"]), int(e["red"]))
    a = _algo_of(ltmod, e["lt"])
    if a is None or list(ltmod.cfg_key(a)) != list(e["key"]):
        return None
    return a


def dump(path: str, pinned, log=print) -> None:
    """Writes this start-up's S6 / SG picks (call after the selection, before the graphs)."""
    plan = {"meta": meta(),
            "algo_r": [[N, K, L, R, _enc(a)] for (N, K, L, R), a in sorted(pinned.algo_r.items())],
            "best_t": [[N, K, L, R, t] for (N, K, L, R), t in sorted(pinned.best_t.items())],
            "fuse_r": [[N, K, L, R, bool(u)] for (N, K, L, R), u in sorted(getattr(pinned, "fuse_r", {}).items())],
            "swiglu_r": [[L, R, c.name, c.drop] for (L, R), c in sorted(getattr(pinned, "swiglu_r", {}).items())]}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(plan, f, separators=(",", ":"))
    os.replace(tmp, path)
    log(f"FP: plan written to {path} ({len(plan['algo_r'])} S6 picks, {len(plan['swiglu_r'])} SG picks)")


def load(spec: str, log=print) -> dict | None:
    """The plan for this start, or None (a skip is logged as "lever not used: plan (<reason>)")."""
    if spec in ("", "0"):
        return None
    try:
        gpu = torch.cuda.get_device_name()
        path = os.path.join(_DIR, BUILTIN[gpu]) if spec == "1" and gpu in BUILTIN else spec
        if spec == "1" and gpu not in BUILTIN:
            log(f"lever not used: plan (no built-in plan for {gpu})")
            return None
        with open(path) as f:
            plan = json.load(f)
        want, have = meta(), plan.get("meta", {})
        for k in ("schema", "gpu", "lt", "torch", "ng", "nsw"):
            if k in ("ng", "nsw") and isinstance(have.get(k), list) and set(have[k]) <= set(want.get(k) or []):
                continue   # COV9b: a plan made with fewer NG kernels stays valid (every pick is byte-checked)
            if have.get(k) != want.get(k):
                log(f"lever not used: plan ({k} differs: plan {str(have.get(k))[:60]!r} vs here {str(want.get(k))[:60]!r})")
                return None
        return plan
    except Exception as exc:  # noqa: BLE001 - never take the batch lanes down: normal timing instead
        log(f"lever not used: plan ({type(exc).__name__}: {exc})")
        return None


def apply_s6(plan: dict, pinned, groups_of, weights: dict, log=print) -> None:
    """S6 without timing: each planned (N, K, bucket, R) algo is restored and kept only if it fits the workspace and
    reproduces the per-decision reference bits at that R (the check lt.tune applies to a winner at every R it serves)."""
    from . import lt as ltmod

    ctx, ws, g = pinned.ctx, pinned._ws, pinned._gen
    by = {}
    for N, K, L, R, e in plan.get("algo_r", []):
        by.setdefault((N, K, L), {})[R] = e
    kept = rej = 0
    why = {}
    for (N, K, L), ref in list(pinned.algo.items()):
        if ref is None or (N, K, L) not in by:
            continue
        groups = groups_of(L)
        if not groups:
            continue
        rs_all = sorted({r for rs in groups.values() for r in rs})
        W = weights[(N, K)][0]
        nb = -(-max(rs_all) // L)
        xs = torch.cat([(torch.randn((L, K), generator=g, device="cuda") * (1 + b % 3)).to(torch.bfloat16)
                        for b in range(nb)])
        ys = torch.cat([pinned.ref_linear(N, K, W, xs[b * L:(b + 1) * L]) for b in range(nb)])
        for r in rs_all:
            e = by[(N, K, L)].get(r)
            if e is None:
                continue
            a = _dec(e)
            if a is None:
                rej += 1
                why["restore"] = why.get("restore", 0) + 1
                continue
            need = ctx.ws_needed(N, K, r, a)
            if need is None or need > ws.numel():
                rej += 1
                why["workspace"] = why.get("workspace", 0) + 1
                continue
            o = torch.empty((r, N), dtype=torch.bfloat16, device="cuda")
            try:
                ctx.run(a, W, xs[:r].contiguous(), o, ws)
            except RuntimeError:
                rej += 1
                why["run"] = why.get("run", 0) + 1
                continue
            if not ltmod._eq(o, ys[:r]):
                rej += 1
                why["bits"] = why.get("bits", 0) + 1
                continue
            pinned.algo_r[(N, K, L, r)] = a
            pinned.ws_bytes = max(pinned.ws_bytes, need)
            kept += 1
    for N, K, L, R, t in plan.get("best_t", []):
        pinned.best_t[(N, K, L, R)] = float(t)
    torch.cuda.synchronize()
    log(f"FP: S6 from plan: {kept} picks kept, {rej} rejected {why or ''} (rejected ones keep the pin); no timing")


def apply_sg(plan: dict, pinned, fz, finfo, groups_of, buckets, log=print) -> None:
    """SG without timing: each planned (bucket, R) SwiGLU kernel is kept only if its output equals the reference
    (F.linear gate / up on M = bucket blocks, then silu_mul) bit for bit at that R (select_swiglu's per-R check)."""
    from . import ng

    ent = [((Nf, K), e) for (Nf, K), e in finfo.items() if e[0] == "gate_up"]
    want = {(L, R): (nm, d) for L, R, nm, d in plan.get("swiglu_r", [])}
    if not ent or not want:
        return
    (Nf, K), (gname, n1, n2, blks) = ent[0]
    H = n1
    cands = {(c.name, c.drop): c for c in (ng.swiglu_candidates(H, K) if n1 == n2 else [])}
    Wb = [b for _, b in blks][:8]
    g = torch.Generator(device="cuda")
    g.manual_seed(4321)
    kept = rej = 0
    for L in buckets:
        groups = groups_of(L)
        rs_all = sorted({r for rs in (groups or {}).values() for r in rs if (L, r) in want})
        if not rs_all or pinned.algo.get((Nf, K, L)) is None or pinned.algo.get((H, K, L)) is None:
            continue
        nb = -(-max(rs_all) // L)
        xs = torch.cat([(torch.randn((L, K), generator=g, device="cuda") * (1 + b % 3)).to(torch.bfloat16)
                        for b in range(nb)])
        y = torch.cat([pinned.ref_linear(Nf, K, Wb[0], xs[b * L:(b + 1) * L]) for b in range(nb)])
        ref = fz.silu_mul(y[:, :H].contiguous(), y[:, H:].contiguous())
        del y
        for r in rs_all:
            c = cands.get(tuple(want[(L, r)]))
            if c is None:
                rej += 1
                continue
            o = torch.empty((r, H), dtype=torch.bfloat16, device="cuda")
            try:
                ng.run_swiglu(c, Wb[0], xs[:r].contiguous(), o)
            except RuntimeError:
                rej += 1
                continue
            if not torch.equal(o.view(torch.int16), ref[:r].view(torch.int16)):
                rej += 1
                continue
            pinned.swiglu_r[(L, r)] = c
            kept += 1
    torch.cuda.synchronize()
    log(f"FP: SG from plan: {kept} picks kept, {rej} rejected (rejected ones keep gate|up GEMM + silu_mul); no timing")
