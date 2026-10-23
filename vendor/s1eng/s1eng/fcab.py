"""FCAB (ruj v14 fixedcost track): in-situ A/B harness for the FCL levers - timing only, opt-in.

Nothing in this module runs unless S1ENG_FCAB is set (a comma list of arm names, "env" excluded). With it set the
engine starts as always (arm "env" = the lanes capture_batch built with every FCL lever the env and its start-up gates
turned on, self-checked and uploaded); then
  * build() (lanes_from_env, after the env lanes are ready) captures one more lane set per distinct FCL lever subset
    an arm names, exactly as capture_batch's lane loop captures the env lanes (packed graphs, the same replay and
    capture stream per lane index, own memory pools and cuBLASLt workspaces, PAR side workspace where the env lane has
    PAR), with the levers the arm switches off removed from PackedForward for the capture only; runs the engine's
    self_check on every set (an arm whose set fails it is dropped), fills the L0T tables again (self_check empties
    them) and uploads the new graphs (S1ENG_GRAPH_UPLOAD);
  * attach() (BatchLoop.__init__) starts a watcher thread that, whenever the control file S1ENG_FCAB_CTL (default
    /run/reax/arm, written by abdriver2.py --ctl) changes, switches BatchLoop.lanes to the arm's set and logs one line
    per segment. Lane i of every set replays on lane i's stream; the batch's lane travels with it to the completer.
Every set is exact by itself: each lever byte-gated at start-up, each set self-checked against the per-decision graphs.

Arms: toggles joined with "+"; "env" (or "") = the env lanes.
  no<lever>   that FCL lever off in the arm's set: noba, nolt, noemb, noro, notg, notgs, noxl
  nofc        every FCL lever off (= v13b's graphs)
  copy        a captured copy of the arm's set (same graphs in an own set: measures the capture-set bias)
  bacfg=<BM/warps/stages>   BA's kernel config for this set (S1ENG_BA_CFG while it is captured; every config gives
              the same bytes: row-independent arithmetic, the gate passed at the env config)
  tgcfg=<BN/warps/stages>, tgscfg=<BN/warps/stages>   the same for TG (S1ENG_TG_CFG) / TGS (S1ENG_TGS_CFG)
  nolean      LL (S1ENG_LEAN) off while the arm is active (scheduling only: no extra lane set; v14 integrate)
  nomx        own-final: no extra lane set (the env set); MX's join is switched by the S1ENG_ARMS arm of the same name
  nokf        KF (ruj gemm2/k2): this set without KF = the base picks + the base RARN / SKR selections (S1+KF: with
              its own MX mixed graphs, so S1's joins run in it as in the env set)
  noskt       skx6t: this set's MX mixed graphs run skx6 itself, not the tail-skip variant (S1ENG_SKX6T=0 while the
              set is captured)
  kfx, nokfx  KFX (kai/kx-k2 #6, S1ENG_KFX): this set's MX mixed graphs with / without the kfx kernels at the keys
              their start-up gate passed (pick lever "kfx": mx.KFX swapped; S1ENG_KFX_ARM=1 = env set without them)
  <pick>, no<pick>   kai/kx pick levers: a lever that replaces exact start-up picks (pinned.swiglu_r, pinned.algo_r,
              ...; e.g. NG7's nsw7l SG picks over the plan's nsw9 picks) calls register(name, obj, attr, on=, off=)
              once its own start-up bit checks passed; "<name>" captures the set with the lever's picks, "no<name>"
              with the picks they replaced, whichever of the two the env set uses (the dicts are swapped in place
              for the capture only; the set is then self-checked like every set). An unregistered name = an unknown
              toggle: the arm is dropped. Registered levers: ng7 (pinned.swiglu_r), tailmb (S1ENG_TAILMB, mb.py:
              PackedForward.tailmb / .tailmb_sw, the last layer's skinny M = B kernels; notailmb = TG / TGS).
              kai/cb-xa adds xa3 (pf.xa_ver + pf.xl_ver: XA3's xa_fwd3q per gated bucket, xa.register_arms;
              registered only when S1ENG_XA3=1 or an arm names xa3 / noxa3).
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading
import time

import torch

# PackedForward attribute of every FCL lever (dict or set of buckets; empty = lever off)
LEVERS = ("ba", "lt", "emb", "ro", "tg", "tgs", "xl", "rn", "rng", "skr", "grp", "late_r")   # + r2 fixedcost2 SKR (noskr)   # + rgh GRP / LATE (nogrp, nolate_r)   # + r2 hotspot RN (norn / norng)   # (no QKR on v15: XA folds the q norm + RoPE)


# kai/kx pick levers: name -> [(obj, attr, on picks, off picks)] (see register)
PICKS: dict = {}
_TOGGLES = ("copy", "nolean", "noonorm", "nomsd", "nomx", "nokf", "noskt", "nofc", "env")


def register(name: str, obj, attr: str, on: dict, off: dict) -> None:
    """A pick lever's two pick sets for arms "<name>" / "no<name>": obj.<attr> (a dict of picks) holds `on` or `off`
    while such an arm's lane set is captured. Call it after the lever's start-up checks, once per (obj, attr)."""
    bad = (not name or not name.isalnum() or name in LEVERS or name in ("fc", "kf", "mx", "lean", "onorm", "msd",
                                                                          "skt") or name in _TOGGLES
           or name.startswith("no"))
    if bad:
        raise ValueError(f"FCAB pick lever name {name!r} is empty or clashes with an FCAB toggle")
    PICKS.setdefault(name, []).append((obj, attr, dict(on), dict(off)))


def picks_of(arm: str) -> tuple:
    """The pick-lever toggles of an arm: (("ng7", "on"), ...) for "ng7", (("ng7", "off"),) for "nong7"."""
    out = []
    for t in toggles(arm):
        if t in PICKS:
            out.append((t, "on"))
        elif t.startswith("no") and t[2:] in PICKS:
            out.append((t[2:], "off"))
    return tuple(out)


@contextlib.contextmanager
def swapped(picks: tuple):
    """obj.<attr> of every pick lever in `picks` holds that arm's picks (in place: other code holds the dicts)."""
    saved = []
    try:
        for name, side in picks:
            for obj, attr, on, off in PICKS[name]:
                d = getattr(obj, attr)
                saved.append((d, dict(d)))
                d.clear()
                d.update(on if side == "on" else off)
        yield
    finally:
        for d, old in reversed(saved):
            d.clear()
            d.update(old)


def arms() -> list:
    return [a for a in os.environ.get("S1ENG_FCAB", "").replace(" ", "").split(",") if a and a != "env"]


def toggles(arm: str) -> list:
    return [t for t in arm.split("+") if t and t != "env"]


def off_of(arm: str) -> frozenset:
    off = set()
    for t in toggles(arm):
        if t == "nofc":
            off |= set(LEVERS) - {"rn", "rng", "skr", "grp", "late_r"}
        elif t.startswith("no") and t[2:] in LEVERS:
            off.add(t[2:])
        elif t in PICKS or (t.startswith("no") and t[2:] in PICKS):   # kai/kx pick levers
            continue
        elif t not in ("copy", "nolean", "noonorm", "nomsd", "nomx", "nokf", "noskt") and not t.startswith(("bacfg=", "tgcfg=", "tgscfg=", "rnk=", "rngh=")):
            raise ValueError(f"unknown FCAB toggle {t!r}")
    return frozenset(off)


def env_of(arm: str) -> dict:
    """env overrides while the arm's set is captured (bacfg= / tgcfg= / tgscfg= -> S1ENG_BA_CFG / _TG_CFG / _TGS_CFG)"""
    ov = {}
    for t in toggles(arm):
        if t == "noonorm":            # r2 hotspot: rgh's ONORM (gated norm in the FFS epilogue) off in this set
            ov["S1ENG_ONORM_OFF"] = "1"
            continue
        if t == "nomsd":              # r2 fixedcost2: MSD (lane 1 in the REMOTE memory sync domain) off in this set
            ov["S1ENG_MSD"] = "0"
            continue
        if t == "noskt":              # skx6t: the set's mixed graphs capture skx6 itself (mx.run reads the switch)
            ov["S1ENG_SKX6T"] = "0"
            continue
        if t.startswith("rnk="):     # r2 hotspot RN: kernel kind of this set (S1ENG_RN_KIND)
            ov["S1ENG_RN_KIND"] = t[4:]
            continue
        if t.startswith("rngh="):    # r2 hotspot RN: gated-norm head-rows per warp (S1ENG_RN_GHPW)
            ov["S1ENG_RN_GHPW"] = t[5:]
            continue
        for pre, var in (("bacfg=", "S1ENG_BA_CFG"), ("tgcfg=", "S1ENG_TG_CFG"), ("tgscfg=", "S1ENG_TGS_CFG")):
            if t.startswith(pre):
                a, w, st = (int(x) for x in t[len(pre):].split("/"))
                ov[var] = f"{a},{w},{st}"
    return ov


def _capture_set(model, ref_lanes, log=print) -> list:
    """One lane set like capture_batch's lane loop (packed graphs only, as the env lanes: S1ENG_BATCH is empty)."""
    from . import kprio
    from .batchfwd import packed_layout, PackedMeta
    from .gov import GOV
    from .model import Lane

    c = model._fcab_ctx
    pack, prow, pfx_of, pinned, bf, pf = c["pack"], c["prow"], c["pfx_of"], c["pinned"], c["bf"], c["pf"]
    buckets, par_all = c["buckets"], c["par_all"]
    out = []
    nwarm = int(os.environ.get("S1ENG_WARMUPS", "1"))
    try:
        for rl in ref_lanes:
            pool = torch.cuda.graph_pool_handle()
            cap_stream = rl.cap_stream
            ws = torch.empty(max(pinned.ws_bytes, 1 << 20), dtype=torch.uint8, device="cuda")
            bf.ws = ws
            use_par = bool(par_all) and len(rl.ws) > 1
            pf.par = par_all if use_par else type(par_all)()
            ws_side = torch.empty_like(ws) if pf.par else None
            pf.ws_side = ws_side

            def cap(fn, *args):
                side = model.warm_stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(nwarm):
                        fn(*args)
                torch.cuda.current_stream().wait_stream(side)
                from . import msd as _msdm
                from . import skr as _skrm

                kp = kprio.mode()
                sk = bool(getattr(pf, "skr", None))         # r2 fixedcost2 SKR
                ms = _msdm.enabled() and rl.idx % 2 == 1    # r2 fixedcost2 MSD
                keep = bool(kp) or sk or ms
                graph = torch.cuda.CUDAGraph(keep_graph=True) if keep else torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=cap_stream):
                    o = fn(*args)
                if sk:
                    try:
                        _skrm.prune(graph)
                    except Exception:  # noqa: BLE001 - the reduce node stays (exact either way)
                        pass
                if ms:
                    try:
                        _msdm.apply(graph, rl.idx)
                    except Exception:  # noqa: BLE001 - scheduling only
                        pass
                if kp:
                    try:
                        kprio.apply(graph, kp)
                    except Exception:  # noqa: BLE001 - scheduling only
                        pass
                if keep:
                    graph.instantiate()
                GOV.tick()
                return graph, o

            packed = {}
            todo = [(L, B, R) for L in buckets if L in pack and L in rl.pbs for B, rs in prow[L].items()
                    if B in rl.pbs[L] for R in rs]
            todo.sort(key=lambda t: -t[2])
            for L, B, R in todo:
                pfx = pfx_of.get(L, 0)
                meta = PackedMeta(B, L, R, pfx=pfx)
                st, sl, rp, la = packed_layout([min(L, R // B + pfx)] * B, B, L, R, pfx)
                meta.start.copy_(torch.from_numpy(st))
                meta.slen.copy_(torch.from_numpy(sl))
                meta.row_pos.copy_(torch.from_numpy(rp))
                meta.last.copy_(torch.from_numpy(la))
                graph, o = cap(pf, meta)
                packed.setdefault((L, B), []).append((R, graph, meta, o))
            for v in packed.values():
                v.sort(key=lambda e: e[0])
            # S1+KF: the env lane's MX mixed graphs too (same (D, S, B, R) set, staged as capture_batch stages them),
            # so the S1ENG_JOIN joins run in this set as in the env set (own-final's _groups guard needs them);
            # build() gates them with the engine's mix_check
            mixed = {}
            mtodo = sorted(((D_, S_, B, e[0]) for (D_, S_, B), es in (getattr(rl, "mixed", None) or {}).items()
                            for e in es), key=lambda t: -t[3])
            for D_, S_, B, R in mtodo:
                pfx = pfx_of.get(D_, 0)
                meta = PackedMeta(B, D_, R, pfx=pfx, mix=S_)
                st, sl, rp, la = packed_layout([min(D_, R // B + pfx)] * B, B, D_, R, pfx)
                meta.start.copy_(torch.from_numpy(st))
                meta.slen.copy_(torch.from_numpy(sl))
                meta.row_pos.copy_(torch.from_numpy(rp))
                meta.last.copy_(torch.from_numpy(la))
                b1 = max(1, B // 2)
                meta.mx.copy_(torch.tensor([int(st[b1]), b1], dtype=torch.int32))
                graph, o = cap(pf, meta)
                mixed.setdefault((D_, S_, B), []).append((R, graph, meta, o))
            for v in mixed.values():
                v.sort(key=lambda e: e[0])
            torch.cuda.synchronize()
            ln = Lane(rl.idx, rl.stream, rl.single, {}, packed,
                      ws=[ws] + ([ws_side] if ws_side is not None else []), cap_stream=cap_stream,
                      pfx=dict(rl.pfx))
            ln.mixed = mixed
            ln.mbs = {}
            for (D_, S_, B) in mixed:
                ln.mbs.setdefault((D_, S_), []).append(B)
            for v in ln.mbs.values():
                v.sort()
            out.append(ln)
            log(f"FCAB lane {rl.idx}: {sum(len(v) for v in packed.values())} packed + "
                f"{sum(len(v) for v in mixed.values())} mixed graphs, PAR "
                f"{sorted(pf.par) if pf.par else 'off'}, GPU memory {torch.cuda.memory_reserved() / 2**30:.2f} GiB "
                f"reserved")
    finally:
        pf.ws_side = None
        pf.par = par_all
        bf.ws = None
    return out


def _snap(pf) -> dict:
    return {k: (dict(v) if isinstance(v, dict) else set(v)) for k in LEVERS
            for v in [getattr(pf, k, None)] if v is not None}


def _restore(pf, s: dict) -> None:
    for k, v in s.items():
        setattr(pf, k, dict(v) if isinstance(v, dict) else set(v))


def build(model, lanes: list, log=print) -> None:
    """lanes_from_env: capture the lane set of every distinct FCL lever subset the arms need."""
    from . import model as mmod

    names = arms()
    c = getattr(model, "_fcab_ctx", None)
    if not names or not lanes or c is None:
        return
    pf = c["pf"]
    t0 = time.time()
    env = _snap(pf)
    sets = {"env": lanes}
    arm_set = {}
    built = {}
    for arm in names:
        try:
            off = off_of(arm)
        except ValueError as exc:
            log(f"FCAB arm {arm}: {exc}; arm dropped")
            continue
        ov = env_of(arm)
        nokf = "nokf" in toggles(arm)   # KF (ruj gemm2/k2): this set without KF (base picks + base RARN)
        pk = picks_of(arm)               # kai/kx pick levers (e.g. NG7): this set with those picks
        key = (off, "copy" in toggles(arm), tuple(sorted(ov.items())), nokf, tuple(sorted(pk)))
        if key == (frozenset(), False, (), False, ()):
            arm_set[arm] = "env"
            continue
        if key in built:
            arm_set[arm] = built[key]
            continue
        name = f"s{len(sets)}"
        st = {k: (type(v)() if k in off else (dict(v) if isinstance(v, dict) else set(v))) for k, v in env.items()}
        _restore(pf, st)
        log(f"FCAB set {name} for arm {arm}: levers off {sorted(off) or '-'}; on "
            f"{ {k: sorted(v) for k, v in st.items() if v} }" + (f"; env {ov}" if ov else ""))
        old_env = {k: os.environ.get(k) for k in ov}
        os.environ.update(ov)
        pinned = c["pinned"]
        kf_swap = None
        if nokf:
            kfb, rnb = getattr(pinned, "_kf_base", None), getattr(pf, "_rarn_nokf", None)
            if kfb is None or rnb is None:
                log(f"FCAB set {name}: arm {arm} needs KF on in the env set (no base picks kept); arm dropped")
                os.environ.update({k: v for k, v in old_env.items() if v is not None})
                for k, v in old_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                _restore(pf, env)
                continue
            kf_swap = (dict(pinned.algo_r), pf.rarn)
            pinned.algo_r.clear()
            pinned.algo_r.update(kfb)
            pf.rarn = dict(rnb)
            skb = getattr(pf, "_skr_nokf", None)   # S1+KF: SKR on the base picks (restored with the env snapshot)
            if skb is not None and "skr" not in off:
                pf.skr = dict(skb)
            elif getattr(pf, "skr", None) and "skr" not in off:
                log(f"FCAB set {name}: no base SKR selection kept; arm {arm} uses the env set's SKR keys")
        if pk:
            log(f"FCAB set {name}: picks {[f'{n}={side}' for n, side in pk]}")
        try:
            with swapped(pk):
                ls = _capture_set(model, lanes, log=log)
        except Exception as exc:  # noqa: BLE001 - timing harness: drop this arm
            log(f"FCAB set {name}: capture failed ({type(exc).__name__}: {exc}); arm {arm} dropped")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            continue
        finally:
            if kf_swap is not None:
                pinned.algo_r.clear()
                pinned.algo_r.update(kf_swap[0])
                pf.rarn = kf_swap[1]
            _restore(pf, env)
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        if os.environ.get("S1ENG_SELFCHECK", "1") == "1":
            res = mmod.self_check(model, ls, log=lambda m, n=name: log(f"FCAB set {n}: {m}"))
            if not all(res.values()):
                log(f"FCAB set {name}: SELF-CHECK FAILED for {res}; arm {arm} dropped")
                continue
        pairs = sorted({(S_, D_) for ln in ls for (D_, S_) in getattr(ln, "mbs", {})
                        if (getattr(model, "join_ok", None) or {}).get((S_, D_))})
        if pairs:   # S1+KF: this set's mixed graphs, the engine's MX gate (every row vs its own per-decision graph)
            try:
                mres = mmod.mix_check(model, ls, pairs, log=lambda m, n=name: log(f"FCAB set {n}: {m}"))
            except Exception as exc:  # noqa: BLE001 - timing harness: no joins in this set
                mres = {}
                log(f"FCAB set {name}: mix check failed ({type(exc).__name__}: {exc})")
            if not all(mres.get(p_) for p_ in pairs):
                for ln in ls:
                    ln.mixed, ln.mbs = {}, {}
                log(f"FCAB set {name}: mixed graphs dropped ({mres}): no joins while arm {arm} is active")
        elif any(getattr(ln, "mixed", None) for ln in ls):
            for ln in ls:
                ln.mixed, ln.mbs = {}, {}
        sets[name] = ls
        built[key] = name
        arm_set[arm] = name
    l0t = getattr(pf, "l0t", None)
    if l0t is not None and len(sets) > 1:   # self_check emptied the L0T tables: fill them again as lanes_from_env
        from . import l0t as l0tmod

        rows_of = {}
        for (L, _B), es_ in lanes[0].packed.items():
            rows_of.setdefault(L, []).extend(e[0] for e in es_)
        l0tmod.prefill(l0t, pf, rows_of, log=log)
    if os.environ.get("S1ENG_GRAPH_UPLOAD", "0") == "1":
        for name, ls in sets.items():
            if name != "env":
                mmod.upload_graphs(ls, log=lambda m, n=name: log(f"FCAB set {n}: {m}"))
                if any(getattr(ln, "mixed", None) for ln in ls):   # S1+KF: the set's mixed graphs too
                    mmod.upload_graphs(ls, log=lambda m, n=name: log(f"FCAB set {n}: {m}"), mixed=True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    model._fcab_sets = sets
    model._fcab_arm_set = arm_set
    log(f"FCAB: arms -> lane sets {arm_set}, sets {list(sets)} ready in {time.time() - t0:.1f}s, GPU memory "
        f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB reserved")


def attach(loop) -> None:
    """BatchLoop.__init__: watcher thread; BatchLoop.lanes follows the control file."""
    sets = getattr(loop.m, "_fcab_sets", None)
    amap = getattr(loop.m, "_fcab_arm_set", None)
    if not sets or amap is None:
        return
    path = os.environ.get("S1ENG_FCAB_CTL", "/run/reax/arm")
    lean0 = bool(getattr(loop, "lean", False))   # v14 integrate: LL as the env set it (nolean arms turn it off)

    def log(msg):
        print(f"[s1eng] {msg}", file=sys.stderr, flush=True)

    def stats_line():
        st = dict(loop.stats)
        loop.stats.clear()
        nb = sum(v for k, v in st.items() if len(k) == 4)
        dec = sum(k[-1] * v for k, v in st.items() if len(k) == 4)
        rows = sum(k[2] * v for k, v in st.items() if len(k) == 4)
        return nb, dec, rows

    def watch():
        last, cur, t_seg = None, "env", time.time()
        stats_line()
        while True:
            time.sleep(0.02)
            try:
                with open(path) as fh:
                    arm = fh.read().strip()
            except OSError:
                continue
            if arm == last or not arm:
                continue
            last = arm
            now = time.time()
            nb, dec, rows = stats_line()
            seg = now - t_seg
            log(f"FCAB segment {cur} {seg:.1f}s: {nb} packed batches, {dec} decisions "
                f"({dec / max(seg, 1e-9):.1f}/s engine-side), mean B {dec / max(nb, 1):.2f}, mean R "
                f"{rows / max(nb, 1):.0f}")
            t_seg = now
            if arm.startswith("_"):
                continue
            if arm != "env" and arm not in amap:
                log(f"FCAB: unknown or dropped arm {arm!r} (have {list(amap)}); staying on {cur}")
                continue
            sname = "env" if arm == "env" else amap[arm]
            with loop.cv:
                loop.lanes = sets[sname]
                loop.lean = lean0 and "nolean" not in toggles(arm)   # v14 integrate: LL per arm
                loop.cv.notify_all()
            log(f"FCAB switch {cur} -> {arm} (set {sname}, lean {loop.lean}) at {now:.3f}")
            cur = arm

    threading.Thread(target=watch, name="fcab", daemon=True).start()
    log(f"FCAB watcher on {path}: arms {amap}")
