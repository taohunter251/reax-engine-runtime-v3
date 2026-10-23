"""s1-fast engine entry point.

The main thread loads the model, captures every CUDA graph and then runs the GPU loop (the reference also captures on
its start-up thread). The asyncio HTTP server runs on a second thread; it does JSON parsing, prompt building and
tokenisation, so none of that holds up the GPU.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path


def _log(msg: str) -> None:
    print(f"[s1eng] {msg}", file=sys.stderr, flush=True)


def _ntv() -> bool:
    """NTV (kai, S1ENG_NTV="1"): nonce-token prefix variants (s1eng/ntv.py). Off: hotspot (tests/test_flag_off.py)."""
    return os.environ.get("S1ENG_NTV", "") == "1"


def _tokx() -> bool:
    """TOKX (kai, S1ENG_TOKX="1"): the prompt encode on a thread pool, off the HTTP event loop (s1eng/tokx.py). Off:
    hotspot (tests/test_flag_off.py)."""
    return os.environ.get("S1ENG_TOKX", "") == "1"


def _ftok(env=os.environ):
    """FTOK (kai, backlog 4a): the ftok module when S1ENG_FTOK=1 or an S1ENG_ARMS arm asks for ftok 1 (its start-up
    check then runs), else None: ftok is not even imported and the tail encode is combine B's."""
    if env.get("S1ENG_FTOK", "") in ("", "0") and not env.get("S1ENG_ARMS"):
        return None
    from . import ftok

    return ftok if ftok.wanted(env)[1] else None


def _token() -> str:
    tok = os.environ.get("S1M_INTERNAL_TOKEN", "")
    path = os.environ.get("S1M_INTERNAL_TOKEN_FILE", "")
    if not tok and path and os.path.exists(path):
        tok = open(path).read().strip()
    if not tok and os.environ.get("RUNTIME_ALLOW_NO_AUTH") != "1":
        sys.exit("s1eng: no token (S1M_INTERNAL_TOKEN or S1M_INTERNAL_TOKEN_FILE)")
    return tok


def _temperature(model_dir: str) -> float:
    p = Path(model_dir) / "jevk5_config.json"
    t = float(json.loads(p.read_text()).get("temperature", 1.0)) if p.exists() else 1.0
    if not 2.06 < t < 2.08:
        sys.exit("s1eng: calibration temperature does not match the pinned release")
    return t


def _switch_s() -> float:
    """GIL switch interval in seconds: S1ENG_SWITCH_US (default 500 us; ARMS key switch_us switches it at run time)."""
    return int(os.environ.get("S1ENG_SWITCH_US", "500") or "500") / 1e6


def main() -> None:
    t0 = time.time()
    sys.setswitchinterval(_switch_s())
    model_dir = os.environ.get("PLUMB_MODEL_DIR", "/model")
    manifest = json.loads(Path(os.environ.get("REAX_MANIFEST", "/opt/reax/s1-fast-v1.json")).read_text())
    token = _token()
    temperature = _temperature(model_dir)
    front = None
    if os.environ.get("S1ENG_MPF", "") not in ("", "0"):   # MPF (ruj): the HTTP workers are forked here, before torch
        try:
            from . import mpf

            front = mpf.start(model_dir, manifest, token, temperature, log=_log)
        except Exception as exc:  # noqa: BLE001 - MPF must never stop the engine from serving
            _log(f"lever not used: mpf ({type(exc).__name__}: {exc})")

    import torch

    from . import fla_pin
    from .model import GpuLoop, Model
    from .prompt import Encoder
    from .server import Engine, serve

    torch.backends.cudnn.benchmark = False
    enc = Encoder(model_dir)
    impl = os.environ.get("S1ENG_IMPL", "ref")
    model = Model(model_dir, enc.slots, impl=impl)
    pins = fla_pin.pin()
    _log(f"loaded in {time.time() - t0:.1f}s impl={impl} fla pins={pins}")
    sets = int(os.environ.get("S1ENG_SETS", "1"))
    model.capture(extra_sets=sets - 1)
    _log(f"captured {sum(len(g) for g in model.graph_sets)} graphs in {sets} set(s) at {time.time() - t0:.1f}s, "
         f"GPU memory {torch.cuda.memory_reserved() / 2**30:.2f} GiB reserved")

    if os.environ.get("S1ENG_PREFIX", "0") == "1":
        model.prefix_ids = enc.template_prefix()
        _log(f"template prefix: {len(model.prefix_ids)} tokens")
        if os.environ.get("S1ENG_PREFIX_VARIANTS", "1") == "1":  # PV (a4)
            model.prefix_variants = enc.prefix_variants(model.prefix_ids)
            _log(f"prefix variants: {[len(v) for v in model.prefix_variants]} tokens")
        if _ntv():   # NTV (kai): + the nonce-token variants, from the template and a fixed hex enumeration (ntv.py);
            # never fatal: a failure logs "lever not used: ntv (...)" and keeps hotspot's variants (setup restores them)
            try:
                from . import ntv as ntvmod

                ntvmod.setup(model, enc.encode_full, log=_log)
            except Exception as exc:  # noqa: BLE001 - NTV must never stop the engine from serving
                _log(f"lever not used: ntv ({type(exc).__name__}: {exc})")
    if os.environ.get("S1ENG_TOK_TAIL", "0") == "1":   # E4: tokenize only the per-request tail, GIL released
        why = enc.enable_tail(getattr(model, "prefix_ids", None) or enc.template_prefix())
        _log("tokenizer: fixed template ids, tails encoded with the GIL released" if not why
             else f"lever not used: tok tail ({why})")
    ftm = _ftok()
    if ftm is not None:   # FTOK (kai): the exact ASCII tail tokenizer; never fatal (a refusal keeps tokenizers')
        _log(ftm.setup(enc))
    if front is not None:   # MPF: every worker's tokenizer and engine constants must equal this process's
        why = front.gate(enc, temperature, token, manifest)
        if why:
            _log(f"lever not used: mpf ({why})")
            front.shutdown()
            front = None
        elif ftm is not None and ftm.OK:   # FTOK: every worker must have it too, else the ARMS arm is refused
            _log(ftm.merge_workers(front.ftok_reports))
    bspec = os.environ.get("S1ENG_BATCH", "")
    if bspec or os.environ.get("S1ENG_PACK"):
        from .model import BatchLoop, lanes_from_env

        try:
            lanes = lanes_from_env(model, log=_log)
        except Exception as exc:  # noqa: BLE001  (E1: a start-up crash is a recorded failure; set 0 is exact)
            import traceback

            _log(f"batch lanes failed, per-decision graphs only: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            lanes = []
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if lanes:
            loop_cls = BatchLoop if front is None else front.loop_class(BatchLoop)   # MPF: a BatchLoop subclass
            gpu = loop_cls(model, lanes, depth=int(os.environ.get("S1ENG_DEPTH", "1")))
            _log(f"{len(lanes)} batch lane(s) ready at {time.time() - t0:.1f}s, GPU memory "
                 f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB reserved")
        else:
            _log("no batch lane fits in free GPU memory: per-decision graphs only")
            gpu = GpuLoop(model, depth=2)
    else:
        gpu = GpuLoop(model, depth=int(os.environ.get("S1ENG_DEPTH", "2")))
    from .gov import GOV

    on, busy, slept, n = GOV.done()   # nothing sleeps while serving
    if on:
        _log(f"start-up governor (S1ENG_START_DUTY {os.environ.get('S1ENG_START_DUTY')}): {n} steps, GPU-busy "
             f"{busy:.1f}s, slept {slept:.1f}s")
    engine = Engine(enc, gpu, temperature, token, manifest["limits"]["max_input_tokens"],
                    manifest["weights"]["revision"])
    # warm the tokenizer and the response path once before serving
    enc.encode("warm-up")
    if _tokx():   # TOKX (kai): the encode of every decision on a small thread pool (tokx.py); never fatal: a refusal or
        # a failure logs "lever not used: tokx (...)" and hotspot's decide serves
        try:
            from . import tokx as tokxmod

            _log(tokxmod.setup(engine))
        except Exception as exc:  # noqa: BLE001 - TOKX must never stop the engine from serving
            _log(f"lever not used: tokx ({type(exc).__name__}: {exc})")

    if front is not None:   # MPF: the workers serve HTTP; on a refusal, #131's server thread below
        why = front.serve(gpu, os.environ.get("RUNTIME_UDS"), t0)
        if not why:
            gpu.run()
            return
        _log(f"lever not used: mpf ({why})")
        front.shutdown()

    uds = os.environ.get("RUNTIME_UDS")
    host = os.environ.get("RUNTIME_HOST", "127.0.0.1")
    port = int(os.environ.get("RUNTIME_PORT", "8101"))

    def http_thread():
        asyncio.run(serve(engine, uds, host, port, ready_cb=lambda: _log(f"serving on {uds or f'{host}:{port}'} "
                                                                          f"at {time.time() - t0:.1f}s")))

    th = threading.Thread(target=http_thread, name="http", daemon=True)
    th.start()
    gpu.run()


if __name__ == "__main__":
    main()
