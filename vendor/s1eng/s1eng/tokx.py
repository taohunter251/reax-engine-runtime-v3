"""TOKX (kai, S1ENG_TOKX="1"): every decision's prompt encode on a small thread pool instead of the HTTP event loop.

hotspot's server (server.Engine.decide) builds the prompt text and encodes it on the event loop's thread. With hotspot's
TOK_TAIL (S1ENG_TOK_TAIL=1, prompt.Encoder.enable_tail) the encode is the template's fixed ids plus the tail through
tokenizers' encode_batch_fast, which releases the GIL while it runs: on a pool thread it overlaps the loop's work on the
other requests (HTTP parsing, JSON, answers). decide below is server.Engine.decide statement for statement but for its
one encode statement, which becomes two: the prompt text is built on the loop as before, the ids come from
loop.run_in_executor(engine.tokx_pool, enc.encode, text). run_in_executor re-raises the encode's own exception, so every
status code and body (400 / 401 / 500) is hotspot's, and ARRIVE's counter still drops at submit (server.Engine._run) or in
handle's finally, on the loop thread only (tests/test_tokx.py). CPU measurement (hotspot's server and tail encoder, plumb-4b
tokenizer, an instant fake GPU, the validator's client): burst slope 183-188 -> 78-86 us per decision, a single decision
unchanged.

setup(engine) switches it on for one Engine: only with the tail encode on (without it the encode holds the GIL and
there is nothing to overlap); S1ENG_TOKX_THREADS threads (default 4, 1..16), named s1eng-tokx_<i>, all started from the
calling (main) thread before serving, each warming the encoder; tail_samples() encoded on the pool must equal the same
encodes on the calling thread. engine.decide is rebound on the instance last, so a refusal or an exception leaves hotspot's
decide and no thread behind. Off (flag unset): main.py never imports this module (tests/test_flag_off.py).
"""
from __future__ import annotations

import hmac
import json
import os
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

from .prompt import options_of, prompt_text, tail_samples
from .server import STATS, TRACE, _dumps

THREADS = 4          # S1ENG_TOKX_THREADS default
THREADS_MAX = 16
NAME = "s1eng-tokx"  # thread name prefix (ThreadPoolExecutor appends _<i>)


async def decide(self, headers: dict, body: bytes, tm: dict | None = None) -> tuple[int, bytes, str]:
    # server.Engine.decide (hotspot) but for the encode: the text on the loop, the ids from the pool (tests/test_tokx.py)
    if self.token and not hmac.compare_digest(headers.get("x-s1m-internal-token", ""), self.token):
        return 401, _dumps({"error": {"message": "unauthorized"}}), "application/json"
    try:
        req = json.loads(body)
        (qid, q), = req["questions"].items()
        if q.get("type") not in ("choice", "noul", "score"):
            raise ValueError("bad question type")
        q.setdefault("instructions", "")
    except Exception as exc:  # noqa: BLE001
        return 400, _dumps({"error": {"message": f"bad request: {type(exc).__name__}"}}), "application/json"
    try:
        options = options_of(q)
        t0 = time.perf_counter() if TRACE else 0.0
        text = prompt_text(req["state"], q["instructions"], [t for _, t in options])
        ids = await self.loop.run_in_executor(self.tokx_pool, self.enc.encode, text)
        if len(ids) > self.max_tokens:
            raise ValueError(f"input exceeds {self.max_tokens} model tokens")
        t1 = time.perf_counter() if TRACE else 0.0
        logits = await self._run(ids, tm)
        if TRACE:
            t2 = time.perf_counter()
            STATS["n"] = STATS.get("n", 0) + 1
            STATS["tok"] = STATS.get("tok", 0.0) + (t1 - t0)
            STATS["gpu"] = STATS.get("gpu", 0.0) + (t2 - t1)
            if tm is not None:
                tm["res"] = t2
        answer = self._answer(q["type"], options, logits)
    except ValueError as exc:
        return 400, _dumps({"error": {"message": str(exc)}}), "application/json"
    except Exception:  # noqa: BLE001  (the reference lets these escape: uvicorn answers 500)
        return 500, b"Internal Server Error", "text/plain; charset=utf-8"
    return 200, _dumps({"model": "s1-fast", "answers": {qid: answer},
                        "usage": {"input_tokens": len(ids), "output_tokens": 0}}), "application/json"


def setup(engine, env=None) -> str:
    """Switch TOKX on for ``engine`` (a server.Engine, before serving). Returns the start-up line: "TOKX on: ..." or
    "lever not used: tokx (<why>)"; an exception leaves the engine as it was (main.py logs it as not used)."""
    env = os.environ if env is None else env
    enc = engine.enc
    if getattr(enc, "_tail", None) is None:
        return "lever not used: tokx (the tail encode is off: S1ENG_TOK_TAIL)"
    n = int(env.get("S1ENG_TOKX_THREADS", "") or THREADS)
    if not 1 <= n <= THREADS_MAX:
        return f"lever not used: tokx (S1ENG_TOKX_THREADS {n} not in 1..{THREADS_MAX})"
    pool = ThreadPoolExecutor(n, thread_name_prefix=NAME)
    try:
        # every thread started now, from this thread: each warm-up task waits at the barrier, so the pool cannot hand
        # two to one thread
        gate = threading.Barrier(n)

        def warm():
            gate.wait(timeout=30)
            return enc.encode("warm-up")
        for f in [pool.submit(warm) for _ in range(n)]:
            f.result(timeout=60)
        texts = tail_samples()
        want = [enc.encode(t) for t in texts]
        got = list(pool.map(enc.encode, texts, timeout=60))
        bad = next((t for t, a, b in zip(texts, got, want) if a != b), None)
        if bad is not None:
            pool.shutdown(wait=True)
            return f"lever not used: tokx (pooled ids differ on {bad[-48:]!r})"
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    engine.tokx_pool = pool
    engine.decide = types.MethodType(decide, engine)
    return f"TOKX on: tail tokenization on {n} threads ({NAME}_0..{n - 1})"
