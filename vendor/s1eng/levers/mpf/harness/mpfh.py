#!/usr/bin/env python3
"""gvhb CPU harness engine (ruj, not shipped; no GPU, no CUDA): main.py's serving flow with the GPU replaced by a
deterministic fake, to test the MPF plumbing (s1eng/mpf.py) against #131's single-process server on the same fake.
Runs INSIDE the #131 image with the gvhb tree mounted over /opt/s1eng (server.py / prompt.py / model.py = #131).

  * the same start-up order as main.py: token, temperature, mpf.start() (S1ENG_MPF=N: fork before torch), Encoder,
    template prefix + prefix variants + NTV (fake model object), TOK_TAIL, mpf gate, BatchLoop (MPFLoop when MPF is
    on) over two fake lanes with the image's packed tables (HOLD_N 8, HOLD_MS 3, ARRIVE, WAKE_FILTER, JOIN 128:192,
    LONE_PACKED from the image ENV), Engine, warm-up, then front.serve() or #131's serve() thread, then gpu.run();
  * fake launch: every job's 16 outputs are a function of its own token ids only (crc32 -> seeded normal(0, 3) ->
    bf16), written into the slot's out rows, so an answer depends on its request only (MPF vs base must be byte-equal
    at any concurrency); completion after 1.5 ms + 4 us per packed row (sleep, GIL released);
  * HB_ERR=1: a lone job whose length is 5 mod 41 raises ValueError("fake launch: length <n>"), 6 mod 43 raises
    RuntimeError (decide must answer 400 with that text / 500, as #131 does).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import zlib

sys.setswitchinterval(0.0005)
os.environ["S1ENG_LEAN"] = "0"           # the lean launch needs CUDA events
os.environ.pop("S1ENG_FCAB", None)
os.environ.pop("S1ENG_ARMS", None)


def log(msg):
    print(f"[s1eng] {msg}", file=sys.stderr, flush=True)


def main():
    t0 = time.time()
    model_dir = os.environ.get("PLUMB_MODEL_DIR", "/model")
    manifest = json.loads(open(os.environ.get("REAX_MANIFEST", "/opt/reax/s1-fast-v1.json")).read())
    from s1eng import main as MAIN
    token = MAIN._token()
    temperature = MAIN._temperature(model_dir)
    front = None
    if os.environ.get("S1ENG_MPF", "") not in ("", "0"):
        try:
            from s1eng import mpf

            front = mpf.start(model_dir, manifest, token, temperature, log=log)
        except Exception as exc:  # noqa: BLE001
            log(f"lever not used: mpf ({type(exc).__name__}: {exc})")
    if os.environ.get("S1ENG_GVH", "0") == "1":   # the gvhb timeline tree (gvh.py with roles, instrumented mpf.py)
        from s1eng import gvh

        gvh.install("gpu" if front is not None else "single")
        if front is not None:
            from s1eng import mpf as _mpf

            _mpf.GV = gvh
        log("gvh timing harness installed (recording off until GET /gvh/mode?on=1)")

    import numpy as np
    import torch

    from s1eng import model as M
    from s1eng import ntv
    from s1eng.prompt import Encoder
    from s1eng.server import Engine, serve

    enc = Encoder(model_dir)

    class FM:
        pass
    fm = FM()
    fm.prefix_ids = enc.template_prefix()
    fm.prefix_variants = enc.prefix_variants(fm.prefix_ids)
    ntv.setup(fm, enc.encode_full, log=log)
    fm.ntv_buckets = (192, 256)
    fm.join_ok = {(128, 192): True}
    why = enc.enable_tail(fm.prefix_ids)
    log("tokenizer: fixed template ids, tails encoded with the GIL released" if not why
        else f"lever not used: tok tail ({why})")
    if front is not None:
        w = front.gate(enc, temperature, token, manifest)
        if w:
            log(f"lever not used: mpf ({w})")
            front.shutdown()
            front = None
    P = len(fm.prefix_ids)
    packb = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16]
    cap = {128: 4, 320: 4}
    lanes = []
    for li in range(2):
        packed = {}
        for L in (128, 192, 256, 320):
            for B in packb:
                if B > cap.get(L, 16):
                    continue
                top = (L - P) * B + 32
                packed[(L, B)] = [(R, None, None, None) for R in range(32, top + 32, 32)]
        single = {L: (None, None, None, None) for L in M.GRAPH_LENGTHS} if li == 0 else {}
        ln = M.Lane(li, None, single, {}, packed, pfx={L: P for L in (128, 192, 256, 320)})
        ln.mbs = {(192, 128): [4, 5, 6, 7, 8, 12]}
        ln.mixed = {(192, 128, B): [(R, None, None, None) for R in range(64, (192 - P) * B + 64, 64)]
                    for B in (4, 5, 6, 7, 8, 12)}
        lanes.append(ln)

    class FakeEvent:
        __slots__ = ("eta", "cuda_event")

        def __init__(self):
            self.eta = 0.0
            self.cuda_event = 1

        def synchronize(self):
            d = self.eta - time.perf_counter()
            if d > 0:
                time.sleep(d)

        def record(self, *a, **k):
            pass

    class FakeBSlot:
        def __init__(self, rows, bmax):
            self.out = torch.zeros((bmax, 16), dtype=torch.bfloat16)
            self.event = FakeEvent()
            self.ev0 = self.ev1 = None
            self.tag = None

    M._BSlot = FakeBSlot
    err = os.environ.get("HB_ERR", "0") == "1"
    lane_t = [0.0, 0.0]

    def det(ids):
        rng = np.random.default_rng(zlib.crc32(np.ascontiguousarray(ids, dtype=np.int64).tobytes()))
        return torch.from_numpy((rng.standard_normal(16) * 3.0).astype(np.float32)).to(torch.bfloat16)

    class FakeLoop(M.BatchLoop):
        def _launch(self, ln, L, jobs, slot, pk):
            B = len(jobs)
            if err and B == 1:
                n = jobs[0].n
                if n % 41 == 5:
                    raise ValueError(f"fake launch: length {n}")
                if n % 43 == 6:
                    raise RuntimeError(f"fake runtime error, length {n}")
            if pk is None:
                R = L
            elif pk[0] == "mx":
                R = pk[4][0]
            else:
                R = pk[1][0]
            for r, j in enumerate(jobs):
                slot.out[r] = det(j.ids)
            now = time.perf_counter()
            lane_t[ln.idx] = max(now, lane_t[ln.idx]) + 0.0015 + 4e-6 * R
            slot.event.eta = lane_t[ln.idx]
            self.stats[(L, B)] += 1

    cls = FakeLoop if front is None else front.loop_class(FakeLoop)
    gpu = cls(fm, lanes, depth=1)
    log(f"BatchLoop {type(gpu).__name__}: hold_n {gpu.hold_n} hold_ms {gpu.hold_s * 1000:.1f} arrive {gpu.arrive} "
        f"wake_filter {gpu.wake_filter} join {gpu.bb_join} lone_packed {gpu.lone_packed} P {gpu.P} variants "
        f"{len(gpu.variants)}")
    engine = Engine(enc, gpu, temperature, token, manifest["limits"]["max_input_tokens"], manifest["weights"]["revision"])
    enc.encode("warm-up")
    if front is not None:
        w = front.serve(gpu, os.environ.get("RUNTIME_UDS"), t0)
        if not w:
            gpu.run()
            return
        log(f"lever not used: mpf ({w})")
        front.shutdown()
    uds = os.environ.get("RUNTIME_UDS")

    def http_thread():
        asyncio.run(serve(engine, uds, "127.0.0.1", 8101,
                          ready_cb=lambda: log(f"serving on {uds} at {time.time() - t0:.1f}s")))

    threading.Thread(target=http_thread, name="http", daemon=True).start()
    gpu.run()


if __name__ == "__main__":
    main()
