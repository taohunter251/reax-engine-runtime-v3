"""Start-up GPU duty governor (E1). During start-up (pin selection, S6 timing, graph captures, start-up checks) the
engine would keep the GPU busy for ~100 s; on a shared GPU that starves whatever else runs there. tick() is called
right after an existing synchronize point of the start-up code: the time since the previous tick (GPU work plus the
host work around it, so an over-estimate) is counted as busy, and the thread sleeps busy * (1 - d) / d (at most
S1ENG_START_GAP_CAP_MS), so start-up uses at most a share d = S1ENG_START_DUTY of the time. Only host sleeps between
already-synchronised steps: no kernel, stream or captured graph changes, and the event-timed S6 measurements exclude
the sleeps. done() switches it off before the engine serves; the serving path never calls tick().
S1ENG_START_DUTY >= 1 turns it off (the S5-S7 behaviour).
SD (rgh): validators give every GPU class the same 900 s start timeout; on a GPU 2-3x slower than the RTX PRO 6000
(GeForce fp32-accumulate GEMMs) busy time and sleeps both grow. The governor stops sleeping once S1ENG_START_SOFT_S
(default 420) have passed since it was created (import time), and elapsed() lets the tuning steps keep their own
deadlines (S6 / SG skip work past theirs; skipped work keeps the exact pinned algos)."""
from __future__ import annotations

import os
import time


class Governor:
    def __init__(self, env=os.environ, clock=time.perf_counter, sleep=time.sleep):
        d = float(env.get("S1ENG_START_DUTY", "1"))
        self.k = 0.0 if d >= 1.0 else (1.0 - d) / max(d, 0.05)
        self.cap = float(env.get("S1ENG_START_GAP_CAP_MS", "50")) / 1000.0
        self.clock, self.sleep = clock, sleep
        self.t0 = clock()
        self.soft = float(env.get("S1ENG_START_SOFT_S", "420"))
        self.last = None
        self.busy = self.slept = 0.0
        self.n = 0
        self.on = self.k > 0.0

    def tick(self) -> None:
        if not self.on:
            return
        now = self.clock()
        if self.last is None:          # the first tick only starts the clock
            self.last = now
            return
        b = now - self.last
        s = min(b * self.k, self.cap) if now - self.t0 < self.soft else 0.0
        self.busy += b
        self.n += 1
        if s > 0:
            self.sleep(s)
            self.slept += s
        self.last = self.clock()       # the sleep never counts as busy

    def elapsed(self) -> float:
        return self.clock() - self.t0

    def done(self):
        on, self.on = self.on, False
        return on, self.busy, self.slept, self.n


GOV = Governor()
