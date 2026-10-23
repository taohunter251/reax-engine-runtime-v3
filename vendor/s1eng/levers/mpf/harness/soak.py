#!/usr/bin/env python3
"""gvhb: validator-style timed runs (reax_subnet UdsSidecarBackend + engrunner.timed_run, 256 items at c16 after
--gap s idle) on fixed seeded items; counts failed answers. Optional: kill one explicit pid (an MPF worker of the
test container, given by the job script) --kill-delay s after timed run --kill-block starts.
  soak.py --sock S --token-file T --out O [--blocks 20 --gap 1.0 --seed 7 --kill-pid P --kill-block 5]"""
import argparse
import asyncio
import json
import os
import signal
import statistics as st
import sys
import time

sys.path.insert(0, "/srv/reax/prof2/abl")
from driver2 import fixed_decisions  # noqa: E402

from reax_subnet.engrunner import UdsSidecarBackend, timed_run  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", required=True)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, default=20)
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--kill-pid", type=int, default=0)
    ap.add_argument("--kill-block", type=int, default=-1)
    ap.add_argument("--kill-delay", type=float, default=0.2)
    ap.add_argument("--ready-timeout", type=float, default=600)
    a = ap.parse_args()
    token = open(a.token_file).read().strip()
    cl = UdsSidecarBackend("s1-fast", a.sock, token, a.conc, 60.0)
    t0 = time.time()
    while True:
        if cl.socket_ok() and await cl.ready() and await cl.probe():
            break
        if time.time() - t0 > a.ready_timeout:
            raise SystemExit("engine not ready")
        await asyncio.sleep(0.5)
    print(f"ready after {time.time() - t0:.1f}s", flush=True)
    pool = fixed_decisions(256 * (a.blocks + 1), a.seed)
    await timed_run(cl, pool[:64])
    rows = []
    for b in range(a.blocks):
        await asyncio.sleep(a.gap)
        items = pool[256 * (b + 1):256 * (b + 2)]
        killer = None
        if b == a.kill_block and a.kill_pid:
            async def kill():
                await asyncio.sleep(a.kill_delay)
                os.kill(a.kill_pid, signal.SIGKILL)
                print(f"killed pid {a.kill_pid} during block {b}", flush=True)
            killer = asyncio.ensure_future(kill())
        tp, ans = await timed_run(cl, items)
        if killer is not None:
            await killer
        fails = sum(1 for v in ans.values() if v is None)
        rows.append({"block": b, "tp": tp, "fails": fails})
        print(f"block {b:3d} {tp:8.2f} items/s fails {fails}", flush=True)
    await cl.close()
    v = [r["tp"] for r in rows]
    nf = sum(r["fails"] for r in rows)
    print(f"mean {st.mean(v):.2f} median {st.median(v):.2f} sd {st.stdev(v) if len(v) > 1 else 0:.2f} "
          f"fails {nf}", flush=True)
    json.dump({"rows": rows, "args": vars(a)}, open(a.out, "w"), indent=1)
    sys.exit(3 if nf else 0)


if __name__ == "__main__":
    asyncio.run(main())
