"""skx6t microbench (ruj skx6t track): the tail-skip variants of our MX two-class GEMM vs skx6 itself.

  python -u skx6t_bench.py exact <out.json>                      exactness at every mixed-graph R bin
  python -u skx6t_bench.py time <out.json> [block_s] [rounds]    two-lane timing + NVML energy (600 W cap regime)
  python -u skx6t_bench.py solo <out.json>                       one-stream event timing per call (below the cap)

Kernels: skx6.cubin (gemm_skx6_<cfg>, the shipped MX kernel = the reference path), skx6t.cubin (gemm_skx6t_<cfg> =
segment tails skipped per m16 sub-tile; gemm_skx6m_<cfg> = one tile grid over [0, M), boundary tile with per-row class
folds). Shapes = the S1+KF two-class launches of 128 -> 192 mixed graphs (class 1 = bucket 192's pin, class 2 = 128's):
down 2560x9216 (2304 / 1344), out_proj / o_proj 2560x4096 (1024 / 512), z 4096x2560 (unsplit 2560 / 512), k|v
2048x2560 and k, v 1024x2560 (448 / 192); config per (N, K, R) as mx.kernel_for.

exact: per (shape, R): two input sets (randn with every 7th row x64; randn x 2^U(-12, 12) per row with every 5th row
x4096 and +-1e4 spikes), weights randn x 0.02 with every 11th column x16. Reference per row = the shipped kernel in
the row's own class: skx6 at M1 = R (all class 1) and at M1 = 0 (all class 2), rows taken by M1 (checked directly
against skx6 at the same M1 on a subset). Every variant at many M1 values (all for R <= 256; else every value near
tile boundaries and the end, 16 q + {0, 1, 8, 15}, random), all R rows compared in all bits; 128 sentinel rows after
R must stay untouched.
time: CUDA graphs of a mixed-graph-like launch sequence per lane (per sample 4 down, 4 out_proj / o_proj, 3 z, 1 k|v
at the sample's (R, M1) = S1's 31 : 31 : 23 : 8 per mixed graph; 32 samples drawn like S1's mixed graphs), two streams
replaying back to back; blocks per variant in seeded random order per round; time and NVML energy sync to sync, SM
clock / power / temp poller (20 ms).
"""
import collections
import ctypes as C
import json
import os
import random
import sys
import threading
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("SKX6T_SRC", "/w/src")
STAGES = {"128x128k64": 3, "128x128k32": 4, "64x256k32": 4, "128x64k64": 3, "64x128k64": 3}
CFGS = list(STAGES)
VARIANTS = ("old", "t", "m")
PREFIX = {"old": "gemm_skx6_", "t": "gemm_skx6t_", "m": "gemm_skx6m_"}
SHAPES = {   # label: (N, K, chunk1 = bucket 192's class, chunk2 = bucket 128's class)
    "down": (2560, 9216, 2304, 1344),
    "oproj": (2560, 4096, 1024, 512),
    "z": (4096, 2560, 2560, 512),
    "kv": (2048, 2560, 448, 192),
    "k": (1024, 2560, 448, 192),
}
R_BINS = sorted(set(range(64, 1793, 64)) | {608, 760, 912, 1064, 1216, 1824})


def cfg_dims(cfg):
    a, b = cfg.split("k")
    bm, bn = (int(x) for x in a.split("x"))
    return bm, bn, int(b), STAGES[cfg]


def kernel_cfg(N, K, R):
    """mx.kernel_for (S1+KF) without the env override"""
    if N == 4096:
        return "64x256k32" if R <= 704 else "128x128k64"
    if N == 2048:
        return "64x256k32"
    if N <= 1024:
        return "64x128k64"
    return "128x128k64"


class Lib:
    def __init__(self):
        self.cu = C.CDLL("libcuda.so.1")
        torch.zeros(1, device="cuda")
        self.fn = {}
        self._keep = []
        for path, variants in ((os.path.join(SRC, "skx6.cubin"), ("old",)),
                               (os.path.join(SRC, "skx6t.cubin"), ("t", "m"))):
            data = open(path, "rb").read()
            img = C.create_string_buffer(data, len(data))
            self._keep.append(img)
            mod = C.c_void_p()
            self._chk(self.cu.cuModuleLoadData(C.byref(mod), img), "load " + path)
            for v in variants:
                for cfg in CFGS:
                    f = C.c_void_p()
                    name = PREFIX[v] + cfg
                    if self.cu.cuModuleGetFunction(C.byref(f), mod, name.encode()) != 0:
                        raise RuntimeError("missing " + name)
                    bm, bn, bk, st = cfg_dims(cfg)
                    smem = st * (bm + bn) * bk * 2
                    self._chk(self.cu.cuFuncSetAttribute(f, 8, smem), "smem")
                    self.fn[(v, cfg)] = (f, smem)

    @staticmethod
    def _chk(r, what):
        if r != 0:
            raise RuntimeError(f"{what}: CUresult {r}")

    def run(self, v, cfg, W, x, out, c1, c2, m1, M=None):
        if v == "t128":   # the tail skip only on the 128-row tile configs
            v = "t" if cfg.startswith("128x") else "old"
        elif v == "m64":   # t on 128-row configs, m on 64-row configs
            v = "t" if cfg.startswith("128x") else "m"
        f, smem = self.fn[(v, cfg)]
        bm, bn, bk, st = cfg_dims(cfg)
        M = x.shape[0] if M is None else M
        K = x.shape[1]
        N = W.shape[0]
        args = [C.c_void_p(x.data_ptr()), C.c_void_p(W.data_ptr()), C.c_void_p(out.data_ptr()), C.c_int(M), C.c_int(N),
                C.c_int(K), C.c_int(x.stride(0)), C.c_int(out.stride(0)), C.c_int(int(c1)), C.c_int(int(c2)),
                C.c_void_p(m1.data_ptr()), C.c_int(0)]
        params = (C.c_void_p * len(args))(*[C.cast(C.pointer(a), C.c_void_p) for a in args])
        gy = -(-M // bm) + (0 if v == "m" else 1)
        self._chk(self.cu.cuLaunchKernel(f, N // bn, gy, 1, 256, 1, 1, smem,
                                         C.c_void_p(torch.cuda.current_stream().cuda_stream), params, None), "launch")


def inputs(kind, R, K, g):
    if kind == 0:
        x = torch.randn((R, K), generator=g, device="cuda")
        x[::7] *= 64
    else:
        e = torch.randint(-12, 13, (R, 1), generator=g, device="cuda").float()
        x = torch.randn((R, K), generator=g, device="cuda") * torch.exp2(e)
        x[::5] *= 4096
        sp = torch.rand((R, K), generator=g, device="cuda") < 1e-3
        x = torch.where(sp, torch.sign(torch.randn((R, K), generator=g, device="cuda")) * 1e4, x)
    return x.to(torch.bfloat16).contiguous()


def weights(N, K, g):
    w = torch.randn((N, K), generator=g, device="cuda") * 0.02
    w[:, ::11] *= 16
    return w.to(torch.bfloat16).contiguous()


def m1_set(R, rng, bm):
    if R <= 256:
        return list(range(0, R + 1))
    s = {0, 1, R - 1, R}
    for q in range(0, R // 16 + 1):
        for d in (0, 1, 8, 15):
            if 0 <= 16 * q + d <= R:
                s.add(16 * q + d)
    for b in range(0, R + 1, bm):   # every value around each tile boundary and in the last 2 tiles
        for d in range(-3, 4):
            if 0 <= b + d <= R:
                s.add(b + d)
    for m in range(max(0, R - 2 * bm), R + 1):
        s.add(m)
    for _ in range(64):
        s.add(rng.randint(0, R))
    return sorted(s)


SENT = -21555   # int16 sentinel (0xABCD as int16): rows R.. must keep it


def exact(out_path):
    L = Lib()
    g = torch.Generator(device="cuda")
    rng = random.Random(4242)
    res = {"cases": [], "fail": 0, "compares": 0}
    t0 = time.time()
    for label, (N, K, c1, c2) in SHAPES.items():
        g.manual_seed(1000 + N + K)
        W = weights(N, K, g)
        for R in R_BINS:
            cfgs = [kernel_cfg(N, K, R)]
            if label in ("down", "z") and R in (448, 640, 704, 768, 832, 1024, 1216):
                cfgs = CFGS   # every config at a few R
            for cfg in cfgs:
                bm = cfg_dims(cfg)[0]
                case = {"shape": label, "N": N, "K": K, "R": R, "cfg": cfg, "m1": 0, "bad": {}, "direct_bad": 0,
                        "oob": 0}
                for kind in (0, 1):
                    x = inputs(kind, R, K, g)
                    outs = {}
                    for nm, m1v in (("all1", R), ("all2", 0)):
                        o = torch.full((R + 128, N), 0, dtype=torch.bfloat16, device="cuda")
                        o.view(torch.int16).fill_(SENT)
                        L.run("old", cfg, W, x, o, c1, c2, torch.tensor([m1v], dtype=torch.int32, device="cuda"))
                        outs[nm] = o
                    ref1, ref2 = outs["all1"][:R].view(torch.int16), outs["all2"][:R].view(torch.int16)
                    ms = m1_set(R, rng, bm)
                    case["m1"] += len(ms)
                    bad = torch.zeros((), dtype=torch.long, device="cuda")
                    dbad = torch.zeros((), dtype=torch.long, device="cuda")
                    oob = torch.zeros((), dtype=torch.long, device="cuda")
                    nb = collections.Counter()
                    o = torch.empty((R + 128, N), dtype=torch.bfloat16, device="cuda")
                    m1all = torch.tensor(ms, dtype=torch.int32, device="cuda")
                    for i, m1v in enumerate(ms):
                        m1 = m1all[i:i + 1]
                        ref = torch.cat([ref1[:m1v], ref2[m1v:]])
                        vs = ["t", "m"] + (["old"] if i % 9 == 0 else [])
                        for v in vs:
                            o.view(torch.int16).fill_(SENT)
                            L.run(v, cfg, W, x, o, c1, c2, m1)
                            ov = o.view(torch.int16)
                            d = (ov[:R] != ref).sum()
                            if v == "old":
                                dbad += d
                            else:
                                bad += d
                                if i % 16 == 0:
                                    torch.cuda.synchronize()
                                    if int(d):
                                        nb[f"{v}@{m1v}"] += int(d)
                            oob += (ov[R:] != SENT).sum()
                            res["compares"] += 1
                    torch.cuda.synchronize()
                    if int(bad):   # find which (variant, M1) differ
                        for m1v in ms:
                            m1 = torch.tensor([m1v], dtype=torch.int32, device="cuda")
                            ref = torch.cat([ref1[:m1v], ref2[m1v:]])
                            for v in ("t", "m"):
                                L.run(v, cfg, W, x, o, c1, c2, m1)
                                d = int((o.view(torch.int16)[:R] != ref).sum())
                                if d:
                                    nb[f"{v}@{m1v}"] = d
                    case["bad"].update({k: v for k, v in nb.items() if v})
                    case["direct_bad"] += int(dbad)
                    case["oob"] += int(oob)
                    res["fail"] += int(bad) + int(dbad) + int(oob)
                res["cases"].append(case)
                print(json.dumps(case)[:400], flush=True)
    res["seconds"] = time.time() - t0
    res["verdict"] = "EXACT" if res["fail"] == 0 else "DIFFER"
    json.dump(res, open(out_path, "w"), indent=1)
    print(f"exact: {res['verdict']} ({res['compares']} compares, {len(res['cases'])} cases, fail {res['fail']}) "
          f"in {res['seconds']:.0f}s", flush=True)


# ---------------------------------------------------------------------------------------------------------- timing
nv = None


def nvml():
    global nv, H
    if nv is None:
        nv = C.CDLL("libnvidia-ml.so.1")
        nv.nvmlInit_v2()
        H = C.c_void_p()
        nv.nvmlDeviceGetHandleByIndex_v2(0, C.byref(H))
    return nv


def energy_mj():
    e = C.c_ulonglong()
    nvml().nvmlDeviceGetTotalEnergyConsumption(H, C.byref(e))
    return e.value


def sample():
    c, p, t = C.c_uint(), C.c_uint(), C.c_uint()
    nvml().nvmlDeviceGetClockInfo(H, 1, C.byref(c))
    nv.nvmlDeviceGetPowerUsage(H, C.byref(p))
    nv.nvmlDeviceGetTemperature(H, 0, C.byref(t))
    return c.value, p.value / 1000.0, t.value


class Poller(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.buf, self.stop = [], False

    def run(self):
        while not self.stop:
            self.buf.append((time.perf_counter(),) + sample())
            time.sleep(0.02)


def mixed_samples(n, seed=777):
    """(R, M1) like S1's 128 -> 192 mixed graphs (s1prof cells: B4/5/6/7/8/12 = 5/21/22/28/45/1 graphs; riders 1 / 2 /
    3 with 70 / 25 / 5 %; bucket-192 decision rows ~N(120, 15) in [93, 152], bucket-128 ~N(81, 8) in [55, 88];
    R = the 64-row bin above the rows, capped at B x 152)"""
    rr = random.Random(seed)
    out = []
    while len(out) < n:
        B = rr.choices([4, 5, 6, 7, 8, 12], weights=[5, 21, 22, 28, 45, 1])[0]
        ns = rr.choices([1, 2, 3], weights=[70, 25, 5])[0]
        ns = min(ns, B - 1)
        r1 = sum(int(min(152, max(93, rr.gauss(120, 15)))) for _ in range(B - ns))
        r2 = sum(int(min(88, max(55, rr.gauss(81, 8)))) for _ in range(ns))
        T = r1 + r2
        R = min(-(-T // 64) * 64, B * 152)
        if R < T:
            R = T
        out.append((R, r1))
    return out


LAUNCHES = (("down", 4), ("oproj", 4), ("z", 3), ("kv", 1))   # per sample ~ S1's mixed graph mix 31 : 31 : 23 : 8


def build_graphs(L, variants, samples, lanes=2):
    g = torch.Generator(device="cuda").manual_seed(99)
    NW = 4   # weight copies per shape, rotated per sample (~400 MB > L2: weights stream from DRAM as in the engine)
    Ws = {(lab, w): weights(N, K, g) for lab, (N, K, c1, c2) in SHAPES.items() if lab != "k" for w in range(NW)}
    Rmax = max(R for R, _ in samples)
    xs, outs = {}, {}
    for lane in range(lanes):
        for lab, (N, K, c1, c2) in SHAPES.items():
            if lab == "k":
                continue
            xs[(lane, lab)] = inputs(0, Rmax, K, g)
            outs[(lane, lab)] = torch.empty((Rmax, N), dtype=torch.bfloat16, device="cuda")
    m1s = torch.tensor([m for _, m in samples], dtype=torch.int32, device="cuda")
    graphs = {}
    calls = 0
    for v in variants:
        for lane in range(lanes):
            st = torch.cuda.Stream()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.stream(st):
                for si, (R, M1) in enumerate(samples[:2]):   # warm once outside the capture
                    for lab, n in LAUNCHES:
                        N, K, c1, c2 = SHAPES[lab]
                        L.run(v, kernel_cfg(N, K, R), Ws[(lab, si % NW)], xs[(lane, lab)][:R], outs[(lane, lab)], c1,
                              c2, m1s[si:si + 1])
                torch.cuda.synchronize()
                with torch.cuda.graph(gr, stream=st):
                    n_calls = 0
                    for si, (R, M1) in enumerate(samples):
                        for lab, n in LAUNCHES:
                            N, K, c1, c2 = SHAPES[lab]
                            for rep in range(n):
                                L.run(v, kernel_cfg(N, K, R), Ws[(lab, (si + rep + lane) % NW)], xs[(lane, lab)][:R],
                                      outs[(lane, lab)], c1, c2, m1s[si:si + 1])
                                n_calls += 1
            graphs[(v, lane)] = gr
            calls = n_calls
    torch.cuda.synchronize()
    return graphs, calls, (Ws, xs, outs, m1s)


def timing(out_path, block_s=4.0, rounds=8, burst=False):
    L = Lib()
    samples = mixed_samples(32)
    variants = [v for v in os.environ.get("SKX6T_VARIANTS", "old,t,m").split(",") if v]
    graphs, calls, keep = build_graphs(L, variants, samples)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    nvml()   # init before the poller thread starts (H is set here)
    pol = Poller()
    pol.start()
    rows = sum(R for R, _ in samples)
    res = {"samples": samples, "calls_per_replay": calls, "block_s": block_s, "rounds": rounds, "blocks": []}
    print(f"samples {samples}\nmean R {rows / len(samples):.0f}, calls per replay {calls}", flush=True)

    def block(v, secs):
        """both lanes replay back to back (<= 3 pairs queued) for secs; time and NVML energy from sync to sync"""
        torch.cuda.synchronize()
        evq = collections.deque()
        i0 = len(pol.buf)
        e0, t0 = energy_mj(), time.perf_counter()
        n = 0
        while time.perf_counter() - t0 < secs:
            evs = []
            for lane, st in enumerate(streams):
                with torch.cuda.stream(st):
                    graphs[(v, lane)].replay()
                ev = torch.cuda.Event()
                ev.record(st)
                evs.append(ev)
            evq.append(evs)
            n += 1
            while len(evq) > 3:
                for ev in evq.popleft():
                    ev.synchronize()
        torch.cuda.synchronize()
        t1, e1 = time.perf_counter(), energy_mj()
        smp = pol.buf[i0:]
        clk = sum(s_[1] for s_ in smp) / max(1, len(smp))
        pw = sum(s_[2] for s_ in smp) / max(1, len(smp))
        tmp = sum(s_[3] for s_ in smp) / max(1, len(smp))
        return {"variant": v, "replay_pairs": n, "wall_s": t1 - t0, "ms_per_replay_pair": (t1 - t0) / n * 1e3,
                "mj_per_replay_pair": (e1 - e0) / n, "clk": clk, "power": pw, "temp": tmp,
                "power_from_energy": (e1 - e0) / 1e3 / (t1 - t0)}

    def burst_block(v, idle, pairs):
        """the official regime: idle gap, then `pairs` replay pairs of both lanes back to back (~0.7 s: power ramps
        from idle, the 600 W cap from ~150 ms on); wall time from the first issue to the last completion"""
        torch.cuda.synchronize()
        time.sleep(idle)
        i0 = len(pol.buf)
        e0, t0 = energy_mj(), time.perf_counter()
        for _ in range(pairs):
            for lane, st in enumerate(streams):
                with torch.cuda.stream(st):
                    graphs[(v, lane)].replay()
        torch.cuda.synchronize()
        t1, e1 = time.perf_counter(), energy_mj()
        smp = pol.buf[i0:]
        clk = sum(s_[1] for s_ in smp) / max(1, len(smp))
        pw = max([s_[2] for s_ in smp] or [0])
        return {"variant": v, "replay_pairs": pairs, "wall_s": t1 - t0, "ms_per_replay_pair": (t1 - t0) / pairs * 1e3,
                "mj_per_replay_pair": (e1 - e0) / pairs, "clk": clk, "power": pw, "temp": smp[-1][3] if smp else 0,
                "power_from_energy": (e1 - e0) / 1e3 / max(t1 - t0, 1e-9)}

    print("warm-up", flush=True)
    block(variants[0], 20.0)
    rr = random.Random(3131)
    for r in range(rounds):
        order = list(variants)
        rr.shuffle(order)
        for v in order:
            b = burst_block(v, 2.0, int(os.environ.get("SKX6T_PAIRS", "13"))) if burst else block(v, block_s)
            b["round"] = r
            res["blocks"].append(b)
            print(json.dumps(b), flush=True)
    pol.stop = True
    summ = {}
    v0 = "old" if "old" in variants else variants[0]
    for v in variants:
        rt, re_ = [], []
        for r in range(rounds):
            b0 = [b for b in res["blocks"] if b["round"] == r and b["variant"] == v0][0]
            bv = [b for b in res["blocks"] if b["round"] == r and b["variant"] == v][0]
            rt.append(bv["ms_per_replay_pair"] / b0["ms_per_replay_pair"])
            re_.append(bv["mj_per_replay_pair"] / b0["mj_per_replay_pair"])
        n = len(rt)
        mt = sum(rt) / n
        se = (sum((x - mt) ** 2 for x in rt) / max(1, n - 1) / n) ** 0.5
        me = sum(re_) / n
        see = (sum((x - me) ** 2 for x in re_) / max(1, n - 1) / n) ** 0.5
        summ[v] = {"time_ratio": mt, "time_se": se, "energy_ratio": me, "energy_se": see, "time_ratios": rt,
                   "energy_ratios": re_}
    res["summary"] = summ
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)


def solo(out_path):
    """one stream, CUDA events: per call time of each shape at a few (R, M1) (no cap: short bursts)"""
    L = Lib()
    g = torch.Generator(device="cuda").manual_seed(5)
    res = []
    cases = [(576, 495), (704, 623), (768, 655), (832, 719), (896, 783), (960, 840), (1024, 943)]
    for lab, (N, K, c1, c2) in SHAPES.items():
        W = weights(N, K, g)
        for R, M1 in cases:
            x = inputs(0, R, K, g)
            o = torch.empty((R, N), dtype=torch.bfloat16, device="cuda")
            m1 = torch.tensor([M1], dtype=torch.int32, device="cuda")
            cfg = kernel_cfg(N, K, R)
            row = {"shape": lab, "R": R, "M1": M1, "cfg": cfg}
            ts = {v: [] for v in VARIANTS}
            for order in (("old", "t", "m"), ("m", "t", "old"), ("t", "old", "m"), ("old", "m", "t")):
                for v in order:
                    for _ in range(20):
                        L.run(v, cfg, W, x, o, c1, c2, m1)
                    for rep in range(3):
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        e0.record()
                        for _ in range(50):
                            L.run(v, cfg, W, x, o, c1, c2, m1)
                        e1.record()
                        e1.synchronize()
                        ts[v].append(e0.elapsed_time(e1) / 50 * 1e3)
            for v in VARIANTS:
                row[v] = sorted(ts[v])[len(ts[v]) // 2]
            res.append(row)
            print(json.dumps(row), flush=True)
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "exact":
        exact(sys.argv[2])
    elif mode == "time":
        timing(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 4.0, int(sys.argv[4]) if len(sys.argv) > 4 else 8)
    elif mode == "burst":
        timing(sys.argv[2], 0.0, int(sys.argv[3]) if len(sys.argv) > 3 else 12, burst=True)
    elif mode == "solo":
        solo(sys.argv[2])
    else:
        raise SystemExit(__doc__)
