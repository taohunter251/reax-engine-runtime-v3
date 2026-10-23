"""Model loading, CUDA-graph capture per padded length, and the pipelined replay loop.

Numerics: the forward that is captured is the reference's (transformers Qwen3_5 text model, bf16, use_cache=False,
batch 1, right-padded with id 0 to the next graph length, hidden state of row len-1 times the 16 letter rows of the
tied embedding). The capture procedure is the reference's too (3 warm-up runs on a side stream, then capture, one
length after the other in ascending order, on the start-up thread). Only what happens around the replays differs:
inputs are staged in pinned host buffers and copied asynchronously, outputs come back asynchronously, and up to
DEPTH replays are queued on the stream so the GPU never waits for Python.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import threading
import time

import numpy as np
import torch

from . import kprio
from .gov import GOV


def _ntv() -> bool:
    """NTV (kai, S1ENG_NTV="1"): nonce-token prefix variants, every variant block bit-checked at start-up
    (s1eng/ntv.py). Off: hotspot (tests/test_flag_off.py)."""
    return os.environ.get("S1ENG_NTV", "") == "1"


GRAPH_LENGTHS = (128, 192, 256, 320, 384, 512, 640, 768, 1024, 1536, 2048, 3072, 4096)

# fusable projection pairs (module paths inside a decoder layer): one GEMM over [W1; W2] gives both outputs side by side
FUSE_GROUPS = {"gate_up": ("mlp.gate_proj", "mlp.up_proj"), "k_v": ("self_attn.k_proj", "self_attn.v_proj"),
               "b_a": ("linear_attn.in_proj_b", "linear_attn.in_proj_a"),
               "qkv_z": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z")}


def relayout(hf, groups, align: int = 512) -> dict:
    """Moves every parameter and buffer of the model into one device arena in which the two weights of each fusable
    pair are adjacent: W1 and W2 become row slices (views) of one [N1 + N2, K] block, bit-identical copies of the loaded
    tensors, so the per-decision path keeps computing with exactly the same values (every offset is a multiple of 512
    bytes: the same pointer alignment class as a fresh allocation for cuBLAS). Device memory does not grow: everything
    is staged in host memory, the device copies are freed (all segments released), then the arena is filled.
    Returns {(layer index, group): fused block}."""
    tensors, seen = [], set()
    for t in list(hf.parameters()) + list(hf.buffers()):
        if t.device.type == "cuda" and id(t) not in seen:
            seen.add(id(t))
            tensors.append(t)
    pairs = []
    for i, layer in enumerate(hf.model.layers):
        for gname in groups:
            p1, p2 = FUSE_GROUPS[gname]
            try:
                w1, w2 = layer.get_submodule(p1).weight, layer.get_submodule(p2).weight
            except AttributeError:
                continue
            if w1.dim() != 2 or w2.dim() != 2 or w1.shape[1] != w2.shape[1] or w1.dtype != w2.dtype:
                continue
            if (w1.numel() * w1.element_size()) % align:
                continue
            pairs.append((i, gname, w1, w2))
    in_pair = {id(w) for _, _, w1, w2 in pairs for w in (w1, w2)}
    order = [w for _, _, w1, w2 in pairs for w in (w1, w2)] + [t for t in tensors if id(t) not in in_pair]
    offs, off = {}, 0
    for t in order:
        offs[id(t)] = off
        off += -(-(t.numel() * t.element_size()) // align) * align
    torch.cuda.synchronize()
    host = {id(t): t.detach().to("cpu", copy=True) for t in order}
    for t in order:
        t.data = torch.empty(0, dtype=t.dtype, device=t.device)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    arena = torch.empty(off, dtype=torch.uint8, device="cuda")
    for t in order:
        h = host.pop(id(t))
        o = offs[id(t)]
        v = arena[o:o + h.numel() * h.element_size()].view(h.dtype).view(h.shape)
        v.copy_(h)
        t.data = v
    torch.cuda.synchronize()
    fused = {}
    for i, gname, w1, w2 in pairs:
        n1, k = w1.shape
        o = offs[id(w1)]
        assert offs[id(w2)] == o + w1.numel() * w1.element_size()
        blk = arena[o:o + (w1.numel() + w2.numel()) * w1.element_size()].view(w1.dtype).view(n1 + w2.shape[0], k)
        assert blk[:n1].data_ptr() == w1.data_ptr() and blk[n1:].data_ptr() == w2.data_ptr()
        fused[(i, gname)] = blk
    return fused


def bucket_of(n: int) -> int | None:
    for g in GRAPH_LENGTHS:
        if g >= n:
            return g
    return None


class Model:
    def __init__(self, model_dir: str, slots: list[int], impl: str = "ref"):
        import transformers

        config = transformers.AutoConfig.from_pretrained(model_dir)
        cls = transformers.AutoModelForCausalLM
        if config.model_type in {"qwen3_5", "qwen3_5_text"}:
            cls, config = transformers.Qwen3_5ForCausalLM, config.get_text_config()
        self.hf = cls.from_pretrained(model_dir, config=config, dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
        groups = [g for g in os.environ.get("S1ENG_FUSE", "").split(",") if g] if impl != "ref" else []
        self.fused_w = relayout(self.hf, groups) if groups else {}
        self.slot_weight = self.hf.lm_head.weight[slots].detach().contiguous()
        self.impl = impl
        self.fwd = None
        if impl != "ref":
            from .fastfwd import FastForward

            self.fwd = FastForward(self.hf, self.slot_weight, impl, fused_w=self.fused_w)
        self.graphs: dict[int, tuple] = {}

    def slot_logits(self, ids: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        if self.fwd is not None:
            return self.fwd(ids, last)
        hidden = self.hf.model(input_ids=ids, use_cache=False).last_hidden_state
        return hidden[torch.arange(ids.shape[0], device=ids.device), last] @ self.slot_weight.T

    def warm_stream(self):
        """The one side stream every warm-up runs on. torch keeps a cuBLAS workspace (32 MiB on this build) for every
        stream that ran a cuBLAS GEMM, for the life of the process: a fresh stream per warm-up cost ~1 GiB in round 3.
        Which stream a warm-up runs on does not change what is captured afterwards."""
        if getattr(self, "_warm", None) is None:
            self._warm = torch.cuda.Stream()
        return self._warm

    @torch.inference_mode()
    def capture(self, lengths=GRAPH_LENGTHS, extra_sets: int = 0, extra_lengths=(128, 192, 256, 320)) -> None:
        """Set 0: every length, captured exactly as the reference does (3 warm-ups on a side stream, then capture on
        torch.cuda.graph's default capture stream). Extra sets (for concurrent replay on other streams): the same
        procedure on their own capture stream, so cuBLAS workspaces and static buffers are not shared between sets."""
        self.set_pools = []
        self.graph_sets = [self._capture_set(lengths, None)]
        for _ in range(extra_sets):
            self.graph_sets.append(self._capture_set([n for n in extra_lengths if n in lengths],
                                                     fresh_stream(capture_streams_in_use(self))))
        self.graphs = self.graph_sets[0]
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _capture_set(self, lengths, cap_stream) -> dict:
        graphs = {}
        # one memory pool per set (graphs of a set replay one after the other on one stream), unless S1ENG_POOL=0
        pool = torch.cuda.graph_pool_handle() if os.environ.get("S1ENG_POOL", "1") == "1" else None
        self.set_pools.append(pool)
        self.set_cap_streams = getattr(self, "set_cap_streams", []) + [cap_stream]
        for n in lengths:
            ids = torch.zeros((1, n), dtype=torch.long, device="cuda")
            last = torch.full((1,), n - 1, dtype=torch.long, device="cuda")
            stream = self.warm_stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.slot_logits(ids, last)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=cap_stream):
                out = self.slot_logits(ids, last)
            GOV.tick()
            graphs[n] = (graph, ids, last, out)
        return graphs

    @torch.inference_mode()
    def logits_direct(self, ids: list[int]) -> np.ndarray:
        """One synchronous replay (tests): the 16 letter logits as float32."""
        graph, s_ids, s_last, out = self.graphs[bucket_of(len(ids))]
        s_ids.zero_()
        s_ids[0, : len(ids)] = torch.tensor(ids, device="cuda")
        s_last.fill_(len(ids) - 1)
        graph.replay()
        return out[0].float().cpu().numpy()


def _default_capture_stream():
    """The stream torch.cuda.graph captures on when no stream is given (created on first use)."""
    if torch.cuda.graph.default_capture_stream is None:
        torch.cuda.graph.default_capture_stream = torch.cuda.Stream()
    return torch.cuda.graph.default_capture_stream


def capture_streams_in_use(model) -> list:
    out = [_default_capture_stream()]
    for st in getattr(model, "set_cap_streams", []):
        if st is not None:
            out.append(st)
    out.append(model.warm_stream())
    return out


def fresh_stream(avoid: list, tries: int = 128):
    """A pool stream whose CUDA handle differs from every stream in `avoid`. torch hands out pool streams round-robin
    (32 per priority), and keeps its cuBLAS workspace per stream: two graph sets captured on one stream would share
    that workspace while they replay concurrently."""
    used = {int(st.cuda_stream) for st in avoid if st is not None}
    for _ in range(tries):
        st = torch.cuda.Stream()
        if int(st.cuda_stream) not in used:
            return st
    raise RuntimeError("no free CUDA pool stream for a new capture / replay stream")


class Job:
    __slots__ = ("ids", "n", "done", "result", "t_enq", "stream", "t_launch", "pok", "pv", "pl", "option_count", "exact", "checkpoint", "lease")

    def __init__(self, ids, done, stream=None):
        self.ids = ids
        self.option_count = 16
        self.exact = False
        self.checkpoint = None
        self.lease = None
        self.stream = stream  # tests: force a graph set / stream
        self.n = len(ids)
        self.done = done  # callable(np.ndarray float32[16]) -> None, called on the GPU thread
        self.result = None
        self.t_enq = time.perf_counter()
        self.t_launch = None
        self.pok = False      # E1: starts with the template prefix (set by BatchLoop.submit)
        self.pv = 0           # PV (a4): the longest cached prefix variant it starts with, and that variant's length
        self.pl = 0


class _Slot:
    def __init__(self, max_len: int):
        self.ids = torch.zeros((1, max_len), dtype=torch.long, pin_memory=True)
        self.ids_np = self.ids.numpy()
        self.last = torch.zeros((1,), dtype=torch.long, pin_memory=True)
        self.out = torch.zeros((16,), dtype=torch.bfloat16, pin_memory=True)
        self.event = torch.cuda.Event()


class GpuLoop:
    """Runs on the thread that captured the graphs. Jobs come from any thread via submit().

    Each graph set has its own CUDA stream; a job is launched on the set with the fewest jobs in flight (set 0 when
    idle, so a lone request takes the same path as always). Results complete in launch order."""

    def __init__(self, model: Model, depth: int = 2):
        self.m = model
        self.sets = model.graph_sets
        self.depth = depth
        self.q: collections.deque = collections.deque()
        self.cv = threading.Condition()
        self.streams = [torch.cuda.current_stream()] + [torch.cuda.Stream() for _ in self.sets[1:]]
        nslots = depth * len(self.sets) + 1
        self.slots = [_Slot(max(GRAPH_LENGTHS)) for _ in range(nslots)]
        self.free = list(self.slots)
        self.busy = [0] * len(self.sets)
        self.stop = False

    def submit(self, job: Job) -> None:
        with self.cv:
            self.q.append(job)
            self.cv.notify()

    def _pick(self, job: Job) -> int | None:
        b = bucket_of(job.n)
        if job.stream is not None:
            return job.stream if self.busy[job.stream] < self.depth else None
        best = None
        for s, gs in enumerate(self.sets):
            if b in gs and self.busy[s] < self.depth and (best is None or self.busy[s] < self.busy[best]):
                best = s
        return best

    def _launch(self, job: Job, slot: _Slot, s: int) -> None:
        n = job.n
        b = bucket_of(n)
        graph, s_ids, s_last, out = self.sets[s][b]
        row = slot.ids_np[0]
        row[:n] = job.ids
        row[n:b] = 0
        slot.last[0] = n - 1
        with torch.cuda.stream(self.streams[s]):
            s_ids.copy_(slot.ids[:, :b], non_blocking=True)
            s_last.copy_(slot.last, non_blocking=True)
            graph.replay()
            slot.out.copy_(out[0], non_blocking=True)
            slot.event.record()

    @torch.inference_mode()
    def run(self) -> None:
        inflight: collections.deque = collections.deque()
        while not self.stop:
            while True:
                with self.cv:
                    if not self.q:
                        if inflight:
                            break
                        self.cv.wait(timeout=0.5)
                        if not self.q:
                            continue
                    job = self.q[0]
                    s = self._pick(job)
                    if s is None:
                        break
                    self.q.popleft()
                slot = self.free.pop()
                try:
                    self._launch(job, slot, s)
                except Exception as exc:  # noqa: BLE001
                    self.free.append(slot)
                    job.done(exc)
                    continue
                self.busy[s] += 1
                inflight.append((job, slot, s))
            if not inflight:
                continue
            job, slot, s = inflight.popleft()
            slot.event.synchronize()
            logits = slot.out.float().numpy().copy()
            self.free.append(slot)
            self.busy[s] -= 1
            job.done(logits)


# ====================================================================================================== S4/S5: batching
BATCH_BUCKETS = (128, 192, 256, 320)


def parse_bmax(spec: str) -> dict:
    """"128:8,192:16,256:16,320:8" -> {128: 8, ...}"""
    out = {}
    for part in spec.split(","):
        if part:
            k, v = part.split(":")
            out[int(k)] = int(v)
    return out


def parse_aff(spec, nlanes: int) -> dict:
    """BB: S1ENG_AFF "192:0,256:1,128:1/0" -> {192: [0], 256: [1], 128: [1, 0]} (lanes >= nlanes dropped)."""
    out = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split(":")
        ls = [int(x) for x in v.split("/") if x != "" and int(x) < nlanes]
        if ls:
            out[int(k)] = list(dict.fromkeys(ls))
    return out


def parse_lanes(spec, nlanes: int):
    """BB: S1ENG_LANES_ON "0" -> {0}; "" -> None (every lane). Lane 0 is always included (per-decision graphs)."""
    if spec is None or str(spec).strip() == "":
        return None
    return {0} | {int(x) for x in str(spec).split(",") if x.strip() != "" and int(x) < nlanes}


def parse_join(spec) -> list:
    """BB4 / MX: S1ENG_JOIN "128:192,192:256" -> [(128, 192), (192, 256)] in priority order: decisions of bucket S
    may ride in bucket D's mixed graphs (MX, ruj r2: every row in its own bucket's bit class; only pairs of batch
    buckets with S < D; each pair is used only after its start-up check passed). The scheduler merges at most one S
    group into a D group and never a group that already took part in a merge (no chains: two classes per graph)."""
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part or ":" not in part:   # "" / "0": off (the base behaviour)
            continue
        s, d = (int(x) for x in part.split("@")[0].split(":"))
        if s in BATCH_BUCKETS and d in BATCH_BUCKETS and s < d and (s, d) not in out:
            out.append((s, d))
    return out


def parse_join_caps(spec) -> dict:
    """MX: the "@k" caps of S1ENG_JOIN items -> {(S, D): k} (the S group merges only while it has <= k jobs)"""
    out = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if "@" in part and ":" in part:
            pair, k = part.split("@")
            s, d = (int(x) for x in pair.split(":"))
            out[(s, d)] = int(k)
    return out


def mix_slots(D: int, S: int, pack_d: list) -> list:
    """MX: the B slots of (D, S) mixed graphs: S1ENG_MIX_B = "4,5,6,7,8,12" (every pair; the default) or per pair
    "S:D=4/5/6,S:D=6/8" (pairs not listed: the default); only D's packed slots >= 2"""
    spec = os.environ.get("S1ENG_MIX_B", "")
    want = [4, 5, 6, 7, 8, 12]
    if spec and "=" not in spec:
        want = [int(b) for b in spec.split(",") if b]
    elif spec:
        for part in spec.split(","):
            k, v = part.split("=")
            s_, d_ = (int(x) for x in k.split(":"))
            if (s_, d_) == (S, D):
                want = [int(b) for b in v.split("/") if b]
    return [b for b in sorted(set(want)) if b in pack_d and b >= 2]


def mix_rows(D: int, S: int, prow_d: dict, pfx: int, slots: list) -> dict:
    """MX: B -> R bins of the (D, S) mixed graphs: D's own packed R values (S6 / SG / RARN picks exist there) on a 64-row
    grid (S1ENG_MIX_RSTEP) from (Bprev + 1) x S1ENG_MIX_LO rows (default 64: shorter batches take the smallest bin) up
    to B full-length D decisions (always included)"""
    step = int(os.environ.get("S1ENG_MIX_RSTEP", "64"))
    lo_rows = int(os.environ.get("S1ENG_MIX_LO", "64"))
    rset = sorted({r for rs in prow_d.values() for r in rs})
    hi = D - pfx
    out, prev = {}, 0
    for B in slots:
        lo, top = max(step, (prev + 1) * lo_rows), B * hi
        rs = [r for r in rset if lo <= r < top and r % step == 0] + [r for r in rset if r == top]
        if top not in rs:
            rs.append(min([r for r in rset if r >= top], default=top))
        out[B] = sorted(set(rs))
        prev = B
    return out


def switch_us_of(v) -> int:
    """switch_us (env S1ENG_SWITCH_US / ARMS key): GIL switch interval in whole microseconds > 0, else ValueError."""
    us = int(v)
    if us <= 0:
        raise ValueError(f"switch_us must be > 0 (got {v!r})")
    return us


def ftok_of(v) -> int:
    """FTOK (kai, ftok.py) arm key ftok / env S1ENG_FTOK: int(v) in (0, 1), else ValueError. The same rule as
    ftok.ftok_of, here so that model.py never imports ftok (it is imported by main.py only when asked for)."""
    x = int(v)
    if x not in (0, 1):
        raise ValueError(f"ftok must be 0 or 1 (got {v!r})")
    return x


def _ftok_mod():
    """FTOK's module if main.py imported it (S1ENG_FTOK=1 or an ARMS arm asks for ftok 1), else None."""
    return sys.modules.get(f"{__package__}.ftok")


def zc_of(v) -> int:
    """ZC-IN (zcin.py) arm key zc / env S1ENG_ZC: 0 or 1 (bool, int or "0" / "1"); anything else raises."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, str):
        if v.strip() not in ("0", "1"):
            raise ValueError(f"zc must be 0 or 1 (got {v!r})")
        return int(v.strip())
    if not isinstance(v, int):
        raise TypeError(f"zc must be 0 or 1 (got {v!r})")
    if v not in (0, 1):
        raise ValueError(f"zc must be 0 or 1 (got {v!r})")
    return int(v)


def _zc_asked(knobs) -> bool:
    try:
        return isinstance(knobs, dict) and zc_of(knobs.get("zc", 0)) == 1
    except (ValueError, TypeError):
        return False


def _l0t_refill(model, lanes: list, log=print) -> None:
    """L0T: fill the tables again after a check emptied them (as lanes_from_env after self_check)."""
    l0t = getattr(getattr(model, "pf", None), "l0t", None)
    if l0t is None or not lanes:
        return
    from . import l0t as l0tmod

    rows_of = {}
    for (L, _B), es in lanes[0].packed.items():
        rows_of.setdefault(L, []).extend(e[0] for e in es)
    l0tmod.prefill(l0t, model.pf, rows_of, log=log)


def join_wanted() -> dict:
    """BB4: every join asked for by S1ENG_JOIN or by any arm of S1ENG_ARMS (the start-up check covers them all)."""
    want = dict((s, d) for s, d in parse_join(os.environ.get("S1ENG_JOIN", "")))
    arms_env = os.environ.get("S1ENG_ARMS", "")
    if arms_env:
        try:
            arms = json.loads(open(arms_env[1:]).read() if arms_env.startswith("@") else arms_env)
            for a in arms.values():
                want.update(dict(parse_join(a.get("join", ""))))
        except Exception:  # noqa: BLE001 - BatchLoop reports a bad arms file
            pass
    return want


def _fc2_counts(acc: dict, what: str, st: dict) -> None:
    """r2 fixedcost2: sum a graph edit's per-graph counts (SKR prune / MSD) into acc[what]."""
    d = acc.setdefault(what, {})
    for k, v in st.items():
        d[k] = d.get(k, 0) + v if k != "domains" else v


def packed_rows(L: int, bs: list, step: int, pfx: int = 0) -> dict:
    """(B -> list of R) for packed graphs of bucket L: a B-slot graph serves groups of Bprev+1..B decisions, each of
    Lprev+1..L tokens, so R covers [(Bprev+1)(Lprev+1), B*L] in multiples of `step` (step(B) may depend on B).
    pfx > 0 (E1): the rows hold positions pfx.. only (decisions of max(Lprev+1, pfx+1)..L tokens)."""
    lprev = max([g for g in GRAPH_LENGTHS if g < L], default=0)
    lo = max(lprev + 1, pfx + 1) - pfx
    hi = L - pfx
    out = {}
    prev = 0
    for B in sorted(bs):
        st = step(B)
        tmin = (prev + 1) * lo
        r0 = -(-tmin // st) * st
        rs = list(range(r0, B * hi, st)) + [B * hi]
        out[B] = sorted(set(min(r, B * hi) for r in rs))
        prev = B
    return out


class Lane:
    """One CUDA stream with its own graphs, memory pool and cuBLASLt workspace; graphs of one lane run one after the
    other on its stream.
      single: L -> (graph, ids[1, L], last[1], out[1, 16])               per-decision graphs (lane 0: set 0)
      batch:  (L, B) -> (graph, ids[B, L], last[B], out[B, 16], ws)      S4 stacked graphs
      packed: (L, B) -> [(R, graph, meta, out), ...] ascending R         S5 packed graphs"""

    def __init__(self, idx: int, stream, single: dict, batch: dict, packed: dict, ws=(), cap_stream=None, pfx=None):
        self.idx = idx
        self.pfx = dict(pfx or {})  # E1: L -> template prefix length of this lane's packed graphs (0: S5 layout)
        self.stream = stream
        self.ws = list(ws)          # cuBLASLt workspace(s) baked into this lane's graphs: alive as long as the lane
        self.cap_stream = cap_stream
        self.single = single
        self.batch = batch
        self.packed = packed
        self.busy = 0
        self.t_free = 0.0           # E3 hold: when this lane last finished a batch
        self.bs = {}
        for (L, B) in batch:
            self.bs.setdefault(L, []).append(B)
        self.pbs = {}
        self.mixed, self.mbs = {}, {}   # MX (capture_batch fills them)
        for (L, B) in packed:
            self.pbs.setdefault(L, []).append(B)
        for d in (self.bs, self.pbs):
            for L in d:
                d[L].sort()

    def best_b(self, L: int, n: int) -> int | None:
        """S4: largest captured B <= n for bucket L (None if no batch graph for L)."""
        best = None
        for b in self.bs.get(L, ()):
            if b <= n:
                best = b
        return best

    def packed_b(self, L: int, n: int) -> int | None:
        """S5: smallest captured B >= n (or the largest B) for bucket L."""
        bs = self.pbs.get(L)
        if not bs:
            return None
        for b in bs:
            if b >= n:
                return b
        return bs[-1]



def select_swiglu(pinned, ctx, fz, finfo, groups_of, buckets, log=print) -> None:
    """SG (rgh): gate|up GEMM + SwiGLU in one native kernel (ng.nsw_*) per (bucket, row-count group). A candidate is
    used for a group only if its output equals the reference (F.linear gate / up on M = bucket blocks, then silu_mul)
    bit for bit at every R of the group and it is at least S1ENG_SG_GAIN faster at the group's anchor than the
    current path (the gate|up GEMM(s) the start-up selection chose + silu_mul); weights rotated over the layers."""
    from . import lt as ltmod
    from . import ng

    ent = [((Nf, K), e) for (Nf, K), e in finfo.items() if e[0] == "gate_up"]
    if not ent:
        return
    (Nf, K), (gname, n1, n2, blks) = ent[0]
    H = n1
    cands = ng.swiglu_candidates(H, K) if n1 == n2 else []
    if not cands:
        log("SG: no SwiGLU kernels (cubin / layout)")
        return
    gain = float(os.environ.get("S1ENG_SG_GAIN", "0.01"))
    reps = int(os.environ.get("S1ENG_TUNE_REPS", "3"))
    sg_deadline = float(os.environ.get("S1ENG_SG_DEADLINE_S", "540"))   # SD (rgh): no SG selection past this
    if GOV.elapsed() > sg_deadline:
        log(f"SG: skipped, start-up deadline {sg_deadline:.0f}s passed (gate|up GEMM + silu_mul everywhere)")
        return

    def graph_ms(fn):
        """Time fn as the engine runs it: captured in a CUDA graph and replayed (no host launch cost in the timing,
        which would otherwise penalise the current path's Triton silu_mul launch)."""
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            fn()
        torch.cuda.current_stream().wait_stream(st)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        gr.replay()
        ms = ltmod._time_ms(gr.replay, reps)
        del gr
        return ms
    # PT (rgh): S1ENG_SG_PAIR=1 (default: S1ENG_TUNE_PAIR) times every candidate as two lanes run it - two graphs of
    # the same work replayed concurrently on two streams (the other copy rotates the weights the other way); the
    # 1-stream winner often fills its own last wave with small tiles and loses once another lane fills it
    # (wavebench2 2026-10-20: the paired winner is 5-11 % faster paired at many R)
    pair = os.environ.get("S1ENG_SG_PAIR", "0") == "1"   # opt-in: Lium A/B 2026-10-20 = wash (262.2 vs 262.9)
    pair_top = int(os.environ.get("S1ENG_TUNE_PAIR_TOP", "0"))
    sa, sb = torch.cuda.Stream(), torch.cuda.Stream()

    def pair_ms(fa, fb):
        grs = []
        for fn in (fa, fb):
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                fn()
            torch.cuda.current_stream().wait_stream(st)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                fn()
            grs.append(gr)
        cur_s = torch.cuda.current_stream()
        grs[0].replay(); grs[1].replay()
        s0, e0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s0.record(cur_s)
        sa.wait_stream(cur_s); sb.wait_stream(cur_s)
        for _ in range(reps):
            with torch.cuda.stream(sa):
                grs[0].replay()
            with torch.cuda.stream(sb):
                grs[1].replay()
        cur_s.wait_stream(sa); cur_s.wait_stream(sb)
        e0.record(cur_s)
        e0.synchronize()
        ms = s0.elapsed_time(e0) / (2 * reps)
        GOV.tick()
        del grs
        return ms
    ws = torch.empty(max(pinned.ws_bytes, 64 << 20), dtype=torch.uint8, device="cuda")
    ws_b = torch.empty_like(ws) if pair else ws
    Wb = [b for _, b in blks][:8]
    nw = len(Wb)
    g = torch.Generator(device="cuda")
    g.manual_seed(4321)
    t0 = time.time()
    used, total, notes = 0, 0, []
    for L in buckets:
        groups = groups_of(L)
        if not groups or pinned.algo.get((Nf, K, L)) is None or pinned.algo.get((H, K, L)) is None:
            continue
        if GOV.elapsed() > sg_deadline:
            notes.append(f"{L}:deadline")
            break
        rmax = max(max(rs) for rs in groups.values())
        nb = -(-rmax // L)
        xs = torch.cat([(torch.randn((L, K), generator=g, device="cuda") * (1 + b % 3)).to(torch.bfloat16)
                        for b in range(nb)])
        y = torch.cat([pinned.ref_linear(Nf, K, Wb[0], xs[b * L:(b + 1) * L]) for b in range(nb)])
        ref = fz.silu_mul(y[:, :H].contiguous(), y[:, H:].contiguous())
        del y

        def cur(x, W, wsx=ws):
            R = x.shape[0]
            if pinned.fuse_r.get((Nf, K, L, R), False):
                a = pinned.algo_r.get((Nf, K, L, R), pinned.algo[(Nf, K, L)])
                yy = torch.empty((R, Nf), dtype=torch.bfloat16, device="cuda")
                ctx.run(a, W, x, yy, wsx)
                return fz.silu_mul(yy[:, :H], yy[:, H:])
            a = pinned.algo_r.get((H, K, L, R), pinned.algo[(H, K, L)])
            gt = torch.empty((R, H), dtype=torch.bfloat16, device="cuda")
            up = torch.empty((R, H), dtype=torch.bfloat16, device="cuda")
            ctx.run(a, W[:H], x, gt, wsx)
            ctx.run(a, W[H:], x, up, wsx)
            return fz.silu_mul(gt, up)

        def t_cur_of(xr):
            if pair:
                return pair_ms(lambda: [cur(xr, Wb[i]) for i in range(nw)],
                               lambda: [cur(xr, Wb[i], ws_b) for i in reversed(range(nw))]) / nw
            return graph_ms(lambda: [cur(xr, Wb[i]) for i in range(nw)]) / nw

        def t_sw_of(c, xr, o):
            if pair:
                o_b = torch.empty_like(o)
                return pair_ms(lambda: [ng.run_swiglu(c, Wb[i], xr, o) for i in range(nw)],
                               lambda: [ng.run_swiglu(c, Wb[i], xr, o_b) for i in reversed(range(nw))]) / nw
            return graph_ms(lambda: [ng.run_swiglu(c, Wb[i], xr, o) for i in range(nw)]) / nw

        for anchor, rs in groups.items():
            total += len(rs)
            x = xs[:anchor].contiguous()
            t_cur = t_cur_of(x)
            out = torch.empty((anchor, H), dtype=torch.bfloat16, device="cuda")
            best = None
            ts1 = []
            for c in cands:
                try:
                    ng.run_swiglu(c, Wb[0], x, out)
                except RuntimeError:
                    continue
                if not torch.equal(out.view(torch.int16), ref[:anchor].view(torch.int16)):
                    continue
                if pair and pair_top:
                    ts1.append((graph_ms(lambda c=c: [ng.run_swiglu(c, Wb[i], x, out) for i in range(nw)]) / nw, c))
                    continue
                t = t_sw_of(c, x, out)
                if best is None or t < best[0]:
                    best = (t, c)
            for _, c in sorted(ts1, key=lambda tc: tc[0])[:pair_top]:   # PT top-k: re-timed paired
                t = t_sw_of(c, x, out)
                if best is None or t < best[0]:
                    best = (t, c)
            if best is None or best[0] >= (1 - gain) * t_cur:
                notes.append(f"{L}/R{anchor}:-")
                continue
            # per R: the anchor's winner must be bit-equal at that R and faster there than the current path (GEMM
            # time is tile-quantised in R, so a group's anchor does not speak for its other row counts)
            n_r, inexact = 0, False
            for r in rs:
                xr = xs[:r].contiguous()
                o = torch.empty((r, H), dtype=torch.bfloat16, device="cuda")
                ng.run_swiglu(best[1], Wb[0], xr, o)
                if not torch.equal(o.view(torch.int16), ref[:r].view(torch.int16)):
                    inexact = True
                    break
                if r != anchor:
                    tc = t_cur_of(xr)
                    ts = t_sw_of(best[1], xr, o)
                    if ts >= (1 - gain) * tc:
                        continue
                pinned.swiglu_r[(L, r)] = best[1]
                n_r += 1
            if inexact:
                for r in rs:
                    pinned.swiglu_r.pop((L, r), None)
                notes.append(f"{L}/R{anchor}:inexact")
                continue
            used += n_r
            notes.append(f"{L}/R{anchor}:{best[1].name}/{best[1].drop}({best[0] * 1000:.0f}/{t_cur * 1000:.0f}us,{n_r}/{len(rs)}R)")
    del ws, ws_b
    log(f"SG gate|up+SwiGLU kernel{' (paired timing)' if pair else ''} at {used}/{total} R in {time.time() - t0:.1f}s: "
        f"{' '.join(notes)}")

def capture_batch(model: Model, lanes_n: int, bmax: dict, log=print, pack: dict | None = None,
                  rstep=lambda B: 64, prune: bool = False, prefix: list | None = None, select_only: bool = False):
    """Builds the lanes. Lane 0 = the per-decision graph set 0 (all lengths, captured as the reference) plus batch
    graphs on the default stream; lanes 1.. = their own stream with batch graphs from B = 1. For buckets in `pack`
    (L -> list of B) the batch graphs are packed (S5), otherwise stacked (S4, B = 1..bmax[L]).
    prefix (E1): the template's token ids; packed buckets then skip those rows (PackedForward.record_prefix) if an
    eager check against the per-decision graphs passes for the bucket, else they keep the S5 layout.
    select_only (kai, spike/s1_table.py): stop after the GEMM selection (pins, S6, fused-or-parts, SG) and return its
    state instead of lanes."""
    from . import lt as ltmod
    from .batchfwd import BatchForward, PackedForward, PackedMeta, packed_layout

    pack = dict(pack or {})
    ff = model.fwd
    ctx = ltmod.Lt()
    pinned = ltmod.Pinned(ctx)
    shapes = {}
    for layer in model.hf.model.layers:
        for name, mod in layer.named_modules():
            if isinstance(mod, torch.nn.Linear):
                N, K = mod.weight.shape
                shapes.setdefault((N, K), mod.weight)
    buckets = sorted(set(L for L in BATCH_BUCKETS if L in bmax) | set(pack))
    P = len(prefix) if prefix else 0
    pfx_of = {L: P for L in pack if P and L > P}
    prow0 = {L: packed_rows(L, bs, rstep) for L, bs in pack.items()}
    prow = {L: packed_rows(L, bs, rstep, pfx_of.get(L, 0)) for L, bs in pack.items()}
    # fused projections (packed buckets only): (Nf, K) -> (group, N1, N2, blocks in layer order)
    finfo = {}
    for (li_, gname), blk in sorted(model.fused_w.items()):
        layer = model.hf.model.layers[li_]
        n1 = layer.get_submodule(FUSE_GROUPS[gname][0]).weight.shape[0]
        e = finfo.setdefault((blk.shape[0], blk.shape[1]), [gname, n1, blk.shape[0] - n1, []])
        e[3].append((li_, blk))
    fused_shapes = []
    for (Nf, K), (gname, n1, n2, blks) in finfo.items():
        li_, blk = blks[0]
        fused_shapes.append((Nf, K, blk, (blk[:n1], blk[n1:]), sorted(pack)))
    t0 = time.time()
    pinned.select([(N, K, W) for (N, K), W in shapes.items()], buckets,
                  lambda L: max(bmax.get(L, 1), max(pack.get(L, [1]))) * L,
                  lambda L: sorted({L} | {r for rs in prow.get(L, {}).values() for r in rs}
                                   | {r for rs in prow0.get(L, {}).values() for r in rs}
                                   | {B * L for B in range(1, bmax.get(L, 0) + 1)}
                                   | (set(range(1, max(pack[L]) + 1)) if prune and L in pack else set())), log=log,
                  fused=fused_shapes)
    for k, v in pinned.report.items():
        log(f"  pin {k}: {v}")
    log(f"pin selection {time.time() - t0:.1f}s")
    per_r = os.environ.get("S1ENG_TUNE_PER_R", "0") == "1"

    def groups_of(L):
        """S6 timing anchors: {anchor R: [R values that use the anchor's winner]}. Default one anchor per packed B
        group (its middle R bin); S1ENG_TUNE_PER_R=1 times every packed R on its own."""
        if L in pack:
            if per_r:
                return {r: [r] for rs in prow[L].values() for r in rs}
            return {rs[len(rs) // 2]: rs for B, rs in prow[L].items()}
        if L in bmax:
            return {B * L: [B * L] for B in range(2, bmax[L] + 1)}
        return {}

    from . import plan as fpmod   # FP (rgh, after u198's shipped plans): frozen S6 / SG picks, byte-checked, untimed

    fplan = fpmod.load(os.environ.get("S1ENG_PLAN", ""), log=log)
    if os.environ.get("S1ENG_TUNE", "1") == "1":
        weights = {}
        for layer in model.hf.model.layers:
            for name, mod in layer.named_modules():
                if isinstance(mod, torch.nn.Linear):
                    weights.setdefault(tuple(mod.weight.shape), []).append(mod.weight)

        for (Nf, K), (gname, n1, n2, blks) in finfo.items():
            weights[(Nf, K)] = [b for _, b in blks]
        table = None
        if fplan is not None:   # FP: the plan's picks instead of S6 timing (each byte-checked at its R)
            try:
                fpmod.apply_s6(fplan, pinned, groups_of, weights, log=log)
            except Exception as exc:  # noqa: BLE001 - never lose the lanes: drop the plan's picks, time as usual
                log(f"lever not used: plan (S6: {type(exc).__name__}: {exc})")
                pinned.algo_r.clear()
                pinned.best_t.clear()
                fplan = None
        if fplan is None:
            if os.environ.get("S1ENG_GEMM_TABLE", "") not in ("", "0"):   # SEL2b (kai): table configs join S6 (gtable)
                from . import gtable

                table = gtable.from_env(os.environ, torch.cuda.get_device_name(), log=log,
                                        loaded=gtable.loaded_lt_version(ltmod.lt))
            ltmod.tune(pinned, weights, groups_of, log=log,
                       pair=os.environ.get("S1ENG_TUNE_PAIR", "0") == "1" and lanes_n > 1,
                       reps=int(os.environ.get("S1ENG_TUNE_REPS", "3")),
                       refine_k=int(os.environ.get("S1ENG_TUNE_REFINE", "0")),
                       refine_reps=int(os.environ.get("S1ENG_TUNE_REFINE_REPS", "2")),
                       refine_min_n=int(os.environ.get("S1ENG_TUNE_REFINE_MIN_N", "2048")), table=table)
        if os.environ.get("S1ENG_2N2560SPLITKPINSMLPDOWNGDNOUTPROJATTENTIONOPROJATTENTIONKV", "0") not in ("", "0"):
            # KF (ruj gemm2/k2): our exact in-CTA split-K GEMM (one CTA per output tile walking all K slices, row
            # tails in small tiles, a fold with no tensor-pipe cost) at the built-in table's (N, K, bucket, R) of the
            # down / out_proj / o_proj / k|v pins, each byte-gated against the per-decision reference (kf.apply);
            # the base picks stay as pinned._kf_base (FCAB arm nokf)
            from . import kf as kfmod

            base_r = dict(pinned.algo_r)
            try:
                if kfmod.apply(pinned, groups_of, weights, log=log):
                    pinned._kf_base = base_r
            except Exception as exc:  # noqa: BLE001 - never lose the lanes: the base picks
                pinned.algo_r.clear()
                pinned.algo_r.update(base_r)
                log(f"lever not used: KF ({type(exc).__name__}: {exc})")
        for k, v in sorted(pinned.tune_report.items()):
            log(f"  tune {k}: {v}")
    elif os.environ.get("S1ENG_GEMM_TABLE", "") not in ("", "0"):
        log("lever not used: gemm_table (S1ENG_TUNE=0: no S6 tuning to offer it to)")
    # fused GEMM or the two part GEMMs, per (bucket, row-count group): whichever is faster at the group's anchor
    fgain = float(os.environ.get("S1ENG_FUSE_GAIN", "0.01"))
    for (Nf, K), (gname, n1, n2, blks) in finfo.items():
        for L in pack:
            if pinned.algo.get((Nf, K, L)) is None:
                log(f"fused {gname}@{L}: not used ({pinned.report.get(f'{Nf}x{K}@{L}')})")
                continue
            used = []
            for anchor, rs in groups_of(L).items():
                for r in rs:
                    bt = pinned.best_t
                    tf = bt.get((Nf, K, L, r), bt.get((Nf, K, L, anchor)))
                    t1 = bt.get((n1, K, L, r), bt.get((n1, K, L, anchor)))
                    t2 = bt.get((n2, K, L, r), bt.get((n2, K, L, anchor)))
                    if os.environ.get("S1ENG_TUNE", "1") != "1":
                        use = True
                    else:
                        use = None not in (tf, t1, t2) and tf < (1 - fgain) * (t1 + t2)
                    pinned.fuse_r[(Nf, K, L, r)] = use
                    if r == anchor:
                        used.append(f"R{anchor}:{'F' if use else '-'}"
                                    + (f"({tf:.0f}/{t1 + t2:.0f}us)" if None not in (tf, t1, t2) else ""))
            nf = sum(1 for (a, b, c, r), u in pinned.fuse_r.items() if (a, b, c) == (Nf, K, L) and u)
            log(f"fused {gname}@{L}: fused at {nf}/{sum(len(v) for v in groups_of(L).values())} R; anchors {' '.join(used)}")
    pinned.swiglu_r = {}
    if os.environ.get("S1ENG_NG7", "0") == "1":
        # NG7 (pjs nsw7l, rgh port): the SiLU table + its check over every bf16 gate value; only then are the nsw7l
        # kernels SG candidates (ng.swiglu_candidates). Unset: nothing loaded, the base engine exactly
        from . import ng

        try:
            ng.check_ng7(ff.fused.silu_mul, log=log)
        except Exception as exc:  # noqa: BLE001 - no nsw7l candidates
            log(f"lever not used: ng7 ({type(exc).__name__}: {exc})")
    if os.environ.get("S1ENG_SG", "1") == "1" and os.environ.get("S1ENG_TUNE", "1") == "1":
        if fplan is not None:   # FP: the plan's SwiGLU kernels, each byte-checked at its R, untimed
            try:
                fpmod.apply_sg(fplan, pinned, ff.fused, finfo, groups_of, sorted(pack), log=log)
            except Exception as exc:  # noqa: BLE001
                log(f"lever not used: plan (SG: {type(exc).__name__}: {exc})")
                pinned.swiglu_r = {}
                fplan = None
        if fplan is None:
            select_swiglu(pinned, ctx, ff.fused, finfo, groups_of, sorted(pack), log=log)
        if os.environ.get("S1ENG_NG7", "0") == "1" and os.environ.get("S1ENG_NG7_TABLE", "0") not in ("", "0"):
            # NG7T (rgh): the in-situ-decided table's nsw7l picks over the SG picks, each byte-gated at its R
            # (2 input sets incl. x64 outliers, launched twice); a failing key or any exception keeps its pick
            from . import ng7t

            sw_base = dict(pinned.swiglu_r)
            try:
                ng7t.apply(pinned, ff.fused, finfo, groups_of, sorted(pack), log=log)
            except Exception as exc:  # noqa: BLE001 - never lose the lanes: the SG picks as they were
                pinned.swiglu_r = sw_base
                log(f"lever not used: NG7T ({type(exc).__name__}: {exc})")
    if os.environ.get("S1ENG_PLAN_DUMP", ""):   # FP: write this start-up's picks (an idle start-up makes the plan)
        try:
            fpmod.dump(os.environ["S1ENG_PLAN_DUMP"], pinned, log=log)
        except Exception as exc:  # noqa: BLE001
            log(f"FP: plan not written ({type(exc).__name__}: {exc})")
    if os.environ.get("S1ENG_GEMM_TABLE", "") not in ("", "0"):   # kai: name this start-up's timing-based picks
        from . import gtable

        fr, sr = getattr(pinned, "fuse_r", {}), getattr(pinned, "swiglu_r", {})
        dg = gtable.selection_digest({k: ltmod.cfg_key(a) for k, a in pinned.algo_r.items()}, fr, sr)
        log(f"selection digest {dg} (S6 algo_r {len(pinned.algo_r)}, fuse_r {len(fr)}, swiglu_r {len(sr)})")
    if select_only:   # kai: spike/s1_table.py reads the selection; no graphs
        return {"pinned": pinned, "ctx": ctx, "finfo": finfo, "groups_of": groups_of, "prow": prow, "pack": pack,
                "buckets": buckets, "shapes": shapes, "pfx_of": pfx_of}
    pinned._ws = None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    bf = BatchForward(ff, pinned, ctx)
    pf = PackedForward(bf)
    for L in list(pack):
        if not pf.available(L):
            log(f"packing disabled for bucket {L}: not every GEMM shape has a verified pinned algo")
            del pack[L]
    pf.prune = prune
    pf.gva = os.environ.get("S1ENG_GVA", "0") == "1"
    if pf.ff_fused:
        # FF (rgh): the fused fla path is used only if it is bit-equal to fla on this GPU (else fla, never a dropped
        # bucket)
        from . import fla_fused

        if pf.ff_mode == "ps":
            # KKT2 (rgh): ps mode's kkt_solve on the native per-key-head kernel if its A is bit-equal to fla's kernel
            # here (checked before ff_check, which then runs the ps forward end to end with it)
            why_k = fla_fused.kkt2_enable()
            log("KKT2 native kkt_solve on (A bit-equal to fla's kernel)" if not why_k else
                f"lever not used: kkt2 ({why_k})")

        def ff_check():
            try:
                return fla_fused.check(pf.ff_cfg, split=pf.ff_split, mode=pf.ff_mode, peel=pf.ff_peel)
            except Exception as exc:  # noqa: BLE001
                return f"{type(exc).__name__}: {exc}"
        why = ff_check()
        if why and fla_fused.KKT2["nat"] is not None:
            # KKT2: never let the native kkt_solve cost ps mode - retry the check on fla's kernel
            log(f"lever not used: kkt2 (ps forward check with it: {why})")
            fla_fused.KKT2["nat"] = None
            why = ff_check()
        from . import ffs

        if why and pf.ff_peel and ffs.lib() is not None:
            # FFS (rgh): the native pass P + S kernel is not bit-equal here: the Triton ps kernels instead
            log(f"lever not used: ffs (start-up check: {why})")
            ffs.disable(why)
            why = ff_check()
        if pf.ff_peel and why:
            # FF-PEEL (kai): the peeled ps kernel is not bit-equal to fla here: job 38's ps kernel (not the fused path)
            log(f"lever not used: ff_peel (start-up check: {why})")
            pf.ff_peel = False
            why = ff_check()
        if why and pf.ff_mode == "ps":
            # ps mode not bit-equal here: the one-kernel fused path may still be
            log(f"FF ps mode OFF on this GPU: {why}")
            pf.ff_mode = ""
            why = ff_check()
        if why:
            pf.ff_fused = False
            log(f"FF fused fla OFF on this GPU: {why}")
        else:
            log(f"FF fused fla on (BV {pf.ff_cfg[0]}, {pf.ff_cfg[1]} warps{', ' + pf.ff_mode if pf.ff_mode else ''}): "
                f"bit-equal to fla at start-up")
    if pf.ff_peel and not (pf.ff_fused and pf.ff_mode == "ps"):
        # FF-PEEL (kai) peels ps mode's pass S only: with ps mode off on this GPU the lanes run job 38's fallback
        log(f"lever not used: ff_peel (FF ps mode is off: "
            f"{'fused, mode ' + repr(pf.ff_mode) if pf.ff_fused else 'fla'})")
        pf.ff_peel = False
    if pf.ff_fused and pf.ff_peel and pf.ff_mode == "ps":
        from . import ffs

        if ffs.lib() is not None:
            log("lever used: ffs (native pass P + S kernel, bit-equal start-up check passed)")
    if pf.ff_peel:   # FF-PEEL (kai): the peeled pass S was part of the bit-equal start-up check above
        log("FF-PEEL on (ps mode pass S: chunk 0 without h dots, no h update after the last chunk)")
    if os.environ.get("S1ENG_PRO", "0") == "1":
        # E4 PRO: conv + l2norm(q, k) + gate + chunk cumsum of g in one launch per GDN layer, per bucket only where
        # check_prologue finds it bit-equal to the four-kernel path; before the prefix recording, so that check_prefix
        # and self_check run it end to end too
        if not (pf.ff_fused and pf.ff_mode == "ps"):
            log("lever not used: pro (it feeds FF ps mode, which is off on this GPU)")
        else:
            t1 = time.time()
            gdn = [d for d in ff.layers if d["type"] == "linear_attention"]
            layers = [gdn[0], gdn[len(gdn) // 2], gdn[-1]]
            pv_lens = [len(v) for v in prefix_variants_of(model, prefix)] if prefix else []
            notes = []
            for L in sorted(pack):
                bs = sorted({b for b in (1, 2, 5, max(pack[L])) if b <= max(pack[L])})
                pp = pfx_of.get(L, 0)
                try:
                    why = pf.check_prologue(L, bs, layers) or \
                        (pf.check_prologue(L, bs, layers, pfx=pp, pmax=max(pv_lens + [pp])) if pp else "")
                except Exception as exc:  # noqa: BLE001 - the lever must never take the batch lanes down with it
                    why = f"{type(exc).__name__}: {exc}"
                if why:
                    notes.append(f"{L}: {why}")
                else:
                    pf.pro.add(L)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            log(f"PRO GDN prologue in one launch on for buckets {sorted(pf.pro)}: bit-equal to conv + gate + fla "
                f"l2norm + cumsum in {time.time() - t1:.1f}s" + (f"; lever not used for {'; '.join(notes)}" if notes
                                                                 else ""))
            if pf.pro and os.environ.get("S1ENG_PRON", "1") == "1":
                # PRON (rgh): the native prologue kernel per PRO bucket where check_pron finds it bit-equal to the PRO
                # kernel (every GDN layer's weights; the bf16 domain sweep once)
                t1 = time.time()
                notes = []
                try:
                    from . import pron as _pron

                    why = "no pron kernel for this GPU" if _pron.lib() is None else pf.check_pron_domain(gdn[0])
                except Exception as exc:  # noqa: BLE001
                    why = f"{type(exc).__name__}: {exc}"
                if why:
                    log(f"lever not used: pron ({why})")
                else:
                    for L in sorted(pf.pro):
                        bs = sorted({b for b in (1, 2, 5, max(pack[L])) if b <= max(pack[L])})
                        pp = pfx_of.get(L, 0)
                        try:
                            why = pf.check_pron(L, bs, gdn[::2]) or \
                                (pf.check_pron(L, bs, gdn[::2], pfx=pp, pmax=max(pv_lens + [pp])) if pp else "")
                        except Exception as exc:  # noqa: BLE001
                            why = f"{type(exc).__name__}: {exc}"
                        if why:
                            notes.append(f"{L}: {why}")
                        else:
                            pf.pron.add(L)
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    log(f"lever used: pron (native GDN prologue) for buckets {sorted(pf.pron)}: bit-equal to the PRO "
                        f"kernel in {time.time() - t1:.1f}s" + (f"; not for {'; '.join(notes)}" if notes else ""))
    if pf.pro and pf.onorm and pf.ff_fused and pf.ff_peel and pf.ff_skip and pf.ff_mode == "ps":
        # ONORM (rgh): the GDN gated norm as the FFS kernel's epilogue, for packed batches of B >= onorm.bmin(L) slots,
        # per PRO bucket where onorm.check finds it bit-equal to FFS + gated_rmsnorm_packed at every such B (before
        # the prefix checks and the self-check, which then run it end to end)
        from . import ffs, onorm

        if ffs.lib() is None or onorm.lib() is None:
            log("lever not used: onorm (needs the FFS kernel" + (")" if ffs.lib() is None else
                                                                 "; onorm.cubin unavailable)"))
        else:
            t1 = time.time()
            try:
                res = onorm.check(sorted(pf.pro), {L: pack[L] for L in pf.pro})
            except Exception as exc:  # noqa: BLE001 - the lever must never take the batch lanes down with it
                onorm.ON.clear()
                res = {L: f"{type(exc).__name__}: {exc}" for L in pf.pro}
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            bad = [f"{L}: {w}" for L, w in sorted(res.items()) if w]
            log(f"ONORM gated norm in the FFS epilogue on for buckets {sorted(onorm.ON)} at B >= "
                f"{ {L: onorm.bmin(L) for L in sorted(onorm.ON)} }: "
                f"bit-equal to FFS + gated_rmsnorm_packed in {time.time() - t1:.1f}s"
                + (f"; lever not used for {'; '.join(bad)}" if bad else ""))
    if getattr(pf, "profuse", False):
        # PROFUSE-v (ruj r3, onormpv.py; S1ENG_PROFUSE=1, default off): ONORM's kernel computes v from the raw qkv rows
        # and PRON writes q / k only, per ONORM + PRON bucket where onormpv.check finds that chain bit-equal to PRON +
        # KKT2 + ONORM at every B ONORM runs at, with and without the template prefix (before the prefix checks and the
        # self-check, which then run it end to end); any other bucket keeps that chain unchanged
        from . import fla_fused as _ffm
        from . import onorm as _onm
        from . import onormpv

        onormpv.ON.clear()
        cand = sorted(L for L in _onm.ON if L in pf.pron and L in pf.pro)
        why = ("onormpv.cubin unavailable" if onormpv.lib() is None else
               "needs KKT2 (the native kkt_solve)" if _ffm.KKT2["nat"] is None else
               "no bucket with ONORM and PRON on" if not cand else "")
        if why:
            log(f"lever not used: profuse ({why})")
        else:
            t1 = time.time()
            gdn = [d for d in ff.layers if d["type"] == "linear_attention"]
            pv_lens = [len(v) for v in prefix_variants_of(model, prefix)] if prefix else []
            notes = []
            for L in cand:
                bs = sorted(b for b in pack[L] if b >= _onm.bmin(L))
                pp = pfx_of.get(L, 0)
                try:
                    why = onormpv.check(pf, L, bs, gdn[::2]) or \
                        (onormpv.check(pf, L, bs, gdn[::2], pfx=pp, pmax=max(pv_lens + [pp])) if pp else "")
                except Exception as exc:  # noqa: BLE001 - the lever must never take the batch lanes down with it
                    why = f"{type(exc).__name__}: {exc}"
                if why:
                    notes.append(f"{L}: {why}")
                else:
                    onormpv.ON.add(L)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            log(f"lever used: profuse (v inside ONORM's kernel from the raw qkv rows, PRON q / k only) for buckets "
                f"{sorted(onormpv.ON)} at B >= { {L: _onm.bmin(L) for L in sorted(onormpv.ON)} }: bit-equal to PRON + "
                f"KKT2 + ONORM (q, k, beta, g, A, v rows, packed output) in {time.time() - t1:.1f}s"
                + (f"; not for {'; '.join(notes)}" if notes else ""))
    if os.environ.get("S1ENG_BA", "0") == "1":
        # FCL BA (ruj v14 fixedcost, fcl.py): the GDN b|a projection as ONE Triton launch (split-slice MMA + in-order
        # fold) instead of the split-K GEMM + reduce, per bucket where ba_check finds it byte-equal to the lanes'
        # pinned path (pair(): pin / S6 pick / fused-or-parts) at every packed R of the bucket
        from . import fcl

        t1 = time.time()
        gdn = [d for d in ff.layers if d["type"] == "linear_attention"]
        layers = [gdn[1 if len(gdn) > 1 else 0], gdn[len(gdn) // 2], gdn[-1]]
        notes = []
        ws0 = bf.ws
        bf.ws = torch.empty(max(pinned.ws_bytes, 1 << 20), dtype=torch.uint8, device="cuda")
        try:
            for L in sorted(pack):
                rs = sorted({r for rr in prow[L].values() for r in rr} | {r for rr in prow0[L].values() for r in rr})
                try:
                    why = fcl.ba_check(pf, L, layers, rs)
                except Exception as exc:  # noqa: BLE001 - the lever must never take the batch lanes down with it
                    why = f"{type(exc).__name__}: {exc}"
                if why:
                    notes.append(f"{L}: {why}")
                else:
                    pf.ba[L] = fcl.BA_MODES[L]
        finally:
            bf.ws = ws0
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        log(f"FCL BA b|a projection in one launch on for buckets {sorted(pf.ba)} (byte-equal to the pinned split-K path "
            f"at every packed R, kernel cfg {fcl.ba_cfg_env()}) in {time.time() - t1:.1f}s"
            + (f"; lever not used for {'; '.join(notes)}" if notes else ""))
    if os.environ.get("S1ENG_CAUSAL", "0") == "1":
        sa = next(d["sa"] for d in ff.layers if d["type"] != "linear_attention")
        for L in pack:
            heads = sa.q_proj.weight.shape[0] // (2 * sa.head_dim)
            pf.causal[L] = pf.check_causal(L, heads, sa.head_dim, sa.scaling, max(pack[L]))
        log(f"causal-skip SDPA (efficient kernel, is_causal=True) bit-equal to the masked call: {pf.causal}")
    if os.environ.get("S1ENG_LR1", "0") == "1" and prune:
        # E4 LR1: the pruned last layer attends with one query row per slot, per bucket where check_lr1 passes
        dl = ff.layers[-1]
        notes = []
        if dl["type"] == "linear_attention":
            notes.append("the last layer is not an attention layer")
        else:
            for L in sorted(pack):
                _m, cos, sin = ff.consts[L]
                try:
                    why = pf.check_lr1(L, dl, cos, sin, max(pack[L]))
                except Exception as exc:  # noqa: BLE001 - the lever must never take the batch lanes down with it
                    why = f"{type(exc).__name__}: {exc}"
                if why:
                    notes.append(f"{L}: {why}")
                else:
                    pf.lr1.add(L)
            for b in range(1, max(max(v) for v in pack.values()) + 1):
                pf._ar.setdefault(b, torch.arange(b, dtype=torch.int32, device="cuda"))
            torch.cuda.synchronize()
        log(f"LR1 last layer attends one query row per slot for buckets {sorted(pf.lr1)}" +
            (f"; lever not used for {'; '.join(notes)}" if notes else ""))
    if os.environ.get("S1ENG_GQA", "0") == "1":
        # E4 GQA: attention layers write K/V once per KV head and run SDPA over KV-head groups (stride-0 heads), per
        # bucket where check_gqa passes (after LR1, whose one-row call it also checks)
        da = next(d for d in ff.layers if d["type"] != "linear_attention")
        notes = []
        for L in sorted(pack):
            _m, cos, sin = ff.consts[L]
            try:
                why = pf.check_gqa(L, da, cos, sin, max(pack[L]))
            except Exception as exc:  # noqa: BLE001 - the lever must never take the batch lanes down with it
                why = f"{type(exc).__name__}: {exc}"
            if why:
                notes.append(f"{L}: {why}")
            else:
                pf.gqa.add(L)
        for b in range(1, max(max(v) for v in pack.values()) + 1):
            pf._ar.setdefault(b, torch.arange(b, dtype=torch.int32, device="cuda"))
        torch.cuda.synchronize()
        log(f"GQA K/V once per KV head for buckets {sorted(pf.gqa)}" +
            (f"; lever not used for {'; '.join(notes)}" if notes else ""))
    if os.environ.get("S1ENG_RARN", "1") == "1":
        # RARN (rgh, after u198's rarn2): out_proj / o_proj / down_proj hand their split-K slice partials to one
        # fold + residual add + RMSNorm kernel (no splitKreduce launch, no y round trip), per (K, bucket, R) where the
        # start-up byte check vs today's path passes; before the prefix recording, so check_prefix and self_check run
        # it end to end too
        from . import rarn as rarnmod

        t1 = time.time()
        try:
            rows_of = {L: {r for rs in prow.get(L, {}).values() for r in rs}
                       | {r for rs in prow0.get(L, {}).values() for r in rs} for L in pack}
            pf.rarn = rarnmod.select(pinned, ctx, shapes, rows_of, ff.layers[0]["ln2_w"], ff.eps,
                                     rarnmod.rule_from_env(log=log), log=log)
            kfb = getattr(pinned, "_kf_base", None)
            if kfb is not None and any("nokf" in a.split("+") for a in os.environ.get("S1ENG_FCAB", "").split(",")):
                # KF (ruj gemm2/k2), FCAB arm nokf: RARN's selection on the base picks (the base engine's pf.rarn)
                cur_r = dict(pinned.algo_r)
                pinned.algo_r.clear()
                pinned.algo_r.update(kfb)
                try:
                    pf._rarn_nokf = rarnmod.select(pinned, ctx, shapes, rows_of, ff.layers[0]["ln2_w"], ff.eps,
                                                   rarnmod.rule_from_env(log=log), log=log)
                finally:
                    pinned.algo_r.clear()
                    pinned.algo_r.update(cur_r)
                log(f"KF: FCAB nokf arm: RARN on the base picks at {len(pf._rarn_nokf)} (K, bucket, R) "
                    f"(with KF: {len(pf.rarn)})")
            why = "" if rarnmod.lib() is not None else "kernels unavailable on this GPU"
        except Exception as exc:  # noqa: BLE001 - the lever must never take the batch lanes down with it
            pf.rarn = {}
            why = f"{type(exc).__name__}: {exc}"
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if pf.rarn:
            kinds = {}
            for (K, L, R), (name, S) in pf.rarn.items():
                kinds[f"{K}@{L}:{name}"] = kinds.get(f"{K}@{L}:{name}", 0) + 1
            log(f"lever used: rarn at {len(pf.rarn)} (K, bucket, R), bit-equal to the S6 pick + add_rmsnorm at "
                f"start-up in {time.time() - t1:.1f}s: {kinds}")
        else:
            log(f"lever not used: rarn ({why or 'no (K, bucket, R) qualified'})")
    from . import skr as _skrm
    if _skrm.enabled():
        # SKR (ruj r2 fixedcost2, skr.py): where the N=2560 pick is a cuBLASLt split-K algo (OUTPUT_TYPE) and RARN is
        # not used, the add_rmsnorm folds cuBLASLt's own workspace partials (byte gate per (K, bucket, R)); the lane
        # graphs then lose the splitKreduce node (skr.prune at capture). Before the prefix recording / self_check.
        t1 = time.time()
        try:
            rows_sk = {L: {r for rs in prow.get(L, {}).values() for r in rs}
                       | {r for rs in prow0.get(L, {}).values() for r in rs} for L in pack}
            pf.skr = _skrm.select(pinned, ctx, shapes, rows_sk, ff.layers[0]["ln2_w"], ff.eps, skip=set(pf.rarn),
                                  log=log)
            kfb = getattr(pinned, "_kf_base", None)
            if kfb is not None and any("nokf" in a.split("+") for a in os.environ.get("S1ENG_FCAB", "").split(",")):
                # S1+KF, FCAB arm nokf: SKR's selection on the base picks and the base RARN sites (= S1's pf.skr;
                # with KF on, select() skips the KF keys)
                cur_r = dict(pinned.algo_r)
                pinned.algo_r.clear()
                pinned.algo_r.update(kfb)
                try:
                    pf._skr_nokf = _skrm.select(pinned, ctx, shapes, rows_sk, ff.layers[0]["ln2_w"], ff.eps,
                                                skip=set(getattr(pf, "_rarn_nokf", pf.rarn)), log=log)
                finally:
                    pinned.algo_r.clear()
                    pinned.algo_r.update(cur_r)
                log(f"KF: FCAB nokf arm: SKR on the base picks at {len(pf._skr_nokf)} (K, bucket, R) "
                    f"(with KF: {len(pf.skr)})")
            why = ""
        except Exception as exc:  # noqa: BLE001 - the lever must never take the batch lanes down with it
            pf.skr = {}
            why = f"{type(exc).__name__}: {exc}"
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if pf.skr:
            kinds = {}
            for (K, L, R), (_a, S) in pf.skr.items():
                kinds[f"{K}@{L}/S{S}"] = kinds.get(f"{K}@{L}/S{S}", 0) + 1
            log(f"lever used: skr at {len(pf.skr)} (K, bucket, R) (cuBLASLt partials folded in add_rmsnorm), "
                f"bit-equal at start-up in {time.time() - t1:.1f}s: {kinds}")
        else:
            log(f"lever not used: skr ({why or 'no (K, bucket, R) qualified'})")
    if os.environ.get("S1ENG_XA", "0") == "1":
        # XA (ATTN): the attention layers run the native computed-rows attention with the gate fused, per bucket where
        # check_xa finds it byte-equal to the reference chain (with and without the template prefix layout)
        t1 = time.time()
        from . import xa as xamod

        pf.xa_fq = xamod.FQ
        # XA3 (kai/rf-xa, xa3.cubin; S1ENG_XA3 default off, or an FCAB arm xa3 / noxa3): the variant gated inside
        # check_xa (and fcl.xl_check, ntv.recheck) against xa_fwd2q, byte for byte; the env lanes use it only with
        # S1ENG_XA3=1
        try:
            pf.xa_want = xamod.xa3_wanted()
        except ValueError as exc:
            log(f"lever not used: xa3 ({exc})")
            pf.xa_want = None
        if pf.xa_want and not pf.xa_fq:
            log("lever not used: xa3 (needs S1ENG_XA_FQ=1: the xa3 kernels are q kernels)")
            pf.xa_want = None
        if pf.xa_want and xamod.lib3() is None:
            log("lever not used: xa3 (xa3.cubin unavailable for this GPU)")
            pf.xa_want = None
        if os.environ.get("S1ENG_XA_GTAB", "0") == "1":
            try:
                pf.xa_gtab = xamod.gate_table("cuda")
            except Exception as exc:  # noqa: BLE001
                log(f"XA gate table unavailable ({type(exc).__name__}: {exc}); per-element sigmoid")
                pf.xa_gtab = None
        das = [d for d in ff.layers if d["type"] != "linear_attention"]
        mid = len(das) // 2   # layers 0 and mid on every B, the others (their norm weights) on the largest B
        layers = [das[0], das[mid]] + [d for i, d in enumerate(das) if i not in (0, mid)]
        pv_lens = [len(v) for v in prefix_variants_of(model, prefix)] if prefix else []
        notes = []
        for L in sorted(pack):
            _m, cos, sin = ff.consts[L]
            pp = pfx_of.get(L, 0)
            try:
                why = pf.check_xa(L, layers, cos, sin, max(pack[L])) or \
                    (pf.check_xa(L, layers, cos, sin, max(pack[L]), pfx=pp, pmax=max(pv_lens + [pp])) if pp else "")
            except Exception as exc:  # noqa: BLE001 - the lever must never take the batch lanes down with it
                why = f"{type(exc).__name__}: {exc}"
            if why:
                notes.append(f"{L}: {why}")
            else:
                pf.xa.add(L)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        log(f"XA native attention (computed rows, gate{' + q norm/RoPE' if pf.xa_fq else ''} fused, kernel "
            f"{xamod.VER}) on for buckets {sorted(pf.xa)}: byte-equal to the masked SDPA chain in "
            f"{time.time() - t1:.1f}s" + (f"; lever not used for {'; '.join(notes)}" if notes else ""))
        if pf.xa_want:   # XA3: the XA buckets whose variant matched xa_fwd2q on every input
            ok = xamod.finish(pf, "xa")
            bad = {L: w for (k, L), w in pf.xa_vbad.items() if k == "xa"}
            log(f"lever {'used' if ok else 'not used'}: xa3 ({pf.xa_want}, {'env lanes' if pf.xa_ver else 'FCAB arms only'})"
                f" XA buckets {sorted(ok)} byte-equal to xa_fwd2q" + (f"; xa_fwd2q kept for {bad}" if bad else ""))
    if os.environ.get("S1ENG_LT", "0") == "1" and prune and ff.layers[-1]["type"] != "linear_attention":
        # FCL LT (ruj v14 fixedcost, fcl.py): the pruned last layer's gathers + LR1 bias in one launch, per bucket where
        # the start-up gate finds it byte-equal to the launches it replaces
        from . import fcl

        t1 = time.time()
        da = next(d for d in ff.layers if d["type"] != "linear_attention")
        hk = da["sa"].k_proj.weight.shape[0] // da["sa"].head_dim
        notes = []
        for L in sorted(pack):
            _m, cos, sin = ff.consts[L]
            G = hk if (L in pf.lr1 and L in pf.gqa) else 0
            try:
                why = fcl.lt_check(_m, L, max(pack[L]), G)
            except Exception as exc:  # noqa: BLE001 - the lever must never take the batch lanes down with it
                why = f"{type(exc).__name__}: {exc}"
            if why:
                notes.append(f"lt {L}: {why}")
            else:
                pf.lt.add(L)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        log(f"FCL LT last-layer gathers + LR1 bias in one launch for buckets {sorted(pf.lt)} ({time.time() - t1:.1f}s)"
            + (f"; lever not used for {'; '.join(notes)}" if notes else ""))
    if os.environ.get("S1ENG_XL", "0") == "1" and prune and ff.layers[-1]["type"] != "linear_attention":
        # FCL XL (ruj v14 fixedcost, fcl.py): the pruned last layer's one-row attention through v15's XA kernel (one
        # computed row per slot), per bucket where LR1 + GQA are on and the gate finds it byte-equal to their chain
        from . import fcl

        t1 = time.time()
        dl = ff.layers[-1]
        notes = []
        for L in sorted(pack):
            _m, cos, sin = ff.consts[L]
            if L not in pf.lr1 or L not in pf.gqa:
                notes.append(f"{L}: LR1 / GQA off")
                continue
            try:
                why = fcl.xl_check(pf, dl, cos, sin, ff.eps, _m, L, sorted(set(pack[L]) | {1}))
            except Exception as exc:  # noqa: BLE001 - the lever must never take the batch lanes down with it
                why = f"{type(exc).__name__}: {exc}"
            if why:
                notes.append(f"{L}: {why}")
            else:
                pf.xl.add(L)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        log(f"FCL XL last-layer attention through XA (one computed row per slot) for buckets {sorted(pf.xl)} "
            f"({time.time() - t1:.1f}s)" + (f"; lever not used for {'; '.join(notes)}" if notes else ""))
        if getattr(pf, "xa_want", None):   # XA3: the XL buckets whose variant matched xa_fwd2q on every input
            from . import xa as xamod3

            okl = xamod3.finish(pf, "xl")
            bad = {L: w for (k, L), w in pf.xa_vbad.items() if k == "xl"}
            log(f"lever {'used' if okl else 'not used'}: xa3 ({pf.xa_want}) XL buckets {sorted(okl)} byte-equal to "
                f"xa_fwd2q" + (f"; xa_fwd2q kept for {bad}" if bad else ""))
    if os.environ.get("S1ENG_TG", "0") == "1" and prune and ff.layers[-1]["type"] != "linear_attention":
        # FCL TG (ruj v14 fixedcost, fcl.py): the pruned last layer's M = B GEMMs (q_proj, o_proj, gate / up or the
        # fused gate|up, down_proj) as 16-row-tile kernels with the pin's arithmetic, per (N, K, bucket) where the gate
        # finds them byte-equal to the pinned path at every packed B
        from . import fcl
        from . import lt as ltmod

        t1 = time.time()
        dl = ff.layers[-1]
        ws_tg = []
        for nm in ("q_proj", "o_proj"):
            ws_tg.append((nm, getattr(dl["sa"], nm).weight))
        for nm in ("gate_proj", "up_proj", "down_proj"):
            ws_tg.append((nm, getattr(dl["mlp"], nm).weight))
        if dl.get("fused", {}).get("gate_up") is not None:
            ws_tg.append(("gate_up", dl["fused"]["gate_up"]))
        notes, kinds = [], {}
        ws0 = bf.ws
        bf.ws = torch.empty(max(pinned.ws_bytes, 1 << 20), dtype=torch.uint8, device="cuda")
        try:
            for L in sorted(pack):
                bs = sorted(set(pack[L]) | {1})
                for nm, W in ws_tg:
                    key = (W.shape[0], W.shape[1], L)
                    if key in pf.tg:
                        continue
                    algo = pinned.algo.get(key)
                    if algo is None:
                        notes.append(f"{nm}@{L}: no pin")
                        continue
                    try:
                        cfg = ltmod.cfg_of(algo)
                        modes = fcl.tg_modes(cfg, W.shape[1])
                        if modes:
                            mode, why = fcl.tg_check(lambda W_, x_, L=L: pf.gemm(W_, x_, L), W, modes, bs)
                        else:
                            mode, why = None, f"split {cfg.get('splitK')} reduction {cfg.get('reduction')} not reproduced"
                    except Exception as exc:  # noqa: BLE001 - the lever must never take the batch lanes down with it
                        mode, why = None, f"{type(exc).__name__}: {exc}"
                    if mode is None:
                        notes.append(f"{nm}@{L}: {why}")
                    else:
                        pf.tg[key] = mode
                        kinds[f"{nm}@{L}"] = f"{mode[1]}x{mode[0]}" if mode[2] else "chain"
        finally:
            bf.ws = ws0
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        log(f"FCL TG last-layer M = B GEMMs as 16-row-tile kernels (cfg {fcl.tg_cfg_env()}) for {kinds} "
            f"({time.time() - t1:.1f}s)" + (f"; lever not used for {'; '.join(notes)}" if notes else ""))
    if os.environ.get("S1ENG_TGS", "0") == "1" and prune and ff.layers[-1]["type"] != "linear_attention":
        # FCL TGS (ruj v14 fixedcost, fcl.py): the pruned last layer's MLP gate + up + SwiGLU at M = B in one launch,
        # per bucket where both pins are non-split and the gate finds it byte-equal to the parts + silu_mul
        from . import fcl
        from . import lt as ltmod

        t1 = time.time()
        mlp = ff.layers[-1]["mlp"]
        wg, wu = mlp.gate_proj.weight, mlp.up_proj.weight
        notes = []
        ws0, tg0 = bf.ws, pf.tg
        bf.ws = torch.empty(max(pinned.ws_bytes, 1 << 20), dtype=torch.uint8, device="cuda")
        pf.tg = {}   # the reference is the pinned path itself (not TG's copy of it)
        try:
            for L in sorted(pack):
                algo = pinned.algo.get((wg.shape[0], wg.shape[1], L))
                try:
                    cfg = ltmod.cfg_of(algo) if algo is not None else {}
                    if algo is None or int(cfg.get("splitK", 1)) > 1:
                        why = "no pin" if algo is None else f"split {cfg.get('splitK')} pin"
                    else:
                        why = fcl.tgs_check(lambda W_, x_, L=L: pf.gemm(W_, x_, L), ff.fused.silu_mul, wg, wu,
                                            sorted(set(pack[L]) | {1}))
                except Exception as exc:  # noqa: BLE001 - the lever must never take the batch lanes down with it
                    why = f"{type(exc).__name__}: {exc}"
                if why:
                    notes.append(f"{L}: {why}")
                else:
                    pf.tgs.add(L)
        finally:
            bf.ws, pf.tg = ws0, tg0
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        log(f"FCL TGS last-layer gate + up + SwiGLU at M = B in one launch (cfg {fcl.tgs_cfg_env()}) for buckets "
            f"{sorted(pf.tgs)} ({time.time() - t1:.1f}s)" + (f"; lever not used for {'; '.join(notes)}" if notes else ""))
    if os.environ.get("S1ENG_GRP", "0") == "1":
        # GRP (rgh 10-21): grouped projection GEMMs (grp.py): at the (site, bucket, R) of the frozen GRP plan the GDN
        # b|a + qkv[|z] and the attention k|v + q_proj run as ONE launch each, only where check_grp finds every output
        # byte-equal to the selected path and to per-decision F.linear; after RARN / XA / PRON (independent sites),
        # before the prefix recording, so that check_prefix and self_check run it end to end too
        try:
            setup_grp(model, pf, pinned, pack, log=log)
        except Exception as exc:  # noqa: BLE001 - the lever must never take the batch lanes down with it
            log(f"lever not used: grp ({type(exc).__name__}: {exc})")
            pf.grp = {}
            pf.late_r = {}
            bf.ws = None
    model.bf, model.pf, model.pinned = bf, pf, pinned
    if _ntv() and pfx_of:   # NTV (kai): only the nonce-token variants whose cached rows leave the lanes' memory free
        # (ntv.prepare: trim; on any failure or none kept NTV is off, hotspot's variants; PRO / PRON / XA re-checked at the
        # runtime pmax if trim changed it); never fatal
        try:
            from . import ntv as ntvmod

            ntv_run = ntvmod.prepare(model, pf, prefix, pfx_of, pack, prow, prow0, lanes_n, prefix_variants_of,
                                     check_prefix, log=log)
        except Exception as exc:  # noqa: BLE001 - NTV must never stop the engine (prepare restores hotspot's variants)
            log(f"lever not used: ntv ({type(exc).__name__}: {exc})")
            ntv_run = None
    if pfx_of:
        t1 = time.time()
        bf.ws = torch.empty(max(pinned.ws_bytes, 1 << 20), dtype=torch.uint8, device="cuda")
        for L in sorted(pfx_of):
            try:
                variants = prefix_variants_of(model, prefix)
                if _ntv() and ntv_run:   # NTV (kai): bucket L's own list: the nonce-token variants only in NTV's buckets
                    # (ntv.bucket_variants), hotspot's own 3 elsewhere (same indices, so pvar < blocks)
                    variants = ntv_run.variants(L)
                pf.record_prefix(L, variants if len(variants) > 1 else prefix)
                if _ntv() and ntv_run and L in pf.xa and ntv_run.caches(L):   # NTV (kai): XA never reads the prefix's
                    # q|gate rows (trim counted them out); only in NTV's buckets (the others keep hotspot's layout)
                    for key in [k for k in pf.pcache if k[0] == L]:
                        pf.pcache[key].pop("qg", None)
                why = check_prefix(model, pf, L, prow[L], prefix, variants=variants)
                if _ntv() and ntv_run:   # NTV (kai): every variant block once (cover; check_prefix samples them); a
                    # failure (or check_prefix's) turns NTV off for every bucket before PRO or the prefix: hotspot's
                    # variants recorded and checked again as below (ntv.Run)
                    why = ntv_run.check(L, why)
                    variants = ntv_run.variants(L)   # the list in force for L (hotspot's after a fallback)
                if why and L in pf.pro:   # E4: PRO runs inside this check; without it before giving up the prefix
                    log(f"lever not used: pro for bucket {L} (template prefix check with it: {why})")
                    pf.pro.discard(L)
                    why = check_prefix(model, pf, L, prow[L], prefix, variants=variants)
            except Exception as exc:  # noqa: BLE001  (any failure: the bucket keeps the S5 layout)
                why = f"{type(exc).__name__}: {exc}"
            if _ntv() and ntv_run and why:   # NTV (kai): record_prefix / check_prefix raised with the nonce-token
                # variants: NTV off for every bucket first, as above
                why = ntv_run.check(L, why)
            if why:
                log(f"template prefix OFF for bucket {L} (S5 layout): {why}")
                del pfx_of[L]
                prow[L] = prow0[L]
                for k in [k for k in pf.pcache if k[0] == L]:
                    del pf.pcache[k]
        bf.ws = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        log(f"template prefix ({P} tokens) on for buckets {sorted(pfx_of)}: eager check vs per-decision graphs in "
            f"{time.time() - t1:.1f}s")
        if _ntv() and ntv_run:   # NTV (kai): its start-up line (variants per bucket, the prefix cache's size)
            log(ntvmod.summary(model, pf, prefix, pfx_of))
    model.pfx_of = dict(pfx_of)
    if os.environ.get("S1ENG_L0T", "0") == "1":   # L0T (kai): layer-0 projection tables, self-checked, before capture
        from . import l0t as l0tmod

        try:
            pf.l0t = l0tmod.setup(model, pf, pinned, ctx, prow, pfx_of, log=log)
        except Exception as exc:  # noqa: BLE001 - the lever must never take the batch lanes down with it
            log(f"lever not used: l0t ({type(exc).__name__}: {exc})")
            pf.l0t = None
            bf.ws = None
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    if os.environ.get("S1ENG_EMB", "0") == "1" or os.environ.get("S1ENG_RO", "0") == "1":
        # FCL EMB / RO (ruj v14 fixedcost, fcl.py): the graph's first gathers (embedding + L0T layer-0 rows) in one
        # launch / the readout's B products + cat in one call, per bucket where the start-up gate finds every byte equal
        from . import fcl

        t1 = time.time()
        notes = []
        if os.environ.get("S1ENG_EMB", "0") == "1":
            d0 = ff.layers[0]
            for L in sorted(pack):
                if pf.l0t is None or L not in pf.l0t.books:
                    notes.append(f"emb {L}: L0T off")
                    continue
                rs = sorted({r for rr in prow[L].values() for r in rr})
                try:
                    why = fcl.emb_check(ff.tm.embed_tokens, pf.l0t, L, rs,
                                        lambda R, L=L: (pf._l0t_fused(d0, "qkv_z", R, L),
                                                        pf._l0t_fused(d0, "b_a", R, L)))
                except Exception as exc:  # noqa: BLE001 - the lever must never take the batch lanes down with it
                    why = f"{type(exc).__name__}: {exc}"
                if why:
                    notes.append(f"emb {L}: {why}")
                else:
                    pf.emb.add(L)
        rop = None
        if os.environ.get("S1ENG_RO", "0") == "1" and os.environ.get("S1ENG_RO_PAD", "0") == "1":
            # ROP (pjs e10): RO's bmm only at row counts whose GEMV is the reference's per-row kernel (fcl.ro_plan:
            # kernel identity + byte gate on random and near-tie rows per graph B; mixed graphs' B are packed B's)
            try:
                rows, rop = fcl.ro_plan(ff.slot_weight, sorted({B for L in pack for B in pack[L]}))
            except Exception as exc:  # noqa: BLE001 - RO stays off: the per-row products (the reference op)
                rows, rop = None, None
                notes.append(f"ro (ROP gate): {type(exc).__name__}: {exc}")
            if rows is not None:
                pf.ro_rows = rows
                for L in pack:
                    pf.ro[L] = "bmm"
        elif os.environ.get("S1ENG_RO", "0") == "1":
            try:
                mode, why = fcl.ro_check(ff.slot_weight, sorted({B for L in pack for B in pack[L]}))
            except Exception as exc:  # noqa: BLE001
                mode, why = None, f"{type(exc).__name__}: {exc}"
            if mode is None:
                notes.append(f"ro: {why}")
            else:
                for L in pack:
                    pf.ro[L] = mode
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        log(f"FCL EMB embedding + layer-0 table rows in one launch for buckets {sorted(pf.emb)}; RO readout in one "
            f"call for buckets {sorted(pf.ro)} (mode {sorted(set(pf.ro.values())) or '-'}) ({time.time() - t1:.1f}s)"
            + (f"; lever not used for {'; '.join(notes)}" if notes else ""))
        if rop:
            log(f"lever used: ROP (S1ENG_RO_PAD) {rop}")

    # MX (ruj r2 mixbatch): S1ENG_MIX="128:192,192:256" -> mixed graphs of these (S, D) pairs: bucket-S decisions ride
    # in bucket-D packed graphs, every row in its own bucket's bit class (two-class GEMMs, joint prefix caches, stacked
    # L0T tables); per pair a start-up gate (PackedForward.mx_setup) before capture and mix_check after it
    mixrows = {}
    mx_want = [(S_, D_) for S_, D_ in parse_join(os.environ.get("S1ENG_MIX", ""))]
    if mx_want:
        t1 = time.time()
        notes = []
        if os.environ.get("S1ENG_META_BLOB", "0") != "1":
            notes.append("needs S1ENG_META_BLOB=1")
            mx_want = []
        try:
            if mx_want and pf.l0t is not None:
                pf.mx_stack_l0t()
        except Exception as exc:  # noqa: BLE001 - the lever must never take the batch lanes down with it
            notes.append(f"l0t stack: {type(exc).__name__}: {exc}")
            mx_want = []
        fz_list = [(blks[0][1], n1) for (Nf, K_), (gname, n1, n2, blks) in finfo.items()]
        bf.ws = torch.empty(max(pinned.ws_bytes, 1 << 20), dtype=torch.uint8, device="cuda")
        for S_, D_ in mx_want:
            if D_ not in pack or S_ not in pack or pfx_of.get(D_, 0) != pfx_of.get(S_, 0):
                notes.append(f"{S_}->{D_}: buckets not both packed with the same prefix layout")
                continue
            try:
                why = pf.mx_setup(D_, S_, shapes, fz_list)
            except Exception as exc:  # noqa: BLE001
                why = f"{type(exc).__name__}: {exc}"
            if why:
                notes.append(f"{S_}->{D_}: {why}")
                continue
            mixrows[(D_, S_)] = mix_rows(D_, S_, prow[D_], pfx_of.get(D_, 0), mix_slots(D_, S_, pack[D_]))
            # skx6t (ruj): the two-class GEMM without its padded tail work (S1ENG_SKX6T), byte-gated per (N, K, R) of
            # these mixed graphs against skx6 itself before they are captured; rejected keys keep skx6
            from . import mx as _mxm

            if _mxm.tail_mode():
                t2 = time.time()
                try:
                    tg = pf.mx_tail_gate(D_, S_, shapes, fz_list,
                                         sorted({r for rs_ in mixrows[(D_, S_)].values() for r in rs_}))
                except Exception as exc:  # noqa: BLE001 - the lever must never take MX down with it
                    _mxm.TAIL.clear()
                    tg = None
                    notes.append(f"skx6t {S_}->{D_}: {type(exc).__name__}: {exc}")
                if tg is not None:
                    log(f"lever used: skx6t (S1ENG_SKX6T, variant {tg[2]}) at {tg[0]} (N, K, R) keys of the {S_}->{D_} "
                        f"mixed graphs, bit-equal to skx6 at start-up in {time.time() - t2:.1f}s; {len(tg[1])} "
                        f"rejected (they keep skx6){': ' + ', '.join(tg[1][:8]) if tg[1] else ''}")
            # KFX (kai/kx-k2 #6): KF's mainloop for these mixed graphs' two-class GEMMs (S1ENG_KFX, default off), byte-gated
            # per (kernel, N, K, R, classes) against skx6 and the skx6t variant the key runs, before they are captured;
            # rejected keys keep skx6t / skx6
            if _mxm.kfx_on():
                t3 = time.time()
                try:
                    xg = pf.mx_kfx_gate(D_, S_, shapes, fz_list,
                                        sorted({r for rs_ in mixrows[(D_, S_)].values() for r in rs_}))
                except Exception as exc:  # noqa: BLE001 - the lever must never take MX down with it
                    xg = None   # the gate already dropped this pair's keys; earlier pairs' passed keys stay
                    notes.append(f"kfx {S_}->{D_}: {type(exc).__name__}: {exc}")
                else:
                    if xg is None:
                        notes.append(f"kfx {S_}->{D_}: kfx kernels unavailable (kfx.cubin / GPU)")
                if xg is not None:
                    log(f"lever used: kfx (S1ENG_KFX, {xg[2]}) at {xg[0]} (N, K, R) keys of the {S_}->{D_} mixed "
                        f"graphs, bit-equal to skx6 / skx6t at start-up in {time.time() - t3:.1f}s; {len(xg[1])} "
                        f"rejected (they keep skx6t / skx6){': ' + ', '.join(xg[1][:8]) if xg[1] else ''}")
        bf.ws = None
        from . import mx as _mxm

        if _mxm.kfx_on():   # KFX: FCAB pick lever kfx / nokfx (S1ENG_KFX_ARM=1: the env set stays on skx6t / skx6)
            _mxm.register_arms(log=log)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        log(f"MX mixed graphs for pairs {[f'{S_}->{D_}' for (D_, S_) in mixrows]}: "
            + ", ".join(f"{S_}->{D_} two-class shapes {sorted(pf.mxp[(D_, S_)].cls)} slots/R bins "
                        f"{ {B: len(r) for B, r in mixrows[(D_, S_)].items()} }" for (D_, S_) in mixrows)
            + f" ({time.time() - t1:.1f}s)" + (f"; lever not used for {'; '.join(notes)}" if notes else ""))
    if os.environ.get("S1ENG_RN", "0") == "1":
        # RN (ruj r2 hotspot, rn.py): the packed graphs' row norms (add + RMSNorm, RARN fold + add + RMSNorm, plain
        # RMSNorm) and the GDN gated norm through rn.cubin's single-pass kernels, where the start-up gate finds them
        # byte-equal to the Triton kernels they replace (every packed R and slot count, every RARN slice count in use)
        from . import rarn as _rarn
        from . import rn

        t1 = time.time()
        parts = rn.parts_env()
        notes = []
        rs = sorted({r for L_ in pack for rr in prow[L_].values() for r in rr} | set(range(1, 17)))
        ss = sorted({v[1] for v in pf.rarn.values() if _rarn.KERNELS.get(v[0], ("i",))[0] == "p"}) if pf.rarn else []
        if "norm" in parts:
            try:
                ws_ = [ff.fn_w] + [d_[k_] for d_ in ff.layers[:3] for k_ in ("ln1_w", "ln2_w")]
                why = rn.check_norm(ff.fused, _rarn, ws_, ff.eps, rs, ss)
            except Exception as exc:  # noqa: BLE001 - the lever must never take the batch lanes down with it
                why = f"{type(exc).__name__}: {exc}"
            if why:
                notes.append(f"norm: {why}")
            else:
                pf.rn = set(pack)
        if "gnorm" in parts:
            try:
                gl_ = [d_ for d_ in ff.layers if d_["type"] == "linear_attention"]
                gw_ = [gl_[0]["la"].norm.weight, gl_[-1]["la"].norm.weight]
                eps_g = gl_[0]["la"].norm.variance_epsilon
                cases = sorted({(B_, L_, R_) for L_ in pack for B_, rr in prow[L_].items() for R_ in rr})
                why = rn.check_gnorm(ff.fused, gw_, eps_g, cases)
            except Exception as exc:  # noqa: BLE001
                why = f"{type(exc).__name__}: {exc}"
            if why:
                notes.append(f"gnorm: {why}")
            else:
                pf.rng = set(pack)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        log(f"RN single-pass row norms for buckets {sorted(pf.rn)} (R {len(rs)} values, RARN slices {ss}), gated norm "
            f"for buckets {sorted(pf.rng)} (hpw {rn.ghpw()}): byte-equal to the Triton kernels in {time.time() - t1:.1f}s"
            + (f"; not used: {'; '.join(notes)}" if notes else ""))

    def used_gib():
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 2**30

    if getattr(pf, "xa_want", None):   # XA3: FCAB pick lever xa3 / noxa3 (after the NTV re-checks, before capture)
        from . import xa as xamod4

        xamod4.register_arms(pf, log=log)
    log(f"before lanes: device memory used {used_gib():.2f} GiB, torch reserved "
        f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB")
    lanes = []
    lane_cost = None   # measured cost of a lane WITHOUT the PAR side branches (PAR's extra is estimated separately)
    margin = float(os.environ.get("S1ENG_MEM_MARGIN_GIB", "2.0"))
    est0 = float(os.environ.get("S1ENG_LANE_EST_GIB", "1.5"))  # lane 0 cost before it is measured (5090: ~1.0 GiB)
    # PAR memory guard (rgh 10-21): PAR's side branches cost ~1.2 GiB per lane (graph branch resources + workspace). A lane is
    # captured with PAR only if that fits beside the usual margin; otherwise it is captured WITHOUT PAR (the v11 path, same
    # bits) instead of being skipped - with tight validator memory the engine keeps its lanes and loses only PAR
    par_extra = float(os.environ.get("S1ENG_PAR_EXTRA_GIB", "1.5"))
    par_all = pf.par
    for li in range(lanes_n):
        free_gib = torch.cuda.mem_get_info()[0] / 2**30
        base = lane_cost if lane_cost is not None else est0
        use_par = bool(par_all) and free_gib >= base + par_extra + margin
        if par_all and not use_par:
            log(f"lane {li}: PAR off for this lane (memory guard: {free_gib:.2f} GiB free < lane {base:.2f} + PAR "
                f"{par_extra:.2f} + margin {margin:.2f} GiB)")
        pf.par = par_all if use_par else type(par_all)()
        need = base + margin
        if free_gib < need:
            # the reference engine shares this GPU during the evaluation: never take the memory it may need
            log(f"lane {li} skipped: {free_gib:.2f} GiB free < estimated lane cost + margin {need:.2f} GiB")
            break
        used0 = used_gib()
        # lane 0 replays its batch graphs on the same stream as the per-decision graphs of set 0, never concurrently,
        # so they can share set 0's memory pool
        share = li == 0 and os.environ.get("S1ENG_SHARE_POOL", "1") == "1" and getattr(model, "set_pools", [None])[0]
        pool = model.set_pools[0] if share else torch.cuda.graph_pool_handle()
        # lane 0 captures on torch.cuda.graph's default capture stream (as set 0: same lane, never concurrent); every
        # other lane on a stream of its own (torch's cuBLAS workspace is per stream and baked into the graphs)
        cap_stream = None if li == 0 else fresh_stream(capture_streams_in_use(model) + [ln.cap_stream for ln in lanes])
        cap_eff = _default_capture_stream() if cap_stream is None else cap_stream
        others = [_default_capture_stream() if ln.cap_stream is None else ln.cap_stream for ln in lanes]
        assert all(int(cap_eff.cuda_stream) != int(o.cuda_stream) for o in others), "lane capture streams must differ"
        assert li == 0 or int(cap_eff.cuda_stream) != int(_default_capture_stream().cuda_stream)
        ws = torch.empty(max(pinned.ws_bytes, 1 << 20), dtype=torch.uint8, device="cuda")
        bf.ws = ws
        # PAR (rgh): the side branch of this lane's graphs gets a workspace of its own (both branches may run split-K
        # GEMMs at the same time); baked into the lane's graphs like ws
        ws_side = torch.empty_like(ws) if pf.par else None
        pf.ws_side = ws_side

        nwarm = int(os.environ.get("S1ENG_WARMUPS", "1"))

        kp_counts: list = []
        fc2_counts: dict = {}   # r2 fixedcost2: SKR prune / MSD counts over this lane's graphs
        from . import msd as _msdm
        from . import skr as _skrm

        def cap(fn, *args):
            # warm-up before capture (lazy initialisation: Triton compiles, cuBLAS handles); does not change what
            # is captured
            side = model.warm_stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(nwarm):
                    fn(*args)
            torch.cuda.current_stream().wait_stream(side)
            kp = kprio.mode()
            sk = bool(pf.skr)                    # r2 fixedcost2 SKR: splitKreduce nodes to prune before instantiation
            ms = _msdm.enabled() and li % 2 == 1   # r2 fixedcost2 MSD: lane 1's kernels in the REMOTE sync domain
            keep = bool(kp) or sk or ms
            graph = torch.cuda.CUDAGraph(keep_graph=True) if keep else torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=cap_stream):
                out = fn(*args)
            if sk:
                try:
                    _fc2_counts(fc2_counts, "skr", _skrm.prune(graph))
                except Exception as exc:  # noqa: BLE001 - the reduce node stays (the fold is exact either way)
                    _fc2_counts(fc2_counts, "skr", {"error": 1})
                    log(f"SKR prune failed: {type(exc).__name__}: {exc}")
            if ms:
                try:
                    _fc2_counts(fc2_counts, "msd", _msdm.apply(graph, li))
                except Exception as exc:  # noqa: BLE001 - scheduling only
                    _fc2_counts(fc2_counts, "msd", {"error": 1})
                    log(f"MSD unavailable: {type(exc).__name__}: {exc}")
            if kp:   # KP (rgh): per-kernel node priorities, then instantiate (scheduling only)
                try:
                    kp_counts.append(kprio.apply(graph, kp))
                except Exception as exc:  # noqa: BLE001 - e.g. an older driver without cuFuncGetName: no priorities
                    kp_counts.append({"gemm": 0, "small": 0, "unknown": 0, "set_failed": 0, "error": 1})
                    log(f"KP node priorities unavailable: {type(exc).__name__}: {exc}")
            if keep:
                graph.instantiate()
            GOV.tick()
            return graph, out

        batch, packed = {}, {}
        # largest graphs first: later (smaller) captures reuse the pool blocks of the earlier ones
        todo = []
        for L in buckets:
            if L in pack:
                todo += [("p", L, B, R) for B, rs in prow[L].items() for R in rs]
            elif L in bmax:
                todo += [("s", L, B, B * L) for B in range(1 if li > 0 else 2, bmax[L] + 1)]
        todo.sort(key=lambda t: -t[3])
        for kind, L, B, R in todo:
            if kind == "p":
                pfx = pfx_of.get(L, 0)
                meta = PackedMeta(B, L, R, pfx=pfx)
                st, sl, rp, la = packed_layout([min(L, R // B + pfx)] * B, B, L, R, pfx)
                meta.start.copy_(torch.from_numpy(st))
                meta.slen.copy_(torch.from_numpy(sl))
                meta.row_pos.copy_(torch.from_numpy(rp))
                meta.last.copy_(torch.from_numpy(la))
                graph, out = cap(pf, meta)
                packed.setdefault((L, B), []).append((R, graph, meta, out))
            else:
                ids = torch.zeros((B, L), dtype=torch.long, device="cuda")
                last = torch.full((B,), L - 1, dtype=torch.long, device="cuda")
                graph, out = cap(bf, ids, last)
                batch[(L, B)] = (graph, ids, last, out, ws)
        for v in packed.values():
            v.sort(key=lambda e: e[0])
        mixed = {}   # MX: (D, S, B) -> [(R, graph, meta, out), ...] ascending R
        mtodo = sorted(((D_, S_, B, R) for (D_, S_), rb in mixrows.items() for B, rs in rb.items() for R in rs),
                       key=lambda t: -t[3])
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
            graph, out = cap(pf, meta)
            mixed.setdefault((D_, S_, B), []).append((R, graph, meta, out))
        for v in mixed.values():
            v.sort(key=lambda e: e[0])
        torch.cuda.synchronize()
        single = model.graph_sets[0] if li == 0 else {}
        if li == 0:
            stream = torch.cuda.current_stream()
        else:
            stream = fresh_stream([ln.stream for ln in lanes] + [model.warm_stream()])
        assert all(int(stream.cuda_stream) != int(ln.stream.cuda_stream) for ln in lanes), "lane streams must differ"
        lanes.append(Lane(li, stream, single, batch, packed, ws=[ws] + ([ws_side] if ws_side is not None else []),
                          cap_stream=cap_stream,
                          pfx={L: pfx_of.get(L, 0) for L in pack}))
        lanes[-1].mixed = mixed   # MX
        lanes[-1].mbs = {}
        for (D_, S_, B) in mixed:
            lanes[-1].mbs.setdefault((D_, S_), []).append(B)
        for v in lanes[-1].mbs.values():
            v.sort()
        lane_cost = max(0.1, used_gib() - used0 - (par_extra if use_par else 0.0))
        if kp_counts:
            tot = {k: sum(c[k] for c in kp_counts) for k in kp_counts[0]}
            log(f"lane {li}: KP node priorities ({kprio.mode()} high) over {len(kp_counts)} graphs: {tot}")
        if fc2_counts:
            log(f"lane {li}: fixedcost2 graph edits (SKR prune / MSD) over its graphs: {fc2_counts}")
        log(f"lane {li}: {len(batch)} stacked + {sum(len(v) for v in packed.values())} packed + "
            f"{sum(len(v) for v in mixed.values())} mixed graphs, "
            f"{len(single)} single graphs, GPU memory {torch.cuda.memory_reserved() / 2**30:.2f} GiB reserved, "
            f"device used {used_gib():.2f} GiB at {time.time() - t0:.1f}s")
    pf.ws_side = None   # PAR: eager forwards after capture run on one stream
    pf.par = par_all
    if pf.par:
        log(f"PAR side-stream fork/join in the lane graphs: {sorted(pf.par)}")
    torch.cuda.empty_cache()
    # FCAB (ruj v14 fixedcost, timing harness, opt-in): what fcab.build needs to capture more lane sets the same way
    model._fcab_ctx = dict(buckets=buckets, pack=pack, prow=prow, pfx_of=pfx_of, pinned=pinned, bf=bf, pf=pf,
                           par_all=par_all)
    return lanes


def setup_grp(model: Model, pf, pinned, pack: dict, log=print) -> None:
    """GRP start-up (S1ENG_GRP=1): the plan's picks per bucket whose split class matches the bucket's pins, checked
    bit for bit (PackedForward.check_grp); sets pf.grp. Any failure drops only the failing picks."""
    from . import grp as grpmod
    from .lt import cfg_of

    t1 = time.time()
    if grpmod.lib() is None:
        log("lever not used: grp (kernels unavailable: GPU, cubin or S1ENG_GRP)")
        return
    plan = grpmod.load_plan(log=log)
    if not plan:
        return
    ff = model.fwd
    gdn = [d for d in ff.layers if d["type"] == "linear_attention"]
    att = [d for d in ff.layers if d["type"] != "linear_attention"]
    gl = [gdn[1], gdn[len(gdn) // 2], gdn[-1]] if len(gdn) > 2 else gdn
    al = [att[0], att[-2]] if len(att) > 2 else att
    la, sa = gdn[0]["la"], att[0]["sa"]
    K = la.in_proj_qkv.weight.shape[1]

    def cls_of(*keys):
        cs = set()
        for k in keys:
            a = pinned.algo.get(k)
            if a is None:
                continue
            c = cfg_of(a)
            cs.add(grpmod.chunk_of(int(c.get("splitK", 1)), int(c.get("reduction", 0)), k[1]))
        return cs.pop() if len(cs) == 1 else None

    def unsplit(*keys):
        return all(pinned.algo.get(k) is not None and int(cfg_of(pinned.algo[k]).get("splitK", 1)) <= 1 for k in keys)

    nq, nz = la.in_proj_qkv.weight.shape[0], la.in_proj_z.weight.shape[0]
    nb2 = la.in_proj_b.weight.shape[0] + la.in_proj_a.weight.shape[0]
    nqa, nk = sa.q_proj.weight.shape[0], sa.k_proj.weight.shape[0]
    has = {"qkvz": all(d["fused"].get("qkv_z") is not None for d in gdn),
           "ba": all(d["fused"].get("b_a") is not None for d in gdn),
           "kv": all(d["fused"].get("k_v") is not None for d in att)}
    bf = pf.bf
    bf.ws = torch.empty(max(pinned.ws_bytes, 1 << 20), dtype=torch.uint8, device="cuda")
    notes, on, nrej = [], {}, [0]
    for L in sorted(pack):
        cba = cls_of((nb2 // 2, K, L), (nb2, K, L))
        ckv = cls_of((nk, K, L), (2 * nk, K, L))
        cand = {}
        for (site, Lp, R), p in plan.items():
            if Lp != L:
                continue
            if site == "late":   # LATE (scheduling only): the PAR sites of (L, R) whose side node goes after the main's
                if p:
                    pf.late_r[(L, R)] = set(p)
                continue
            if site == "gdn":
                if not has["ba"] or cba is None or grpmod.x_mode(p.name) != cba[0]:
                    continue
                if p.main == "qkvz" and not (has["qkvz"] and unsplit((nq + nz, K, L), (nq, K, L), (nz, K, L))
                                             and grpmod.supported_main(p.name, nq + nz, K)):
                    continue
                if p.main in ("qkv", "qkv2", "qkv3") and not (unsplit((nq, K, L)) and grpmod.supported_main(p.name, nq, K)):
                    continue
                p.chunk2 = cba[1]
            elif site == "att":
                if not has["kv"] or ckv is None or grpmod.x_mode(p.name) != ckv[0] or not unsplit((nqa, K, L)) \
                        or not grpmod.supported_main(p.name, nqa, K):
                    continue
                p.chunk2 = ckv[1]
            else:
                continue
            cand[(site, R)] = p
        nplan = sum(1 for (s_, Lp, _R) in plan if Lp == L and s_ in ("gdn", "att"))
        if nplan > len(cand):   # picks whose split class / shape does not match this bucket's pins (count as rejected)
            notes.append(f"{L}: {nplan - len(cand)} picks rejected (class/shape)")
        nrej[0] += nplan - len(cand)
        bad = pf.check_grp(L, cand, gl, al)
        nrej[0] += len(bad)
        for (site, R), p in cand.items():
            if (site, R) not in bad:
                pf.grp[(site, L, R)] = p
                on[site] = on.get(site, 0) + 1
        if bad:
            notes.append(f"{L}: {len(bad)} picks failed ({next(iter(bad.values()))})")
    bf.ws = None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    log(f"GRP grouped projection GEMMs on at {sum(on.values())} (site, bucket, R) {on}: bit-equal to the selected "
        f"path and per-decision F.linear in {time.time() - t1:.1f}s; LATE side-node order at {len(pf.late_r)} (bucket, R); "
        f"{nrej[0]} plan picks rejected" + (f"; {'; '.join(notes)}" if notes else ""))


UPLOAD_L = {192: 0, 256: 1, 128: 2, 320: 3}          # E4 graph upload order: buckets by traffic share,
UPLOAD_B = (5, 4, 6, 3, 7, 8, 2, 1, 12, 16)            # then packed B slots by how often c16 serving launches them


def upload_graphs(lanes: list, log=print, mixed: bool = False) -> int:
    """E4 (S1ENG_GRAPH_UPLOAD): upload the packed graphs' executables to the device at start-up (cuGraphUpload on the
    lane's stream) instead of inside their first replay, which would otherwise happen in a timed run for most of them
    (the self-check replays one R per (lane, bucket, B)). Order: UPLOAD_L, UPLOAD_B, then R; stops once the device
    memory the uploads took exceeds S1ENG_GRAPH_UPLOAD_MIB (default 192). Scheduling only: no kernel, argument or
    buffer changes. Returns the number of graphs uploaded."""
    import ctypes as C

    budget = float(os.environ.get("S1ENG_MX_UPLOAD_MIB" if mixed else "S1ENG_GRAPH_UPLOAD_MIB",
                                  "160" if mixed else "192")) * 2**20
    rank_b = {b: i for i, b in enumerate(UPLOAD_B)}
    if mixed:   # MX: mixed graphs, by slot rank then R
        todo = sorted(((UPLOAD_L.get(D, 9), rank_b.get(B, 99), e[0], ln.idx, ln, e[1])
                       for ln in lanes for (D, S, B), es in ln.mixed.items() for e in es), key=lambda t: t[:4])
    else:
        todo = sorted(((UPLOAD_L.get(L, 9), rank_b.get(B, 99), e[0], ln.idx, ln, e[1])
                       for ln in lanes for (L, B), es in ln.packed.items() for e in es), key=lambda t: t[:4])
    t0 = time.time()
    torch.cuda.synchronize()
    free0 = torch.cuda.mem_get_info()[0]
    n, used, why = 0, 0, "all"
    try:
        cu = kprio._cu()
        for *_k, ln, graph in todo:
            st = cu.cuGraphUpload(C.c_void_p(graph.raw_cuda_graph_exec()), C.c_void_p(ln.stream.cuda_stream))
            if st != 0:
                why = f"cuGraphUpload error {st}"
                break
            n += 1
            if n % 16 == 0:
                torch.cuda.synchronize()
                used = free0 - torch.cuda.mem_get_info()[0]
                if used > budget:
                    why = f"memory budget {budget / 2**20:.0f} MiB reached"
                    break
        torch.cuda.synchronize()
        used = free0 - torch.cuda.mem_get_info()[0]
    except Exception as exc:  # noqa: BLE001 - an upload problem costs the lever only (graphs upload at first replay)
        why = f"{type(exc).__name__}: {exc}"
    log(f"graph upload: {n}/{len(todo)} {'mixed' if mixed else 'packed'} graphs uploaded at start-up ({why}), device memory "
        f"{used / 2**20:+.0f} MiB, {time.time() - t0:.1f}s")
    return n


def lanes_from_env(model: Model, log=print) -> list:
    """S1ENG_BATCH="128:8,192:16,..." (S4 stacked buckets and their max B), S1ENG_PACK="192,256" (S5 packed
    buckets), S1ENG_PACKB="1,2,3,4,6,8,12,16" (packed B slots), S1ENG_RSTEP="64" or "64/128" (R step for B <= 8 / B > 8),
    S1ENG_LANES (number of lanes)."""
    cap = parse_bmax(os.environ.get("S1ENG_PACKB_CAP", ""))   # E1: "128:8,320:8" -> fewer B slots for rare buckets
    pack = {int(L): [int(b) for b in os.environ.get("S1ENG_PACKB", "1,2,3,4,6,8,12,16").split(",")
                     if int(b) <= cap.get(int(L), 1 << 30)]
            for L in os.environ.get("S1ENG_PACK", "").split(",") if L}
    rs = [int(x) for x in os.environ.get("S1ENG_RSTEP", "64").split("/")]
    lanes = capture_batch(model, int(os.environ.get("S1ENG_LANES", "2")), parse_bmax(os.environ.get("S1ENG_BATCH", "")),
                          log=log, pack=pack, rstep=lambda B: rs[0] if B <= 8 or len(rs) == 1 else rs[1],
                          prune=os.environ.get("S1ENG_PRUNE", "0") == "1", prefix=getattr(model, "prefix_ids", None))
    model.join_ok = {}
    if os.environ.get("S1ENG_SELFCHECK", "1") == "1":
        self_check(model, lanes, log=log)
        mxp = getattr(getattr(model, "pf", None), "mxp", None) or {}
        if mxp and lanes:   # MX: a mixed pair is used only when this start-up check passed
            try:
                model.join_ok = mix_check(model, lanes, [(S_, D_) for (D_, S_) in sorted(mxp)], log=log)
            except Exception as exc:  # noqa: BLE001 - a failed check costs the lever only
                log(f"lever not used: mix ({type(exc).__name__}: {exc})")
                model.join_ok = {}
        joins = {s_: d_ for s_, d_ in join_wanted().items() if (s_, d_) not in model.join_ok}
        if joins and lanes:   # BB4: a cross-bucket join is used only when this start-up check passed
            try:
                model.join_ok.update(join_check(model, lanes, sorted(joins.items()), log=log))
            except Exception as exc:  # noqa: BLE001 - a failed check costs the lever only
                log(f"lever not used: join ({type(exc).__name__}: {exc})")
    l0t = getattr(getattr(model, "pf", None), "l0t", None)
    if l0t is not None and lanes:   # E4: fill the L0T tables after the self-check emptied them (S1ENG_L0T_PREFILL)
        from . import l0t as l0tmod

        rows_of = {}
        for (L, _B), es in lanes[0].packed.items():
            rows_of.setdefault(L, []).extend(e[0] for e in es)
        l0tmod.prefill(l0t, model.pf, rows_of, log=log)
    if lanes and os.environ.get("S1ENG_GRAPH_UPLOAD", "0") == "1":   # E4: no implicit upload inside a timed run
        upload_graphs(lanes, log=log)
        if any(ln.mixed for ln in lanes):   # MX: the mixed graphs too (their own budget)
            upload_graphs(lanes, log=log, mixed=True)
    if lanes and os.environ.get("S1ENG_FCAB"):   # FCAB (ruj v14 fixedcost): in-situ A/B lane sets (timing harness)
        from . import fcab

        fcab.build(model, lanes, log=log)
    return lanes


def prefix_variants_of(model, prefix: list) -> list:
    """PV (a4): the cached prefix variants: the template prefix first, then every longer prefix of
    model.prefix_variants that starts with it (each must leave at least one token in a bucket)."""
    out = [list(prefix)]
    for v in getattr(model, "prefix_variants", None) or []:
        v = list(v)
        if len(v) > len(prefix) and v[:len(prefix)] == list(prefix) and v not in out:
            out.append(v)
    return out


def pick_variant(ids, variants_np) -> int:
    """Index of the longest variant ids starts with (and outgrows), or -1."""
    best, best_len = -1, -1
    for vi, v in enumerate(variants_np):
        k = len(v)
        if len(ids) > k > best_len and bool(np.array_equal(ids[:k], v)):
            best, best_len = vi, k
    return best


if _ntv():   # NTV (kai), the flag read once at import (flag off: pick_variant is hotspot's function, no per-request work)
    _scan = pick_variant

    def pick_variant(ids, variants_np) -> int:
        """NTV: an ntv.Index walks its token trie (the scan's answer, tests/test_ntv.py); anything else hotspot's scan."""
        if hasattr(variants_np, "pick"):
            return variants_np.pick(ids)
        return _scan(ids, variants_np)


@torch.inference_mode()
def check_prefix(model: Model, pf, L: int, prow_l: dict, prefix: list, seed: int = 93, variants=None) -> str:
    """E1 start-up gate of the template prefix for bucket L, before any graph is captured: random decisions that start
    with the template run eagerly through prefix-packed forwards (every B slot, full and smallest groups) and one by
    one through the per-decision graph of set 0. Returns "" if every logit is bit-equal, else the first difference."""
    import numpy as np

    from .batchfwd import PackedMeta, packed_layout

    P = len(prefix)
    g = torch.Generator().manual_seed(seed + L)
    vocab = model.hf.config.vocab_size
    lprev = max([x for x in GRAPH_LENGTHS if x < L], default=0)
    variants = variants or [list(prefix)]
    heads = [np.asarray(v, dtype=np.int64) for v in variants]
    graph, s_ids, s_last, out1 = model.graph_sets[0][L]
    bs = sorted(prow_l)
    prev = 0
    for B in bs:
        for k in sorted({prev + 1, B}):
            seqs, vis = [], []
            for j in range(k):
                vi = (j + int(torch.randint(0, len(heads), (1,), generator=g))) % len(heads)
                head = heads[vi]
                n = int(torch.randint(max(lprev + 1, len(head) + 1), L + 1, (1,), generator=g))
                tail = torch.randint(10, min(vocab, 150000), (n - len(head),), generator=g).numpy()
                seqs.append(np.concatenate([head, tail]))
                vis.append(vi)
            ref = []
            for ids in seqs:
                s_ids.zero_()
                s_ids[0, :len(ids)] = torch.from_numpy(ids).cuda()
                s_last.fill_(len(ids) - 1)
                graph.replay()
                ref.append(out1[0].clone())
            pls = [len(heads[vi]) for vi in vis]
            T = sum(len(x) - pl for x, pl in zip(seqs, pls))
            R = next((r for r in prow_l[B] if r >= T), None)
            if R is None:
                return f"no R bin for B={B} T={T}"
            m = PackedMeta(B, L, R, pfx=P)
            st, sl, rp, la = packed_layout([len(x) for x in seqs], B, L, R, P, plens=pls)
            ids = torch.zeros((R,), dtype=torch.long)
            ids[:T] = torch.from_numpy(np.concatenate([x[pl:] for x, pl in zip(seqs, pls)]))
            m.ids.copy_(ids)
            m.start.copy_(torch.from_numpy(st))
            m.slen.copy_(torch.from_numpy(sl))
            m.row_pos.copy_(torch.from_numpy(rp))
            m.last.copy_(torch.from_numpy(la))
            m.pvar[:k].copy_(torch.tensor(vis, dtype=torch.int32))
            m.plen[:k].copy_(torch.tensor(pls, dtype=torch.int32))
            got = pf(m)
            torch.cuda.synchronize()
            GOV.tick()
            for r in range(k):
                if not torch.equal(got[r].view(torch.int16), ref[r].view(torch.int16)):
                    return f"B={B} R={R}: decision {r} of {k} (length {len(seqs[r])}) differs from the per-decision graph"
        prev = B
    return ""


@torch.inference_mode()
def self_check(model: Model, lanes: list, log=print, seed: int = 92, zc=None, drop: bool = True) -> dict:
    """Start-up check on this GPU: random-token decisions of every batch bucket, computed one by one through the
    per-decision graph (set 0, the reference-identical path) and inside batch graphs of every lane (several B, random
    companions). A bucket whose batch graphs disagree in any bit on any lane is dropped from all lanes (those decisions
    then run per decision). Returns {bucket: ok}.
    ZC-IN gate (zc = a zcin.Zc, drop=False): the packed blob graphs get their inputs as served (stage_blob into pinned
    staging, then the copy kernel on the lane stream) instead of the per-field copies; a failure drops nothing."""
    from .batchfwd import packed_layout

    import numpy as np

    g = torch.Generator().manual_seed(seed)
    vocab = model.hf.config.vocab_size
    single = model.graph_sets[0]
    res = {}
    buckets = sorted({L for ln in lanes for L in list(ln.pbs) + list(ln.bs)})
    prefix = np.asarray(getattr(model, "prefix_ids", None) or [], dtype=np.int64)
    variants = [np.asarray(v, dtype=np.int64) for v in prefix_variants_of(model, list(prefix))] if len(prefix) else []
    l0t = getattr(getattr(model, "pf", None), "l0t", None)   # L0T (kai)
    if zc is not None:   # ZC-IN: pinned staging the copy kernel reads through UVA
        from . import zcin as zcmod

        nz = zcmod.meta_sizes([lanes])
        zv64 = torch.zeros((max([1 << 15] + [a for a, _ in nz]),), dtype=torch.long, pin_memory=True)
        zv32 = torch.zeros((max([1 << 15] + [b for _, b in nz]),), dtype=torch.int32, pin_memory=True)
        zp64, zp32 = zc.dev_ptr(zv64.data_ptr()), zc.dev_ptr(zv32.data_ptr())
        if not zp64 or not zp32:
            raise RuntimeError("ZC check: staging not device-accessible")
    for L in buckets:
        if _ntv() and len(prefix):   # NTV (kai): bucket L's own cached variants (the nonce-token ones only in NTV's
            # buckets), so no self-check slot gets a pvar its bucket's pcache does not hold
            try:
                from . import ntv as ntvmod

                variants = [np.asarray(v, dtype=np.int64)
                            for v in ntvmod.bucket_variants(model, list(prefix), L, prefix_variants_of)]
            except Exception:  # noqa: BLE001 - the variants every bucket caches
                variants = variants[:getattr(model, "ntv_common", len(variants))]
        lprev = max([x for x in GRAPH_LENGTHS if x < L], default=0)
        P = lanes[0].pfx.get(L, 0)          # E1: the same on every lane
        n_seq = max([max(ln.pbs.get(L, [1])) for ln in lanes] + [max(ln.bs.get(L, [1])) for ln in lanes])
        seqs, svar = [], []
        for j in range(n_seq):
            vi = (j + int(torch.randint(0, max(1, len(variants)), (1,), generator=g))) % max(1, len(variants))
            head = variants[vi] if P else prefix[:0]
            n = int(torch.randint(max(lprev + 1, len(head) + 1), L + 1, (1,), generator=g))
            tail = torch.randint(10, min(vocab, 150000), (n - len(head),), generator=g).numpy()
            seqs.append(np.concatenate([head, tail]) if P else tail)
            svar.append(vi if P else 0)
        ref = []
        for ids in seqs:
            graph, s_ids, s_last, out = single[L]
            s_ids.zero_()
            s_ids[0, :len(ids)] = torch.from_numpy(ids).cuda()
            s_last.fill_(len(ids) - 1)
            graph.replay()
            ref.append(out[0].clone())
        torch.cuda.synchronize()
        ok = True
        for ln in lanes:
            ln.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(ln.stream):
                if L in ln.pbs:
                    for B in ln.pbs[L]:
                        k = min(B, n_seq)
                        order = torch.randperm(n_seq, generator=g)[:k].tolist()
                        grp = [seqs[i] for i in order]
                        pls = [len(variants[svar[i]]) for i in order] if P else [0] * len(order)
                        T = sum(len(x) - pl for x, pl in zip(grp, pls))
                        R, graph, meta, out = next(e for e in ln.packed[(L, B)] if e[0] >= T)
                        st, sl, rp, la = packed_layout([len(x) for x in grp], B, L, R, P, plens=pls if P else None)
                        if zc is not None and meta.blob:   # ZC-IN: the serving staging, copied by the kernel
                            v64n, v32n = zv64.numpy(), zv32.numpy()
                            zjobs = [_MixJob(seqs[i], svar[i] if P else 0, pl_) for i, pl_ in zip(order, pls)]
                            idsv = stage_blob(meta, v64n, v32n, zjobs, st, sl, rp, la)
                            if l0t is not None and L in l0t.books:
                                o, n = meta.lay64["l0t_slot"]
                                v64n[o:o + n] = l0t.prepare(L, idsv)
                            if zc.copy_meta(meta, zp64, zp32, ln.stream.cuda_stream):
                                raise RuntimeError("ZC check: copy kernel launch failed")
                        else:
                            ids = torch.zeros((R,), dtype=torch.long)
                            ids[:T] = torch.from_numpy(np.concatenate([x[pl:] for x, pl in zip(grp, pls)]))
                            meta.ids.copy_(ids)
                            if l0t is not None and L in l0t.books:   # L0T (kai): misses inserted on this lane's stream
                                meta.l0t_slot.copy_(torch.from_numpy(l0t.prepare(L, ids.numpy())))
                            meta.start.copy_(torch.from_numpy(st))
                            meta.slen.copy_(torch.from_numpy(sl))
                            meta.row_pos.copy_(torch.from_numpy(rp))
                            meta.last.copy_(torch.from_numpy(la))
                            if P:
                                pv = torch.zeros((B,), dtype=torch.int32)
                                pl = torch.full((B,), P, dtype=torch.int32)
                                pv[:len(order)] = torch.tensor([svar[i] for i in order], dtype=torch.int32)
                                pl[:len(order)] = torch.tensor(pls, dtype=torch.int32)
                                meta.pvar.copy_(pv)
                                meta.plen.copy_(pl)
                        graph.replay()
                        got = out.clone()
                        ln.stream.synchronize()
                        GOV.tick()
                        for r, i in enumerate(order):
                            if not torch.equal(got[r].view(torch.int16), ref[i].view(torch.int16)):
                                ok = False
                for B in ln.bs.get(L, ()):
                    order = [int(torch.randint(0, n_seq, (1,), generator=g)) for _ in range(B)]
                    graph, s_ids, s_last, out, _ws = ln.batch[(L, B)]
                    s_ids.zero_()
                    for r, i in enumerate(order):
                        s_ids[r, :len(seqs[i])] = torch.from_numpy(seqs[i]).cuda()
                        s_last[r] = len(seqs[i]) - 1
                    graph.replay()
                    got = out.clone()
                    ln.stream.synchronize()
                    GOV.tick()
                    for r, i in enumerate(order):
                        if not torch.equal(got[r].view(torch.int16), ref[i].view(torch.int16)):
                            ok = False
            torch.cuda.synchronize()
        res[L] = ok
        if not ok and not drop:   # ZC-IN gate: nothing dropped (the copy kernel stays off)
            log(f"zc check failed for bucket {L} (copy kernel inputs; graphs kept, zc stays off)")
        elif not ok:
            log(f"SELF-CHECK FAILED for bucket {L}: batch graphs disabled for it (per-decision path)")
            for ln in lanes:
                for key in [k for k in ln.packed if k[0] == L]:
                    del ln.packed[key]
                for key in [k for k in ln.batch if k[0] == L]:
                    del ln.batch[key]
                ln.pbs.pop(L, None)
                ln.bs.pop(L, None)
    log(f"self-check (batch vs per-decision bits on random decisions): {res}")
    if l0t is not None:   # L0T (kai): serving starts with empty tables (not the self-check's random tokens)
        l0t.reset_all()
    return res


@torch.inference_mode()
def join_check(model: Model, lanes: list, pairs: list, log=print, seed: int = 94, mixes: int = 3) -> dict:
    """BB4 (S1ENG_JOIN) start-up gate of a cross-bucket join S -> D on this GPU: random decisions of bucket S
    (lprev(S)+1..S tokens, random prefix variants) ride inside bucket D's packed graphs of every lane - every B slot
    >= 2, `mixes` groups per slot of 1..B-1 S decisions among random D companions in random order - and every output
    row is compared in all bits with the per-decision graph of the decision's OWN bucket (set 0, the
    reference-identical path, as in self_check). A pair passes only if every row of every graph on every lane is
    equal; otherwise it is never used. Returns {(S, D): ok}."""
    from .batchfwd import packed_layout

    g = torch.Generator().manual_seed(seed)
    vocab = model.hf.config.vocab_size
    single = model.graph_sets[0]
    prefix = np.asarray(getattr(model, "prefix_ids", None) or [], dtype=np.int64)
    variants = [np.asarray(v, dtype=np.int64) for v in prefix_variants_of(model, list(prefix))] if len(prefix) else []
    l0t = getattr(getattr(model, "pf", None), "l0t", None)   # L0T (kai)
    res = {}

    def draw(L, k, P):
        lprev = max([x for x in GRAPH_LENGTHS if x < L], default=0)
        seqs, svar = [], []
        for _ in range(k):
            vi = int(torch.randint(0, max(1, len(variants)), (1,), generator=g)) if P else 0
            head = variants[vi] if P else prefix[:0]
            n = int(torch.randint(max(lprev + 1, len(head) + 1), L + 1, (1,), generator=g))
            tail = torch.randint(10, min(vocab, 150000), (n - len(head),), generator=g).numpy()
            seqs.append(np.concatenate([head, tail]) if P else tail)
            svar.append(vi)
        return seqs, svar

    def refs(L, seqs):
        out = []
        graph, s_ids, s_last, o = single[L]
        for ids in seqs:
            s_ids.zero_()
            s_ids[0, :len(ids)] = torch.from_numpy(ids).cuda()
            s_last.fill_(len(ids) - 1)
            graph.replay()
            out.append(o[0].clone())
        return out

    for S, D in pairs:
        P = lanes[0].pfx.get(D, 0)
        if (S not in single or D not in single or any(D not in ln.pbs for ln in lanes)
                or any(ln.pfx.get(S, 0) != P for ln in lanes)):
            res[(S, D)] = False
            log(f"join {S}->{D}: not checked (bucket graphs or prefix layout differ): not used")
            continue
        nmax = max(max(ln.pbs[D]) for ln in lanes)
        s_seqs, s_var = draw(S, nmax, P)
        d_seqs, d_var = draw(D, nmax, P)
        s_ref, d_ref = refs(S, s_seqs), refs(D, d_seqs)
        torch.cuda.synchronize()
        ok, ngraph, nrows, nbad = True, 0, 0, 0
        for ln in lanes:
            ln.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(ln.stream):
                for B in ln.pbs[D]:
                    if B < 2:
                        continue
                    for _m in range(mixes):
                        ns = int(torch.randint(1, B, (1,), generator=g))
                        grp = ([("s", i) for i in torch.randperm(nmax, generator=g)[:ns].tolist()]
                               + [("d", i) for i in torch.randperm(nmax, generator=g)[:B - ns].tolist()])
                        grp = [grp[p] for p in torch.randperm(len(grp), generator=g).tolist()]
                        seqs = [s_seqs[i] if k == "s" else d_seqs[i] for k, i in grp]
                        vs = [s_var[i] if k == "s" else d_var[i] for k, i in grp]
                        pls = [len(variants[v]) for v in vs] if P else [0] * len(grp)
                        T = sum(len(x) - pl for x, pl in zip(seqs, pls))
                        R, graph, meta, out = next(e for e in ln.packed[(D, B)] if e[0] >= T)
                        st, sl, rp, la = packed_layout([len(x) for x in seqs], B, D, R, P, plens=pls if P else None)
                        ids = torch.zeros((R,), dtype=torch.long)
                        ids[:T] = torch.from_numpy(np.concatenate([x[pl:] for x, pl in zip(seqs, pls)]))
                        meta.ids.copy_(ids)
                        if l0t is not None and D in l0t.books:   # L0T (kai): misses inserted on this lane's stream
                            meta.l0t_slot.copy_(torch.from_numpy(l0t.prepare(D, ids.numpy())))
                        meta.start.copy_(torch.from_numpy(st))
                        meta.slen.copy_(torch.from_numpy(sl))
                        meta.row_pos.copy_(torch.from_numpy(rp))
                        meta.last.copy_(torch.from_numpy(la))
                        if P:
                            pv = torch.zeros((B,), dtype=torch.int32)
                            plt = torch.full((B,), P, dtype=torch.int32)
                            pv[:len(vs)] = torch.tensor(vs, dtype=torch.int32)
                            plt[:len(pls)] = torch.tensor(pls, dtype=torch.int32)
                            meta.pvar.copy_(pv)
                            meta.plen.copy_(plt)
                        graph.replay()
                        got = out.clone()
                        ln.stream.synchronize()
                        GOV.tick()
                        ngraph += 1
                        for r, (k, i) in enumerate(grp):
                            ref = s_ref[i] if k == "s" else d_ref[i]
                            nrows += 1
                            if not torch.equal(got[r].view(torch.int16), ref.view(torch.int16)):
                                ok = False
                                nbad += 1
            torch.cuda.synchronize()
        res[(S, D)] = ok and ngraph > 0
        log(f"join {S}->{D} check (bucket-{S} decisions inside bucket-{D} packed graphs vs their own per-decision "
            f"graphs, every bit): {'PASS' if res[(S, D)] else 'FAIL: not used'} ({ngraph} graphs, {nrows} rows, "
            f"{nbad} differ)")
    if l0t is not None:   # serving starts with empty tables (not this check's random tokens)
        l0t.reset_all()
    return res


def stage_mix(meta, v64, v32, jobs: list, nd: int, st, sl, rp, la, voff: int) -> tuple:
    """MX: stage_blob for a mixed graph - jobs = the nd bucket-D jobs then the bucket-S jobs (slot order); S slots read
    their prefix variant from the S half of the joint caches (pvar + voff); mx = [r1 (first S row), nd]. -> (ids, r1)"""
    ids = stage_blob(meta, v64, v32, jobs, st, sl, rp, la)
    P = meta.pfx
    o, n = meta.lay32["pvar"]
    if P:
        for b in range(nd, len(jobs)):
            v32[o + b] += voff
    r1 = sum(j.n - (j.pl if P else 0) for j in jobs[:nd])
    o, n = meta.lay32["mx"]
    v32[o] = r1
    v32[o + 1] = nd
    return ids, r1


def mix_l0t(l0t, offs: dict, D: int, S: int, ids, r1: int) -> np.ndarray:
    """MX: absolute L0T slots of a mixed graph's rows: rows [0, r1) in bucket D's table, the rest (S rows and the
    padding) in bucket S's; misses are inserted on the current stream (Table.prepare)"""
    a = l0t.prepare(D, ids[:r1]) + offs[D] if r1 > 0 else np.zeros(0, dtype=np.int64)
    b = l0t.prepare(S, ids[r1:]) + offs[S] if len(ids) > r1 else np.zeros(0, dtype=np.int64)
    return np.concatenate([a, b])


class _MixJob:
    """MX check: the fields of a Job that the staging reads"""
    __slots__ = ("ids", "n", "pv", "pl")

    def __init__(self, ids, pv, pl):
        self.ids, self.n, self.pv, self.pl = ids, len(ids), pv, pl


@torch.inference_mode()
def mix_check(model: Model, lanes: list, pairs: list, log=print, seed: int = 95, mixes: int = 3, zc=None) -> dict:
    """MX start-up gate of each mixed pair S -> D on this GPU: random decisions of buckets D and S (random prefix
    variants, lprev+1..L tokens) in the pair's mixed graphs of every lane - every B slot, `mixes` mixes per slot
    (1..B-1 S decisions, plus one all-D and one all-S group per lane), staged by the serving code (stage_mix /
    mix_l0t) - and every output row compared in all bits with the per-decision graph of the decision's own bucket
    (set 0, the reference-identical path). A pair passes only if every row of every graph on every lane is equal;
    otherwise it is never used. Returns {(S, D): ok}. ZC-IN gate (zc = a zcin.Zc): the staged blobs reach the device
    through the copy kernel on the lane stream instead of the two H2D copies."""
    g = torch.Generator().manual_seed(seed)
    vocab = model.hf.config.vocab_size
    single = model.graph_sets[0]
    pf = model.pf
    prefix = np.asarray(getattr(model, "prefix_ids", None) or [], dtype=np.int64)
    variants = [np.asarray(v, dtype=np.int64) for v in prefix_variants_of(model, list(prefix))] if len(prefix) else []
    l0t = getattr(pf, "l0t", None)
    offs = getattr(pf, "l0t_off", {})
    res = {}
    nl = len(lanes)

    def nvar_of(L):
        """S1 (MX x NTV): the variants bucket L's pcache holds (NTV buckets: all; the others: hotspot's common ones)"""
        if not _ntv() or not len(prefix):
            return len(variants)
        try:
            from . import ntv as ntvmod

            return len(ntvmod.bucket_variants(model, list(prefix), L, prefix_variants_of))
        except Exception:  # noqa: BLE001 - the variants every bucket caches
            return min(len(variants), getattr(model, "ntv_common", len(variants)))

    def draw(L, k, P):
        lprev = max([x for x in GRAPH_LENGTHS if x < L], default=0)
        out = []
        nv = nvar_of(L)
        for _ in range(k):
            vi = int(torch.randint(0, max(1, nv), (1,), generator=g)) if P else 0
            head = variants[vi] if P else prefix[:0]
            n = int(torch.randint(max(lprev + 1, len(head) + 1), L + 1, (1,), generator=g))
            tail = torch.randint(10, min(vocab, 150000), (n - len(head),), generator=g).numpy()
            ids = np.concatenate([head, tail]) if P else tail
            out.append(_MixJob(ids, vi, len(head) if P else 0))
        return out

    def refs(L, jobs):
        out = []
        graph, s_ids, s_last, o = single[L]
        for j in jobs:
            s_ids.zero_()
            s_ids[0, :j.n] = torch.from_numpy(j.ids).cuda()
            s_last.fill_(j.n - 1)
            graph.replay()
            out.append(o[0].clone())
        return out

    v64 = torch.zeros((1 << 15,), dtype=torch.long, pin_memory=True)
    v32 = torch.zeros((1 << 15,), dtype=torch.int32, pin_memory=True)
    if zc is not None:   # ZC-IN: the kernel reads this staging through UVA
        zp64, zp32 = zc.dev_ptr(v64.data_ptr()), zc.dev_ptr(v32.data_ptr())
        if not zp64 or not zp32:
            raise RuntimeError("ZC check: staging not device-accessible")
    for S, D in pairs:
        mp = pf.mxp.get((D, S))
        P = lanes[0].pfx.get(D, 0)
        if mp is None or any(not ln.mbs.get((D, S)) for ln in lanes):
            res[(S, D)] = False
            log(f"mix {S}->{D}: not checked (no mixed graphs on every lane): not used")
            continue
        nmax = max(max(ln.mbs[(D, S)]) for ln in lanes)
        sj, dj = draw(S, nmax, P), draw(D, nmax, P)
        s_ref, d_ref = refs(S, sj), refs(D, dj)
        torch.cuda.synchronize()
        ok, ngraph, nrows, nbad = True, 0, 0, 0
        for ln in lanes:
            ln.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(ln.stream):
                cases = []
                for B in ln.mbs[(D, S)]:
                    for _m in range(mixes):
                        ns = int(torch.randint(1, B, (1,), generator=g))
                        cases.append((B, ns, B - ns))
                bmax = ln.mbs[(D, S)][-1]
                cases += [(bmax, 0, bmax), (bmax, bmax, 0), (ln.mbs[(D, S)][0], 1, 0)]
                for B, ns, nd in cases:
                    di = torch.randperm(nmax, generator=g)[:nd].tolist()
                    si = torch.randperm(nmax, generator=g)[:ns].tolist()
                    jobs = [dj[i] for i in di] + [sj[i] for i in si]
                    want = [d_ref[i] for i in di] + [s_ref[i] for i in si]
                    T = sum(j.n - (j.pl if P else 0) for j in jobs)
                    R, graph, meta, out = next(e for e in ln.mixed[(D, S, B)] if e[0] >= T)
                    from .batchfwd import packed_layout

                    st, sl, rp, la = packed_layout([j.n for j in jobs], B, D, R, P,
                                                   plens=[j.pl for j in jobs] if P else None)
                    v64n, v32n = v64.numpy(), v32.numpy()
                    ids, r1 = stage_mix(meta, v64n, v32n, jobs, nd, st, sl, rp, la, mp.voff)
                    if l0t is not None and D in l0t.books:
                        o, n = meta.lay64["l0t_slot"]
                        v64n[o:o + n] = mix_l0t(l0t, offs, D, S, ids, r1)
                    if zc is not None:
                        if zc.copy_meta(meta, zp64, zp32, ln.stream.cuda_stream):
                            raise RuntimeError("ZC check: copy kernel launch failed")
                    else:
                        meta.b64.copy_(v64[:meta.n64])
                        meta.b32.copy_(v32[:meta.n32])
                    graph.replay()
                    got = out.clone()
                    ln.stream.synchronize()
                    GOV.tick()
                    ngraph += 1
                    for r, ref in enumerate(want):
                        nrows += 1
                        if not torch.equal(got[r].view(torch.int16), ref.view(torch.int16)):
                            ok = False
                            nbad += 1
            torch.cuda.synchronize()
        res[(S, D)] = ok and ngraph > 0
        log(f"mix {S}->{D} check (bucket-{S} + bucket-{D} decisions in mixed graphs on {nl} lanes vs their own "
            f"per-decision graphs, every bit): {'PASS' if res[(S, D)] else 'FAIL: not used'} ({ngraph} graphs, "
            f"{nrows} rows, {nbad} differ)")
    if l0t is not None:   # serving starts with empty tables (not this check's random tokens)
        l0t.reset_all()
    return res


class _BSlot:
    def __init__(self, rows: int, bmax: int):
        self.ids = torch.zeros((rows,), dtype=torch.long, pin_memory=True)
        self.ids_np = self.ids.numpy()
        self.last = torch.zeros((bmax,), dtype=torch.long, pin_memory=True)
        self.last_np = self.last.numpy()
        self.start = torch.zeros((bmax,), dtype=torch.int32, pin_memory=True)
        self.slen = torch.zeros((bmax,), dtype=torch.int32, pin_memory=True)
        self.row_pos = torch.zeros((rows,), dtype=torch.int32, pin_memory=True)
        self.pvar = torch.zeros((bmax,), dtype=torch.int32, pin_memory=True)   # PV (a4)
        self.plen = torch.zeros((bmax,), dtype=torch.int32, pin_memory=True)
        self.l0t_slot = torch.zeros((rows,), dtype=torch.long, pin_memory=True) \
            if os.environ.get("S1ENG_L0T", "0") == "1" else None   # L0T (kai): only with the flag on
        if os.environ.get("S1ENG_META_BLOB", "0") == "1":   # E4: staging for PackedMeta's two blobs (any layout)
            self.b64 = torch.zeros((2 * rows + bmax + 96,), dtype=torch.long, pin_memory=True)
            self.b32 = torch.zeros((rows + 4 * bmax + 320,), dtype=torch.int32, pin_memory=True)
            self.b64_np, self.b32_np = self.b64.numpy(), self.b32.numpy()
        self.zc64 = self.zc32 = None   # ZC-IN: device addresses of b64 / b32 (UVA), set by the start-up gate
        self.out = torch.zeros((bmax, 16), dtype=torch.bfloat16, pin_memory=True)
        self.event = torch.cuda.Event()
        self.ev0 = self.ev1 = None   # S1ENG_TRACE: timing events around the batch on the lane stream
        self.tag = None


def stage_blob(meta, v64, v32, jobs: list, st, sl, rp, la):
    """E4 (S1ENG_META_BLOB): write a packed launch's inputs into pinned staging blobs (numpy views v64 / v32) at the
    meta's offsets: the values the eight-copy path writes into the eight fields (ids of the packed rows and zeros up to
    R, start, slen, row_pos, last, and pvar / plen with unused slots at variant 0 / the template length). Returns the
    ids view (R rows), which L0T maps to table slots."""
    P, R = meta.pfx, meta.R

    def f(v, lay, name):
        o, n = lay[name]
        return v[o:o + n]
    ids = f(v64, meta.lay64, "ids")
    r = 0
    for j in jobs:
        p = j.pl if P else 0
        ids[r:r + j.n - p] = j.ids[p:]
        r += j.n - p
    ids[r:R] = 0
    f(v32, meta.lay32, "start")[:] = st
    f(v32, meta.lay32, "slen")[:] = sl
    f(v32, meta.lay32, "row_pos")[:] = rp
    f(v64, meta.lay64, "last")[:] = la
    pv, pl = f(v32, meta.lay32, "pvar"), f(v32, meta.lay32, "plen")
    pv[:] = 0
    pl[:] = P
    if P:
        for b, j in enumerate(jobs):
            pv[b], pl[b] = j.pv, j.pl
    return ids


class _LeanCu:
    """LL (ruj batching): a packed launch's five CUDA calls through the driver API with the GIL held (ctypes.PyDLL;
    each call only enqueues work): the two H2D copies of the staged blobs, the graph launch, the D2H copy of the
    outputs and the completion event record, on the lane's stream. The same calls as torch's copy_ / replay() /
    record() (minus their Python dispatch, the caching host allocator's event per copy and the GIL hand-offs)."""

    def __init__(self):
        import ctypes as C

        lib = C.PyDLL("libcuda.so.1")

        def fn(name, args):
            f = getattr(lib, name)
            f.argtypes = args
            f.restype = C.c_int
            return f
        self.htod = fn("cuMemcpyHtoDAsync_v2", [C.c_uint64, C.c_void_p, C.c_size_t, C.c_void_p])
        self.dtoh = fn("cuMemcpyDtoHAsync_v2", [C.c_void_p, C.c_uint64, C.c_size_t, C.c_void_p])
        self.launch = fn("cuGraphLaunch", [C.c_void_p, C.c_void_p])
        self.record = fn("cuEventRecord", [C.c_void_p, C.c_void_p])


class BatchLoop:
    """Scheduler for lanes. Runs on the capture thread; one completion thread per lane waits for the lane's events
    in launch order and hands results back. When a lane has room, the queued jobs are grouped by bucket and the
    largest group (or the oldest job's group, once it waited STARVE_S) is launched as one batch. A lone request is
    launched at once (B = 1). Jobs longer than the batch buckets use lane 0's per-decision graphs."""

    STARVE_S = 0.25

    def __init__(self, model: Model, lanes: list, depth: int = 1):
        self.m = model
        self.lanes = lanes
        self.depth = depth
        self.q: collections.deque = collections.deque()
        self.cv = threading.Condition()
        rows = max([max(GRAPH_LENGTHS)] + [L * B for ln in lanes for (L, B) in ln.batch]
                   + [e[0] for ln in lanes for v in ln.packed.values() for e in v])
        bm = max([B for ln in lanes for (L, B) in list(ln.batch) + list(ln.packed)] + [1])
        self.free = [_BSlot(rows, bm) for _ in range(depth * len(lanes) + 1)]
        self.stop = False
        self.done_q = [collections.deque() for _ in lanes]
        self.done_cv = [threading.Condition() for _ in lanes]
        self.stats = collections.Counter()
        self.stats_every = int(os.environ.get("S1ENG_STATS", "0"))
        self.minb = int(os.environ.get("S1ENG_MINB", "1"))
        self.maxwait = float(os.environ.get("S1ENG_MAXWAIT_MS", "0")) / 1000.0
        self.split = os.environ.get("S1ENG_SPLIT", "0") == "1"
        # E3 scheduling (exactness-neutral: every packed graph is verified for any companions): while the GPU is busy,
        # hold a small batch group for up to S1ENG_HOLD_MS after a lane frees (or after its head arrived) until it has
        # S1ENG_HOLD_N jobs; S1ENG_FIT=1: a group one job larger than a captured slot runs that slot (the extra job
        # goes in the next batch) instead of padding up to the next slot. Both off by default.
        self.hold_n = int(os.environ.get("S1ENG_HOLD_N", "0"))
        self.hold_s = float(os.environ.get("S1ENG_HOLD_MS", "0")) / 1000.0
        self.fit = os.environ.get("S1ENG_FIT", "0") == "1"
        # ARRIVE (rgh, u198 idea): a hold also ends as soon as nothing more is on the way: no request between HTTP
        # arrival and the queue (arriving, written by the HTTP thread) and every client answered within the last
        # S1ENG_RETURN_MS has sent its next decision (back: answers delivered minus decisions submitted, under cv)
        self.arrive = os.environ.get("S1ENG_ARRIVE", "1") == "1"   # default on (Verda 10-21 idle read +1.4 %)
        self.arriving = 0
        self.back = 0
        self.t_back = 0.0
        self.ret_s = float(os.environ.get("S1ENG_RETURN_MS", "10")) / 1000.0
        self.arrive_idle = self.arrive and os.environ.get("S1ENG_ARRIVE_IDLE", "0") == "1"
        self.idle_hold_s = float(os.environ.get("S1ENG_IDLE_HOLD_MS", "1.5")) / 1000.0
        # RAMP (kai/kx-sched, default off): S1ENG_ARRIVE_YIELD=1 - the HTTP handler yields once after counting a
        # request as arriving (server.py), and an ARRIVE idle hold ends once the largest group has HOLD_N jobs
        self.arrive_yield = os.environ.get("S1ENG_ARRIVE_YIELD", "0") == "1"
        # MIX_RSTEP (kai/kx-sched): the R grid the mixed picks use (default S1ENG_MIX_RSTEP, the captured grid); an
        # arm "mix_rstep": 64 on a step-32 capture picks only the step-64 bins (scheduling only: every bin is captured)
        self.mix_rstep = max(1, int(os.environ.get("S1ENG_MIX_RSTEP", "64")))
        # HOST (kai/kx-host, backlog #4 stage 0, host timing only): arm keys host_delay_us (server.HOST_DELAY_NS: a
        # busy-wait before each submit, GIL held while < switch_us; default S1ENG_HOST_DELAY_US = 0) and switch_us
        # (sys.setswitchinterval, default S1ENG_SWITCH_US = 500, set by main.py at start-up)
        from .server import host_delay_us_of

        self.host_delay_us = host_delay_us_of(os.environ.get("S1ENG_HOST_DELAY_US", "0") or "0")
        self.switch_us = switch_us_of(os.environ.get("S1ENG_SWITCH_US", "500") or "500")
        self.wake_at = None
        self.njobs = 0
        self.starve = float(os.environ.get("S1ENG_STARVE_MS", str(self.STARVE_S * 1000))) / 1000.0
        # E1: buckets whose packed graphs skip the template prefix take only jobs that start with it (others run the
        # per-decision graphs); S1ENG_LONE_PACKED=1: a lone job runs a packed B = 1 graph instead of set 0
        self.pfx = dict(lanes[0].pfx) if lanes else {}
        pids = getattr(model, "prefix_ids", None) or []
        self.prefix = np.asarray(pids, dtype=np.int64)
        self.P = len(pids) if any(self.pfx.values()) else 0
        # PV (a4): the cached prefix variants (index 0 = the template prefix), as recorded by capture_batch
        self.variants = [np.asarray(v, dtype=np.int64) for v in prefix_variants_of(model, pids)] if self.P else []
        if _ntv() and self.variants:   # NTV (kai): pick_variant walks a token trie instead of scanning every variant,
            # restricted per bucket to the variants its pcache holds (ntv.index: no nonce-token pvar outside
            # NTV's buckets)
            try:
                from . import ntv as ntvmod

                self.variants = ntvmod.index(model, self.variants, pids, self.pfx, prefix_variants_of, bucket_of)
            except Exception as exc:  # noqa: BLE001 - the plain list of the variants every bucket caches: hotspot's scan
                import sys

                print(f"[s1eng] lever not used: ntv ({type(exc).__name__}: {exc})", file=sys.stderr, flush=True)
                self.variants = self.variants[:getattr(model, "ntv_common", len(self.variants))]
        self.lone_packed = os.environ.get("S1ENG_LONE_PACKED", "0") == "1"
        self.wake_filter = os.environ.get("S1ENG_WAKE_FILTER", "0") == "1"
        self.l0t = getattr(getattr(model, "pf", None), "l0t", None)   # L0T (kai): layer-0 tables (None: off)
        self.trace_every = int(os.environ.get("S1ENG_TRACE", "0"))
        self.trace = []
        self.trace_t0 = None
        if self.trace_every:
            for sl in self.free:
                sl.ev0 = torch.cuda.Event(enable_timing=True)
                sl.ev1 = torch.cuda.Event(enable_timing=True)
        # LL (ruj batching, S1ENG_LEAN=1; v13x): packed launches through _LeanCu (scheduling and graphs unchanged)
        self.lean = os.environ.get("S1ENG_LEAN", "0") == "1"
        self._cu = None
        self._lean_ok = None
        if self.lean:   # LL: every slot's completion event exists (torch creates it at its first record)
            for sl_ in self.free:
                if not sl_.event.cuda_event:
                    sl_.event.record()
            torch.cuda.synchronize()
        # BB (ruj v14 bigbatch): more decisions per lane graph. Scheduling only: every packed graph the scheduler can
        # pick was captured and self-checked at start-up, so no knob below changes a bit of any answer. All default
        # off; with none set, _choose is v13b's rule unchanged.
        #   S1ENG_AFF="192:0,256:1,128:1/0,320:1/0": lane affinity, bucket -> lanes (primary first, "/" = also
        #     allowed after the lane's own buckets). A lane launches only groups of buckets it is listed for, so each
        #     lane collects all of its buckets' traffic (bigger batches than two lanes splitting every bucket). The
        #     E3 hold runs per lane (from that lane's own free time). Unlisted buckets (inputs > 320 tokens: lane 0's
        #     per-decision graphs) run as before.
        #   S1ENG_AFF_STEAL_MS: a group whose lanes are all busy may run on another free lane once its oldest job
        #     waited this long (0 = never).
        #   S1ENG_LANES_ON="0": the scheduler uses only these lanes (lane 0 always: it holds the per-decision graphs);
        #     one-lane mode inside a LANES=2 capture (S1ENG_LANES=1 is the start-up equivalent).
        #   S1ENG_IDLE_BACK=1: the hold also applies with no lane busy, while the decisions just answered are on their
        #     way back (ARRIVE's back counter within RETURN_MS, or a request between HTTP arrival and the queue): the
        #     returning burst joins the launch. A lone request is never held (its own submit clears the counter).
        #   S1ENG_SYNC_MS: a free lane holds while another lane is busy, up to this long after it freed (the lanes
        #     launch together, each with a whole bucket group).
        #   S1ENG_HOLD_BY="192:12": per-bucket hold size (the chosen group's bucket; others HOLD_N), as v13x's HB.
        #   S1ENG_PCAP="128:4,320:4": the largest packed B slot the scheduler uses per bucket (default every captured
        #     slot; S1ENG_PACKB_CAP decides which slots are captured). Applies to v13b's rule as well.
        self.bb_aff, self.bb_lanes, self.bb_pcap, self.bb_hold_by = {}, None, {}, {}
        self.bb_steal_s = self.bb_sync_s = 0.0
        self.bb_idle_back = False
        try:
            self.bb_aff = parse_aff(os.environ.get("S1ENG_AFF", ""), len(lanes))
            self.bb_lanes = parse_lanes(os.environ.get("S1ENG_LANES_ON", ""), len(lanes))
            self.bb_steal_s = float(os.environ.get("S1ENG_AFF_STEAL_MS", "0")) / 1000.0
            self.bb_sync_s = float(os.environ.get("S1ENG_SYNC_MS", "0")) / 1000.0
            self.bb_idle_back = os.environ.get("S1ENG_IDLE_BACK", "0") == "1"
            self.bb_pcap = parse_bmax(os.environ.get("S1ENG_PCAP", ""))
            self.bb_hold_by = parse_bmax(os.environ.get("S1ENG_HOLD_BY", ""))
        except Exception as exc:  # noqa: BLE001 - a bad knob costs the lever only
            print(f"[s1eng] lever not used: bigbatch knobs ({type(exc).__name__}: {exc})", flush=True)
            self.bb_aff, self.bb_lanes, self.bb_pcap, self.bb_hold_by = {}, None, {}, {}
            self.bb_steal_s = self.bb_sync_s = 0.0
            self.bb_idle_back = False
        # BB4 S1ENG_JOIN="128:192": when both groups are queued, bucket-128 decisions join the bucket-192 group and
        # ride in its packed graphs (fewer, larger launches; a 128 decision no longer waits for a 128 group of its own).
        # Exact only where join_check (start-up, this GPU) found every bit of S decisions inside D graphs equal to
        # their own per-decision graph: unverified pairs are never used. A lone S decision runs its own graph.
        self.join_ok = {k for k, v in (getattr(model, "join_ok", None) or {}).items() if v}
        self.bb_join = self._verified_join(os.environ.get("S1ENG_JOIN", ""))
        self.arm_join = 0
        self._bb_update()
        # ARMS (ruj diagnostics, default off): runtime-switchable scheduling knobs for in-situ A/B inside one engine.
        # S1ENG_ARMS = JSON or @file {"<arm>": {"hold_n": 16, "aff": "192:0,256:1", "lanes": "0", ...}} (keys: hold_n
        # hold_ms starve_ms arrive ret_ms aff steal_ms lanes idle_back sync_ms pcap hold_by arrive_idle idle_hold_ms
        # split join arrive_yield mix_rstep host_delay_us switch_us zc ftok; missing keys = the image's env values); S1ENG_CTL =
        # a file whose first line names the active arm (polled every 20 ms).
        self.arm = "env"
        self.arms = {}
        self.arm_stats = collections.Counter()
        self.arm_t = time.perf_counter()
        # BB3 diagnostics (only while S1ENG_ARMS is in use; nothing else reads them): per arm, why each launch left the
        # queue when it did (arm_why: idle = no lane busy, size = group >= the hold size, arr = ARRIVE: nothing more on
        # the way, cut = the hold deadline passed while more was on the way, starve / forced / mixed = no hold rule),
        # and the client turnaround (completion -> that client's next submit; FIFO over completions, entries older
        # than 50 ms dropped as clients that did not come back), in 0.25 ms bins. Printed at every arm switch.
        self.arm_why = collections.Counter()
        self.tr_led = collections.deque()
        self.tr_hist = collections.Counter()
        self._why = None
        arms_env = os.environ.get("S1ENG_ARMS", "")
        if arms_env:
            try:
                self.arms = json.loads(open(arms_env[1:]).read() if arms_env.startswith("@") else arms_env)
            except Exception as exc:  # noqa: BLE001
                print(f"[s1eng] arms not used: {type(exc).__name__}: {exc}", flush=True)
                self.arms = {}
        # ZC-IN (kai/kx-host, zcin.py; arm key zc, default S1ENG_ZC=0): lean launches copy their two input blobs with
        # one kernel on the lane stream (pinned staging read through UVA) instead of two H2D copies. The start-up gate
        # runs only when env or an arm asks for it; zc_ok = it passed (an arm with zc 1 is refused otherwise).
        self._zc = None
        self.zc_ok = False
        try:
            zc_env = zc_of(os.environ.get("S1ENG_ZC", "0") or "0")
        except (ValueError, TypeError) as exc:
            print(f"[s1eng] S1ENG_ZC ignored ({exc}): env default zc 0", flush=True)   # not the gate's refusal (check-start)
            zc_env = 0
        if zc_env or (isinstance(self.arms, dict) and any(_zc_asked(a) for a in self.arms.values())):
            self.zc_ok = bool(self._zc_setup())
        self.zc = bool(zc_env and self.zc_ok)
        # FTOK (kai, ftok.py; arm key ftok, default S1ENG_FTOK=0): the exact ASCII tail tokenizer. main.py ran its
        # start-up check (here and, under MPF, in every worker) when env or an arm asks for it; ftok_ok = it passed
        # everywhere (an arm with ftok 1 is refused otherwise), ftok = the switch as start-up left it
        ftm = _ftok_mod()
        self.ftok_ok = bool(ftm is not None and ftm.OK)
        self.ftok = bool(ftm is not None and ftm.ACTIVE)
        self.arm_defaults = {"hold_n": self.hold_n, "hold_ms": self.hold_s * 1000, "starve_ms": self.starve * 1000,
                             "arrive": int(self.arrive), "ret_ms": self.ret_s * 1000,
                             "aff": {L: list(v) for L, v in self.bb_aff.items()},
                             "steal_ms": self.bb_steal_s * 1000,
                             "lanes": None if self.bb_lanes is None else sorted(self.bb_lanes),
                             "idle_back": int(self.bb_idle_back), "sync_ms": self.bb_sync_s * 1000,
                             "pcap": dict(self.bb_pcap), "hold_by": dict(self.bb_hold_by),
                             "arrive_idle": int(self.arrive_idle), "idle_hold_ms": self.idle_hold_s * 1000,
                             "split": int(self.split),
                             "arrive_yield": int(self.arrive_yield), "mix_rstep": self.mix_rstep,
                             "host_delay_us": self.host_delay_us, "switch_us": self.switch_us, "zc": int(self.zc),
                             "ftok": int(self.ftok),
                             "join": ",".join(f"{s}:{d}" + (f"@{self.bb_join_cap[(s, d)]}" if (s, d) in
                                                              getattr(self, "bb_join_cap", {}) else "")
                                              for s, d in self.bb_join)}
        self.ctl_path = os.environ.get("S1ENG_CTL", "")
        # BB idle-back ledger: (completion time, decisions answered and not yet followed by a new request), FIFO,
        # entries older than RETURN_MS dropped. ARRIVE's back counter keeps a residue across the end of a timed run
        # (answers with no next request), harmless for holds that need a busy GPU but not for a hold of an idle one
        # (a c1 request would be held), so the idle-back hold counts only this ledger. Kept only if some arm uses it.
        self.bb_ret = collections.deque()
        self.bb_ledger = self.bb_idle_back or any(int(a.get("idle_back", 0)) for a in self.arms.values())
        if self.arms and self.ctl_path:
            threading.Thread(target=self._ctl_watch, name="ctl", daemon=True).start()
        for i in range(len(lanes)):
            threading.Thread(target=self._completer, args=(i,), name=f"done{i}", daemon=True).start()
        if os.environ.get("S1ENG_FCAB"):   # FCAB (ruj v14 fixedcost): lane sets on the control file (timing harness)
            from . import fcab

            fcab.attach(self)

    def submit(self, job: Job) -> None:
        P = self.P
        vi = pick_variant(job.ids, self.variants) if P > 0 else -1
        job.pok = vi >= 0
        job.pv = max(vi, 0)
        job.pl = len(self.variants[vi]) if vi >= 0 else 0
        with self.cv:
            self.q.append(job)
            if self.back > 0:
                self.back -= 1
            if self.bb_ledger and self.bb_ret:   # BB: this client's answer is back
                now = time.perf_counter()
                while self.bb_ret and now - self.bb_ret[0][0] > self.ret_s:
                    self.bb_ret.popleft()
                if self.bb_ret:
                    self.bb_ret[0][1] -= 1
                    if self.bb_ret[0][1] <= 0:
                        self.bb_ret.popleft()
            if self.arms and self.tr_led:   # BB3 diagnostics: this client's turnaround
                now = time.perf_counter()
                while self.tr_led and now - self.tr_led[0][0] > 0.05:
                    self.tr_led.popleft()
                if self.tr_led:
                    self.tr_hist[min(int((now - self.tr_led[0][0]) * 4000), 40)] += 1
                    self.tr_led[0][1] -= 1
                    if self.tr_led[0][1] <= 0:
                        self.tr_led.popleft()
            # E4 (S1ENG_WAKE_FILTER): with every lane full nothing can launch, so the scheduler is not woken (its
            # _choose would return None); the completer that frees a lane wakes it and it then sees this job
            if not self.wake_filter or any(ln.busy < self.depth for ln in self.lanes):
                self.cv.notify()

    def _key(self, j: Job):
        """(bucket, packed graphs usable): a job that does not start with the template prefix cannot join a prefix
        bucket's packed graphs."""
        L = bucket_of(j.n)
        return L, (not self.pfx.get(L)) or j.pok

    # ---------------------------------------------------------------- choice (called with self.cv held)
    def _choose(self):
        """-> (lane, L, jobs, packed entry or None) or None; sets self.wake_at when a small group was deferred."""
        if self.bb_on and self.q[0].stream is None:
            return self._choose_bb()
        return self._choose_v13()

    def _bb_update(self) -> None:
        """BB: the bigbatch rule is used only when one of its knobs is set (S1ENG_PCAP alone keeps v13b's rule)."""
        self.bb_on = bool(self.bb_aff or self.bb_lanes is not None or self.bb_idle_back or self.bb_sync_s > 0
                          or self.bb_steal_s > 0 or self.bb_hold_by)

    def _bb_packed_b(self, ln: Lane, L: int, n: int) -> int:
        """Lane.packed_b with S1ENG_PCAP: the smallest captured slot >= n among the slots <= the bucket's cap (else the
        largest of those)."""
        bs = ln.pbs[L]
        cap = self.bb_pcap.get(L)
        if cap:
            bs = [b for b in bs if b <= cap] or bs[:1]
        for b in bs:
            if b >= n:
                return b
        return bs[-1]

    def _verified_join(self, spec) -> dict:
        """BB4: the S1ENG_JOIN pairs that passed join_check at start-up (others are reported once and dropped)."""
        try:
            want = parse_join(spec)
        except Exception as exc:  # noqa: BLE001 - a bad knob costs the lever only
            print(f"[s1eng] lever not used: join ({type(exc).__name__}: {exc})", flush=True)
            return {}
        ok = [(s, d) for s, d in want if (s, d) in self.join_ok]
        caps = parse_join_caps(spec)
        self.bb_join_cap = {k: v for k, v in caps.items() if k in ok}   # MX: S-group size caps of the used pairs
        warned = self.__dict__.setdefault("_join_warned", set())
        for s, d in want:
            if (s, d) not in ok and (s, d) not in warned:
                warned.add((s, d))
                print(f"[s1eng] lever not used: join {s}:{d} (no passed start-up check, or a chained pair)", flush=True)
        return ok

    def _groups(self):
        """Queued jobs by _key; BB4 S1ENG_JOIN: a usable S group joins the usable D group when both are queued (one
        group in queue order, launched in D's packed graphs). -> (groups, {key merged away: key it joined})."""
        groups: dict = {}
        for j in self.q:
            groups.setdefault(self._key(j), []).append(j)
        moved, took = {}, set()
        for s, d in self.bb_join:   # MX: in priority order; one S group per D group, no chains
            ks, kd = (s, True), (d, True)
            if ks in groups and kd in groups and not ({ks, kd} & took) and ks not in moved and kd not in moved:
                cap = self.bb_join_cap.get((s, d)) if getattr(self, "bb_join_cap", None) else None
                if cap is not None and len(groups[ks]) > cap:   # MX cap: a big S group keeps its own graphs
                    continue
                if not all(ln.mbs.get((d, s)) for ln in self.lanes):
                    continue   # own-final: never merge on a lane set without this pair's mixed graphs (FCAB sets)
                groups[kd] = sorted(groups[kd] + groups.pop(ks), key=lambda j: j.t_enq)
                moved[ks] = kd
                took |= {ks, kd}
        return groups, moved

    def _choose_bb(self):
        """BB rule (see __init__): per free lane, its candidate groups - own buckets (S1ENG_AFF primary lane, or any
        unlisted bucket) largest first, then buckets it may also take, then groups it may steal; the oldest job's group
        first once it starves - then the E3 hold on the first candidate, per lane, ended early by ARRIVE; then the same
        slot / R-bin choice as v13b's rule."""
        now = time.perf_counter()
        groups, moved = self._groups()
        oldest = self.q[0]
        self.wake_at = None
        lanes = self.lanes if self.bb_lanes is None else [ln for ln in self.lanes if ln.idx in self.bb_lanes]
        free = [ln for ln in lanes if ln.busy < self.depth]
        if not free:
            return None
        busy = [ln for ln in lanes if ln.busy]
        okey = self._key(oldest)
        okey = moved.get(okey, okey)
        starving = now - oldest.t_enq > self.starve
        order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[1][0].t_enq))
        # more on the way: a request between HTTP arrival and the queue, or answers of the last RETURN_MS not yet back
        coming = self.arriving > 0 or (self.back > 0 and now - self.t_back <= self.ret_s)
        # idle-back: only decisions answered within RETURN_MS and not yet back (ledger), or requests being parsed
        returning = self.arriving > 0 or any(now - e[0] <= self.ret_s and e[1] > 0 for e in self.bb_ret)
        if (self.arrive_idle and not busy and self.arriving > 0 and now - oldest.t_enq < self.idle_hold_s
                and not self._idle_full(groups)):
            # ARRIVE idle (v13b's knob, same rule): an idle GPU lets a burst being tokenized join the first launch
            self.wake_at = oldest.t_enq + self.idle_hold_s
            return None
        aff = self.bb_aff
        busy_idx = {ln.idx for ln in self.lanes if ln.busy >= self.depth}
        if aff and starving:   # the starving group's own lane first
            pl = aff.get(okey[0], [0])[0]
            free.sort(key=lambda ln: ln.idx != pl)
        wake = None
        for ln in free:
            cands = []
            for key, jobs in order:
                L = key[0]
                if starving and key == okey:
                    cands.append((0, key, jobs))
                    continue
                lst = aff.get(L)
                if lst is None or lst[0] == ln.idx:
                    cands.append((1, key, jobs))
                elif ln.idx in lst:
                    cands.append((2, key, jobs))
                elif (self.bb_steal_s > 0 and all(i in busy_idx for i in lst)
                      and now - jobs[0].t_enq >= self.bb_steal_s):
                    cands.append((3, key, jobs))
            if not cands:
                continue
            cands.sort(key=lambda c: c[0])   # stable: size / age order within a rank
            rank0, (L0, u0), jobs0 = cands[0]
            why = "starve"
            if rank0 > 0:
                # E3 hold (per lane): a small group waits for more of its bucket while the GPU is busy (BB idle-back:
                # or while answered decisions are on their way back), at most HOLD_MS after this lane freed / its head
                # arrived; ARRIVE ends it once nothing more is on the way. SYNC: the lane also waits for the others.
                dl = None
                hn = self.bb_hold_by.get(L0, self.hold_n)
                batchable = all(c[1][1] and c[1][0] in BATCH_BUCKETS for c in cands)
                if hn and len(jobs0) < hn and batchable and (busy or (self.bb_idle_back and returning)):
                    d = max(ln.t_free, jobs0[0].t_enq) + self.hold_s
                    more = not self.arrive or (coming if busy else returning)
                    if now < d and more:
                        dl = d
                    why = "cut" if more else "arr"
                else:
                    why = ("size" if hn and len(jobs0) >= hn else "mixed" if not batchable else
                           "idle" if not busy else "nohold")
                if self.bb_sync_s > 0 and any(o is not ln for o in busy):
                    d = ln.t_free + self.bb_sync_s
                    if now < d:
                        dl = d if dl is None else max(dl, d)
                if dl is not None:
                    wake = dl if wake is None else min(wake, dl)
                    continue
            self._why = why
            for _rank, (L, usable), jobs in cands:
                n = len(jobs)
                if ((n == 1 or not usable) and L in ln.single
                        and not (self.lone_packed and usable and L in ln.pbs)):
                    return ln, L, jobs[:1], None
                if not usable:
                    continue
                if L in ln.pbs:
                    mxpick = self._mix_pick(ln, L, jobs, n) if self.bb_join else None
                    if mxpick is not None:   # MX: a merged group with bucket-S jobs: a mixed graph
                        return mxpick
                    B = self._bb_packed_b(ln, L, n)
                    take = jobs[:min(n, B)]
                    T = sum(j.n - (j.pl if ln.pfx.get(L, 0) else 0) for j in take)
                    for e in ln.packed[(L, B)]:
                        if e[0] >= T:
                            return ln, L, take, (B, e)
                    continue
                b = ln.best_b(L, n)
                if b is not None:
                    return ln, L, jobs[:b], None
                if L in ln.single:
                    return ln, L, jobs[:1], None
        self.wake_at = wake
        return None

    def _idle_full(self, groups: dict) -> bool:
        """RAMP (S1ENG_ARRIVE_YIELD / arm key arrive_yield): an ARRIVE idle hold ends once the largest batchable group
        has HOLD_N jobs (a launch that size goes now; the rest of the burst fills the next one)."""
        if not (self.arrive_yield and self.hold_n):
            return False
        return max((len(js) for (L, u), js in groups.items() if u and L in BATCH_BUCKETS), default=0) >= self.hold_n

    def _apply_arm(self, name: str) -> None:
        """ARMS: the scheduling knobs of arm `name` (missing keys: the image's env values). Called with cv held."""
        c = {**self.arm_defaults, **self.arms[name]}
        from . import server as _srv

        host_delay_us = _srv.host_delay_us_of(c["host_delay_us"])   # HOST: checked before any knob changes
        switch_us = switch_us_of(c["switch_us"])
        zc = zc_of(c["zc"])   # ZC-IN: checked before any knob changes
        if zc and not self.zc_ok:
            raise RuntimeError("zc unavailable (start-up gate failed: see 'lever not used: zc')")
        ft = ftok_of(c.get("ftok", 0))   # FTOK: checked before any knob changes
        if ft and not self.ftok_ok:
            raise RuntimeError("ftok unavailable (start-up check failed or not run: see 'lever not used: ftok')")
        self.hold_n = int(c["hold_n"])
        self.hold_s = float(c["hold_ms"]) / 1000.0
        self.starve = float(c["starve_ms"]) / 1000.0
        self.arrive = bool(int(c["arrive"]))
        self.ret_s = float(c["ret_ms"]) / 1000.0
        a = c["aff"]
        self.bb_aff = parse_aff(a, len(self.lanes)) if isinstance(a, str) else \
            {int(k): [int(x) for x in v] for k, v in (a or {}).items()}
        self.bb_steal_s = float(c["steal_ms"]) / 1000.0
        lo = c["lanes"]
        self.bb_lanes = parse_lanes(lo, len(self.lanes)) if isinstance(lo, str) else \
            (None if lo is None else {0} | {int(x) for x in lo})
        self.bb_idle_back = bool(int(c["idle_back"]))
        self.bb_sync_s = float(c["sync_ms"]) / 1000.0
        p = c["pcap"]
        self.bb_pcap = parse_bmax(p) if isinstance(p, str) else {int(k): int(v) for k, v in (p or {}).items()}
        h = c["hold_by"]
        self.bb_hold_by = parse_bmax(h) if isinstance(h, str) else {int(k): int(v) for k, v in (h or {}).items()}
        self.arrive_idle = self.arrive and bool(int(c["arrive_idle"]))
        self.idle_hold_s = float(c["idle_hold_ms"]) / 1000.0
        self.split = bool(int(c["split"]))
        self.arrive_yield = bool(int(c["arrive_yield"]))
        self.mix_rstep = max(1, int(c["mix_rstep"]))
        j = c.get("join", "")
        self.bb_join = self._verified_join(j if isinstance(j, str) else
                                           ",".join(f"{s}:{d}" for s, d in (j or [])))
        self._bb_update()
        self.host_delay_us, self.switch_us = host_delay_us, switch_us
        self.zc = bool(zc)
        _srv.HOST_DELAY_NS = host_delay_us * 1000
        sys.setswitchinterval(switch_us / 1e6)
        if self.ftok_ok:
            self._ftok_switch(ft)
        self.arm = name

    def _ftok_switch(self, v: int) -> None:
        """FTOK: the switch in this process (its HTTP thread when MPF is off) and, under MPF, in every worker (a
        control record on its result pipe: mpf.Front.ctl_all -> mpf.worker_ctl)."""
        ftm = _ftok_mod()
        if ftm is not None:
            ftm.set_active(v)
        self.ftok = bool(v)
        f = getattr(self, "_mpf", None)
        if f is not None:
            f.ctl_all({"ftok": int(v)})

    def _ctl_watch(self) -> None:
        import sys

        last = None
        while True:
            try:
                with open(self.ctl_path) as fh:
                    name = fh.readline().strip()
            except OSError:
                name = None
            if name and name != last and name in self.arms:
                with self.cv:
                    st, self.arm_stats = self.arm_stats, collections.Counter()
                    why, self.arm_why = self.arm_why, collections.Counter()
                    trh, self.tr_hist = self.tr_hist, collections.Counter()
                    njoin, self.arm_join = self.arm_join, 0
                    prev, dt = self.arm, time.perf_counter() - self.arm_t
                    try:
                        self._apply_arm(name)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[s1eng] arm {name} not applied: {type(exc).__name__}: {exc}", file=sys.stderr,
                              flush=True)
                    self.arm_t = time.perf_counter()
                    self.cv.notify_all()
                per = {}
                for (L, B, li), v in st.items():
                    e = per.setdefault(L, [0, 0, collections.Counter()])
                    e[0] += v
                    e[1] += B * v
                    e[2][li] += v
                nb, dec = sum(e[0] for e in per.values()), sum(e[1] for e in per.values())
                print(f"[s1eng] arm {prev} -> {name} after {dt:.1f}s: {nb} batches, {dec} decisions, mean B "
                      f"{dec / max(nb, 1):.2f}; per bucket (batches, mean B, lanes): "
                      + ", ".join(f"{L}: {e[0]} {e[1] / max(e[0], 1):.2f} {dict(e[2])}" for L, e in sorted(per.items())),
                      file=sys.stderr, flush=True)
                ntr = sum(trh.values())
                print(f"[s1eng] armdiag {prev} dec {dec} why {json.dumps(dict(sorted(why.items())))} turnaround n {ntr} "
                      f"hist025 {json.dumps([trh.get(i, 0) for i in range(41)])} joined {njoin}", file=sys.stderr,
                      flush=True)
                last = name
            time.sleep(0.02)

    def _choose_v13(self):
        """v13b's rule (unchanged except S1ENG_PCAP in the slot choice and the BB4 S1ENG_JOIN grouping)."""
        now = time.perf_counter()
        groups, moved = self._groups()
        oldest = self.q[0]
        forced = oldest.stream
        order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[1][0].t_enq))
        if now - oldest.t_enq > self.starve:
            ob = self._key(oldest)
            ob = moved.get(ob, ob)
            order = [(ob, groups[ob])] + [kv for kv in order if kv[0] != ob]
        if forced is not None:
            order = [(self._key(oldest), [oldest])]
        gpu_busy = any(ln.busy for ln in self.lanes)
        nfree = sum(1 for ln in self.lanes if ln.busy < self.depth)
        self.wake_at = None
        why = ("forced" if forced is not None else "idle" if not gpu_busy else "nohold" if not self.hold_n
               else "starve")
        if self.hold_n and forced is None and gpu_busy and now - oldest.t_enq <= self.starve:
            bg = [js for (L, u), js in groups.items() if u and L in BATCH_BUCKETS]
            big = max(bg, key=len) if bg else None
            free_t = [ln.t_free for ln in self.lanes if ln.busy < self.depth]
            if big is not None and free_t and len(big) < self.hold_n and len(bg) == len(groups):
                dl = max(min(free_t), big[0].t_enq) + self.hold_s
                more = not (self.arrive and self.arriving <= 0 and (self.back <= 0 or now - self.t_back > self.ret_s))
                if now < dl and more:
                    self.wake_at = dl
                    return None
                why = "cut" if more else "arr"
            else:
                why = "size" if big is not None and len(big) >= self.hold_n else "mixed"
        self._why = why
        if (self.arrive_idle and forced is None and not gpu_busy and self.arriving > 0
                and now - oldest.t_enq < self.idle_hold_s and not self._idle_full(groups)):
            # ARRIVE idle: the GPU is idle but more decisions are between HTTP arrival and the queue (a burst being
            # tokenized): let them join this launch, at most S1ENG_IDLE_HOLD_MS after the oldest. A lone request
            # (nothing arriving, e.g. concurrency 1) is never held.
            self.wake_at = oldest.t_enq + self.idle_hold_s
            return None
        for (L, usable), jobs in order:
            if (forced is None and gpu_busy and len(jobs) < self.minb and L in BATCH_BUCKETS
                    and now - jobs[0].t_enq < self.maxwait):
                # the GPU has work: let a small group grow a little (a lone request on an idle GPU never waits)
                w = jobs[0].t_enq + self.maxwait
                self.wake_at = w if self.wake_at is None else min(self.wake_at, w)
                continue
            for ln in self.lanes:
                if ln.busy >= self.depth:
                    continue
                if forced is not None and ln.idx != forced:
                    continue
                n = len(jobs)
                if self.split and forced is None and nfree >= 2 and len(order) == 1 and n >= 2:
                    # every lane is free and there is one group: share it out so that all lanes start now
                    n = -(-n // nfree)
                if ((n == 1 or not usable) and L in ln.single
                        and not (self.lone_packed and usable and L in ln.pbs)):
                    return ln, L, jobs[:1], None
                if not usable:
                    continue
                if L in ln.pbs:
                    mxpick = self._mix_pick(ln, L, jobs, n) if self.bb_join else None
                    if mxpick is not None:   # MX: a merged group with bucket-S jobs: a mixed graph
                        return mxpick
                    B = self._bb_packed_b(ln, L, n)   # = ln.packed_b(L, n) unless S1ENG_PCAP caps the bucket
                    if self.fit and gpu_busy and forced is None and B != n and now - jobs[0].t_enq <= self.starve:
                        lo = max((b for b in ln.pbs[L] if b < n), default=0)
                        if lo >= 2 and n - lo == 1:
                            B = lo
                    take = jobs[:min(n, B)]
                    T = sum(j.n - (j.pl if ln.pfx.get(L, 0) else 0) for j in take)
                    for e in ln.packed[(L, B)]:
                        if e[0] >= T:
                            return ln, L, take, (B, e)
                    continue
                b = ln.best_b(L, n)
                if b is not None:
                    return ln, L, jobs[:b], None
                if L in ln.single:
                    return ln, L, jobs[:1], None
        return None

    def _mix_pick(self, ln: Lane, L: int, jobs: list, n: int):
        """MX: the pick for a merged group (D = L) that holds bucket-S jobs: the oldest jobs up to the largest mixed slot,
        in slot order (D jobs first), the smallest mixed (B, R) graph that takes them -> (ln, L, jobs, ("mx", S, B, nd,
        entry)); a pure group -> None (the usual packed path); a taken set with no D job runs as a bucket-S group."""
        S = next((b for b in (bucket_of(j.n) for j in jobs) if b != L), None)
        if S is None:
            return None
        bs = ln.mbs.get((L, S))
        if not bs:
            return None
        B = next((b for b in bs if b >= n), bs[-1])
        take = jobs[:min(n, B)]
        dj = [j for j in take if bucket_of(j.n) == L]
        sj = [j for j in take if bucket_of(j.n) != L]
        P = ln.pfx.get(L, 0)
        if not sj:
            Bp = self._bb_packed_b(ln, L, len(dj))
            dj = dj[:Bp]
            T = sum(j.n - (j.pl if P else 0) for j in dj)
            e = next((e for e in ln.packed[(L, Bp)] if e[0] >= T), None)
            return (ln, L, dj, (Bp, e)) if e is not None else None
        if not dj:
            if S not in ln.pbs:
                return None
            Bp = self._bb_packed_b(ln, S, len(sj))
            sj = sj[:Bp]
            T = sum(j.n - (j.pl if ln.pfx.get(S, 0) else 0) for j in sj)
            e = next((e for e in ln.packed[(S, Bp)] if e[0] >= T), None)
            return (ln, S, sj, (Bp, e)) if e is not None else None
        jobs2 = dj + sj
        T = sum(j.n - (j.pl if P else 0) for j in jobs2)
        es = ln.mixed[(L, S, B)]
        st, last = self.mix_rstep, len(es) - 1
        for i, e in enumerate(es):
            # MIX_RSTEP arm: only bins on the arm's grid; the slot's top bin (B full-length decisions) is on every grid
            if e[0] >= T and (i == last or e[0] % st == 0):
                return ln, L, jobs2, ("mx", S, B, len(dj), e)
        return None

    def _launch_mix(self, ln: Lane, D: int, jobs: list, slot: _BSlot, pk) -> None:
        """MX: a mixed launch - stage_mix + mix_l0t into the slot's pinned blobs, then the lean (driver API) or the
        torch path, as _launch_lean / _launch_blob."""
        from .batchfwd import packed_layout

        _mxk, S, Bs, nd, (R, graph, meta, out) = pk
        B = len(jobs)
        P = meta.pfx
        st, sl, rp, la = packed_layout([j.n for j in jobs], Bs, D, R, P, plens=[j.pl for j in jobs] if P else None)
        mp = self.m.pf.mxp[(D, S)]
        ids, r1 = stage_mix(meta, slot.b64_np, slot.b32_np, jobs, nd, st, sl, rp, la, mp.voff)
        if self.trace_every:
            self._trace_begin(ln, slot, (D, Bs, R, B), jobs)
        lean = self.lean and not self.trace_every and not getattr(self, "timing", False) and self._lean_setup()
        with torch.cuda.stream(ln.stream):
            if slot.ev0 is not None and self.trace_every:
                slot.ev0.record()
            if self.l0t is not None and D in self.l0t.books:
                o, n = meta.lay64["l0t_slot"]
                slot.b64_np[o:o + n] = mix_l0t(self.l0t, self.m.pf.l0t_off, D, S, ids, r1)
            if lean:
                cu, h = self._cu, ln.stream.cuda_stream
                if self.zc:   # ZC-IN: one copy kernel on the lane stream instead of the two H2D copies
                    r = self._zc.launch(meta.b64.data_ptr(), slot.zc64, meta.n64 * 8, meta.b32.data_ptr(), slot.zc32,
                                        meta.n32 * 4, h)
                else:
                    r = (cu.htod(meta.b64.data_ptr(), slot.b64.data_ptr(), meta.n64 * 8, h)
                         or cu.htod(meta.b32.data_ptr(), slot.b32.data_ptr(), meta.n32 * 4, h))
                if (r
                        or cu.launch(graph.raw_cuda_graph_exec(), h)
                        or cu.dtoh(slot.out.data_ptr(), out.data_ptr(), B * out.shape[1] * out.element_size(), h)):
                    raise RuntimeError("mixed lean launch: CUDA driver call failed")
                ev = slot.event.cuda_event
                if not ev:
                    slot.event.record(ln.stream)
                elif cu.record(ev, h):
                    raise RuntimeError("mixed lean launch: cuEventRecord failed")
            else:
                meta.b64.copy_(slot.b64[:meta.n64], non_blocking=True)
                meta.b32.copy_(slot.b32[:meta.n32], non_blocking=True)
                graph.replay()
                slot.out[:B].copy_(out[:B], non_blocking=True)
                if self.trace_every:
                    slot.ev1.record()
                slot.event.record()
        self.stats[(D, Bs, R, B)] += 1

    def _launch(self, ln: Lane, L: int, jobs: list, slot: _BSlot, pk) -> None:
        B = len(jobs)
        if pk is not None and pk[0] == "mx":   # MX
            self._launch_mix(ln, L, jobs, slot, pk)
            return
        if pk is not None:
            from .batchfwd import packed_layout

            Bs, (R, graph, meta, out) = pk
            P = meta.pfx
            pls = [j.pl for j in jobs] if P else None
            st, sl, rp, la = packed_layout([j.n for j in jobs], Bs, L, R, P, plens=pls)
            if meta.blob:
                if self.lean and not self.trace_every and not getattr(self, "timing", False) and self._lean_setup():
                    self._launch_lean(ln, L, jobs, slot, graph, meta, out, st, sl, rp, la)
                    self.stats[(L, Bs, R, B)] += 1
                    return
                self._launch_blob(ln, L, jobs, slot, Bs, R, graph, meta, out, st, sl, rp, la)
                return
            ids = slot.ids_np
            r = 0
            for j in jobs:
                p = j.pl if P else 0
                ids[r:r + j.n - p] = j.ids[p:]
                r += j.n - p
            ids[r:R] = 0
            slot.start.numpy()[:Bs] = st
            slot.slen.numpy()[:Bs] = sl
            slot.row_pos.numpy()[:R] = rp
            slot.last_np[:Bs] = la
            if P:  # PV: each slot's variant and prefix length (unused slots: variant 0, the template length)
                pv, pl = slot.pvar.numpy(), slot.plen.numpy()
                pv[:Bs] = 0
                pl[:Bs] = P
                for b, j in enumerate(jobs):
                    pv[b], pl[b] = j.pv, j.pl
            if self.trace_every:
                self._trace_begin(ln, slot, (L, Bs, R, B), jobs)
            with torch.cuda.stream(ln.stream):
                if slot.ev0 is not None and self.trace_every:
                    slot.ev0.record()
                if self.l0t is not None and L in self.l0t.books:   # L0T (kai): misses computed on this stream first
                    slot.l0t_slot.numpy()[:R] = self.l0t.prepare(L, ids[:R])
                    meta.l0t_slot.copy_(slot.l0t_slot[:R], non_blocking=True)
                meta.ids.copy_(slot.ids[:R], non_blocking=True)
                meta.start.copy_(slot.start[:Bs], non_blocking=True)
                meta.slen.copy_(slot.slen[:Bs], non_blocking=True)
                meta.row_pos.copy_(slot.row_pos[:R], non_blocking=True)
                meta.last.copy_(slot.last[:Bs], non_blocking=True)
                if P:
                    meta.pvar.copy_(slot.pvar[:Bs], non_blocking=True)
                    meta.plen.copy_(slot.plen[:Bs], non_blocking=True)
                graph.replay()
                slot.out[:B].copy_(out[:B], non_blocking=True)
                if self.trace_every:
                    slot.ev1.record()
                slot.event.record()
            self.stats[(L, Bs, R, B)] += 1
            return
        if B == 1 and L in ln.single:
            graph, s_ids, s_last, out = ln.single[L]
        else:
            graph, s_ids, s_last, out, _ws = ln.batch[(L, B)]
        ids = slot.ids_np
        for r, j in enumerate(jobs):
            row = ids[r * L:(r + 1) * L]
            row[:j.n] = j.ids
            row[j.n:] = 0
            slot.last_np[r] = j.n - 1
        if self.trace_every:
            self._trace_begin(ln, slot, (L, B, 0, B), jobs)
        with torch.cuda.stream(ln.stream):
            if self.trace_every:
                slot.ev0.record()
            s_ids.copy_(slot.ids[:B * L].view(B, L), non_blocking=True)
            s_last.copy_(slot.last[:B], non_blocking=True)
            graph.replay()
            slot.out[:B].copy_(out, non_blocking=True)
            if self.trace_every:
                slot.ev1.record()
            slot.event.record()
        self.stats[(L, B)] += 1

    def _lean_setup(self) -> bool:
        """LL: driver handle and the slots' events (created by one torch record each); False = the torch path."""
        if self._lean_ok is None:
            try:
                self._cu = _LeanCu()
                self._lean_ok = True
            except Exception as exc:  # noqa: BLE001 - the lever must never take serving down
                import sys

                print(f"[s1eng] lever not used: lean launch ({type(exc).__name__}: {exc})", file=sys.stderr, flush=True)
                self._lean_ok = False
        return self._lean_ok

    def _zc_setup(self) -> bool:
        """ZC-IN start-up gate (zcin.py): the lean launch is on, zcin.cubin loads, every staging slot is
        device-accessible and the kernel copies random blobs of every meta size in use exactly like the H2D path
        (zcin.gate), then self_check / mix_check pass on every lane set with their inputs copied by the kernel.
        Never raises: any failure logs 'lever not used: zc (...)' and returns False."""
        from . import zcin

        zc, why, sizes = None, None, []
        try:
            if not self.lean:
                why = "needs the lean launch (S1ENG_LEAN=1, S1ENG_META_BLOB=1)"
            elif not self._lean_setup():
                why = "lean launch unavailable"
            else:
                zc = zcin.load()
                if zc is None:
                    why = "zcin.cubin not loadable on this GPU"
                else:
                    sets = getattr(self.m, "_fcab_sets", None) or {"env": self.lanes}
                    sizes = zcin.meta_sizes(list(sets.values()))
                    why = zcin.gate(zc, self._cu.htod, self.free, sizes, 0)
                    if why is None:
                        why = self._zc_checks(zc, sets)
        except Exception as exc:  # noqa: BLE001 - the lever must never take serving down
            why = f"{type(exc).__name__}: {exc}"
        if why:
            print(f"[s1eng] lever not used: zc ({why})", flush=True)
            return False
        self._zc = zc
        print(f"[s1eng] ZC-IN: copy kernel gate passed ({len(sizes)} meta sizes, {len(self.free)} staging slots, "
              f"self_check / mix_check with zc on): arm key zc available", flush=True)
        return True

    def _zc_checks(self, zc, sets: dict):
        """ZC-IN: self_check (and mix_check for the set's checked mixed pairs) on every lane set with the inputs
        copied by the kernel; nothing is dropped. None = pass, else the reason. The L0T tables are filled again."""
        join_ok = getattr(self.m, "join_ok", None) or {}
        try:
            for name, ls in sets.items():
                def lg(msg, n=name):
                    print(f"[s1eng] ZC set {n}: {msg}", flush=True)
                res = self_check(self.m, ls, log=lg, zc=zc, drop=False)
                if not all(res.values()):
                    return f"self_check with zc failed on set {name}: {res}"
                pairs = sorted({(S_, D_) for ln in ls for (D_, S_) in (getattr(ln, "mbs", None) or {})
                                if join_ok.get((S_, D_))})
                if pairs:
                    mres = mix_check(self.m, ls, pairs, log=lg, zc=zc)
                    if not all(mres.get(p_) for p_ in pairs):
                        return f"mix_check with zc failed on set {name}: {mres}"
        finally:
            _l0t_refill(self.m, self.lanes, log=lambda msg: print(f"[s1eng] {msg}", flush=True))
        return None

    def _launch_lean(self, ln: Lane, L: int, jobs: list, slot: _BSlot, graph, meta, out, st, sl, rp, la) -> None:
        """LL: _launch_blob's work with the CUDA calls through _LeanCu; L0T table slots by a plain lookup when every id
        is in the table and no insert is pending (else Table.prepare on the lane stream, as before)."""
        B = len(jobs)
        ids = stage_blob(meta, slot.b64_np, slot.b32_np, jobs, st, sl, rp, la)
        if self.l0t is not None and L in self.l0t.books:
            o, n = meta.lay64["l0t_slot"]
            bk = self.l0t.books[L]
            s = bk.map.slot_of[ids]
            if not bk.pending.items and int(s.min()) >= 0:
                slot.b64_np[o:o + n] = s
                self.l0t.stats["batches"] += 1
            else:
                with torch.cuda.stream(ln.stream):
                    slot.b64_np[o:o + n] = self.l0t.prepare(L, ids)
        cu, h = self._cu, ln.stream.cuda_stream
        if self.zc:   # ZC-IN: one copy kernel on the lane stream (pinned staging read through UVA), no H2D copies
            r = self._zc.launch(meta.b64.data_ptr(), slot.zc64, meta.n64 * 8, meta.b32.data_ptr(), slot.zc32,
                                meta.n32 * 4, h)
        else:
            r = (cu.htod(meta.b64.data_ptr(), slot.b64.data_ptr(), meta.n64 * 8, h)
                 or cu.htod(meta.b32.data_ptr(), slot.b32.data_ptr(), meta.n32 * 4, h))
        if (r
                or cu.launch(graph.raw_cuda_graph_exec(), h)
                or cu.dtoh(slot.out.data_ptr(), out.data_ptr(), B * out.shape[1] * out.element_size(), h)):
            raise RuntimeError("lean launch: CUDA driver call failed")
        ev = slot.event.cuda_event
        if not ev:
            slot.event.record(ln.stream)       # first use of this slot's event: torch creates and records it
        elif cu.record(ev, h):
            raise RuntimeError("lean launch: cuEventRecord failed")

    def _launch_blob(self, ln: Lane, L: int, jobs: list, slot: _BSlot, Bs: int, R: int, graph, meta, out, st, sl, rp,
                     la) -> None:
        """E4 (S1ENG_META_BLOB): _launch's packed path with the inputs staged in the slot's pinned blobs at the meta's
        offsets (the same values as the eight-copy path, padding and unused slots included) and two H2D copies."""
        B = len(jobs)
        ids = stage_blob(meta, slot.b64_np, slot.b32_np, jobs, st, sl, rp, la)
        if self.trace_every:
            self._trace_begin(ln, slot, (L, Bs, R, B), jobs)
        with torch.cuda.stream(ln.stream):
            if slot.ev0 is not None and self.trace_every:
                slot.ev0.record()
            if self.l0t is not None and L in self.l0t.books:   # L0T (kai): misses computed on this stream first
                o, n = meta.lay64["l0t_slot"]
                slot.b64_np[o:o + n] = self.l0t.prepare(L, ids)
            meta.b64.copy_(slot.b64[:meta.n64], non_blocking=True)
            meta.b32.copy_(slot.b32[:meta.n32], non_blocking=True)
            graph.replay()
            slot.out[:B].copy_(out[:B], non_blocking=True)
            if self.trace_every:
                slot.ev1.record()
            slot.event.record()
        self.stats[(L, Bs, R, B)] += 1

    def _trace_begin(self, ln, slot, tag, jobs=()):
        now = time.perf_counter()
        for j in jobs:
            j.t_launch = now
        if self.trace_t0 is None:
            self.trace_t0 = torch.cuda.Event(enable_timing=True)
            self.trace_t0.record(ln.stream)
        slot.tag = (ln.idx,) + tag + (time.perf_counter(),)

    @torch.inference_mode()
    def run(self) -> None:
        while not self.stop:
            with self.cv:
                pick = None
                while not self.stop:
                    self.wake_at = None
                    if self.q and self.free:
                        pick = self._choose()
                        if pick is not None:
                            break
                    if self.wake_at is not None:
                        self.cv.wait(timeout=max(self.wake_at - time.perf_counter(), 0.0) + 1e-4)
                    else:
                        self.cv.wait(timeout=0.5)
                if pick is None:
                    continue
                ln, L, jobs, pk = pick
                for j in jobs:
                    self.q.remove(j)
                slot = self.free.pop()
                ln.busy += 1
                if self.arms:   # ARMS: per-arm batch statistics (printed at each arm switch)
                    self.arm_stats[(L, len(jobs), ln.idx)] += 1
                    self.arm_why[self._why or "?"] += 1
                    if self.bb_join and pk is not None:
                        self.arm_join += sum(1 for j in jobs if bucket_of(j.n) != L)
            try:
                self._launch(ln, L, jobs, slot, pk)
            except Exception as exc:  # noqa: BLE001
                with self.cv:
                    ln.busy -= 1
                    ln.t_free = time.perf_counter()
                    self.free.append(slot)
                for j in jobs:
                    j.done(exc)
                continue
            with self.done_cv[ln.idx]:
                self.done_q[ln.idx].append((jobs, slot, ln))   # FCAB: the lane travels with the batch
                self.done_cv[ln.idx].notify()

    def _completer(self, i: int) -> None:
        dq, dcv = self.done_q[i], self.done_cv[i]
        while True:
            with dcv:
                while not dq:
                    dcv.wait()
                jobs, slot, ln = dq.popleft()   # FCAB: the lane the batch ran on (lane sets may be switched)
            slot.event.synchronize()
            res = slot.out[:len(jobs)].float().numpy().copy()
            if self.trace_every and slot.tag is not None:
                self._trace_end(slot, jobs)
            with self.cv:
                ln.busy -= 1
                ln.t_free = time.perf_counter()
                self.free.append(slot)
                self.back += len(jobs)    # ARRIVE: these clients send their next decision soon
                self.t_back = ln.t_free
                if self.bb_ledger:
                    self.bb_ret.append([ln.t_free, len(jobs)])
                if self.arms:   # BB3 diagnostics: these clients' turnaround starts now
                    self.tr_led.append([ln.t_free, len(jobs)])
                self.cv.notify()
            for r, j in enumerate(jobs):
                j.done(res[r])
            if self.stats_every:
                self.njobs += len(jobs)
                if self.njobs >= self.stats_every:
                    self.njobs = 0
                    self._dump_stats()

    def _trace_end(self, slot, jobs):
        import sys

        t0 = self.trace_t0
        a, b = t0.elapsed_time(slot.ev0), t0.elapsed_time(slot.ev1)
        now = time.perf_counter()
        wait = max(now - j.t_enq for j in jobs)
        qwait = sum((j.t_launch or now) - j.t_enq for j in jobs) / len(jobs)
        T = sum(j.n for j in jobs)
        with self.cv:
            self.trace.append(slot.tag[:5] + (a, b, wait, qwait, T))
            if sum(t[4] for t in self.trace) < self.trace_every:
                return
            tr, self.trace = self.trace, []
        tr.sort(key=lambda t: t[5])
        span = max(t[6] for t in tr) - min(t[5] for t in tr)
        # union of busy intervals and time with >= 2 batches running
        ev = sorted([(t[5], 1) for t in tr] + [(t[6], -1) for t in tr])
        cur, last, busy, two = 0, ev[0][0], 0.0, 0.0
        for x, d in ev:
            if cur >= 1:
                busy += x - last
            if cur >= 2:
                two += x - last
            cur += d
            last = x
        dec = sum(t[4] for t in tr)
        rows = sum(t[3] if t[3] else t[1] * t[2] for t in tr)
        toks = sum(t[9] for t in tr)
        qw = sum(t[8] * t[4] for t in tr) / max(dec, 1)
        gpu = sum(t[6] - t[5] for t in tr)
        from . import server as _srv
        hs = _srv.STATS.copy() if getattr(_srv, "STATS", None) else {}
        if hs:
            _srv.STATS.clear()
        print(f"[s1eng] trace {len(tr)} batches {dec} decisions in {span:.0f} ms: {1000 * dec / span:.1f} dec/s, "
              f"GPU busy {100 * busy / span:.1f} %, 2+ lanes {100 * two / span:.1f} %, mean batch {gpu / len(tr):.2f} ms "
              f"for {dec / len(tr):.2f} dec / {rows / len(tr):.0f} rows ({1000 * gpu / max(rows, 1):.1f} us batch time per row), "
              f"max queue wait {max(t[7] for t in tr) * 1000:.0f} ms, mean queue wait {qw * 1000:.1f} ms, real tokens "
              f"{100 * toks / max(rows, 1):.1f} % of rows"
              + (f"; http: {hs['n']} requests, tokenize {1000 * hs['tok'] / hs['n']:.2f} ms, submit->result "
                 f"{1000 * hs['gpu'] / hs['n']:.1f} ms, result->sent {1000 * hs['send'] / hs['n']:.2f} ms" if hs.get('n') else ""),
              file=sys.stderr, flush=True)

    def _dump_stats(self):
        import sys

        st = dict(self.stats)
        self.stats.clear()
        dec = sum(k[-1] * v for k, v in st.items() if len(k) == 4) + sum(k[1] * v for k, v in st.items() if len(k) == 2)
        nb = sum(st.values())
        hist = collections.Counter()
        for k, v in st.items():
            hist[k[-1] if len(k) == 4 else k[1]] += v
        print(f"[s1eng] batches {nb}, decisions {dec}, mean B {dec / max(nb, 1):.2f}, B histogram "
              f"{dict(sorted(hist.items()))}", file=sys.stderr, flush=True)
