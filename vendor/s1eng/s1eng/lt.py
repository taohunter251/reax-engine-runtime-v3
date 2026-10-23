"""Pinned cuBLASLt GEMMs for stacked / packed decision rows (S4/S5).

The reference computes every decoder linear layer per decision as torch F.linear on [1, bucket, K] (cuBLAS
cublasGemmEx, whose kernel choice depends on M = bucket). Running the rows of several decisions through ONE GEMM is
bit-exact only if every row is computed by the kernel/tile/split-K/reduction the reference gets for M = bucket.

At start-up, on the actual GPU, for every (N, K, bucket):
  1. ask cuBLASLt's heuristic for the reference problem (M = bucket, the layout torch passes to cuBLAS);
  2. run each candidate at M = bucket on random data and keep the first one whose output equals torch's F.linear
     bit for bit (two different inputs);
  3. run that algo on stacked rows (Mmax = Bmax * bucket, and a short packed M) and check every row block against
     torch at M = bucket, bit for bit, inside a CUDA graph capture.
Shapes that fail any step are not pinned: the batched forward then runs them per decision through torch (exact by
construction). Nothing here is hard-coded per GPU.

torch's F.linear layout: out[M, N] = x[M, K] @ W[N, K]^T, column-major D(N x M) = op_T(W: K x N, ld K) * op_N(x: K x M,
ld K), COMPUTE_32F, scale 32F, bf16 matrices, beta 0.
"""
from __future__ import annotations

import ctypes as C
import glob
import os
import sys
import time

import torch

from . import ng
from .gov import GOV

COMPUTE_32F = 68
R_32F, R_16BF = 0, 14
OP_N, OP_T = 0, 1
DESC_TRANSA, DESC_TRANSB, DESC_SM_COUNT_TARGET = 3, 4, 15
PREF_MAX_WS = 1
PREF_ALIGN_A, PREF_ALIGN_B, PREF_ALIGN_C, PREF_ALIGN_D = 5, 6, 7, 8
HEUR_WS = 32 * 1024 * 1024  # what cuBLAS uses for its own heuristic query on this torch build (S1 trace)


def _lib():
    import torch as _t  # noqa: F401  (torch loads its cuBLASLt first; dlopen of the same file shares the handle)
    cands = []
    for base in (os.path.dirname(_t.__file__) + "/../nvidia/cu13/lib", os.path.dirname(_t.__file__) + "/../nvidia/cublas/lib",
                 os.path.dirname(_t.__file__) + "/lib"):
        cands += sorted(glob.glob(os.path.join(base, "libcublasLt.so*")))
    if not cands:
        raise OSError("libcublasLt not found next to torch")
    return C.CDLL(os.path.realpath(cands[0]))


lt = _lib()


class Algo(C.Structure):
    _fields_ = [("data", C.c_uint64 * 8)]


class HeurResult(C.Structure):
    _fields_ = [("algo", Algo), ("workspaceSize", C.c_size_t), ("state", C.c_int), ("wavesCount", C.c_float),
                ("reserved", C.c_int * 4)]


lt.cublasLtMatmul.argtypes = [C.c_void_p] * 12 + [C.POINTER(Algo), C.c_void_p, C.c_size_t, C.c_void_p]
lt.cublasLtMatmul.restype = C.c_int
lt.cublasLtMatrixLayoutCreate.argtypes = [C.POINTER(C.c_void_p), C.c_int, C.c_uint64, C.c_uint64, C.c_int64]
lt.cublasLtMatmulAlgoGetHeuristic.argtypes = [C.c_void_p] * 7 + [C.c_int, C.POINTER(HeurResult), C.POINTER(C.c_int)]
lt.cublasLtMatmulAlgoCheck.argtypes = [C.c_void_p] * 6 + [C.POINTER(Algo), C.POINTER(HeurResult)]
lt.cublasLtMatmulAlgoConfigGetAttribute.argtypes = [C.POINTER(Algo), C.c_int, C.c_void_p, C.c_size_t,
                                                    C.POINTER(C.c_size_t)]
lt.cublasLtMatmulAlgoConfigSetAttribute.argtypes = [C.POINTER(Algo), C.c_int, C.c_void_p, C.c_size_t]

CFG = {"algoId": (0, C.c_int32), "tile": (1, C.c_uint32), "splitK": (2, C.c_int32), "reduction": (3, C.c_uint32),
       "swizzle": (4, C.c_uint32), "custom": (5, C.c_uint32), "stages": (6, C.c_uint32), "inner": (7, C.c_uint16),
       "cluster": (8, C.c_uint16)}


def _chk(st, what):
    if st != 0:
        raise RuntimeError(f"cuBLASLt {what} failed with status {st}")


def _set(fn, obj, attr, val, ctype):
    v = ctype(val)
    _chk(fn(obj, attr, C.byref(v), C.sizeof(v)), f"{fn.__name__}({attr})")


def cfg_of(algo: Algo) -> dict:
    if getattr(algo, "is_kf", False):  # KF (ruj gemm2/k2): our exact in-CTA split-K GEMM of the pin's bit class
        return {"algoId": -3, "tile": -1, "splitK": algo.split, "reduction": algo.red, "swizzle": 0, "custom": 0,
                "stages": -1, "inner": 0, "cluster": 0, "kf": f"{algo.name}/{'pad' if algo.pad else 'tail'}"}
    if isinstance(algo, ng.NgAlgo):  # NG3 (rgh): a native sm_120 GEMM of the pin's bit class
        return {"algoId": -2, "tile": -1, "splitK": algo.split, "reduction": algo.red, "swizzle": 0, "custom": 0,
                "stages": -1, "inner": 0, "cluster": 0, "ng": f"{algo.name}/{algo.drop}"}
    out = {}
    for k, (attr, ct) in CFG.items():
        v = ct(0)
        w = C.c_size_t(0)
        if lt.cublasLtMatmulAlgoConfigGetAttribute(C.byref(algo), attr, C.byref(v), C.sizeof(v), C.byref(w)) == 0:
            out[k] = int(v.value)
    return out


def cfg_key(algo: Algo) -> tuple:
    if getattr(algo, "is_kf", False):  # KF (ruj gemm2/k2)
        return algo.key()
    if isinstance(algo, ng.NgAlgo):
        return algo.key()
    c = cfg_of(algo)
    return tuple(c.get(k, -1) for k in CFG)


def with_split(algo: Algo, split_k: int, reduction: int) -> Algo | None:
    """Copy of algo with the given split-K count and reduction scheme (None if cuBLASLt refuses the attributes)."""
    a = Algo()
    C.memmove(C.byref(a), C.byref(algo), C.sizeof(Algo))
    for attr, val in ((CFG["splitK"][0], C.c_int32(split_k)), (CFG["reduction"][0], C.c_uint32(reduction))):
        if lt.cublasLtMatmulAlgoConfigSetAttribute(C.byref(a), attr, C.byref(val), C.sizeof(val)) != 0:
            return None
    return a


class Problem:
    def __init__(self, ctx: "Lt", N: int, K: int, M: int):
        self.N, self.K, self.M = N, K, M
        self.op = C.c_void_p()
        _chk(lt.cublasLtMatmulDescCreate(C.byref(self.op), COMPUTE_32F, R_32F), "DescCreate")
        _set(lt.cublasLtMatmulDescSetAttribute, self.op, DESC_TRANSA, OP_T, C.c_int32)
        _set(lt.cublasLtMatmulDescSetAttribute, self.op, DESC_TRANSB, OP_N, C.c_int32)
        _set(lt.cublasLtMatmulDescSetAttribute, self.op, DESC_SM_COUNT_TARGET, ctx.sms, C.c_int32)
        self.A, self.B, self.D = C.c_void_p(), C.c_void_p(), C.c_void_p()
        _chk(lt.cublasLtMatrixLayoutCreate(C.byref(self.A), R_16BF, K, N, K), "LayoutA")
        _chk(lt.cublasLtMatrixLayoutCreate(C.byref(self.B), R_16BF, K, M, K), "LayoutB")
        _chk(lt.cublasLtMatrixLayoutCreate(C.byref(self.D), R_16BF, N, M, N), "LayoutD")


class Lt:
    def __init__(self):
        self.handle = C.c_void_p()
        _chk(lt.cublasLtCreate(C.byref(self.handle)), "Create")
        self.sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self.alpha, self.beta = C.c_float(1.0), C.c_float(0.0)
        self._probs: dict = {}
        self.pref = C.c_void_p()
        _chk(lt.cublasLtMatmulPreferenceCreate(C.byref(self.pref)), "PrefCreate")
        _set(lt.cublasLtMatmulPreferenceSetAttribute, self.pref, PREF_MAX_WS, HEUR_WS, C.c_uint64)
        for a in (PREF_ALIGN_A, PREF_ALIGN_B, PREF_ALIGN_C, PREF_ALIGN_D):
            _set(lt.cublasLtMatmulPreferenceSetAttribute, self.pref, a, 16, C.c_uint32)

    def problem(self, N, K, M) -> Problem:
        key = (N, K, M)
        if key not in self._probs:
            self._probs[key] = Problem(self, N, K, M)
        return self._probs[key]

    def heuristic(self, N, K, M, n=32) -> list[Algo]:
        prob = self.problem(N, K, M)
        res = (HeurResult * n)()
        got = C.c_int(0)
        st = lt.cublasLtMatmulAlgoGetHeuristic(self.handle, prob.op, prob.A, prob.B, prob.D, prob.D, self.pref, n, res,
                                               C.byref(got))
        if st != 0:
            return []
        out = []
        for i in range(got.value):
            if res[i].state == 0:
                a = Algo()
                C.memmove(C.byref(a), C.byref(res[i].algo), C.sizeof(Algo))
                out.append(a)
        return out

    def ws_needed(self, N, K, M, algo: Algo) -> int | None:
        if getattr(algo, "is_kf", False):  # KF (ruj gemm2/k2): no workspace
            return 0
        if isinstance(algo, ng.NgAlgo):
            return 0
        prob = self.problem(N, K, M)
        r = HeurResult()
        st = lt.cublasLtMatmulAlgoCheck(self.handle, prob.op, prob.A, prob.B, prob.D, prob.D, C.byref(algo), C.byref(r))
        return int(r.workspaceSize) if st == 0 else None

    def run(self, algo: Algo, W: torch.Tensor, x: torch.Tensor, out: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
        """out[M, N] = x[M, K] @ W[N, K]^T on the current stream; x/W/out contiguous bf16."""
        if getattr(algo, "is_kf", False):  # KF (ruj gemm2/k2)
            from . import kf

            return kf.run(algo, W, x, out)
        if isinstance(algo, ng.NgAlgo):
            return ng.run(algo, W, x, out, ws)
        M, K = x.shape
        N = W.shape[0]
        prob = self.problem(N, K, M)
        _chk(lt.cublasLtMatmul(self.handle, prob.op, C.byref(self.alpha), C.c_void_p(W.data_ptr()), prob.A,
                               C.c_void_p(x.data_ptr()), prob.B, C.byref(self.beta), C.c_void_p(out.data_ptr()),
                               prob.D, C.c_void_p(out.data_ptr()), prob.D, C.byref(algo),
                               C.c_void_p(ws.data_ptr()), ws.numel(), C.c_void_p(torch.cuda.current_stream().cuda_stream)),
             "Matmul")
        return out


def _time_ms(fn, reps: int) -> float:
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn()
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    e.synchronize()
    ms = s.elapsed_time(e) / reps
    GOV.tick()
    return ms


def _eq(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


class Pinned:
    """Per (N, K, bucket): the verified algo (or None = per-decision torch fallback) + workspace sizing."""

    def __init__(self, ctx: Lt):
        self.ctx = ctx
        self.algo: dict = {}
        self.cfg: dict = {}
        self.ws_bytes = 0
        self.report: dict = {}
        self.algo_r: dict = {}   # (N, K, bucket, R) -> faster algo of the same bit class (S6), else the pinned one
        self.tune_report: dict = {}
        self.parts: dict = {}    # (N, K) of a fused projection -> its part weights (layer 0) whose outputs it concatenates
        self.best_t: dict = {}   # (N, K, bucket, anchor R) -> time (us) of the algo used at that anchor (S6 timing)
        self.fuse_r: dict = {}   # (N, K, bucket, R) of a fused projection -> True when the fused GEMM is used at R

    def ref_linear(self, N: int, K: int, W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """The reference op for one M = bucket block: torch F.linear (per part for a fused projection: the reference
        computes each projection on its own)."""
        parts = self.parts.get((N, K))
        if parts is None:
            return torch.nn.functional.linear(x.unsqueeze(0), W).squeeze(0)
        return torch.cat([torch.nn.functional.linear(x.unsqueeze(0), p).squeeze(0) for p in parts], dim=1)

    def base_keys(self, M: int) -> list:
        return [k for k in self.algo if k[2] == M and (k[0], k[1]) not in self.parts]

    def unpin(self, key, reason: str) -> None:
        """(N, K, bucket) failed a check: no pinned algo (stacked buckets run that shape per decision through torch,
        packed buckets are not packed) and no S6 alternative."""
        self.algo[key] = None
        self.cfg.pop(key, None)
        for k in [k for k in self.algo_r if k[:3] == key]:
            del self.algo_r[k]
        N, K, M = key
        self.report[f"{N}x{K}@{M}"] = f"UNPINNED: {reason}"

    def select(self, shapes, buckets, mmax_of, mins_of, log=print, fused=()) -> None:
        """shapes: iterable of (N, K, W) with a real weight W [N, K]; buckets: iterable of M; mmax_of(bucket) -> the
        largest stacked M used; mins_of(bucket) -> extra (small) M values used (packed / pruned rows).
        fused: iterable of (N, K, W, parts, buckets): a fused projection (W = the parts' weights stacked as rows) whose
        reference is the parts' F.linear outputs side by side; candidates are the parts' pinned algos (one bit class
        for both parts) and the fused shape's heuristic algos forced to that class."""
        ctx = self.ctx
        g = torch.Generator(device="cuda")
        g.manual_seed(1234)
        ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        todo = [(N, K, W, None, buckets) for N, K, W in shapes] + [tuple(f) for f in fused]
        for N, K, W, parts, bks in todo:
            if parts is not None:
                assert (N, K) not in {(n, k) for n, k, *_ in shapes}, "fused shape collides with a base shape"
                self.parts[(N, K)] = parts
            for M in bks:
                key = (N, K, M)
                ref_in, ref_out = [], []
                for scale in (1.0, 8.0):
                    x = (torch.randn((M, K), generator=g, device="cuda", dtype=torch.float32) * scale).to(torch.bfloat16)
                    ref_in.append(x)
                    ref_out.append(self.ref_linear(N, K, W, x))
                chosen = None
                if parts is None:
                    cands = ctx.heuristic(N, K, M)
                else:
                    pa = [self.algo.get((p.shape[0], p.shape[1], M)) for p in parts]
                    if any(a is None for a in pa):
                        self.algo[key] = None
                        self.report[f"{N}x{K}@{M}"] = "fused: a part has no pinned algo"
                        continue
                    pc = {(cfg_of(a)["splitK"], cfg_of(a)["reduction"]) for a in pa}
                    if len(pc) != 1:
                        self.algo[key] = None
                        self.report[f"{N}x{K}@{M}"] = f"fused: parts in different bit classes {sorted(pc)}"
                        continue
                    (sk, rd), = pc
                    cands = list(pa)
                    for a in ctx.heuristic(N, K, M):
                        v = with_split(a, sk, rd)
                        if v is not None:
                            cands.append(v)
                for i, a in enumerate(cands):
                    need = ctx.ws_needed(N, K, M, a)
                    if need is None or need > ws.numel():
                        continue
                    ok = True
                    for x, y in zip(ref_in, ref_out):
                        out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
                        try:
                            ctx.run(a, W, x, out, ws)
                        except RuntimeError:
                            ok = False
                            break
                        if not _eq(out, y):
                            ok = False
                            break
                    if ok:
                        chosen = (i, a)
                        break
                if chosen is None:
                    self.algo[key] = None
                    self.report[f"{N}x{K}@{M}"] = f"no exact heuristic algo among {len(cands)}"
                    continue
                i, a = chosen
                # stacked / packed rows: every row block must equal torch at M = bucket
                mmax = mmax_of(M)
                bmax = mmax // M
                xs = torch.cat([(torch.randn((M, K), generator=g, device="cuda") * (1 + b % 3)).to(torch.bfloat16)
                                for b in range(bmax)])
                ys = torch.cat([self.ref_linear(N, K, W, xs[b * M:(b + 1) * M]) for b in range(bmax)])
                ok = True
                why = ""
                mlist = sorted({mmax} | {m for m in mins_of(M) if m <= mmax})
                need_max = 0
                for Mx in mlist:
                    need = ctx.ws_needed(N, K, Mx, a)
                    if need is None or need > ws.numel():
                        ok, why = False, f"algo check failed at M={Mx}"
                        break
                    need_max = max(need_max, need)
                    out = torch.empty((Mx, N), dtype=torch.bfloat16, device="cuda")
                    try:
                        ctx.run(a, W, xs[:Mx].contiguous(), out, ws)
                    except RuntimeError as exc:
                        ok, why = False, f"matmul failed at M={Mx}: {exc}"
                        break
                    if not _eq(out, ys[:Mx]):
                        ok, why = False, f"NOT exact on stacked rows at M={Mx}"
                        break
                if ok:
                    # once more inside a CUDA graph capture (replayed with new inputs)
                    xin = xs[:mmax].clone()
                    out = torch.empty((mmax, N), dtype=torch.bfloat16, device="cuda")
                    try:
                        s = torch.cuda.Stream()
                        s.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(s):
                            ctx.run(a, W, xin, out, ws)
                        torch.cuda.current_stream().wait_stream(s)
                        gr = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(gr):
                            ctx.run(a, W, xin, out, ws)
                        xin.copy_(torch.roll(xs[:mmax], M, 0))
                        out.zero_()
                        gr.replay()
                        torch.cuda.synchronize()
                        GOV.tick()
                        ok = _eq(out, torch.roll(ys[:mmax], M, 0))
                        why = "" if ok else "NOT exact inside a CUDA graph replay"
                        del gr
                    except RuntimeError as exc:
                        ok, why = False, f"graph capture/replay failed: {exc}"
                if ok:
                    self.algo[key] = a
                    self.cfg[key] = cfg_of(a)
                    self.ws_bytes = max(self.ws_bytes, need_max)
                    self.report[f"{N}x{K}@{M}"] = f"heur#{i} {self.cfg[key]}"
                else:
                    self.algo[key] = None
                    self.report[f"{N}x{K}@{M}"] = f"heur#{i} exact at M=bucket but {why}"
        self._ws = ws
        self._gen = g
        torch.cuda.synchronize()
        nb = [k for k in self.algo if (k[0], k[1]) not in self.parts]
        nf = [k for k in self.algo if (k[0], k[1]) in self.parts]
        log(f"cuBLASLt pins: {sum(self.algo[k] is not None for k in nb)}/{len(nb)} (N,K,bucket) verified, fused "
            f"projections {sum(self.algo[k] is not None for k in nf)}/{len(nf)}; workspace {self.ws_bytes / 2**20:.1f} MiB")


def _time_pair_ms(fn_a, fn_b, reps: int, sa, sb) -> float:
    """Wall time per call pair with fn_a on stream sa and fn_b on stream sb running concurrently (aggregate efficiency
    when two lanes share the GPU)."""
    cur = torch.cuda.current_stream()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(cur)
    sa.wait_stream(cur)
    sb.wait_stream(cur)
    with torch.cuda.stream(sa):
        for _ in range(reps):
            fn_a()
    with torch.cuda.stream(sb):
        for _ in range(reps):
            fn_b()
    cur.wait_stream(sa)
    cur.wait_stream(sb)
    e.record(cur)
    e.synchronize()
    ms = s.elapsed_time(e) / reps
    GOV.tick()
    return ms


def _table_offer(pinned: "Pinned", table, N: int, K: int, L: int, W: torch.Tensor, nb: int, rc: dict, ws, cands: dict,
                 tg, log=print, ws_cap: int | None = None) -> tuple[dict, dict, dict]:
    """SEL2b (kai, gtable): the table's configurations of (N, K, bucket L) in the pin's split class join ``cands``
    (those not already there), each rebuilt from its attributes, AlgoCheck'ed for nb * L rows with a workspace need of
    at most ``ws_cap`` (default: all of ``ws``; with pair timing the smaller ws_b) and proven on two extra inputs (scale
    8; randn x 0.5 with 1 % of the entries x 64: the inputs on which SEL2-quick's unstable configs failed), each
    compared with the per-decision F.linear, the pair twice. ``cands`` changes only at the end (an exception leaves it
    as it was); a shape without table configurations returns before any GPU work. Returns (table-only key -> R values
    it is the table's pick for, R -> key of the table's pick, rejections by reason)."""
    from . import gtable

    offered = table.configs(N, K, L, cls=gtable.pin_class(rc), limit=getattr(table, "max", gtable.DEFAULT_MAX))
    if not offered:
        return {}, {}, {}
    cap = ws.numel() if ws_cap is None else ws_cap
    rows = nb * L
    xs8 = (torch.randn((rows, K), generator=tg, device="cuda") * 8).to(torch.bfloat16)
    o = torch.randn((rows, K), generator=tg, device="cuda") * 0.5
    xo = torch.where(torch.rand((rows, K), generator=tg, device="cuda") < 0.01, o * 64, o).to(torch.bfloat16)
    del o
    refs = [torch.cat([pinned.ref_linear(N, K, W, x[b * L:(b + 1) * L]) for b in range(nb)]) for x in (xs8, xo)]
    out = torch.empty((rows, N), dtype=torch.bfloat16, device="cuda")
    ctx = pinned.ctx

    def check(v, x, y):
        ctx.run(v, W, x, out, ws)
        return _eq(out, y)
    table_rs, pick_at, rejected, add = {}, {}, {}, {}
    for cfg, rs in offered:
        v = gtable.build(sys.modules[__name__], ctx.handle, cfg)
        if v is None:
            rejected["build"] = rejected.get("build", 0) + 1
            continue
        need = ctx.ws_needed(N, K, rows, v)
        if need is None or need > cap:
            why = "algocheck" if need is None else "workspace"
            rejected[why] = rejected.get(why, 0) + 1
            continue
        k = cfg_key(v)
        if k not in cands and k not in add:
            why = gtable.prove([lambda v=v: check(v, xs8, refs[0]), lambda v=v: check(v, xo, refs[1])])
            if why:
                rejected[why] = rejected.get(why, 0) + 1
                continue
            add[k] = v
            table_rs[k] = list(rs)
        for r in rs:
            pick_at[r] = k
    torch.cuda.synchronize()
    cands.update(add)
    return table_rs, pick_at, rejected


def tune(pinned: "Pinned", weights: dict, groups_of, log=print, min_gain: float = 0.02, reps: int = 3,
         pair: bool = False, refine_k: int = 0, refine_reps: int = 2, refine_min_n: int = 0, table=None) -> None:
    """S6: per (N, K, bucket) and row-count group, use the FASTEST algo of the pinned algo's bit class.

    Configs that differ from the reference algo only in tile shape / stages (same split-K and reduction) compute every
    output element with the same K order, so they give the same bits; they are found here, not assumed: candidates are
    the heuristic's algos for M = bucket .. 16 * bucket, each also forced to the reference split-K and reduction; a
    candidate is kept only if it reproduces torch's per-decision output (M = bucket blocks) bit for bit on stacked
    random rows and at every row count it is assigned to. Timing at one anchor row count per group, weights rotated
    over the layers (streamed from DRAM as in the model). A candidate replaces the pinned algo only if it is at least
    min_gain faster at that anchor.
      weights: (N, K) -> list of weight tensors (all layers of that shape)
      groups_of(bucket) -> {anchor R: [R values that use the anchor's winner]}
    refine_k > 0: afterwards every other R of a group is timed on its own with the anchor's refine_k fastest exact
    candidates (and the pinned algo), each verified bit-exact at that R (shapes with N >= refine_min_n).
    table (SEL2b, kai; gtable.Table or None = off): its configurations of each (N, K, bucket) join the candidates after
    extra checks (_table_offer); an anchor times those that are the table's pick at an R of its group, and refinement
    adds the table's pick for each R to the anchor's top. Winners are logged as table / ng / heur / pin."""
    ctx = pinned.ctx
    ws = pinned._ws
    g = pinned._gen
    t0 = time.time()
    nrep = 0
    pair_top = int(os.environ.get("S1ENG_TUNE_PAIR_TOP", "0"))   # 0: every exact candidate timed paired
    # SD (rgh): past this many seconds since start-up began, remaining (shape, bucket) groups keep their exact pins
    deadline = float(os.environ.get("S1ENG_TUNE_DEADLINE_S", "500"))
    n_late = 0
    if pair:
        sa, sb = torch.cuda.Stream(), torch.cuda.Stream()
        ws_b = torch.empty_like(ws[: max(pinned.ws_bytes, 64 << 20)])
    if table is not None:   # SEL2b (kai)
        from . import gtable

        tg = torch.Generator(device="cuda")
        tg.manual_seed(gtable.SEED)
        pinned.table_kinds, pinned.table_stats = {}, {"offered": 0, "rejected": {}, "timed": 0, "refine_picks": 0}
        t_cap = gtable.ws_cap(ws.numel(), ws_b.numel() if pair else None)   # pair timing runs on ws_b too
        t_all = gtable.anchor_fallback(table)
    for (N, K, L), ref in list(pinned.algo.items()):
        if ref is None:
            continue
        if GOV.elapsed() > deadline:
            n_late += 1
            continue
        groups = groups_of(L)
        if not groups:
            continue
        Ws = weights[(N, K)][:8]
        W = Ws[0]
        rmax = max(max(rs) for rs in groups.values())
        nb = -(-rmax // L)
        xs = torch.cat([(torch.randn((L, K), generator=g, device="cuda") * (1 + b % 3)).to(torch.bfloat16)
                        for b in range(nb)])
        ys = torch.cat([pinned.ref_linear(N, K, W, xs[b * L:(b + 1) * L]) for b in range(nb)])
        rc = cfg_of(ref)
        cands = {cfg_key(ref): ref}
        for mq in sorted({L, 2 * L, 4 * L, 8 * L, 16 * L}):
            for a in ctx.heuristic(N, K, mq, n=32):
                for v in (a, with_split(a, rc["splitK"], rc["reduction"])):
                    if v is None:
                        continue
                    k = cfg_key(v)
                    if k in cands:
                        continue
                    need = ctx.ws_needed(N, K, nb * L, v)
                    if need is None or need > ws.numel():
                        continue
                    cands[k] = v
        # NG3 (rgh): native sm_120 GEMMs that follow the pin's rule join the candidates; exactness and speed decide
        for v in ng.candidates(N, K, rc["splitK"], rc["reduction"]):
            cands[v.key()] = v
        if table is not None:   # SEL2b (kai): the table's configurations join the pool after their extra checks
            kinds = {k: ("ng" if isinstance(v, ng.NgAlgo) else "heur") for k, v in cands.items()}
            try:
                table_rs, pick_at, rej = _table_offer(pinned, table, N, K, L, W, nb, rc, ws, cands, tg, log=log,
                                                      ws_cap=t_cap)
            except Exception as exc:  # noqa: BLE001 - the lever must never take the batch lanes down with it
                why = f"{type(exc).__name__}: {exc}"
                log(f"lever not used: gemm_table (from {N}x{K}@{L} on: {why})")
                pinned.table_stats["failed"] = f"{N}x{K}@{L}: {why}"
                table = gtable.disabled(table, why)
                table_rs, pick_at, rej = {}, {}, {}
            kinds.update({k: "table" for k in table_rs})
            pinned.table_kinds[(N, K, L)] = kinds
            pinned.table_stats["offered"] += len(table_rs)
            for why, n in rej.items():
                pinned.table_stats["rejected"][why] = pinned.table_stats["rejected"].get(why, 0) + n
        exact = {}
        for k, v in cands.items():
            out = torch.empty((nb * L, N), dtype=torch.bfloat16, device="cuda")
            try:
                ctx.run(v, W, xs, out, ws)
            except RuntimeError:
                continue
            if _eq(out, ys):
                exact[k] = v
        if pinned.algo.get((N, K, L)) is None:
            continue
        for anchor, rs in groups.items():
            if pinned.algo.get((N, K, L)) is None or GOV.elapsed() > deadline:
                break
            x = xs[:anchor].contiguous()
            out = torch.empty((anchor, N), dtype=torch.bfloat16, device="cuda")
            times = {}
            if table is not None:   # SEL2b (kai): table-only configs timed here: the picks of this group's R
                tgroup = gtable.group_keys({k: rs_ for k, rs_ in table_rs.items() if k in exact}, rs,
                                           fallback_all=t_all)
                t_fit = set()
            for k, v in exact.items():
                if table is not None and k in table_rs and k not in tgroup:
                    continue
                if table is not None and k in table_rs:   # SEL2b (kai): fits ws (and ws_b) and exact at every R
                    why = gtable.fits_group(
                        sorted(set(rs) | {anchor}), lambda r, v=v: ctx.ws_needed(N, K, r, v),
                        lambda r, v=v: _eq(ctx.run(v, W, xs[:r].contiguous(),
                                                   torch.empty((r, N), dtype=torch.bfloat16, device="cuda"), ws),
                                           ys[:r]), t_cap)
                    if why:
                        rj = pinned.table_stats["rejected"]
                        rj["group:" + why.split("@")[0]] = rj.get("group:" + why.split("@")[0], 0) + 1
                        continue
                    t_fit.add(k)
                need = ctx.ws_needed(N, K, anchor, v)
                if need is None or need > ws.numel():
                    continue
                try:
                    ctx.run(v, W, x, out, ws)
                except RuntimeError:
                    continue
                if not _eq(out, ys[:anchor]):
                    continue

                def run(v=v):
                    for w in Ws:
                        ctx.run(v, w, x, out, ws)
                if pair and not pair_top:
                    out_b = torch.empty_like(out)

                    def run_b(v=v):
                        for w in Ws[::-1]:
                            ctx.run(v, w, x, out_b, ws_b)
                    run()
                    run_b()
                    times[k] = _time_pair_ms(run, run_b, reps, sa, sb) / (2 * len(Ws))
                else:
                    times[k] = _time_ms(run, reps) / len(Ws)
                nrep += 1
            kref = cfg_key(ref)
            if table is not None:   # SEL2b (kai): start-up cost of the table (anchor contests it entered)
                pinned.table_stats["timed"] += len(t_fit & set(times))
            if pair and pair_top and times:
                # PT top-k (rgh): the pair_top fastest 1-stream candidates (and the pin) re-timed as two lanes run them
                top = sorted(times, key=times.get)[:pair_top]
                if kref in times and kref not in top:
                    top.append(kref)
                tp = {}
                for k in top:
                    v = exact[k]
                    out_b = torch.empty_like(out)

                    def run(v=v):
                        for w in Ws:
                            ctx.run(v, w, x, out, ws)

                    def run_b(v=v):
                        for w in Ws[::-1]:
                            ctx.run(v, w, x, out_b, ws_b)
                    run()
                    run_b()
                    tp[k] = _time_pair_ms(run, run_b, reps, sa, sb) / (2 * len(Ws))
                    nrep += 1
                times = tp
            if kref not in times:
                # the pinned algo itself does not reproduce the per-decision bits at this row count (select() checks
                # every row count, so this means it failed here): nothing of this (shape, bucket) is used
                pinned.unpin((N, K, L), f"pinned algo not exact or not runnable at R={anchor} in S6 tuning")
                break
            best = min(times, key=times.get)
            if kref in times and times[best] > (1 - min_gain) * times[kref]:
                best = kref
            win = exact[best]
            ok = True
            if best != kref:
                need_max = 0
                for r in rs:  # the winner must also reproduce the bits at every row count it is used for
                    need = ctx.ws_needed(N, K, r, win)
                    if need is None or need > ws.numel():
                        ok = False
                        break
                    need_max = max(need_max, need)
                    o = torch.empty((r, N), dtype=torch.bfloat16, device="cuda")
                    try:
                        ctx.run(win, W, xs[:r].contiguous(), o, ws)
                    except RuntimeError:
                        ok = False
                        break
                    if not _eq(o, ys[:r]):
                        ok = False
                        break
            if ok and best != kref:
                pinned.ws_bytes = max(pinned.ws_bytes, need_max)
                for r in rs:
                    pinned.algo_r[(N, K, L, r)] = win
            pinned.best_t[(N, K, L, anchor)] = 1000.0 * (times[best] if ok else times[kref])
            if refine_k and len(rs) > 1 and N >= refine_min_n:
                top = sorted(times, key=times.get)[:refine_k]
                if kref not in top:
                    top.append(kref)
                if table is not None:
                    top0 = list(top)
                for r in rs:
                    if r == anchor:
                        continue
                    if table is not None:   # SEL2b (kai): the table's own pick for this R competes too
                        top = gtable.with_pick(top0, pick_at.get(r),
                                               {k: v for k, v in exact.items() if k not in table_rs or k in t_fit})
                        pinned.table_stats["refine_picks"] += len(top) - len(top0)
                    xr = xs[:r].contiguous()
                    outr = torch.empty((r, N), dtype=torch.bfloat16, device="cuda")
                    tr = {}
                    for k in top:
                        v = exact[k]
                        need = ctx.ws_needed(N, K, r, v)
                        if need is None or need > ws.numel():
                            continue
                        try:
                            ctx.run(v, W, xr, outr, ws)
                        except RuntimeError:
                            continue
                        if not _eq(outr, ys[:r]):
                            continue

                        def runr(v=v):
                            for w in Ws:
                                ctx.run(v, w, xr, outr, ws)
                        if pair:
                            outr_b = torch.empty_like(outr)

                            def runr_b(v=v):
                                for w in Ws[::-1]:
                                    ctx.run(v, w, xr, outr_b, ws_b)
                            runr()
                            runr_b()
                            tr[k] = _time_pair_ms(runr, runr_b, refine_reps, sa, sb) / (2 * len(Ws))
                        else:
                            tr[k] = _time_ms(runr, refine_reps) / len(Ws)
                        nrep += 1
                    if kref not in tr:
                        pinned.unpin((N, K, L), f"pinned algo not exact or not runnable at R={r} in S6 refinement")
                        break
                    b = min(tr, key=tr.get)
                    if tr[b] > (1 - min_gain) * tr[kref]:
                        b = kref
                    if b == kref:
                        pinned.algo_r.pop((N, K, L, r), None)
                    else:
                        pinned.algo_r[(N, K, L, r)] = exact[b]
                        pinned.ws_bytes = max(pinned.ws_bytes, ctx.ws_needed(N, K, r, exact[b]))
                    pinned.best_t[(N, K, L, r)] = 1000.0 * tr[b]
                if pinned.algo.get((N, K, L)) is None:
                    break
            pinned.tune_report[f"{N}x{K}@{L}/R{anchor}"] = (
                f"{len(exact)}/{len(cands)} exact; ref {times.get(kref, float('nan')) * 1000:.1f} us -> "
                f"{'ref' if best == kref or not ok else cfg_of(win)} {times[best] * 1000:.1f} us")
    torch.cuda.synchronize()
    log(f"S6 tuning{' (paired' + (f' top {pair_top}' if pair_top else '') + ')' if pair else ''}: "
        f"{len(pinned.algo_r)} (shape, bucket, R) entries use a faster exact algo; {nrep} timings in "
        f"{time.time() - t0:.1f}s" + (f"; start-up deadline {deadline:.0f}s passed: {n_late} (shape, bucket) "
                                       f"groups keep their pins" if n_late else ""))
    if table is not None:   # SEL2b (kai): which kind of algo won at every (shape, bucket, R) that S6 timed
        _table_log(pinned, groups_of, log, deadline=deadline)


def _table_log(pinned: "Pinned", groups_of, log=print, deadline: float | None = None) -> None:
    from . import gtable

    every = []
    for (N, K, L), kinds in sorted(pinned.table_kinds.items()):
        ref = pinned.algo.get((N, K, L))
        if ref is None:
            continue
        kref = cfg_key(ref)
        per_r = {}
        for rs in (groups_of(L) or {}).values():
            for r in rs:
                a = pinned.algo_r.get((N, K, L, r))
                per_r[r] = "pin" if a is None else gtable.kind_of(cfg_key(a), kref, kinds)
        every += list(per_r.values())
        log(gtable.winners_line(f"{N}x{K}@{L}", per_r))
    st = pinned.table_stats
    log(f"gemm_table: {st['offered']} table configs offered, rejected {st['rejected']}; S6 winners "
        f"{gtable.tally(every)} over {len(every)} (shape, bucket, R)")
    # SD (job 38): what the table cost and how far S6 ended from the start-up deadline (later groups keep their pins)
    log(f"gemm_table start-up: {st.get('timed', 0)} anchor timings + {st.get('refine_picks', 0)} refinement timings "
        f"of table configs; S6 ended {GOV.elapsed():.0f}s after start-up"
        + (f" (S1ENG_TUNE_DEADLINE_S {deadline:.0f}s)" if deadline is not None else ""))
