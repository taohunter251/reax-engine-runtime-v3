"""KF (ruj gemm2/k2, 2026-10-21): our exact in-CTA split-K GEMM for the split-K pins of the N = 2560 GEMMs (MLP down
2560x9216, GDN out_proj / attention o_proj 2560x4096) and the fused attention k|v GEMM (2048x2560).

The pins of these shapes are cuBLASLt split-K algos with the OUTPUT_TYPE reduction (scheme 4): S ceil64 K slices
(chunk = ceil(ceil(K / 64) / S) * 64; S' = ceil(K / chunk) non-empty slices), each one ascending-k chain of mma.sync
m16n8k16 (fp32 from +0) rounded once to bf16, folded in slice order with fp32 RN adds and rounded once. kf.cubin
(kf.cu, nvcc 13.4 -arch=sm_120a) computes exactly that per output element with one CTA per 128x128 tile walking ALL
slices through ONE continuous cp.async pipeline (the load stream of an unsplit GEMM), rows [mb, M) in small 32 / 64 /
96 x 128 tiles of the same launch (no MMA work on the padded rows of a last 128-row band), and a fold with no tensor-
pipe cost (fp32 totals start at -0.0f; the first k16 step of a slice is a zero-C MMA; the fold of a cell = one F2FP +
one FADD issued before that cell's zero-C MMA). Kernels: gemm_kf_b128_s{32,64,96}[_e] (_e: smem-staged epilogue).

Use: S1ENG_2N2560SPLITKPINSMLPDOWNGDNOUTPROJATTENTIONOPROJATTENTIONKV=1 -> after the S6 / FP picks, every (N, K, bucket,
R) of the built-in table for this GPU model (kf_table_pro6000.json: where the microbench found KF faster than the
engine's path there, two lanes) gets a KfAlgo in pinned.algo_r, kept ONLY where its output equals the per-decision
reference (pinned.ref_linear on M = bucket blocks, as FP's apply_s6) byte for byte at that R on two inputs (random;
x64 outlier rows / columns), each launched twice; every other key keeps its pick. RARN does not take a KF key (its rule
returns today's path = KF GEMM + add_rmsnorm, which the table measured faster). =<path>: that table file. Unset / 0:
the base engine exactly (nothing loaded).
"""
from __future__ import annotations

import ctypes as C
import functools
import json
import os

import torch

ENV = "S1ENG_2N2560SPLITKPINSMLPDOWNGDNOUTPROJATTENTIONOPROJATTENTIONKV"
_DIR = os.path.dirname(os.path.abspath(__file__))
CUBIN = os.path.join(_DIR, "kf.cubin")
TABLES = {"NVIDIA RTX PRO 6000 Blackwell Server Edition": "kf_table_pro6000.json"}
SMEM = 98304   # 128x128x64 x 3 stages (the small-tile configs fit in it)
BM, BN, BK = 128, 128, 64
KERNELS = {f"gemm_kf_b128_{s}{e}": sbm for s, sbm in (("s32", 32), ("s64", 64), ("s96", 96)) for e in ("", "_e")}


def spec() -> str:
    return os.environ.get(ENV, "0")


class KfAlgo:
    """A pseudo cuBLASLt algo (lt.run / ws_needed / cfg_of / cfg_key recognise it by is_kf). pad: big tiles only
    (rows padded); else rows [0, (M // 128) * 128) in 128x128 tiles, the rest in the kernel's small tiles."""

    __slots__ = ("name", "pad", "chunk", "split", "red")
    is_kf = True

    def __init__(self, name: str, pad: bool, chunk: int, split: int, red: int):
        self.name, self.pad, self.chunk, self.split, self.red = name, bool(pad), int(chunk), int(split), int(red)

    def key(self):
        return ("kf", self.name, self.pad, self.chunk)

    def __repr__(self):
        return f"KfAlgo({self.name}, pad={self.pad}, chunk={self.chunk})"


class _Lib:
    def __init__(self):
        self.cu = C.CDLL("libcuda.so.1")
        torch.zeros(1, device="cuda")
        data = open(CUBIN, "rb").read()
        self._img = C.create_string_buffer(data, len(data))
        self.mod = C.c_void_p()
        self._chk(self.cu.cuModuleLoadData(C.byref(self.mod), self._img), "cuModuleLoadData")
        self.fn = {}
        for name in KERNELS:
            f = C.c_void_p()
            self._chk(self.cu.cuModuleGetFunction(C.byref(f), self.mod, name.encode()), name)
            self._chk(self.cu.cuFuncSetAttribute(f, 8, SMEM), "smem attribute")
            self.fn[name] = f

    @staticmethod
    def _chk(r, what):
        if r != 0:
            raise RuntimeError(f"KF {what}: CUresult {r}")


@functools.lru_cache(maxsize=1)
def lib():
    if spec() in ("", "0") or not os.path.exists(CUBIN):
        return None
    try:
        if tuple(torch.cuda.get_device_capability(0)) != (12, 0):
            return None
        return _Lib()
    except Exception as exc:  # noqa: BLE001 - no KF on any loading problem
        print(f"[s1eng] KF kernels unavailable: {exc!r}", flush=True)
        return None


def chunk_of(K: int, S: int) -> int:
    return K if S <= 1 else -(-(K // 64) // S) * 64


def run(algo: KfAlgo, W: torch.Tensor, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """out[M, N] = x[M, K] @ W[N, K]^T with the pin's split-K arithmetic, on the current stream."""
    L = lib()
    M, K = x.shape
    N = W.shape[0]
    if (L is None or x.dtype != torch.bfloat16 or W.dtype != torch.bfloat16 or out.dtype != torch.bfloat16
            or N % BN or K % BK or algo.chunk % BK or W.stride(1) != 1 or W.stride(0) != K or x.stride(1) != 1
            or out.stride(1) != 1 or x.data_ptr() % 16 or W.data_ptr() % 16 or out.data_ptr() % 16
            or (x.stride(0) * 2) % 16 or (out.stride(0) * 2) % 16 or out.shape[0] != M or out.shape[1] != N or M <= 0):
        raise RuntimeError("KF GEMM: unavailable or unsupported layout")
    f = L.fn[algo.name]
    sbm = KERNELS[algo.name]
    nb = N // BN
    if algo.pad or M % BM == 0:
        mb = 1 << 30 if algo.pad else M
        grid = -(-M // BM) * nb
    else:
        mb = (M // BM) * BM
        grid = (mb // BM) * nb + (-(-(M - mb) // sbm)) * nb
    args = [C.c_void_p(x.data_ptr()), C.c_void_p(W.data_ptr()), C.c_void_p(out.data_ptr()), C.c_int(M), C.c_int(N),
            C.c_int(K), C.c_int(x.stride(0)), C.c_int(out.stride(0)), C.c_int(algo.chunk), C.c_int(mb)]
    params = (C.c_void_p * len(args))(*[C.cast(C.pointer(a), C.c_void_p) for a in args])
    L._chk(L.cu.cuLaunchKernel(f, grid, 1, 1, 256, 1, 1, SMEM, C.c_void_p(torch.cuda.current_stream().cuda_stream),
                               params, None), "launch")
    return out


def load_table(log=print) -> dict:
    """{(N, K, L, R): (kernel, pad)} from the table file the switch names ("1": the built-in one for this GPU)."""
    s = spec()
    if s in ("", "0"):
        return {}
    try:
        if s == "1":
            name = TABLES.get(torch.cuda.get_device_name())
            if name is None:
                log("lever not used: KF (no built-in table for this GPU model)")
                return {}
            path = os.path.join(_DIR, name)
        else:
            path = s
        with open(path) as f:
            t = json.load(f)
        if t.get("meta", {}).get("gpu") not in (None, torch.cuda.get_device_name()):
            log(f"lever not used: KF (table for {t['meta']['gpu']})")
            return {}
        return {(int(N), int(K), int(L), int(R)): (str(k), bool(p)) for N, K, L, R, k, p, *_ in t["table"]
                if str(k) in KERNELS}
    except Exception as exc:  # noqa: BLE001
        log(f"lever not used: KF (table: {type(exc).__name__}: {exc})")
        return {}


def _outliers(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    x = x.clone().float()
    R, K = x.shape
    x[torch.randint(0, R, (max(1, R // 50),), generator=g, device="cuda")] *= 64
    x[:, torch.randint(0, K, (32,), generator=g, device="cuda")] *= 64
    return x.to(torch.bfloat16)


@torch.inference_mode()
def apply(pinned, groups_of, weights: dict, log=print) -> int:
    """Byte-gated overlay of the table's KF picks onto pinned.algo_r (call after S6 / FP's apply_s6). Returns the
    number of (N, K, bucket, R) now running KF."""
    from . import lt as ltmod

    table = load_table(log)
    if not table:
        return 0
    if lib() is None:
        log("lever not used: KF (kernels unavailable on this GPU)")
        return 0
    g = torch.Generator(device="cuda")
    g.manual_seed(2560)
    kept, rej, why = 0, 0, {}
    by = {}
    for (N, K, L, R), v in table.items():
        by.setdefault((N, K, L), {})[R] = v
    for (N, K, L), ents in sorted(by.items()):
        pin = pinned.algo.get((N, K, L))
        if pin is None or (N, K) not in weights:
            why["no pin"] = why.get("no pin", 0) + len(ents)
            continue
        rc = ltmod.cfg_of(pin)
        S, red = int(rc.get("splitK", 1)), int(rc.get("reduction", 0))
        if S > 1 and red != 4:
            why[f"reduction {red}"] = why.get(f"reduction {red}", 0) + len(ents)
            continue
        chunk = chunk_of(K, S)
        if chunk % BK:
            why["chunk"] = why.get("chunk", 0) + len(ents)
            continue
        served = sorted({r for rs in (groups_of(L) or {}).values() for r in rs} & set(ents))
        if not served:
            continue
        wl = weights[(N, K)]
        W = wl[0] if isinstance(wl, (list, tuple)) else wl
        nb = -(-max(served) // L)
        xs = torch.cat([(torch.randn((L, K), generator=g, device="cuda") * (1 + b % 3)).to(torch.bfloat16)
                        for b in range(nb)])
        xo = _outliers(xs, g)
        refs = [torch.cat([pinned.ref_linear(N, K, W, xx[b * L:(b + 1) * L]) for b in range(nb)]) for xx in (xs, xo)]
        for r in served:
            name, pad = ents[r]
            a = KfAlgo(name, pad, chunk, S, red)
            ok = True
            try:
                for xx, yy in zip((xs, xo), refs):
                    xr = xx[:r].contiguous()
                    for _ in range(2):
                        o = torch.full((r, N), float("nan"), dtype=torch.bfloat16, device="cuda")
                        run(a, W, xr, o)
                        if not ltmod._eq(o, yy[:r]):
                            ok = False
            except RuntimeError as exc:
                ok = False
                why[str(exc)[:40]] = why.get(str(exc)[:40], 0) + 1
            if ok:
                pinned.algo_r[(N, K, L, r)] = a
                kept += 1
            else:
                rej += 1
                why["bits"] = why.get("bits", 0) + 1
    torch.cuda.synchronize()
    log(f"lever used: KF ({ENV}) at {kept} (N, K, bucket, R), bit-equal to the per-decision reference; {rej} "
        f"rejected {why or ''} (they keep their pick)" if kept else
        f"lever not used: KF (0 of the table's keys passed: {rej} rejected {why or ''})")
    return kept
