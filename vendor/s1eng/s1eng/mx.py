"""MX (ruj r2 mixbatch, 2026-10-21): the two-class GEMM of cross-bucket ("mixed") packed batches.

A mixed batch carries decisions of two buckets D > S in one bucket-D packed graph: the D decisions' rows first
([0, r1)), the S decisions' rows after them ([r1, R)). Every GEMM whose pinned bit class differs between the two
buckets (the split-K projections: down_proj, out_proj, o_proj, k / v, z) must give each row its own bucket's class.
skx6.cubin (skx6.cu, nvcc 13.4 -arch=sm_120a) does that in ONE launch: rows [0, M1) folded as class-1 slices, rows
[M1, M) as class-2 slices (unsplit allowed), M1 read from device memory (the graph's meta blob), so one captured graph
serves every split. Arithmetic = SKX5's / ns9's (skx6.cu header): per ceil64 K slice an ascending-k mma.sync chain
from +0, bf16 partials folded in slice order in fp32, one RN round. Used only where start-up byte checks pass
(PackedForward.mx_setup: every shape and class pair against F.linear at M = bucket blocks; then the mixed graphs
end to end against the per-decision graphs, model.mix_check).
Loading needs libcuda and the cubin next to this file; anything missing means no mixed batches.
"""
from __future__ import annotations

import ctypes as C
import functools
import os
import sys

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
CUBIN = os.path.join(_DIR, os.environ.get("MX_CUBIN", "skx6.cubin"))
# skx6t (ruj skx6t track): the same two-class GEMM without the padded tail work (skx6t.cu / skx6t.cubin): "t" =
# skx6's segment grids with the m16 row sub-tiles past a segment's end skipped, "m" = one tile grid over [0, M) whose
# boundary tile folds every row at its own class's slice ends. Switch S1ENG_SKX6T ("" / "0" = skx6 only, as before;
# "1" = DEFAULT_TAIL; "t" / "m"); a variant runs only at the (kernel, N, K, M, c1, c2) keys that tail_gate found
# byte-equal to skx6 at start-up (TAIL), every other launch stays skx6.
TCUBIN = os.path.join(_DIR, os.environ.get("MX_TCUBIN", "skx6t.cubin"))
TVARIANTS = {"t": "gemm_skx6t_", "m": "gemm_skx6m_"}
DEFAULT_TAIL = "t"
TAIL = {}   # (skx6 kernel name, N, K, M, class 1 slice width, class 2 slice width) -> variant kernel name
# KFX (kai/kx-k2, backlog #6 KF-MX): KF's mainloop / epilogue (kf.cu's kf_tile) for the same two-class GEMMs (kfx.cu /
# kfx.cubin): each class segment in KF's 128x128 tiles plus one 32 / 64 / 96-row tail tile (kfx_map.h), KF's continuous
# cp.async pipeline, zero-C fold and (_e) 16-byte smem epilogue. Switch S1ENG_KFX ("" / "0" = off, nothing loaded;
# "1" / "all" = every two-class shape; "2560,2048" = only these N); S1ENG_KFX_EPI ("1" default: gemm_kfx_b128_e, the
# smem epilogue KF's table picks almost everywhere; "0": gemm_kfx_b128). A kfx kernel runs only at the (kernel, N, K, M,
# c1, c2) keys that the start-up gate (PackedForward.mx_kfx_gate -> kfx_check) found byte-equal to skx6 and to the
# skx6t variant the key runs (KFX); every other launch keeps skx6t / skx6. FCAB pick lever "kfx" (register_arms):
# arms kfx / nokfx; S1ENG_KFX_ARM=1 keeps the env set on skx6t / skx6.
XCUBIN = os.path.join(_DIR, os.environ.get("MX_XCUBIN", "kfx.cubin"))
XKERNELS = ("gemm_kfx_b128_e", "gemm_kfx_b128")
XBM, XBN, XBK = 128, 128, 64
XSMEM = 98304   # kf.cu's 128x128x64 x 3 stages (the 32 / 64 / 96-row tile configs fit in it)
KFX = {}   # (skx6 kernel name, N, K, M, class 1 slice width, class 2 slice width) -> kfx kernel name
# name: (BM, BN, BK, stages, threads)
KERNELS = {
    "gemm_skx6_128x128k64": (128, 128, 64, 3, 256),
    "gemm_skx6_128x128k32": (128, 128, 32, 4, 256),
    "gemm_skx6_64x256k32": (64, 256, 32, 4, 256),
    "gemm_skx6_128x64k64": (128, 64, 64, 3, 256),
    "gemm_skx6_64x128k64": (64, 128, 64, 3, 256),
}


def _smem(bm, bn, bk, st):
    return st * (bm + bn) * bk * 2


class _Lib:
    def __init__(self):
        self.cu = C.CDLL("libcuda.so.1")
        torch.zeros(1, device="cuda")
        data = open(CUBIN, "rb").read()
        self._img = C.create_string_buffer(data, len(data))
        self.mod = C.c_void_p()
        self._chk(self.cu.cuModuleLoadData(C.byref(self.mod), self._img), "cuModuleLoadData")
        self.fn = {}
        for name, (bm, bn, bk, st, th) in KERNELS.items():
            f = C.c_void_p()
            if self.cu.cuModuleGetFunction(C.byref(f), self.mod, name.encode()) != 0:
                continue
            smem = _smem(bm, bn, bk, st)
            self._chk(self.cu.cuFuncSetAttribute(f, 8, smem), "smem attribute")
            self.fn[name] = (f, smem)
        if not self.fn:
            raise RuntimeError("MX: no kernels in " + CUBIN)
        self.tfn = {}   # skx6t variants, loaded only with the switch on (off: skx6.cubin alone, as before)
        if tail_mode() and os.path.exists(TCUBIN):
            tdata = open(TCUBIN, "rb").read()
            self._timg = C.create_string_buffer(tdata, len(tdata))
            self.tmod = C.c_void_p()
            self._chk(self.cu.cuModuleLoadData(C.byref(self.tmod), self._timg), "cuModuleLoadData skx6t")
            for name, (bm, bn, bk, st, th) in KERNELS.items():
                for pre in TVARIANTS.values():
                    tname = pre + name[len("gemm_skx6_"):]
                    f = C.c_void_p()
                    if self.cu.cuModuleGetFunction(C.byref(f), self.tmod, tname.encode()) != 0:
                        continue
                    smem = _smem(bm, bn, bk, st)
                    self._chk(self.cu.cuFuncSetAttribute(f, 8, smem), "smem attribute")
                    self.tfn[tname] = (f, smem)
        self.xfn = {}   # kfx kernels, loaded only with S1ENG_KFX on; any loading problem: none (skx6t / skx6 stay)
        if kfx_on() and os.path.exists(XCUBIN):
            try:
                xdata = open(XCUBIN, "rb").read()
                self._ximg = C.create_string_buffer(xdata, len(xdata))
                self.xmod = C.c_void_p()
                self._chk(self.cu.cuModuleLoadData(C.byref(self.xmod), self._ximg), "cuModuleLoadData kfx")
                xfn = {}
                for xname in XKERNELS:
                    f = C.c_void_p()
                    self._chk(self.cu.cuModuleGetFunction(C.byref(f), self.xmod, xname.encode()), xname)
                    self._chk(self.cu.cuFuncSetAttribute(f, 8, XSMEM), "smem attribute kfx")
                    xfn[xname] = (f, XSMEM)
                self.xfn = xfn
            except Exception as exc:  # noqa: BLE001 - no kfx, MX itself unaffected
                print(f"[s1eng] KFX kernels unavailable: {exc!r}", flush=True)

    @staticmethod
    def _chk(r, what):
        if r != 0:
            raise RuntimeError(f"MX {what}: CUresult {r}")


@functools.lru_cache(maxsize=1)
def lib():
    if not os.path.exists(CUBIN):
        return None
    try:
        if tuple(torch.cuda.get_device_capability(0)) != (12, 0):
            return None
        return _Lib()
    except Exception as exc:  # noqa: BLE001 - no MX on any loading problem
        print(f"[s1eng] MX two-class GEMMs unavailable: {exc!r}", flush=True)
        return None


def chunk_of(K: int, split: int) -> int:
    """the slice width of a split pin (ceil64 slices; K itself for an unsplit pin)"""
    return K if split <= 1 else -(-(K // 64) // split) * 64


def supported(name: str, x: torch.Tensor, W: torch.Tensor, out: torch.Tensor, c1: int, c2: int) -> bool:
    L = lib()
    if L is None or name not in L.fn:
        return False
    bm, bn, bk, st, th = KERNELS[name]
    M, K = x.shape
    N = W.shape[0]
    return (x.dtype == torch.bfloat16 and W.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
            and x.stride(1) == 1 and out.stride(1) == 1 and W.stride(1) == 1 and W.stride(0) == K
            and x.data_ptr() % 16 == 0 and out.data_ptr() % 16 == 0 and W.data_ptr() % 16 == 0
            and (x.stride(0) * 2) % 16 == 0 and (out.stride(0) * 2) % 16 == 0
            and N % bn == 0 and K % bk == 0 and c1 % bk == 0 and c2 % bk == 0 and 0 < c1 <= K and 0 < c2 <= K
            and M >= 1)


def run(name: str, W: torch.Tensor, x: torch.Tensor, out: torch.Tensor, c1: int, c2: int, m1=None, m1s: int = 0):
    """out[:M] = x @ W^T with rows [0, M1) in class c1 and [M1, M) in class c2 (c = slice width, K = unsplit);
    M1 = the int32 at m1 (a device tensor) or m1s. Launches on the current stream."""
    L = lib()
    if L is None or not supported(name, x, W, out, c1, c2):
        raise RuntimeError("MX GEMM: unavailable or unsupported layout")
    M, K = x.shape
    N = W.shape[0]
    key = (name, N, K, M, int(c1), int(c2))
    xv = KFX.get(key) if KFX else None   # kfx: gated key (read while a graph is captured; FCAB swaps the dict)
    if xv is not None and xv in getattr(L, "xfn", {}) and kfx_supported(x, W, out, c1, c2):
        _launch(L, xv, W, x, out, c1, c2, m1, m1s)
        return out
    tv = TAIL.get(key) if TAIL and tail_mode() else None   # skx6t: gated key (capture)
    _launch(L, tv if tv is not None and tv in L.tfn else name, W, x, out, c1, c2, m1, m1s)
    return out


def _args(W, x, out, c1, c2, m1=None, m1s: int = 0) -> list:
    """the kernel arguments of skx6 / skx6t / kfx: (A, B, C, M, N, K, lda, ldc, chunk1, chunk2, m1p, m1s)"""
    M, K = x.shape
    N = W.shape[0]
    return [x.data_ptr(), W.data_ptr(), out.data_ptr(), M, N, K, x.stride(0), out.stride(0), int(c1), int(c2),
            m1.data_ptr() if m1 is not None else 0, int(m1s)]


def _cfg(kname: str, M: int, N: int) -> tuple:
    """(grid, threads, dynamic smem) of a launch: skx6 / skx6t one spare row tile (gemm_skx6m_*: none), kfx
    kfx_row_tiles(M) row tiles (enough for every M1)"""
    if kname in XKERNELS:
        return (N // XBN, kfx_row_tiles(M), 1), 256, XSMEM
    base = kname if kname in KERNELS else "gemm_skx6_" + kname.split("_", 2)[2]
    bm, bn, bk, st, th = KERNELS[base]
    return (N // bn, -(-M // bm) + (0 if kname.startswith(TVARIANTS["m"]) else 1), 1), th, _smem(bm, bn, bk, st)


def _launch(L, kname, W, x, out, c1, c2, m1=None, m1s: int = 0, stream=None) -> None:
    """one launch of an skx6 kernel, an skx6t variant or a kfx kernel"""
    if kname in XKERNELS:
        f, _ = L.xfn[kname]
    else:
        f, _ = L.fn[kname] if kname in L.fn else L.tfn[kname]
    a = _args(W, x, out, c1, c2, m1, m1s)
    args = [C.c_void_p(a[0]), C.c_void_p(a[1]), C.c_void_p(a[2])] + [C.c_int(v) for v in a[3:10]] + \
        [C.c_void_p(a[10]), C.c_int(a[11])]
    params = (C.c_void_p * len(args))(*[C.cast(C.pointer(v), C.c_void_p) for v in args])
    (gx, gy, gz), th, smem = _cfg(kname, a[3], a[4])
    h = (stream or torch.cuda.current_stream()).cuda_stream
    L._chk(L.cu.cuLaunchKernel(f, gx, gy, gz, th, 1, 1, smem, C.c_void_p(h), params, None), "launch")


def tail_mode() -> str:
    """skx6t switch S1ENG_SKX6T: "" (also unset, "0") = skx6 only; "1" = DEFAULT_TAIL; "t" / "m" = that variant"""
    v = os.environ.get("S1ENG_SKX6T", "0").strip().lower()
    if v in ("", "0"):
        return ""
    v = DEFAULT_TAIL if v == "1" else v
    return v if v in TVARIANTS else ""


def tail_name(name: str, variant: str) -> str:
    return TVARIANTS[variant] + name[len("gemm_skx6_"):]


@torch.inference_mode()
def tail_check(name: str, W: torch.Tensor, c1: int, c2: int, R: int, variant: str, g) -> int:
    """skx6t start-up byte gate of one (kernel, N, K, M = R) key: the variant vs skx6 itself, every one of the R output
    rows in all bits, for M1 at one value in every 16-row block (residue 7 q mod 16: all residues mod 16 occur), around
    every tile edge, at 0 / 1 / R - 1 / R, on two inputs (randn with every 7th row x64; rows at 2^U(-12, 12) with every
    5th row x4096 and sparse +-1e4 spikes; the second input at every 4th M1). Reference row = skx6 in the row's own class
    (skx6 at M1 = R and at M1 = 0: a row's bits depend on its class only; checked by running skx6 itself at every 11th
    M1). 128 sentinel rows after R must stay untouched. -> number of differing values (0 = byte-equal), -1 = unavailable"""
    L = lib()
    tv = tail_name(name, variant)
    if L is None or tv not in L.tfn or not supported(name, torch.empty((1, W.shape[1]), dtype=torch.bfloat16,
                                                                          device="cuda"), W,
                                                     torch.empty((1, W.shape[0]), dtype=torch.bfloat16, device="cuda"),
                                                     c1, c2):
        return -1
    N, K = W.shape
    bm = KERNELS[name][0]
    ms = {0, 1, R - 1, R} | {min(R, 16 * q + (7 * q) % 16) for q in range(R // 16 + 1)}
    ms |= {b + d for b in range(0, R + 1, bm) for d in (-1, 0, 1) if 0 <= b + d <= R}
    ms = sorted(m for m in ms if 0 <= m <= R)
    sent = -21555
    bad = torch.zeros((), dtype=torch.long, device="cuda")
    out = torch.empty((R + 128, N), dtype=torch.bfloat16, device="cuda")
    for kind in (0, 1):
        if kind == 0:
            x = torch.randn((R, K), generator=g, device="cuda")
            x[::7] *= 64
        else:
            e = torch.randint(-12, 13, (R, 1), generator=g, device="cuda").float()
            x = torch.randn((R, K), generator=g, device="cuda") * torch.exp2(e)
            x[::5] *= 4096
            sp = torch.rand((R, K), generator=g, device="cuda") < 1e-3
            x = torch.where(sp, torch.sign(torch.randn((R, K), generator=g, device="cuda")) * 1e4, x)
        x = x.to(torch.bfloat16).contiguous()
        refs = []
        for mv in (R, 0):
            o = torch.empty((R, N), dtype=torch.bfloat16, device="cuda")
            _launch(L, name, W, x, o, c1, c2, torch.tensor([mv], dtype=torch.int32, device="cuda"))
            refs.append(o.view(torch.int16))
        mk = ms if kind == 0 else ms[::4]
        m1all = torch.tensor(mk, dtype=torch.int32, device="cuda")
        for i, mv in enumerate(mk):
            ref = torch.cat([refs[0][:mv], refs[1][mv:]])
            for kn in ((tv, name) if i % 11 == 0 else (tv,)):
                out.view(torch.int16).fill_(sent)
                _launch(L, kn, W, x, out, c1, c2, m1all[i:i + 1])
                ov = out.view(torch.int16)
                bad += (ov[:R] != ref).sum() + (ov[R:] != sent).sum()
    torch.cuda.synchronize()
    return int(bad)


def kernel_for(N: int, K: int, R: int) -> str:
    """the SKX6 config of a two-class GEMM (mx1 twin-pair timings at mixed-graph row counts; S1ENG_MX_KERN="<name>" or
    "N:<name>,..." overrides): 128x128 tiles for N = 2560 (down / out_proj / o_proj) and for z (N = 4096) above 704
    rows, 64x256 tiles for z below and for the fused k|v block (N = 2048), 64x128 tiles for k / v (N = 1024)"""
    spec = os.environ.get("S1ENG_MX_KERN", "")
    if spec:
        if ":" not in spec:
            return spec
        for part in spec.split(","):
            n, name = part.split(":")
            if int(n) == N:
                return name
    if N == 4096:
        return "gemm_skx6_64x256k32" if R <= 704 else "gemm_skx6_128x128k64"
    if N == 2048:
        return "gemm_skx6_64x256k32"
    if N <= 1024:
        return "gemm_skx6_64x128k64"
    return "gemm_skx6_128x128k64"


# ---- KFX (kai/kx-k2, backlog #6 KF-MX) ------------------------------------------------------------------------------
def kfx_spec(env=os.environ):
    """S1ENG_KFX -> None (off: "" / "0" / unset / unparsable), frozenset() (every two-class shape: "1" / "all") or the
    frozenset of the N it may serve ("2560,2048")"""
    v = env.get("S1ENG_KFX", "0").strip().lower()
    if v in ("", "0"):
        return None
    if v in ("1", "all"):
        return frozenset()
    try:
        ns = frozenset(int(p) for p in v.split(","))
    except ValueError:
        return None
    return ns if ns and all(n > 0 for n in ns) else None


def kfx_on(env=os.environ) -> bool:
    return kfx_spec(env) is not None


def kfx_name(env=os.environ) -> str:
    """S1ENG_KFX_EPI: "1" (default) -> the smem-staged epilogue kernel, "0" -> the plain one"""
    return XKERNELS[1] if env.get("S1ENG_KFX_EPI", "1").strip() == "0" else XKERNELS[0]


def kfx_row_tiles(M: int) -> int:
    """row tiles of a kfx grid (kfx_map.h): ceil(M1 / 128) + ceil((M - M1) / 128) <= ceil(M / 128) + 1 for every M1"""
    return -(-M // XBM) + 1


def kfx_tile(t: int, M: int, M1: int):
    """Python mirror of kfx_map.h's kfx_tile (tests/test_kfx.py checks them equal): row tile t -> (m0, mend, cls, h)
    or None (past both segments)"""
    L2 = M - M1
    b1, b2 = M1 // XBM, L2 // XBM
    r1, r2 = M1 - b1 * XBM, L2 - b2 * XBM
    if t < b1:
        return t * XBM, M1, 1, XBM
    t -= b1
    if t < b2:
        return M1 + t * XBM, M, 2, XBM
    t -= b2
    if r1 > 0:
        if t == 0:
            return b1 * XBM, M1, 1, -(-r1 // 32) * 32
        t -= 1
    if r2 > 0 and t == 0:
        return M1 + b2 * XBM, M, 2, -(-r2 // 32) * 32
    return None


def kfx_tiles(M: int, M1: int) -> list:
    """the live row tiles of a kfx launch at (M, M1), in blockIdx.y order"""
    out = []
    for t in range(kfx_row_tiles(M)):
        v = kfx_tile(t, M, M1)
        if v is None:
            break
        out.append(v)
    return out


def kfx_supported(x: torch.Tensor, W: torch.Tensor, out: torch.Tensor, c1: int, c2: int) -> bool:
    M, K = x.shape
    N = W.shape[0]
    return (x.dtype == torch.bfloat16 and W.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
            and x.stride(1) == 1 and out.stride(1) == 1 and W.stride(1) == 1 and W.stride(0) == K
            and x.data_ptr() % 16 == 0 and out.data_ptr() % 16 == 0 and W.data_ptr() % 16 == 0
            and (x.stride(0) * 2) % 16 == 0 and (out.stride(0) * 2) % 16 == 0
            and N % XBN == 0 and K % XBK == 0 and c1 % XBK == 0 and c2 % XBK == 0 and 0 < c1 <= K and 0 < c2 <= K
            and M >= 1)


def kfx_m1s(R: int) -> list:
    """the M1 points of the kfx gate at R rows: 0 / 1 / R - 1 / R, one per 16-row block at residue 7 q mod 16 (every
    residue mod 16 once R >= about 241; smaller R get fewer residues, which does not matter for the bits: tile shapes
    depend only on the 32-row tail edges), and every 32-row edge -1 / 0 / +1 counted from either end (every tail height
    32 / 64 / 96 / 128 and none, of both segments, and every 128-row big-tile edge)"""
    ms = {0, 1, R - 1, R} | {min(R, 16 * q + (7 * q) % 16) for q in range(R // 16 + 1)}
    for b in range(0, R + 1, 32):
        for d in (-1, 0, 1):
            ms |= {b + d, R - b - d}
    return sorted(m for m in ms if 0 <= m <= R)


def kfx_witness(x: torch.Tensor, chunk: int) -> torch.Tensor:
    """the order witness of one class: the K columns of every odd slice (width chunk) scaled by 2^24 (exact in bf16), so
    a slice bound, the fold order or a class mix-up changes the rounded bits"""
    K = x.shape[1]
    s = torch.ones((K,), dtype=torch.float32, device=x.device)
    for k0 in range(chunk, K, 2 * chunk):
        s[k0:k0 + chunk] = 2.0 ** 24
    return (x.float() * s).to(x.dtype)


@torch.inference_mode()
def kfx_check(name: str, W: torch.Tensor, c1: int, c2: int, R: int, g, kname: str) -> int:
    """kfx start-up byte gate of one (kernel, N, K, M = R, c1, c2) key: the kfx kernel vs skx6 (the key's skx6 config
    `name`), every one of the R output rows in all bits, at every M1 of kfx_m1s(R), on four inputs (randn with every
    7th row x64; rows at 2^U(-12, 12) with every 5th row x4096 and sparse +-1e4 spikes; the order witness of class 1
    and of class 2), each kfx launch done twice. Reference row = skx6 in the row's own class (skx6 at M1 = R and at
    M1 = 0); at every 7th M1 skx6 itself and the skx6t variant the key runs (TAIL, when the switch is on) are checked
    against the same reference. 128 sentinel rows after R must stay untouched. -> number of differing values (0 =
    byte-equal), -1 = unavailable"""
    L = lib()
    x0 = torch.empty((1, W.shape[1]), dtype=torch.bfloat16, device="cuda")
    o0 = torch.empty((1, W.shape[0]), dtype=torch.bfloat16, device="cuda")
    if (L is None or kname not in getattr(L, "xfn", {}) or not supported(name, x0, W, o0, c1, c2)
            or not kfx_supported(x0, W, o0, c1, c2)):
        return -1
    N, K = W.shape
    tv = TAIL.get((name, N, K, R, int(c1), int(c2))) if TAIL and tail_mode() else None
    served = [name] + ([tv] if tv is not None and tv in L.tfn else [])
    ms = kfx_m1s(R)
    sent = -21555
    bad = torch.zeros((), dtype=torch.long, device="cuda")
    out = torch.empty((R + 128, N), dtype=torch.bfloat16, device="cuda")
    ov = out.view(torch.int16)
    rows = torch.arange(R, device="cuda").unsqueeze(1)
    for kind in range(4):
        if kind == 0:
            x = torch.randn((R, K), generator=g, device="cuda")
            x[::7] *= 64
        elif kind == 1:
            e = torch.randint(-12, 13, (R, 1), generator=g, device="cuda").float()
            x = torch.randn((R, K), generator=g, device="cuda") * torch.exp2(e)
            x[::5] *= 4096
            sp = torch.rand((R, K), generator=g, device="cuda") < 1e-3
            x = torch.where(sp, torch.sign(torch.randn((R, K), generator=g, device="cuda")) * 1e4, x)
        else:
            x = kfx_witness(torch.randn((R, K), generator=g, device="cuda").to(torch.bfloat16), c1 if kind == 2 else c2)
        x = x.to(torch.bfloat16).contiguous()
        refs = []
        for mv in (R, 0):
            o = torch.empty((R, N), dtype=torch.bfloat16, device="cuda")
            _launch(L, name, W, x, o, c1, c2, torch.tensor([mv], dtype=torch.int32, device="cuda"))
            refs.append(o.view(torch.int16))
        mk = ms if kind == 0 else ms[::4] + ([ms[-1]] if (len(ms) - 1) % 4 else [])
        m1all = torch.tensor(mk, dtype=torch.int32, device="cuda")
        for i, mv in enumerate(mk):
            m1 = m1all[i:i + 1]
            ref = torch.where(rows < m1, refs[0], refs[1])
            for kn in (kname, kname) + (tuple(served) if i % 7 == 0 else ()):
                ov.fill_(sent)
                _launch(L, kn, W, x, out, c1, c2, m1)
                bad += (ov[:R] != ref).sum() + (ov[R:] != sent).sum()
    torch.cuda.synchronize()
    return int(bad)


def register_arms(log=print, env=os.environ) -> None:
    """kai/kx-k2 FCAB pick lever "kfx" (after every MX pair's kfx gate, before the lanes are captured): arm "kfx" captures
    its lane set with the gated kfx keys (KFX), "nokfx" without them (every key back on skx6t / skx6: each kfx key is
    byte-equal to them, so the two sets are bit-equal; each FCAB set is self-checked and its mixed graphs mix-checked).
    S1ENG_KFX_ARM=1: the env set without them. Nothing gated: nothing registered (the arms are dropped). Never raises."""
    try:
        from . import fcab

        on = dict(KFX)
        if not on:
            log("KFX FCAB arms not registered: no kfx key passed the start-up gate")
            return
        fcab.register("kfx", sys.modules[__name__], "KFX", on=on, off={})
        arm = env.get("S1ENG_KFX_ARM", "0") == "1"
        if arm:
            KFX.clear()
        log(f"KFX FCAB arms: kfx / nokfx registered ({len(on)} keys); env set on "
            f"{'skx6t / skx6 (S1ENG_KFX_ARM=1)' if arm else 'kfx'}")
    except Exception as exc:  # noqa: BLE001
        log(f"KFX FCAB arms not registered ({type(exc).__name__}: {exc})")
