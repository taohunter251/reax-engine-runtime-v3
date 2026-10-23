"""NG3 (rgh): native sm_120 GEMMs (ngemm3.cubin, built from pod/ngemm/ngemm3.cu with nvcc 13.4 -arch=sm_120a), offered
to S6 tuning as candidate algorithms of a pin's bit class.

Arithmetic: mma.sync m16n8k16 with fp32 accumulation in ascending K and one RN round to bf16 (an unsplit cuBLASLt
algo), or, for a split-K pin, its S "ceil64" K slices each accumulated that way and folded in slice order by the pin's
reduction scheme (1 INPLACE, 2 COMPUTE_TYPE, 4 OUTPUT_TYPE). Kernel families offered:
  * nh_*: unsplit; rows [0, mb) in big tiles and the rest in small tiles of one launch (the small tiles fill the last
    wave that a whole number of big tiles leaves idle): 4-13 % faster than the fastest exact cuBLASLt algo at row
    counts just past a 128-row boundary (ngtest3, 2026-10-19);
  * ns*_*: split; every slice of a tile in one CTA, in order (no reduce kernel): faster where the tiles fill about
    one wave (M ~ 1024 for down / out_proj).
The serial-fixup split kernels (nf*_*, one CTA per slice) are in the cubin but not offered (10-20 % slower).
Like every S6 candidate an NgAlgo is used only where its output equals the per-decision reference bit for bit (at the
anchor and at every row count it serves) and it is faster. Loading needs libcuda and the cubin next to this file;
anything missing just means no candidates (S1ENG_NG=0 disables them).
"""
from __future__ import annotations

import ctypes as C
import functools
import os
import time

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
CUBIN = os.path.join(_DIR, "ngemm5.cubin")
# MA (rgh): the same source built per architecture (nvcc 13.0 -O3 -arch=sm_89 / sm_90; mma.sync, ldmatrix and cp.async
# exist since sm_80), so 4090 / H100 validators also get the candidates; S6 / SG keep a kernel only where it is
# bit-equal to that GPU's reference pins and faster (on H100 the reference's wgmma GEMMs may leave none exact)
CUBINS = {(12, 0): CUBIN, (8, 9): os.path.join(_DIR, "ngemm5_sm89.cubin"), (9, 0): os.path.join(_DIR, "ngemm5_sm90.cubin")}


# NG9B (rgh 2026-10-20): BK=64 two-stage cp.async kernels (ngemm9b.cu, nvcc 13.4 -arch=sm_120a), a second module
# loaded on sm_120 only (S1ENG_NG9=0 disables). Same bit class as nh_* / nsw_* (unsplit, ascending-k mma.sync, one RN
# round; SwiGLU = swiglu1); ~3 % faster paired at the serving row counts (power-capped GEMMs: full 128-byte L2 lines
# per row per k-tile cut data-movement energy -> higher SM clock). Offered to S6 / SG like any NG candidate.
CUBIN9 = os.path.join(_DIR, "ngemm9b.cubin")
KERNELS9 = {
    "nh9_b128x256k64_s64x256k64": ("h", (128, 256, 64, 2), (64, 256, 64, 2), 256, 0),
    "nh9_b128x256k64_s32x256k64": ("h", (128, 256, 64, 2), (32, 256, 64, 2), 256, 0),
    "nh9_b128x256k64_s64x256": ("h", (128, 256, 64, 2), (64, 256, 32, 4), 256, 0),
    "nh9_b128x256k64_s32x256": ("h", (128, 256, 64, 2), (32, 256, 32, 4), 256, 0),
    "nh9_b128x128k64_s64x128k64": ("h", (128, 128, 64, 3), (64, 128, 64, 3), 256, 0),
    "nh9_b128x128k64_s32x128": ("h", (128, 128, 64, 3), (32, 128, 32, 4), 256, 0),
    # in-CTA split-K (all slices of a tile in one CTA, folded in slice order; slices are ceil64 -> BK=64 divides them)
    "ns9_4_128x128k64": ("s", (128, 128, 64, 3), None, 256, 4),
    "ns9_1_128x128k64": ("s", (128, 128, 64, 3), None, 256, 1),
    "ns9_2_128x128k64": ("s", (128, 128, 64, 3), None, 256, 2),
}
SW_KERNELS9 = {
    "nsw9_b128x256k64_s64x256k64": ((128, 256, 64, 2), (64, 256, 64, 2), 256),
    "nsw9_b128x256k64_s32x256k64": ((128, 256, 64, 2), (32, 256, 64, 2), 256),
    "nsw9_b128x256k64_s64x256": ((128, 256, 64, 2), (64, 256, 32, 4), 256),
    "nsw9_b128x256k64_s32x256": ((128, 256, 64, 2), (32, 256, 32, 4), 256),
    "nsw9_b128x128k64_s64x128k64": ((128, 128, 64, 3), (64, 128, 64, 3), 256),
    "nsw9_b128x128k64_s32x128": ((128, 128, 64, 3), (32, 128, 32, 4), 256),
}


# COV9b (rgh 2026-10-21): grouped lastfold split-K GEMMs (ngemm9l.cubin, built from ngemm9l.cu with nvcc 13.4
# -arch=sm_120a), a third module on sm_120 only (S1ENG_NG9L=0 disables). Bit class = a split-K pin with reduction
# scheme 4 (OUTPUT_TYPE; every split pin of this model): the pin's S ceil64 K slices, each the ngemm9b BK=64 mainloop
# (ascending-k mma.sync, fp32 from +0) rounded once to bf16, folded IN SLICE ORDER (fp32 RN adds) and rounded once.
# Grid = NG groups x tiles; a group computes G = ceil(S / NG) consecutive slices of a tile in one continuous pipeline
# (bf16 partial stored at each slice boundary), the CTA of a tile that finishes last folds all S partials and writes
# the tile (tile counter, no reduce kernel, no spin-wait). G = 1: one CTA per slice (cuBLASLt's split-K grid); G = S:
# one CTA per tile (in-CTA split with 128x256 tiles). Names "<kernel>_g<G>" select G; drop -1 = big tiles only (rows
# padded), else rows [0, (M // BM - drop) * BM) in big tiles and the rest in small tiles (as nh9_*). Partials and
# counters live in fix_bufs(ws) (one workspace per lane). Offered for split-K scheme-4 pins; used only where the
# start-up check finds them bit-equal to the per-decision reference at every R they serve (the frozen plan's picks).
CUBIN9L = os.path.join(_DIR, "ngemm9l.cubin")
_LG_BASE = {   # cubin function: (big (BM, BN, BK, stages), small (BM, BN, BK, stages), threads)
    "nlg9_4_b128x256_s32x256": ((128, 256, 64, 2), (32, 256, 64, 2), 256),
    "nlg9_4_b128x256_s64x256": ((128, 256, 64, 2), (64, 256, 64, 2), 256),
    "nlg9_4_b128x128_s32x128": ((128, 128, 64, 3), (32, 128, 64, 2), 256),
    "nlg9_4_b128x128_s64x128": ((128, 128, 64, 3), (64, 128, 64, 3), 256),
    # tall tails (see CUBIN9T below): last band = 128 + R mod 128 rows of 160/192/224 x 128 tiles (drop 1), or 96x256
    "nlg9_4_b128x256_t160x128": ((128, 256, 64, 2), (160, 128, 64, 2), 256),
    "nlg9_4_b128x256_t192x128": ((128, 256, 64, 2), (192, 128, 64, 2), 256),
    "nlg9_4_b128x256_t224x128": ((128, 256, 64, 2), (224, 128, 64, 2), 256),
    "nlg9_4_b128x256_s96x256": ((128, 256, 64, 2), (96, 256, 64, 2), 256),
}
LG_GROUPS = (1, 2, 3, 4, 6, 8)
# name: (kind, big, small, threads, mode, G, cubin function)
KERNELS9L = {f"{b}_g{G}": ("LG", big, small, th, 4, G, b) for b, (big, small, th) in _LG_BASE.items() for G in LG_GROUPS}


# COV9b tall tails (rgh 2026-10-21): ngemm9t.cubin (ngemm9t.cu, nvcc 13.4 -arch=sm_120a), a fourth module on sm_120
# only (S1ENG_NG9T=0 disables): the nh9_ / nsw9_ 128x256 kernels with the last row band in ONE band of taller tiles,
# 160 / 192 / 224 rows x 128 columns (offered with drop 1: that band holds 128 + R mod 128 rows) or 96x256 (drop 0: R mod
# 128 = 96). Graph-paired GEMM cost is ~linear in 128-row bands and a 32/64-row band costs ~84 % of a full one (it
# streams the whole weight panel), while served R are 32/64/96 mod 128 for 78 % of the rows. Same bit class as nh9_ /
# nsw9_ (tile geometry never changes an element's ascending-k mma.sync chain).
CUBIN9T = os.path.join(_DIR, "ngemm9t.cubin")
KERNELS9T = {
    "nh9_b128x256k64_t160x128k64": ("h", (128, 256, 64, 2), (160, 128, 64, 2), 256, 0),
    "nh9_b128x256k64_t192x128k64": ("h", (128, 256, 64, 2), (192, 128, 64, 2), 256, 0),
    "nh9_b128x256k64_t224x128k64": ("h", (128, 256, 64, 2), (224, 128, 64, 2), 256, 0),
    "nh9_b128x256k64_s96x256k64": ("h", (128, 256, 64, 2), (96, 256, 64, 2), 256, 0),
}
SW_KERNELS9T = {
    "nsw9_b128x256k64_t160x128k64": ((128, 256, 64, 2), (160, 128, 64, 2), 256),
    "nsw9_b128x256k64_t192x128k64": ((128, 256, 64, 2), (192, 128, 64, 2), 256),
    "nsw9_b128x256k64_t224x128k64": ((128, 256, 64, 2), (224, 128, 64, 2), 256),
    "nsw9_b128x256k64_s96x256k64": ((128, 256, 64, 2), (96, 256, 64, 2), 256),
}


# NG7 (pjs e6 nsw7l kernels, team code, ported by rgh 10-21 as an in-situ-decided SG overlay; ngemm7.cu rebuilt with
# nvcc 13.4 -cubin -O3 -std=c++17 -arch=sm_120a: same SASS as pjs' shipped cubin), a fifth module loaded on sm_120 only
# with S1ENG_NG7=1 (off by default: unset = the base engine exactly). Only the nsw7l SwiGLU kernels and the table check
# kernel are loaded (the np7 NP-FOLD kernels in the cubin are not used). nsw7l_*: gate|up GEMM + SwiGLU, the same
# mainloop bit class as nsw9 (one ascending-k fp32 mma.sync chain per element, one RN round), the SiLU read from a
# 64K-entry table filled from torch's own F.silu over every bf16 value (_LUT7, passed in the ws argument): out =
# bf16(float(lut[bits(g)]) * float(u)) = the reference's act_fn(gate) * up. Offered to SG only after check_ng7() found
# the epilogue equal to F.silu(g) * u and silu_mul for every bf16 g (x 8 u); ng7t.apply byte-checks every table key.
# The A and W row strides must equal K (the kernels do not read lda; a row-strided input is copied first).
CUBIN7 = os.path.join(_DIR, "ngemm7.cubin")
SW_KERNELS7 = {  # name: (big (BM, BN, BK, stages), small, threads, K)
    "nsw7l_b128x256_s64x256_k2560": ((128, 256, 64, 2), (64, 256, 64, 2), 256, 2560),
    "nsw7l_b128x256_s32x256_k2560": ((128, 256, 64, 2), (32, 256, 64, 2), 256, 2560),
    # NG7T tall tails (rgh 10-21, added to ngemm7.cu): last band 160 / 192 / 224 x 128 (drop 1) or 96x256 (drop 0)
    "nsw7l_b128x256_t160x128_k2560": ((128, 256, 64, 2), (160, 128, 64, 2), 256, 2560),
    "nsw7l_b128x256_t192x128_k2560": ((128, 256, 64, 2), (192, 128, 64, 2), 256, 2560),
    "nsw7l_b128x256_t224x128_k2560": ((128, 256, 64, 2), (224, 128, 64, 2), 256, 2560),
    "nsw7l_b128x256_s96x256_k2560": ((128, 256, 64, 2), (96, 256, 64, 2), 256, 2560),
}
CHECK7 = "swiglu7_elem_lut"   # the SwiGLU epilogue arithmetic elementwise (check_ng7)


def ng7_on(env=os.environ) -> bool:
    """S1ENG_NG7=1 loads the NG7 module (off by default)."""
    return env.get("S1ENG_NG7", "0") == "1"


def is_ng7(name: str) -> bool:
    return name in SW_KERNELS7


def _tail_drops(name: str):
    """drops offered for a tall-tail kernel (None: not a tall-tail kernel)."""
    if "_t160x128" in name or "_t192x128" in name or "_t224x128" in name:
        return (1,)
    if "_s96x256" in name:
        return (0,)
    return None


def _lg_smem(big, small, th):
    # pipeline / epilogue tile / own partial (one 16-byte vector per thread per 8 fp32 sums)
    def one(bm, bn, bk, st):
        return max(_smem(bm, bn, bk, st), (bm * bn // th // 8) * th * 16)
    return max(one(*big), one(*small))


def _kspec(name: str):
    if name in KERNELS:
        return KERNELS[name]
    if name in KERNELS9:
        return KERNELS9[name]
    if name in KERNELS9T:
        return KERNELS9T[name]
    return KERNELS9L[name][:5]


def _swspec(name: str):
    if name in SW_KERNELS7:
        return SW_KERNELS7[name][:3]
    if name in SW_KERNELS:
        return SW_KERNELS[name]
    return SW_KERNELS9[name] if name in SW_KERNELS9 else SW_KERNELS9T[name]


def _cubin():
    try:
        return CUBINS.get(tuple(torch.cuda.get_device_capability(0)))
    except Exception:  # noqa: BLE001
        return None


def _smem(bm, bn, bk, st):
    return max(st * (bm + bn) * bk * 2, bm * (bn + 8) * 2)


# name: (kind, big (BM, BN, BK, stages), small or None, threads, mode)
KERNELS = {
    "nh_b128x128w8_s32x256": ("h", (128, 128, 32, 3), (32, 256, 32, 4), 256, 0),
    "nh_b128x128w8_s64x256": ("h", (128, 128, 32, 3), (64, 256, 32, 4), 256, 0),
    "nh_b128x128w8_s32x128": ("h", (128, 128, 32, 3), (32, 128, 32, 4), 256, 0),
    "nh_b128x128w8_s64x128": ("h", (128, 128, 32, 3), (64, 128, 32, 4), 256, 0),
    "nh_b128x256_s32x256": ("h", (128, 256, 32, 4), (32, 256, 32, 4), 256, 0),
    "nh_b128x256_s64x256": ("h", (128, 256, 32, 4), (64, 256, 32, 4), 256, 0),
    "nh_b128x256s3_s32x256": ("h", (128, 256, 32, 3), (32, 256, 32, 4), 256, 0),
    "ns4_128x128w8": ("s", (128, 128, 32, 3), None, 256, 4), "ns1_128x128w8": ("s", (128, 128, 32, 3), None, 256, 1),
    # NG4: 4-warp 128x128 big tiles at 2 CTAs per SM (one CTA's epilogue overlaps the other's mainloop)
    "nh4_b128x128_s32x128": ("h", (128, 128, 32, 3), (32, 128, 32, 3), 128, 0),
    "nh4_b128x128_s64x128": ("h", (128, 128, 32, 3), (64, 128, 32, 3), 128, 0),
    "nh4_b64x128_s32x128": ("h", (64, 128, 32, 4), (32, 128, 32, 3), 128, 0),
}
FLAGS = 1 << 16


class NgAlgo:
    """A pseudo cuBLASLt algo (Lt.run / ws_needed / cfg_of / cfg_key recognise it). drop: for nh_*, how many big-tile
    rows fewer than fit go to small tiles (mb = (M // BM - drop) * BM); chunk: the pin's slice width (K if unsplit)."""

    __slots__ = ("name", "drop", "chunk", "split", "red")

    def __init__(self, name: str, drop: int, chunk: int, split: int, red: int):
        self.name, self.drop, self.chunk, self.split, self.red = name, drop, chunk, split, red

    def key(self):
        return ("ng", self.name, self.drop, self.chunk)

    def __repr__(self):
        return f"NgAlgo({self.name}, drop={self.drop}, chunk={self.chunk})"


class _Lib:
    def __init__(self):
        self.cu = C.CDLL("libcuda.so.1")
        torch.zeros(1, device="cuda")
        data = open(_cubin(), "rb").read()
        self._img = C.create_string_buffer(data, len(data))
        self.mod = C.c_void_p()
        self._chk(self.cu.cuModuleLoadData(C.byref(self.mod), self._img), "cuModuleLoadData")
        self.fn = {}
        for name, (kind, big, small, th, mode) in KERNELS.items():
            f = C.c_void_p()
            self._chk(self.cu.cuModuleGetFunction(C.byref(f), self.mod, name.encode()), name)
            smem = max(_smem(*big), _smem(*small) if small else 0)
            self._chk(self.cu.cuFuncSetAttribute(f, 8, smem), "smem attribute")
            self.fn[name] = (f, smem)
        self.sw = {}
        for name, (big, small, th) in SW_KERNELS.items():
            f = C.c_void_p()
            if self.cu.cuModuleGetFunction(C.byref(f), self.mod, name.encode()) != 0:
                self.sw = {}
                break
            smem = max(_smem(*big), _smem(*small))
            self._chk(self.cu.cuFuncSetAttribute(f, 8, smem), "smem attribute")
            self.sw[name] = (f, smem)
        self.ng9 = False
        if (os.environ.get("S1ENG_NG9", "1") == "1" and tuple(torch.cuda.get_device_capability(0)) == (12, 0)
                and os.path.exists(CUBIN9)):
            try:
                data9 = open(CUBIN9, "rb").read()
                self._img9 = C.create_string_buffer(data9, len(data9))
                self.mod9 = C.c_void_p()
                self._chk(self.cu.cuModuleLoadData(C.byref(self.mod9), self._img9), "cuModuleLoadData (ng9)")
                fn9, sw9 = {}, {}
                for name, (kind, big, small, th, mode) in KERNELS9.items():
                    f = C.c_void_p()
                    self._chk(self.cu.cuModuleGetFunction(C.byref(f), self.mod9, name.encode()), name)
                    smem = max(_smem(*big), _smem(*small) if small else 0)
                    self._chk(self.cu.cuFuncSetAttribute(f, 8, smem), "smem attribute")
                    fn9[name] = (f, smem)
                for name, (big, small, th) in SW_KERNELS9.items():
                    f = C.c_void_p()
                    self._chk(self.cu.cuModuleGetFunction(C.byref(f), self.mod9, name.encode()), name)
                    smem = max(_smem(*big), _smem(*small))
                    self._chk(self.cu.cuFuncSetAttribute(f, 8, smem), "smem attribute")
                    sw9[name] = (f, smem)
                self.fn.update(fn9)
                if self.sw:
                    self.sw.update(sw9)
                self.ng9 = True
            except Exception as exc:  # noqa: BLE001 - the NG3/NG5 candidates stay
                print(f"[s1eng] NG9 GEMMs unavailable: {exc!r}", flush=True)
        self.ng9l = False
        if (os.environ.get("S1ENG_NG9L", "1") == "1" and tuple(torch.cuda.get_device_capability(0)) == (12, 0)
                and os.path.exists(CUBIN9L)):
            try:   # COV9b grouped lastfold split-K kernels
                data9l = open(CUBIN9L, "rb").read()
                self._img9l = C.create_string_buffer(data9l, len(data9l))
                self.mod9l = C.c_void_p()
                self._chk(self.cu.cuModuleLoadData(C.byref(self.mod9l), self._img9l), "cuModuleLoadData (ng9l)")
                base = {}
                for b, (big, small, th) in _LG_BASE.items():
                    f = C.c_void_p()
                    self._chk(self.cu.cuModuleGetFunction(C.byref(f), self.mod9l, b.encode()), b)
                    smem = _lg_smem(big, small, th)
                    self._chk(self.cu.cuFuncSetAttribute(f, 8, smem), "smem attribute")
                    base[b] = (f, smem)
                self.fn.update({name: base[e[6]] for name, e in KERNELS9L.items()})
                self.ng9l = True
            except Exception as exc:  # noqa: BLE001 - the other NG candidates stay
                print(f"[s1eng] NG9L GEMMs unavailable: {exc!r}", flush=True)
        self.ng9t = False
        if (self.ng9 and self.sw and os.environ.get("S1ENG_NG9T", "1") == "1" and os.path.exists(CUBIN9T)):
            try:   # COV9b tall-tail kernels
                data9t = open(CUBIN9T, "rb").read()
                self._img9t = C.create_string_buffer(data9t, len(data9t))
                self.mod9t = C.c_void_p()
                self._chk(self.cu.cuModuleLoadData(C.byref(self.mod9t), self._img9t), "cuModuleLoadData (ng9t)")
                fn9t, sw9t = {}, {}
                for name, (kind, big, small, th, mode) in KERNELS9T.items():
                    f = C.c_void_p()
                    self._chk(self.cu.cuModuleGetFunction(C.byref(f), self.mod9t, name.encode()), name)
                    smem = max(_smem(*big), _smem(*small))
                    self._chk(self.cu.cuFuncSetAttribute(f, 8, smem), "smem attribute")
                    fn9t[name] = (f, smem)
                for name, (big, small, th) in SW_KERNELS9T.items():
                    f = C.c_void_p()
                    self._chk(self.cu.cuModuleGetFunction(C.byref(f), self.mod9t, name.encode()), name)
                    smem = max(_smem(*big), _smem(*small))
                    self._chk(self.cu.cuFuncSetAttribute(f, 8, smem), "smem attribute")
                    sw9t[name] = (f, smem)
                self.fn.update(fn9t)
                self.sw.update(sw9t)
                self.ng9t = True
            except Exception as exc:  # noqa: BLE001 - the other NG candidates stay
                print(f"[s1eng] NG9T GEMMs unavailable: {exc!r}", flush=True)
        self._load_ng7()
        self.bufs: dict = {}   # workspace data_ptr -> (fp32 totals, int32 flags); kept alive (graphs bake pointers)
        self.keep: list = []

    def _load_ng7(self):
        """NG7: the nsw7l SwiGLU kernels + the table check kernel (switch on, sm_120, cubin present); anything missing
        or failing -> no NG7 candidates (nothing else changes)."""
        self.ng7 = False
        self.ng7_why = ""         # why the NG7 module is not loaded
        self.ng7_sw_ok = False    # check_ng7 passed: the nsw7l SwiGLU kernels may join the SG candidates
        self.ng7_sw_why = "table check not run"
        self.sw7, self.chk7 = {}, None
        cap = tuple(torch.cuda.get_device_capability(0))
        if not ng7_on():
            self.ng7_why = "switch off"
        elif cap != (12, 0):
            self.ng7_why = f"sm_{''.join(map(str, cap))} (the cubin is sm_120a only)"
        elif not os.path.exists(CUBIN7):
            self.ng7_why = "ngemm7.cubin missing"
        else:
            try:
                data7 = open(CUBIN7, "rb").read()
                self._img7 = C.create_string_buffer(data7, len(data7))
                self.mod7 = C.c_void_p()
                self._chk(self.cu.cuModuleLoadData(C.byref(self.mod7), self._img7), "cuModuleLoadData (ng7)")
                sw7 = {}
                for name, (big, small, th, kc) in SW_KERNELS7.items():
                    f = C.c_void_p()
                    self._chk(self.cu.cuModuleGetFunction(C.byref(f), self.mod7, name.encode()), name)
                    smem = max(_smem(*big), _smem(*small))
                    self._chk(self.cu.cuFuncSetAttribute(f, 8, smem), "smem attribute")
                    sw7[name] = (f, smem)
                chk = C.c_void_p()
                self._chk(self.cu.cuModuleGetFunction(C.byref(chk), self.mod7, CHECK7.encode()), CHECK7)
                self.sw7, self.chk7 = sw7, chk
                self.ng7 = True
            except Exception as exc:  # noqa: BLE001 - the other NG candidates stay
                self.ng7_why = f"{type(exc).__name__}: {exc}"
                print(f"[s1eng] NG7 GEMMs unavailable: {exc!r}", flush=True)

    @staticmethod
    def _chk(r, what):
        if r != 0:
            raise RuntimeError(f"NG {what}: CUresult {r}")

    def fix_bufs(self, ws: torch.Tensor, need: int):
        key = ws.data_ptr()
        cur = self.bufs.get(key)
        if cur is None or cur[0].numel() < need:
            tot = torch.empty(max(need, 1 << 20, 2 * cur[0].numel() if cur is not None else 0), dtype=torch.float32,
                              device="cuda")
            flags = torch.zeros(FLAGS, dtype=torch.int32, device="cuda")
            self.bufs[key] = (tot, flags)
            self.keep.append((tot, flags))
        return self.bufs[key]


@functools.lru_cache(maxsize=1)
def lib():
    path = _cubin()
    if os.environ.get("S1ENG_NG", "1") != "1" or path is None or not os.path.exists(path):
        return None
    try:
        return _Lib()
    except Exception as exc:  # noqa: BLE001 - no candidates on any loading problem
        print(f"[s1eng] NG GEMMs unavailable: {exc!r}", flush=True)
        return None


# S1ENG_NG_SET=lean: only the candidates that ever won S6 / SG timing on the RTX PRO 6000 (start-up logs 2026-10-19:
# every NG anchor winner and nearly every SG winner is below), so start-up times far fewer configs
LEAN_NG = {("ns4_128x128w8", 0), ("ns1_128x128w8", 0), ("nh_b128x256_s64x256", 0), ("nh_b128x256_s32x256", 0)}
LEAN_SW = {("nsw_b128x256_s32x256", 0), ("nsw_b128x256_s32x256", 1), ("nsw_b128x256_s64x256", 0),
           ("nsw_b128x256_s64x256", 1), ("nsw_b128x256s3_s32x256", 1), ("nsw_b128x128w8_s64x128", 0)}


# NG9B lean entries (the (name, drop) pairs that won the 2026-10-20 Lium paired / 1-stream tables)
LEAN_NG |= {("nh9_b128x256k64_s64x256", 0), ("nh9_b128x256k64_s64x256", 1), ("nh9_b128x256k64_s32x256", 0),
            ("nh9_b128x256k64_s32x256k64", 0), ("nh9_b128x256k64_s64x256k64", 0), ("nh9_b128x128k64_s32x128", 0),
            ("ns9_4_128x128k64", 0), ("ns9_1_128x128k64", 0), ("ns9_2_128x128k64", 0)}
LEAN_SW |= {("nsw9_b128x256k64_s64x256", 0), ("nsw9_b128x256k64_s64x256", 1), ("nsw9_b128x256k64_s32x256", 0),
            ("nsw9_b128x256k64_s32x256k64", 0), ("nsw9_b128x256k64_s64x256k64", 0)}


LEAN_NG |= {(name, d) for name in KERNELS9L for d in (0, 1, -1)}   # COV9b grouped lastfold split-K
LEAN_NG |= {(name, d) for name in KERNELS9T for d in _tail_drops(name)}   # COV9b tall tails
LEAN_SW |= {(name, d) for name in SW_KERNELS9T for d in _tail_drops(name)}


def _lean() -> bool:
    return os.environ.get("S1ENG_NG_SET", "full") == "lean"


# S1ENG_NG_DROPS="0,1,2,3": nh_* / nsw_* also offered with 2 / 3 big-tile rows moved to small tiles (finer last-wave
# fill; wavebench 2026-10-20: 3-7 % faster single-stream at some R for qkvz / q / gate_up); the lean set then keeps the
# s64x256 kernels at the extra drops. Default "0,1" = the candidates before.
def _drops() -> tuple:
    try:
        d = tuple(sorted({int(v) for v in os.environ.get("S1ENG_NG_DROPS", "0,1").split(",") if v.strip()}))
    except ValueError:
        d = (0, 1)
    return d or (0, 1)


def _lean_ok(name: str, drop: int, lean: set) -> bool:
    return (name, drop) in lean or (drop >= 2 and name in ("nh_b128x256_s64x256", "nsw_b128x256_s64x256"))


def candidates(N: int, K: int, split: int, red: int) -> list:
    """NgAlgos that may reproduce a pin with this split-K count and reduction scheme (exactness is checked by S6)."""
    if lib() is None:
        return []
    if _lean():
        return [a for a in _candidates(N, K, split, red) if _lean_ok(a.name, a.drop, LEAN_NG)]
    return _candidates(N, K, split, red)


def _candidates(N: int, K: int, split: int, red: int) -> list:
    if lib() is None:
        return []
    mode = 0 if split <= 1 else red
    if mode not in (0, 1, 2, 4):
        return []
    chunk = K if split <= 1 else -(-(K // 64) // split) * 64
    out = []
    nh4 = os.environ.get("S1ENG_NG_NH4", "0") == "1"   # NG4 2-CTA/SM kernels: wash / -0.7 % in-engine, opt-in
    table = dict(KERNELS)
    if lib().ng9:
        table.update(KERNELS9)
    if lib().ng9t:
        table.update(KERNELS9T)
    for name, (kind, big, small, th, kmode) in table.items():
        if kmode != mode or N % big[1] or K % big[2] or (small is not None and (N % small[1] or K % small[2])):
            continue
        if name.startswith("nh4_") and not nh4:
            continue
        if kind == "h":
            out += [NgAlgo(name, d, chunk, split, red) for d in (_tail_drops(name) or _drops())]
        else:
            out.append(NgAlgo(name, 0, chunk, split, red))

    if lib().ng9l and mode == 4 and split > 1 and chunk % 64 == 0:   # COV9b: one G per distinct group count
        S = -(-K // chunk)
        seen = set()
        for name, (kind, big, small, th, kmode, G, base) in KERNELS9L.items():
            if N % big[1] or N % small[1] or G > S:
                continue
            ngr = -(-S // G)
            if (base, ngr) in seen:
                continue
            seen.add((base, ngr))
            pad = base.endswith("_s32x256") or base.endswith("_s32x128")   # drop -1 ignores the small tiles
            drops = _tail_drops(base) or ((0, 1, -1) if pad else (0, 1))
            out += [NgAlgo(name, d, chunk, split, red) for d in drops]
    return out


def supported(x: torch.Tensor, out: torch.Tensor) -> bool:
    return (x.dtype == torch.bfloat16 and x.stride(1) == 1 and out.stride(1) == 1 and x.data_ptr() % 16 == 0
            and out.data_ptr() % 16 == 0 and (x.stride(0) * 2) % 16 == 0 and (out.stride(0) * 2) % 16 == 0)


def run(algo: NgAlgo, W: torch.Tensor, x: torch.Tensor, out: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    L = lib()
    if L is None or not supported(x, out):
        raise RuntimeError("NG GEMM: unavailable or unsupported layout")
    kind, big, small, th, mode = _kspec(algo.name)
    f, smem = L.fn[algo.name]
    M, K = x.shape
    N = W.shape[0]
    mb = 0
    if kind == "h":
        mb = max(0, M // big[0] - algo.drop) * big[0]
        grid = (mb // big[0]) * (N // big[1]) + (-(-(M - mb) // small[0])) * (N // small[1])
        tot = flags = None
    elif kind == "s":
        grid = -(-M // big[0]) * (N // big[1])
        tot = flags = None
    elif kind == "LG":   # COV9b grouped lastfold split-K: NG * tiles CTAs, bf16 partials + tile counters in fix_bufs
        G = KERNELS9L[algo.name][5]
        S = -(-K // algo.chunk)
        ngr = -(-S // min(G, S))
        if algo.drop < 0:   # big tiles only (rows padded)
            mb = -(-M // big[0]) * big[0]
            ns = 0
        else:
            mb = max(0, M // big[0] - algo.drop) * big[0]
            ns = -(-(M - mb) // small[0])
        tiles = (mb // big[0]) * (N // big[1]) + ns * (N // small[1])
        if tiles > FLAGS:
            raise RuntimeError("NG GEMM: too many tiles")
        grid = ngr * tiles
        tot, flags = L.fix_bufs(ws, (S * (mb + ns * small[0]) * N) // 2 + 1024)
    else:
        tiles = -(-M // big[0]) * (N // big[1])
        if tiles > FLAGS:
            raise RuntimeError("NG GEMM: too many tiles")
        grid = tiles * (-(-K // algo.chunk))
        tot, flags = L.fix_bufs(ws, M * N)
    args = [C.c_void_p(x.data_ptr()), C.c_void_p(W.data_ptr()), C.c_void_p(out.data_ptr()), C.c_int(M), C.c_int(N),
            C.c_int(K), C.c_int(x.stride(0)), C.c_int(out.stride(0)), C.c_int(algo.chunk), C.c_int(mb),
            C.c_void_p(tot.data_ptr() if tot is not None else 0), C.c_void_p(flags.data_ptr() if flags is not None else 0)]
    params = (C.c_void_p * len(args))(*[C.cast(C.pointer(a), C.c_void_p) for a in args])
    L._chk(L.cu.cuLaunchKernel(f, grid, 1, 1, th, 1, 1, smem, C.c_void_p(torch.cuda.current_stream().cuda_stream),
                               params, None), "launch")
    return out


# ------------------------------------------------------------------------------- SG: gate|up GEMM + SwiGLU (nsw_*)
# W = [Wg; Wu] stacked ([2H, K], the fused gate_up block); one launch computes g = x Wg^T and u = x Wu^T tile by tile
# (each tile holds BN/2 gate columns and the same BN/2 up columns, both unsplit = the gate / up pins' bit class), rounds
# both to bf16 and writes bf16(bf16(silu(g)) * u) (opmath f32 silu: x / (1 + expf(-x)), IEEE division), i.e. pjs's
# exact silu_mul (every bf16 gate value checked, ngtest5 2026-10-19). Out [M, H]. Used per (bucket, R) only where
# start-up selection found it bit-equal to the reference and faster than gate|up GEMM(s) + silu_mul.
SW_KERNELS = {  # name: (big (BM, BN, BK, stages), small, threads)
    "nsw_b128x256_s32x256": ((128, 256, 32, 4), (32, 256, 32, 4), 256),
    "nsw_b128x256_s64x256": ((128, 256, 32, 4), (64, 256, 32, 4), 256),
    "nsw_b128x256s3_s32x256": ((128, 256, 32, 3), (32, 256, 32, 4), 256),
    "nsw_b128x128w8_s32x256": ((128, 128, 32, 3), (32, 256, 32, 4), 256),
    "nsw_b128x128w8_s32x128": ((128, 128, 32, 3), (32, 128, 32, 4), 256),
    "nsw_b128x128w8_s64x128": ((128, 128, 32, 3), (64, 128, 32, 4), 256),
}


class NgSwAlgo:
    __slots__ = ("name", "drop")

    def __init__(self, name: str, drop: int):
        self.name, self.drop = name, drop

    def key(self):
        return ("nsw", self.name, self.drop)

    def __repr__(self):
        return f"NgSwAlgo({self.name}, drop={self.drop})"


def swiglu_candidates(H: int, K: int) -> list:
    L = lib()
    if L is None or not L.sw:
        return []
    out = []
    table = dict(SW_KERNELS)
    if L.ng9:
        table.update(SW_KERNELS9)
    if L.ng9t:
        table.update(SW_KERNELS9T)
    for name, (big, small, th) in table.items():
        if name not in L.sw or H % (big[1] // 2) or H % (small[1] // 2) or K % big[2] or K % small[2]:
            continue
        out += [NgSwAlgo(name, d) for d in (_tail_drops(name) or _drops())]
    if _lean():
        out = [a for a in out if _lean_ok(a.name, a.drop, LEAN_SW)]
    return out + swiglu_candidates7(H, K)


def swiglu_candidates7(H: int, K: int) -> list:
    """NG7: the nsw7l kernels (drops 0 and 1 of S1ENG_NG_DROPS; the tall tails at their one drop), only once check_ng7 passed on this GPU and only for
    K = the kernel's template K; not subject to the lean set (two kernels). [] with S1ENG_NG7 unset."""
    L = lib()
    if L is None or not getattr(L, "ng7", False) or not getattr(L, "ng7_sw_ok", False) or _LUT7[0] is None:
        return []
    drops = [d for d in _drops() if d <= 1] or [0]
    out = []
    for name, (big, small, th, kc) in SW_KERNELS7.items():
        if name not in L.sw7 or K != kc or H % (big[1] // 2) or H % (small[1] // 2):
            continue
        out += [NgSwAlgo(name, d) for d in (_tail_drops(name) or drops)]
    return out


def _a_rows(x: torch.Tensor) -> torch.Tensor:
    """NG7 kernels read A rows at stride K (no lda): a row-strided view is copied first (same values, same bits)."""
    return x if x.stride(0) == x.shape[1] or x.stride(1) != 1 else x.contiguous()


def run_swiglu(algo: NgSwAlgo, W: torch.Tensor, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """out[M, H] = SwiGLU(x @ W[:H]^T, x @ W[H:]^T) on the current stream; W = the stacked [2H, K] gate|up block."""
    L = lib()
    ng7 = algo.name in SW_KERNELS7
    if ng7:   # NG7: the SiLU table in ws, K fixed by the kernel, A rows at stride K
        if L is None or algo.name not in getattr(L, "sw7", {}) or _LUT7[0] is None \
                or x.shape[1] != SW_KERNELS7[algo.name][3]:
            raise RuntimeError("NG7 SwiGLU: unavailable (module / SiLU table) or K is not the kernel's")
        x = _a_rows(x)
    if L is None or not supported(x, out) or W.stride(1) != 1 or W.stride(0) != x.shape[1]:
        raise RuntimeError("NG SwiGLU: unavailable or unsupported layout")
    big, small, th = _swspec(algo.name)
    f, smem = L.sw7[algo.name] if ng7 else L.sw[algo.name]
    ws = _LUT7[0].data_ptr() if ng7 else 0
    M, K = x.shape
    N = W.shape[0]
    H = N // 2
    mb = max(0, M // big[0] - algo.drop) * big[0]
    grid = (mb // big[0]) * (H // (big[1] // 2)) + (-(-(M - mb) // small[0])) * (H // (small[1] // 2))
    args = [C.c_void_p(x.data_ptr()), C.c_void_p(W.data_ptr()), C.c_void_p(out.data_ptr()), C.c_int(M), C.c_int(N),
            C.c_int(K), C.c_int(x.stride(0)), C.c_int(out.stride(0)), C.c_int(K), C.c_int(mb), C.c_void_p(ws),
            C.c_void_p(0)]
    params = (C.c_void_p * len(args))(*[C.cast(C.pointer(a), C.c_void_p) for a in args])
    L._chk(L.cu.cuLaunchKernel(f, grid, 1, 1, th, 1, 1, smem, C.c_void_p(torch.cuda.current_stream().cuda_stream),
                               params, None), "SwiGLU launch")
    return out


# ------------------------------------------------------------------------------- NG7: SiLU table, check, report
_LUT7: list = [None]   # the nsw7l SiLU table (int16[65536] on the GPU, index = the bf16 bits of g); kept alive for
#                        the process: captured graphs bake its address into the nsw7l launches


def all_bf16(device="cuda") -> torch.Tensor:
    """Every bf16 bit pattern once, in bit order (element i holds the bits i)."""
    a = torch.arange(65536, dtype=torch.int32, device=device)
    return torch.where(a >= 32768, a - 65536, a).to(torch.int16).view(torch.bfloat16)


def lut7_values(device="cuda") -> torch.Tensor:
    """The nsw7l SiLU table: torch's own F.silu of every bf16 value on this device, as int16 bits (index = the
    input's bits), so the table holds exactly the reference's act_fn(gate) bytes."""
    return torch.nn.functional.silu(all_bf16(device)).view(torch.int16).contiguous()


CHECK7_U = (1.0, 0.01, 50.0)                       # randn scales of the check's u values ...
CHECK7_UC = (0.0, -0.0, 1.0, -3.5, 1e-30)          # ... and constant u values (8 u per g in all)


def _eq_num(a: torch.Tensor, b: torch.Tensor) -> int:
    """Outputs whose bits differ, except where both are NaN (NaN payloads are not compared; no forward value is NaN)."""
    return int(((a.view(torch.int16) != b.view(torch.int16)) & ~(torch.isnan(a) & torch.isnan(b))).sum())


def check_ng7(silu_mul=None, log=print) -> bool:
    """NG7 start-up check (S1ENG_NG7=1, before SG): builds the SiLU table (_LUT7) and runs the nsw7l epilogue's
    arithmetic (swiglu7_elem_lut = the kernels' own swiglu_pair) over every bf16 g (all 65536 bit patterns) x 8 u values
    against torch F.silu(g) * u (the reference's act_fn(gate) * up) and the engine's silu_mul: equal bit for bit
    wherever the result is a number -> lib().ng7_sw_ok and the nsw7l kernels join the SG candidates (every use is then
    byte-checked at its R). Any difference or error: no nsw7l candidates. Returns whether the check passed."""
    L = lib()
    if not ng7_on() or L is None or not getattr(L, "ng7", False) or L.chk7 is None:
        if ng7_on():
            log(f"lever not used: ng7 ({'the NG module is not loaded' if L is None else getattr(L, 'ng7_why', '')})")
        return False
    if L.ng7_sw_ok:
        return True
    t0 = time.time()
    why = ""
    try:
        if silu_mul is None:
            from .fused import silu_mul
        n = 65536
        g = all_bf16().repeat(len(CHECK7_U) + len(CHECK7_UC))
        gen = torch.Generator(device="cuda")
        gen.manual_seed(707)
        us = [(torch.randn(n, generator=gen, device="cuda") * s).to(torch.bfloat16) for s in CHECK7_U]
        us += [torch.full((n,), v, dtype=torch.bfloat16, device="cuda") for v in CHECK7_UC]
        u = torch.cat(us)
        lut = lut7_values()
        o = torch.empty_like(g)
        args = [C.c_void_p(g.data_ptr()), C.c_void_p(u.data_ptr()), C.c_void_p(o.data_ptr()),
                C.c_void_p(lut.data_ptr()), C.c_int(g.numel())]
        params = (C.c_void_p * len(args))(*[C.cast(C.pointer(a), C.c_void_p) for a in args])
        L._chk(L.cu.cuLaunchKernel(L.chk7, -(-g.numel() // 256), 1, 1, 256, 1, 1, 0,
                                   C.c_void_p(torch.cuda.current_stream().cuda_stream), params, None), "NG7 check")
        want = torch.nn.functional.silu(g) * u
        want2 = silu_mul(g, u)
        torch.cuda.synchronize()
        b1, b2 = _eq_num(o, want), _eq_num(o, want2)
        if b1 or b2:
            why = f"table check: {b1} / {b2} of {g.numel()} outputs differ from F.silu(g) * u / silu_mul"
    except Exception as exc:  # noqa: BLE001 - no nsw7l candidates, nothing else changes
        why = f"table check: {type(exc).__name__}: {exc}"
    if why:
        L.ng7_sw_why = why
        log(f"lever not used: ng7 (no nsw7l SwiGLU candidates: {why})")
        return False
    _LUT7[0] = lut
    L.ng7_sw_ok, L.ng7_sw_why = True, ""
    log(f"NG7 (S1ENG_NG7) SiLU table check: the nsw7l epilogue == F.silu(g) * u == silu_mul for every bf16 g x "
        f"{len(CHECK7_U) + len(CHECK7_UC)} u ({g.numel()} outputs) in {(time.time() - t0) * 1000:.0f} ms: nsw7l "
        f"kernels offered to SG")
    return True
