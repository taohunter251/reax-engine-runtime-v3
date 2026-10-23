"""XA (ATTN): native sm_120 attention + output gate for the packed attention layers (xa.cubin, built from xa.cu with
nvcc 13.x -O3 -arch=sm_120a), used by batchfwd.PackedForward._attn in place of SDPA (efficient kernel) +
attn_gate_grouped when S1ENG_XA=1 and the start-up check (PackedForward.check_xa) passed for the bucket.

Only the computed rows (positions plen .. n-1 of every decision) get query rows; one CTA = 16 positions x the 4 query
heads of one KV head (K/V tiles shared by the 4 heads); the per-row arithmetic replays the reference's masked SDPA call
(mem-efficient cutlass kernel, 32x128 gmem variant, HMMA.1688, all ceil(L / 128) key blocks of its walk) and the
sigmoid gate bit for bit; see the xa.cu header. S1ENG_XA_FQ=1 (default) also folds the q norm + RoPE
(fused.q_norm_rope_rows) into the kernel.

XA3 (kai/rf-xa, BACKLOG2 2.4 XA-LAT; S1ENG_XA3, default off): xa3.cubin (kai's xa3.cu, derived from xa.cu) holds
xa_fwd3q (the gate tile copied into shared memory once the Q tile is dead + the Q-build's loads issued together) and
xa_fwd3qe (the gate change only), launched exactly as xa_fwd2q. S1ENG_XA3=1 runs S1ENG_XA3_VER (3q default, 3qe) in the
env lanes; an FCAB arm xa3 / noxa3 (S1ENG_FCAB) gates it with the env lanes on xa_fwd2q. Gate (variant_check, inside
check_xa / fcl.xl_check and so inside ntv.recheck too): each output of xa_fwd2q against the variant's on the same
inputs, byte for byte, plus outlier and order-witness inputs (witness_inputs); a miss or an error keeps xa_fwd2q for
that (kind, bucket). The graphs read pf.xa_ver (XA layers) / pf.xl_ver (XL) at capture; FCAB swaps them (register_arms).
"""
from __future__ import annotations

import ctypes as C
import functools
import os

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
CUBINS = {(12, 0): os.path.join(_DIR, "xa.cubin")}
# kernels: "2" = xa_fwd2 (16 warps, 3-stage cp.async ring, 1-D grid with the heaviest position tiles first + XB_NZ
# zero-fill CTAs); "2q" = xa_fwd2q, the same with fused.q_norm_rope_rows folded in (raw q_proj rows in)
KERNELS = {"2": ("xa_fwd2", 100352, 512), "2q": ("xa_fwd2q", 100352, 512)}
# XA3 (kai): xa3.cubin, loaded only when a variant is asked for (S1ENG_XA3 / an FCAB xa3 arm)
CUBINS3 = {(12, 0): os.path.join(_DIR, "xa3.cubin")}
KERNELS3 = {"3q": ("xa_fwd3q", 100352, 512), "3qe": ("xa_fwd3qe", 100352, 512)}
VER = "2"
FQ = os.environ.get("S1ENG_XA_FQ", "1") == "1"   # fused q norm + RoPE (the "q" variant of VER)
XB_NZ = 4
HD, NH, NG = 256, 16, 4
INV_256 = 1.0 / 256.0


class _Lib:
    def __init__(self, path: str, kernels: dict | None = None):
        self.cu = C.CDLL("libcuda.so.1")
        torch.zeros(1, device="cuda")
        data = open(path, "rb").read()
        self._img = C.create_string_buffer(data, len(data))
        self.mod = C.c_void_p()
        self._chk(self.cu.cuModuleLoadData(C.byref(self.mod), self._img), "cuModuleLoadData")
        self.fns = {}
        for ver, (name, smem, _threads) in (KERNELS if kernels is None else kernels).items():
            fn = C.c_void_p()
            self._chk(self.cu.cuModuleGetFunction(C.byref(fn), self.mod, name.encode()), name)
            self._chk(self.cu.cuFuncSetAttribute(fn, 8, smem), "smem attribute")
            self.fns[ver] = fn

    @staticmethod
    def _chk(r, what):
        if r != 0:
            raise RuntimeError(f"XA {what}: CUresult {r}")


@functools.lru_cache(maxsize=1)
def _lib():
    try:
        path = CUBINS.get(tuple(torch.cuda.get_device_capability(0)))
        if path is None or not os.path.exists(path):
            return None
        return _Lib(path)
    except Exception as exc:  # noqa: BLE001 - no native path on any loading problem
        print(f"[s1eng] XA kernel unavailable: {exc!r}", flush=True)
        return None


def lib():
    return _lib()


@functools.lru_cache(maxsize=1)
def _lib3():
    try:
        path = CUBINS3.get(tuple(torch.cuda.get_device_capability(0)))
        if path is None or not os.path.exists(path):
            return None
        return _Lib(path, KERNELS3)
    except Exception as exc:  # noqa: BLE001 - no XA3 on any loading problem (xa_fwd2q stays)
        print(f"[s1eng] XA3 kernels unavailable: {exc!r}", flush=True)
        return None


def lib3():
    return _lib3()


def kernel_of(ver) -> tuple:
    """variant -> (cubin "xa" / "xa3", kernel name, dynamic smem, threads); ValueError for an unknown variant"""
    if isinstance(ver, str) and ver in KERNELS:
        return ("xa",) + KERNELS[ver]
    if isinstance(ver, str) and ver in KERNELS3:
        return ("xa3",) + KERNELS3[ver]
    raise ValueError(f"unknown XA variant {ver!r} (have {sorted(KERNELS) + sorted(KERNELS3)})")


def lib_for(ver):
    """the loaded module that holds variant ver's kernel (None: not loadable)"""
    return lib() if kernel_of(ver)[0] == "xa" else lib3()


def xa3_spec(env=os.environ):
    """S1ENG_XA3: "" / "0" / unset -> None (off); "1" -> S1ENG_XA3_VER ("3q" default, or "3qe"). ValueError else."""
    v = env.get("S1ENG_XA3", "0").strip()
    if v in ("", "0"):
        return None
    if v != "1":
        raise ValueError(f"S1ENG_XA3={v!r} (want 0 or 1)")
    ver = env.get("S1ENG_XA3_VER", "3q").strip() or "3q"
    if ver not in KERNELS3:
        raise ValueError(f"S1ENG_XA3_VER={ver!r} (want one of {sorted(KERNELS3)})")
    return ver


def xa3_wanted(env=os.environ):
    """the variant to gate at start-up: the env's (S1ENG_XA3=1), else S1ENG_XA3_VER (3q) when an FCAB arm toggles xa3
    or noxa3, else None. ValueError for a malformed S1ENG_XA3 / S1ENG_XA3_VER."""
    ver = xa3_spec(env)
    if ver is not None:
        return ver
    arms = [a for a in env.get("S1ENG_FCAB", "").replace(" ", "").split(",") if a and a != "env"]
    if any(t in ("xa3", "noxa3") for a in arms for t in a.split("+")):
        ver = env.get("S1ENG_XA3_VER", "3q").strip() or "3q"
        if ver not in KERNELS3:
            raise ValueError(f"S1ENG_XA3_VER={ver!r} (want one of {sorted(KERNELS3)})")
        return ver
    return None


def _fail(pf, kind: str, L: int, why: str) -> None:
    pf.xa_vbad[(kind, L)] = why
    for d in ((pf.xa_ver, pf.xa3_ok) if kind == "xa" else (pf.xl_ver, pf.xl3_ok)):
        d.pop(L, None)


def _unlike(base):
    """a buffer for the variant's output: base's shape and dtype, every byte base's complement, so a byte the variant
    does not write can never match xa_fwd2q's (review fix: not a reused torch.empty block that may hold the same)"""
    b = base.contiguous()
    out = torch.empty_like(b, memory_format=torch.contiguous_format)
    out.view(torch.uint8).copy_(b.view(torch.uint8) ^ 0xFF)
    return out


def vcount(pf, kind: str, L: int) -> int:
    """the number of variant comparisons that passed for (kind, bucket L) so far"""
    return (getattr(pf, "xa_vn", None) or {}).get((kind, L), 0)


def variant_check(pf, kind: str, L: int, base, run) -> None:
    """XA3 start-up gate step: base = xa_fwd2q's output, run(ver, out) = the variant's output on the same inputs,
    written into out (a buffer of base's shape whose every byte differs from base's, _unlike: an unwritten byte
    misses). Every byte must agree (all rows, the zero rows past the used ones and NaN patterns included, same shape);
    a pass counts in pf.xa_vn[(kind, L)]; a miss or an exception turns the variant off for (kind "xa" / "xl", bucket
    L) for good (pf.xa_vbad; out of pf.xa_ver / xa3_ok or pf.xl_ver / xl3_ok). No variant wanted (pf.xa_want None)
    or already missed: nothing runs. Never raises."""
    ver = getattr(pf, "xa_want", None)
    if not ver or (kind, L) in pf.xa_vbad:
        return
    try:
        got = run(ver, _unlike(base))
        if tuple(got.shape) != tuple(base.shape) or got.dtype != base.dtype:
            why = f"shape / dtype {tuple(got.shape)} {got.dtype} vs {tuple(base.shape)} {base.dtype}"
        else:
            a = base.contiguous().view(torch.uint8)
            b = got.contiguous().view(torch.uint8)
            if torch.equal(a, b):
                vn = getattr(pf, "xa_vn", None)
                if vn is None:
                    vn = pf.xa_vn = {}
                vn[(kind, L)] = vn.get((kind, L), 0) + 1
                return
            ia = base.contiguous().view(torch.int16) if base.element_size() == 2 else a
            ib = got.contiguous().view(torch.int16) if got.element_size() == 2 else b
            why = f"{int((ia != ib).sum())} values differ"
    except Exception as exc:  # noqa: BLE001 - the variant only: xa_fwd2q stays
        why = f"{type(exc).__name__}: {exc}"
    _fail(pf, kind, L, f"{ver}: {why}")


def require_new(pf, kind: str, L: int, before: int) -> None:
    """after a re-run of a start-up check (NTV re-check): the variant must have passed at least one more comparison
    for (kind, L) than ``before`` (vcount before the re-run), else it is turned off there ("no comparison")"""
    ver = getattr(pf, "xa_want", None)
    if ver and (kind, L) not in pf.xa_vbad and vcount(pf, kind, L) <= before:
        _fail(pf, kind, L, f"{ver}: no comparison in the re-check")


def finish(pf, kind: str, env=os.environ) -> dict:
    """after the XA ("xa", pf.xa) / XL ("xl", pf.xl) start-up block: the buckets whose variant passed every check
    (pf.xa3_ok / xl3_ok); with S1ENG_XA3=1 the env lanes use them (pf.xa_ver / xl_ver). -> the passed map."""
    ver = getattr(pf, "xa_want", None)
    members = pf.xa if kind == "xa" else pf.xl
    if ver:
        for L in sorted(members):   # a member whose variant was never compared has not passed (review fix)
            if (kind, L) not in pf.xa_vbad and vcount(pf, kind, L) < 1:
                _fail(pf, kind, L, f"{ver}: no comparison ran")
    ok = {L: ver for L in sorted(members) if ver and (kind, L) not in pf.xa_vbad}
    tgt, use = (pf.xa3_ok, pf.xa_ver) if kind == "xa" else (pf.xl3_ok, pf.xl_ver)
    tgt.clear()
    tgt.update(ok)
    try:
        on = xa3_spec(env) is not None
    except ValueError:
        on = False
    if on:
        use.update(ok)
    return ok


def register_arms(pf, log=print) -> None:
    """FCAB pick lever "xa3": arm xa3 captures its lane set with the gated variant buckets (pf.xa_ver = xa3_ok,
    pf.xl_ver = xl3_ok), noxa3 with none (xa_fwd2q everywhere); the dicts are swapped in place for the capture only
    and each set is self-checked. Nothing gated: nothing registered (an xa3 arm is then dropped). Never raises."""
    if not getattr(pf, "xa_want", None):
        return
    try:
        from . import fcab

        for ok, use, members in ((pf.xa3_ok, pf.xa_ver, getattr(pf, "xa", ())), (pf.xl3_ok, pf.xl_ver,
                                                                                 getattr(pf, "xl", ()))):
            for L in [L for L in ok if L not in members]:   # left XA / XL since finish (NTV re-check)
                ok.pop(L, None)
                use.pop(L, None)
        if not pf.xa3_ok and not pf.xl3_ok:
            log("XA3 FCAB arms not registered: no bucket passed the xa3 start-up gate")
            return
        fcab.register("xa3", pf, "xa_ver", on=dict(pf.xa3_ok), off={})
        fcab.register("xa3", pf, "xl_ver", on=dict(pf.xl3_ok), off={})
        log(f"XA3 FCAB arms: xa3 / noxa3 registered (XA buckets {sorted(pf.xa3_ok)}, XL buckets {sorted(pf.xl3_ok)}, "
            f"kernel {pf.xa_want}); env set on {'xa3' if pf.xa_ver or pf.xl_ver else 'xa_fwd2q'}")
    except Exception as exc:  # noqa: BLE001
        log(f"XA3 FCAB arms not registered ({type(exc).__name__}: {exc})")


def witness_inputs(kind: str, gen, R: int, B: int, L: int, H: int, HK: int, hd: int, slen, device="cuda"):
    """XA3 gate inputs beyond the XA gate's random draws -> (qg [R, H, 2 hd], k4, v4 [B, HK, L, hd]) bf16, K / V zero
    past each slot's slen:
      "outlier": random rows with about 1 % of the values scaled to 10^2..10^4 (q, k, v and gate halves), a few
                 whole rows large, some exact zeros and saturating gates (|g| up to 10^4);
      "witness": order witnesses: within every q / k / v row values 2^-12 .. 2^10 apart in magnitude (alternating), so
                 any change in the order of a fp32 sum (q norm, Q.K, P.V, row sums) changes its bits."""
    if kind not in ("outlier", "witness"):
        raise ValueError(f"unknown witness kind {kind!r}")
    f32 = dict(device=device, dtype=torch.float32)

    def rnd(*shape):
        return torch.randn(shape, generator=gen, device=device)

    if kind == "outlier":
        qg = rnd(R, H, 2 * hd)
        k4 = rnd(B, HK, L, hd)
        v4 = rnd(B, HK, L, hd) * 2
        for t in (qg, k4, v4):
            hit = torch.rand(t.shape, generator=gen, device=device) < 0.01
            mag = torch.pow(10.0, 2.0 + 2.0 * torch.rand(t.shape, generator=gen, device=device))
            t.copy_(torch.where(hit, t.sign() * mag, t))
            t.masked_fill_(torch.rand(t.shape, generator=gen, device=device) < 0.01, 0.0)
        if R > 3:
            qg[1] *= 300.0
            qg[2, :, :hd] *= 1e-4
        qg[..., hd:] = torch.where(torch.rand((R, H, hd), generator=gen, device=device) < 0.05,
                                   rnd(R, H, hd).sign() * 1e4, qg[..., hd:] * 8)
    else:
        def alt(*shape):
            n = shape[-1]
            e = torch.where(torch.arange(n, device=device) % 2 == 0, torch.tensor(10.0, **f32),
                            torch.tensor(-12.0, **f32))
            e = e + torch.randint(-2, 3, shape, generator=gen, device=device).to(torch.float32)
            sgn = torch.where(torch.rand(shape, generator=gen, device=device) < 0.5, -1.0, 1.0)
            return sgn * torch.pow(2.0, e) * (1.0 + torch.rand(shape, generator=gen, device=device))

        qg = alt(R, H, 2 * hd) * 2.0 ** -8
        k4 = alt(B, HK, L, hd) * 2.0 ** -8
        v4 = alt(B, HK, L, hd)
    keep = (torch.arange(L, device=device)[None, :] < slen.to(device).long()[:, None]).to(torch.float32)
    keep = keep[:, None, :, None]
    return (qg.to(torch.bfloat16).contiguous(), (k4 * keep).to(torch.bfloat16).contiguous(),
            (v4 * keep).to(torch.bfloat16).contiguous())


def supported(heads: int, kv_heads: int, hd: int, L: int) -> bool:
    return lib() is not None and heads == NH and kv_heads == NG and hd == HD and L % 16 == 0


_GTAB = {}


def gate_table(device) -> torch.Tensor:
    """The gate's sigmoid by bf16 bit pattern (fused.sigmoid_table), built once per device."""
    key = str(device)
    if key not in _GTAB:
        from . import fused

        _GTAB[key] = fused.sigmoid_table(device)
    return _GTAB[key]


def _launch(ver, q, k4, v4, qg, start, slen, plen, L, scale, out, gtab, qw=None, cos=None, sin=None, eps=0.0):
    if gtab is not None:
        assert gtab.dtype == torch.bfloat16 and gtab.numel() == 65536 and gtab.is_contiguous()
    lb = lib()
    R = qg.shape[0]
    B = k4.shape[0]
    assert qg.is_contiguous() and qg.shape[1:] == (NH, 2 * HD) and k4.is_contiguous() and v4.is_contiguous()
    assert tuple(k4.shape) == (B, NG, L, HD) and v4.shape == k4.shape
    for t in (start, slen) + ((plen,) if plen is not None else ()):
        assert t.dtype == torch.int32 and t.numel() >= B
    if out is None:
        out = torch.empty((R, NH * HD), dtype=qg.dtype, device=qg.device)
    vp = lambda t: C.c_void_p(t.data_ptr() if t is not None else 0)
    head = [vp(q), vp(k4), vp(v4), vp(qg), vp(out), vp(start), vp(slen), vp(plen), C.c_int(L), C.c_int(B), C.c_int(R),
            C.c_float(scale)]
    args = head + [vp(gtab), vp(qw), vp(cos), vp(sin), C.c_float(eps), C.c_float(INV_256)]
    grid = ((L // 16) * NG * B + XB_NZ, 1, 1)
    params = (C.c_void_p * len(args))(*[C.cast(C.pointer(x), C.c_void_p) for x in args])
    _cub, _name, smem, threads = kernel_of(ver)
    if _cub == "xa3":   # XA3 (kai): the variant's own module
        lb = lib3()
        if lb is None:
            raise RuntimeError("XA3 kernels unavailable (xa3.cubin / GPU)")
    lb._chk(lb.cu.cuLaunchKernel(lb.fns[ver], grid[0], grid[1], grid[2], threads, 1, 1, smem,
                                 C.c_void_p(torch.cuda.current_stream().cuda_stream), params, None), "launch")
    return out


def attn(q, k4, v4, qg, start, slen, plen, L: int, scale: float, out=None, gtab=None,
         ver: str | None = None) -> torch.Tensor:
    """q [R, 16, 256] (normed + RoPE'd rows), k4 / v4 [B, 4, L, 256], qg [R, 16, 512], start / slen / plen int32 [B]
    (plen None: 0) -> gated attention output [R, 4096] (rows past the used ones zero). gtab: the gate table
    (gate_table) instead of evaluating the sigmoid per element (same bits)."""
    assert q.is_contiguous() and tuple(q.shape) == (qg.shape[0], NH, HD)
    return _launch(VER if ver is None else ver, q, k4, v4, qg, start, slen, plen, L, scale, out, gtab)


def attn_q(qg, qw, cos, sin, eps: float, k4, v4, start, slen, plen, L: int, scale: float, out=None, gtab=None,
           ver: str | None = None) -> torch.Tensor:
    """attn with fused.q_norm_rope_rows folded in: the raw q_proj rows qg [R, 16, 512] (q = first 256 of each head),
    qw = 1 + q_norm weight [256] fp32, cos / sin [L, 64] bf16 (rows = positions), eps of the norm. ver: "2q" (None,
    xa_fwd2q) or an XA3 variant ("3q" / "3qe", xa3.cubin); ValueError for any other."""
    ver = "2q" if ver is None else ver
    if ver not in ("2q",) + tuple(KERNELS3):
        kernel_of(ver)   # unknown: ValueError
        raise ValueError(f"XA variant {ver!r} is not a q variant")
    assert qw.dtype == torch.float32 and qw.is_contiguous() and qw.numel() == HD
    assert cos.dtype == torch.bfloat16 and sin.dtype == torch.bfloat16 and cos.is_contiguous() and sin.is_contiguous()
    assert cos.shape[-1] == 64 and cos.numel() >= L * 64 and sin.numel() >= L * 64
    return _launch(ver, None, k4, v4, qg, start, slen, plen, L, scale, out, gtab, qw, cos, sin, eps)
