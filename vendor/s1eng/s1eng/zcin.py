"""ZC-IN (kai/kx-host, mining r2 lever; ARMS key "zc", default S1ENG_ZC=0): a lean packed launch's two input blobs
reach the device through one tiny copy kernel on the lane stream instead of two cuMemcpyHtoDAsync.

Why: under serving load the copy engine is shared with the serving models, so each launch's H2D copy waited p50 1.6
ms before it started (idle: 6 us; d96 loaded profile), and the graph cannot start before its inputs are there. A copy
kernel needs only the compute engine. The kernel (zcin.cu / zcin.cubin, nvcc -arch=sm_120a) reads the slot's pinned
staging blobs through UVA (pinned host memory is device-accessible) and writes the same bytes to meta.b64 / meta.b32.
Data movement only: graphs, kernels and logits are unchanged.

Start-up gate (BatchLoop._zc_setup, only when S1ENG_ZC=1 or an ARMS arm asks for zc): the cubin loads on this GPU,
every slot's staging blobs have a device pointer (cuPointerGetAttribute DEVICE_POINTER), and gate() copies random
blobs of every meta size in use through both paths and compares the bytes (and that nothing past the end is
written); then self_check / mix_check run on every lane set with their inputs copied by the kernel. Any failure: zc
stays unavailable, and an arm with "zc": 1 is refused ("arm X not applied", the harness's exit 4).

Slot reuse: the kernel reads host memory asynchronously, exactly like the async H2D copy it replaces, and runs on the
lane stream before the graph launch, the D2H copy and the slot's completion event record. The slot goes back to the
free list only after that event completed (BatchLoop._completer), so the host never rewrites a staging blob the
kernel may still read."""
from __future__ import annotations

import ctypes as C
import functools
import os

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
CUBIN = os.path.join(_DIR, "zcin.cubin")
KERNEL = "zcin_copy2"
THREADS = 256
MAX_CTAS = 8
CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3


def grid_x(n0: int, n1: int) -> int:
    """CTAs per blob: one 16-byte vector per thread and pass, at most MAX_CTAS (the blobs are 4-100 KB)."""
    return max(1, min(MAX_CTAS, -(-max(int(n0), int(n1)) // (THREADS * 16))))


class Zc:
    """zcin.cubin's kernel through the driver API with the GIL held (ctypes.PyDLL, as model._LeanCu)."""

    def __init__(self, cubin: str = CUBIN, lib=None):
        self.cu = lib if lib is not None else C.PyDLL("libcuda.so.1")
        cu = self.cu
        cu.cuModuleLoadData.argtypes = [C.c_void_p, C.c_void_p]
        cu.cuModuleGetFunction.argtypes = [C.c_void_p, C.c_void_p, C.c_char_p]
        cu.cuPointerGetAttribute.argtypes = [C.c_void_p, C.c_int, C.c_uint64]
        cu.cuLaunchKernel.argtypes = [C.c_void_p, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint,
                                      C.c_uint, C.c_void_p, C.c_void_p, C.c_void_p]
        for f in (cu.cuModuleLoadData, cu.cuModuleGetFunction, cu.cuPointerGetAttribute, cu.cuLaunchKernel):
            f.restype = C.c_int
        data = open(cubin, "rb").read()
        self._img = C.create_string_buffer(data, len(data))
        self.mod = C.c_void_p()
        r = cu.cuModuleLoadData(C.byref(self.mod), self._img)
        if r:
            raise RuntimeError(f"ZC cuModuleLoadData: CUresult {r}")
        self.fn = C.c_void_p()
        r = cu.cuModuleGetFunction(C.byref(self.fn), self.mod, KERNEL.encode())
        if r:
            raise RuntimeError(f"ZC cuModuleGetFunction {KERNEL}: CUresult {r}")
        # one parameter block, reused (cuLaunchKernel copies the values at the call; launches come from one thread)
        self._a = [C.c_uint64() for _ in range(6)]
        self._params = (C.c_void_p * 6)(*[C.cast(C.pointer(a), C.c_void_p) for a in self._a])

    def dev_ptr(self, host_ptr: int) -> int | None:
        """The device address of pinned host memory (UVA); None = not device-accessible."""
        d = C.c_uint64(0)
        r = self.cu.cuPointerGetAttribute(C.byref(d), CU_POINTER_ATTRIBUTE_DEVICE_POINTER, C.c_uint64(int(host_ptr)))
        return int(d.value) if r == 0 and d.value else None

    def launch(self, d0: int, s0: int, n0: int, d1: int, s1: int, n1: int, stream) -> int:
        """Enqueue zcin_copy2 on `stream`: n0 bytes s0 -> d0 and n1 bytes s1 -> d1. Returns the CUresult (0 = ok)."""
        a = self._a
        a[0].value, a[1].value, a[2].value = int(d0), int(s0), int(n0)
        a[3].value, a[4].value, a[5].value = int(d1), int(s1), int(n1)
        return self.cu.cuLaunchKernel(self.fn, grid_x(n0, n1), 2, 1, THREADS, 1, 1, 0, C.c_void_p(stream or 0),
                                      self._params, None)

    def copy_meta(self, meta, p64: int, p32: int, stream) -> int:
        """A PackedMeta's two blobs from staging at device addresses p64 / p32 (the lean launch's byte counts)."""
        return self.launch(meta.b64.data_ptr(), p64, meta.n64 * 8, meta.b32.data_ptr(), p32, meta.n32 * 4, stream)


@functools.lru_cache(maxsize=1)
def load():
    """Zc on this GPU, or None (no cubin, not sm_120, or any loading problem)."""
    if not os.path.exists(CUBIN):
        return None
    try:
        import torch

        if tuple(torch.cuda.get_device_capability(0)) != (12, 0):
            return None
        torch.zeros(1, device="cuda")   # the primary context is current
        return Zc()
    except Exception as exc:  # noqa: BLE001 - no ZC on any loading problem
        print(f"[s1eng] ZC kernel unavailable: {exc!r}", flush=True)
        return None


def _torch_buf(nbytes: int, fill: int):
    import torch

    t = torch.full((nbytes,), fill, dtype=torch.uint8, device="cuda")
    return t.data_ptr(), (lambda: t.cpu().numpy().tobytes())


def _torch_sync():
    import torch

    torch.cuda.synchronize()


def meta_sizes(lane_sets) -> list:
    """Every distinct (n64, n32) of the blob metas (packed and mixed graphs) of these lane sets."""
    out = set()
    for ls in lane_sets:
        for ln in ls:
            groups = list(getattr(ln, "packed", {}).values()) + list(getattr(ln, "mixed", {}).values())
            for es in groups:
                for e in es:
                    meta = e[2]
                    if getattr(meta, "blob", False):
                        out.add((int(meta.n64), int(meta.n32)))
    return sorted(out)


GUARD = 64       # bytes after each scratch destination that must stay untouched
SENTINEL = (0xA5, 0x5A)


def gate(zc, htod, slots: list, sizes: list, stream, new_buf=None, sync=None, rounds: int = 2,
         seed: int = 96) -> str | None:
    """ZC-IN byte gate. slots: the BatchLoop's _BSlot staging (pinned b64 / b32 tensors and their numpy views);
    sizes: (n64, n32) element counts in use. For every slot: its staging blobs' device pointers (stored as
    slot.zc64 / slot.zc32). Then for every size (and each slot's full capacity), `rounds` times with fresh random
    bytes: the H2D path (htod) and the kernel (zc.launch) copy the blobs into two scratch buffers; both must equal the
    staged bytes, and the kernel must not write past the end. The staging blobs are zeroed afterwards. Returns None
    (pass) or the reason it failed."""
    new_buf = new_buf or _torch_buf
    sync = sync or _torch_sync
    if not slots:
        return "no staging slots"
    for i, sl in enumerate(slots):
        if getattr(sl, "b64", None) is None or getattr(sl, "b32", None) is None:
            return "no staging blobs (S1ENG_META_BLOB=1)"
        p64, p32 = zc.dev_ptr(sl.b64.data_ptr()), zc.dev_ptr(sl.b32.data_ptr())
        if not p64 or not p32:
            return f"slot {i}: staging blob not device-accessible (cuPointerGetAttribute DEVICE_POINTER)"
        sl.zc64, sl.zc32 = p64, p32
    cap = (min(len(sl.b64_np) for sl in slots), min(len(sl.b32_np) for sl in slots))
    big = [s for s in sizes if s[0] > cap[0] or s[1] > cap[1]]
    if big:
        return f"meta blob larger than the staging slots: {big[:3]} > {cap}"
    rng = np.random.default_rng(seed)
    todo = sorted(set(map(tuple, sizes)) | {cap})
    k = 0
    try:
        for n64, n32 in todo:
            for _r in range(rounds):
                for sl in (slots if k == 0 else [slots[k % len(slots)]]):   # every slot once, then round-robin
                    sl.b64_np[:n64] = rng.integers(-(1 << 63), (1 << 63) - 1, n64, dtype=np.int64, endpoint=True)
                    sl.b32_np[:n32] = rng.integers(-(1 << 31), (1 << 31) - 1, n32, dtype=np.int32, endpoint=True)
                    w64, w32 = sl.b64_np[:n64].tobytes(), sl.b32_np[:n32].tobytes()
                    a64, ra64 = new_buf(n64 * 8 + GUARD, SENTINEL[0])
                    a32, ra32 = new_buf(n32 * 4 + GUARD, SENTINEL[0])
                    b64, rb64 = new_buf(n64 * 8 + GUARD, SENTINEL[1])
                    b32, rb32 = new_buf(n32 * 4 + GUARD, SENTINEL[1])
                    if htod(a64, sl.b64.data_ptr(), n64 * 8, stream) or htod(a32, sl.b32.data_ptr(), n32 * 4, stream):
                        return "H2D copy failed"
                    if zc.launch(b64, sl.zc64, n64 * 8, b32, sl.zc32, n32 * 4, stream):
                        return "copy kernel launch failed"
                    sync()
                    ga64, ga32, gb64, gb32 = ra64(), ra32(), rb64(), rb32()
                    if ga64[:n64 * 8] != w64 or ga32[:n32 * 4] != w32:
                        return f"H2D bytes differ from the staging (n64 {n64}, n32 {n32})"
                    if gb64[:n64 * 8] != w64 or gb32[:n32 * 4] != w32:
                        return f"copy kernel bytes differ from the H2D path (n64 {n64}, n32 {n32})"
                    if gb64[n64 * 8:] != bytes([SENTINEL[1]]) * GUARD or gb32[n32 * 4:] != bytes([SENTINEL[1]]) * GUARD:
                        return f"copy kernel wrote past the end (n64 {n64}, n32 {n32})"
                k += 1
    finally:
        sync()
        for sl in slots:   # serving starts from zeroed staging, as before the gate
            sl.b64_np[:] = 0
            sl.b32_np[:] = 0
    return None
