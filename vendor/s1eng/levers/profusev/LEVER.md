# PROFUSE-v on ONORM (ruj r3, 2026-10-23) - switch `S1ENG_PROFUSE=1` (default off in the code)

**What:** in packed GDN layers, ONORM's FFS + gated-norm kernel (`ffs_fused5n`, rgh) computes the value rows v itself
from the raw packed qkv GEMM rows, so the native prologue PRON (rgh) runs with `qk_only=True` (its q / k head blocks
and gate units only). v is no longer computed by a separate set of PRON CTAs, written to global memory and read back
by ONORM. New kernel `ffs_pv5n` in `onormpv.cubin` (source `onormpv.cu` = `#include "onorm.cu"` unchanged + the PV
section), loader / gate `onormpv.py`, dispatch `batchfwd.PackedForward._profuse` / `_gdn`, start-up gate block in
`model.capture_batch` (after ONORM's) and a re-check in `ntv.recheck`.

**Credit:** the idea and the raw-row / halo staging are rgh's dev_profuse (`ffsp.cu` `ffs_pv5`, on `ffs.cu`'s fused5
body, measured but never shipped). This lever is ruj's port onto ONORM's fused5n body (the body the engine runs for
packed graphs), with a column-slab conversion (each warp converts exactly the v columns its own pass P reads and its
own w columns overwrite, so no CTA barrier is added), int-only raw-row constants (no spill beyond ONORM's own 8 bytes),
the gate and the dispatch.

**Exactness:** v is PRON's `conv_tile4` arithmetic op for op (FHFMA 4-tap causal conv of the raw rows - prefix-variant
cache rows below plen, packed rows above, +0 outside [0, slen) - then PRON's F9 SiLU, rows >= slen +0); every other
operation is `ffs_fused5n`'s as `onorm.cubin` compiles it. Start-up gate `onormpv.check` per bucket at every B ONORM
runs at: both chains (PRON full -> KKT2 -> ONORM vs PRON qk_only -> KKT2 -> PV) on random packed layouts with the
model's GDN layer weights (every other layer), prefix layouts (none / template pfx..pmax over 3 variant blocks /
chunk-edge prefixes), scales 1 / 8 / 0.02, unused slots, z beside the qkv rows or separate, extreme z; q, k, beta, g, A,
v (gate kernel `ffs_pvx` vs PRON's v) and the packed output compared bit for bit. A failing bucket keeps the unchanged
chain. The engine's own prefix / self / mix checks then run the PV chain end to end.

**Measured (Verda VM, RTX PRO 6000, microbench of the GDN chain PRON + KKT2 + ONORM per served (L, B), weighted by the
served mix):** solo +2.9 % (the conversion's 2 MUFU ops per element sit on the kernel's sequential path), two lanes
running the chain -5.9 %, fixed work on both lanes (chains + a gate|up GEMM loop worth ~1.7x / ~3.4x their solo time)
-1.5 % / -1.3 % makespan. In-engine results: build/r3/PROFUSEV/PROGRESS.md (VPS).

**Patch:** `profusev_vs_k151.patch` (against kai #151's /opt/s1eng tree; `onormpv.cubin` is built from `onormpv.cu` with
the KERNELS.md line at /opt/s1eng/s1eng).
