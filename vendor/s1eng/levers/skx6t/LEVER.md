# skx6t: the MX two-class GEMM (skx6) without its padded tail work

Owner: ruj, skx6t track (MXT lever B), 2026-10-21. Our own code: a new kernel file `skx6t.cu` next to our `skx6.cu`
(MX, r2 mixbatch), plus small hooks in `mx.py`, `batchfwd.py`, `model.py`, `fcab.py`. No rgh kernel code, nothing from
UID 198 / 177 / 7.

Switch: `S1ENG_SKX6T`
* `=1`: the default variant (`t`, below).
* `=t` / `=m`: that variant.
* `=0` or unset: the base engine exactly. skx6t.cubin is not even loaded, no gate runs, every mixed graph runs skx6.

## What it does

In a 128 -> 192 mixed graph (MX), rows `[0, M1)` belong to bucket-192 decisions and rows `[M1, R)` to bucket-128
decisions plus the graph's R padding. Five projections have a different pinned split-K class in the two buckets
(down 2560x9216: slices 2304 / 1344; out_proj and o_proj 2560x4096: 1024 / 512; z 4096x2560: unsplit / 512; fused
k|v 2048x2560: 448 / 192). They run skx6, which tiles each segment separately: `ceil(M1/BM) + ceil((R-M1)/BM)` row
tiles of BM = 128 (or 64) rows, so both segment tails are padded to whole tiles. A typical mixed graph has one or two
riders: about 113 rows in segment 2, so a 128-row launch computes about R + 79 rows (R + 47 for 64-row tiles), and the
padded rows are zero-filled rows that still go through ldmatrix + mma.sync. At the 600 W cap a zero row costs about as
much as a real one (rows1 zb: 92 %), so the padding has to be skipped, not just zeroed.

`skx6t.cu` keeps skx6's arithmetic for every stored element and removes the work on rows that are never stored:
* `gemm_skx6t_<cfg>` (variant `t`, the default): skx6's grid and segments. In the last (partial) tile of a segment each
  warp runs only its m16 row sub-tiles that hold a row below the segment end: their ldmatrix + mma.sync, nothing
  for the others. A warp with no row left skips its B fragments too; it still issues its share of the CTA's cp.async
  loads and joins every barrier.
* `gemm_skx6m_<cfg>` (variant `m`): one tile grid over `[0, R)`. The tile that holds row M1 carries rows of both
  classes; the k-loop runs to the union of both classes' slice ends and each m16 sub-tile folds at its own class's
  ends (per row in the sub-tile that straddles M1). Exact too, but slower than `t` (below).
* Per output element the chain is skx6's: per class, ceil64 slices of the pin's width, each one ascending-k mma.sync
  chain from +0 rounded once to bf16, folded in slice order with fp32 RN adds from the first partial itself, one RN
  round to bf16 (unsplit: bf16 of the chain). The rows of mma.sync m16n8k16 are independent, so skipping (or
  zero-filling) other rows cannot change a stored row.
* Implementation notes:
  * The active sub-tile count is a compile-time template argument (0..MT), dispatched per warp. A runtime
    `if (i < act)` made ptxas predicate each ldmatrix / mma.sync group behind a WARPSYNC and lose the load / MMA
    interleave: +5..+19 % per call.
  * The dispatch value is broadcast with `__shfl_sync(.., 0)`, so ptxas can prove it warp-uniform: 0 WARPSYNC.
  * Full tiles run skx6's hot loop and slice fold unchanged: 77 / 82 instructions per k-tile (skx6 78 / 82).
  * `t`: 144-240 registers, 0 spills. `m`: spills only in its boundary-tile event loop, outside the k-tile loops.
* Build: `nvcc -arch=sm_120a -O3 -cubin -Xptxas -v -o skx6t.cubin skx6t.cu` (nvcc 13.4 V13.4.92). It rebuilds the
  shipped cubin byte for byte (md5 `23ef7c9cd4485022635a2440a5f5093d`). The image carries `skx6t.cu`, `skx6.cu` (the
  source of the base's skx6.cubin, which S1 shipped without it) and `KERNELS.md` (exact build lines + md5 of every cubin
  of ours: skx6, skx6t, kf, rn) in `/opt/s1eng/s1eng/`.

## Start-up byte gate (fallback: the key keeps skx6)

`PackedForward.mx_tail_gate` runs in `capture_batch` right after `mx_setup` and `mix_rows`, before any mixed graph is
captured.
* It covers every two-class shape of the pair (and the fused k|v block) at every R of the pair's mixed graphs: the
  (kernel, N, K, R) keys that `_mx_gemm` will launch. It uses the model's own weights.
* Per key, `mx.tail_check` compares the variant with skx6 itself, on every one of the R output rows, in all bits:
  * M1 at one value in every 16-row block (residue 7q mod 16, so every residue mod 16 occurs), around every tile edge,
    and at 0 / 1 / R-1 / R;
  * two inputs: randn with every 7th row x64; rows at 2^U(-12, 12) with every 5th row x4096 and sparse +-1e4 spikes.
    The second input is used at every 4th M1;
  * the reference row is skx6 in the row's own class (skx6 at M1 = R and at M1 = 0). skx6 itself is also run at every
    11th M1 (a row's bits depend only on its class);
  * 128 sentinel rows after R must stay untouched.
* A key that passes enters `mx.TAIL`, and `mx.run` launches the variant there. `mx.run` reads the switch at call time
  (= capture time), so the FCAB `noskt` set captures skx6.
* Then the engine's own MX gate `mix_check` runs on the captured mixed graphs: every row of random mixed batches
  against its own bucket's per-decision graph.
* Log line: `lever used: skx6t (S1ENG_SKX6T, variant t) at <n> (N, K, R) keys of the 128->192 mixed graphs, bit-equal
  to skx6 at start-up in <s>s; 0 rejected (they keep skx6)`.

## Measured

Microbench (`skx6t_bench.py`, runc container of the S1+KF image, VM RTX PRO 6000):
* **exact:** 407930 compares in 221 cases, 0 differ, 0 out-of-bounds writes.
  * 5 shapes x 34 R bins (64..1792 step 64 + 608 / 760 / 912 / 1064 / 1216 / 1824); every tile config at 7 R.
  * 2 inputs; every M1 for R <= 256, else every value near tile edges, in the last two tiles, 16q + {0, 1, 8, 15}
    and 64 random values.
  * Each compare covers all R rows x N bits against skx6 in the row's own class. Measured on each kernel version (v1,
    v2, v3 = the shipped one).
* **time per call, two lanes** (CUDA graphs per lane):
  * workload per sample: 4 down + 4 out_proj / o_proj + 3 z + 1 k|v (S1's 31 : 31 : 23 : 8 per mixed graph);
  * 32 (R, M1) samples drawn like S1's mixed graphs (mean R 790);
  * 4 weight copies rotated (> L2); seeded variant order per round.

  | variant | capped: 8 rounds x 4 s at 600 W, ~2115 MHz | burst: 12 rounds, 2 s idle then ~0.68 s of work (official-like) |
  |---|---|---|
  | `t` | **-7.48 % time** (se 0.02), -7.38 % energy | **-7.42 % time** (se 0.06), -6.35 % energy (se 1.6) |
  | `m` | -7.17 % / -6.80 % | -6.97 % / -5.36 % |
* **one stream, no cap:**
  * `t` matches skx6 within +-1.5 % on the 128-row configs (down, out_proj, z at R > 704); k|v (64x256k32) is
    +4.5 %; z at R 704 is -24.5 %, because skx6 needs 2 waves there;
  * `m` is +4..+16 % on the 128-row configs and +40 % on k|v (18 fold events per boundary CTA).
* Expected in the engine: skx6 is 198.5 us of 3047 us per decision (s1prof). -7.4 % of it is about -0.5 % of GPU time
  per decision. The in-situ screen decides.

In-situ screen and exactness of the image: see the VPS copy of this file and `README.md` of `levers_for_rgh` (the
image copy was written before the screen ran).

## Files (patch `skx6t_vs_s1kf.patch`, `patch -p1` from the engine root, paths `s1eng/...`, against the S1+KF tree =
## #127 / #129 `/opt/s1eng/s1eng`)

* `skx6t.cu` + `skx6t.cubin` (new): ten kernels `gemm_skx6t_*` / `gemm_skx6m_*` for skx6's five tile configs.
* `skx6.cu` (new in the image): the source of the existing `skx6.cubin`, byte-identical to `levers_for_rgh/mx/skx6.cu`.
* `KERNELS.md` (new): sources, exact nvcc lines and md5 of our cubins.
* `mx.py`:
  * `TCUBIN` / `TVARIANTS` / `DEFAULT_TAIL` / `TAIL`;
  * the loader loads skx6t only with the switch on;
  * `run` -> `_launch` (the variant at a gated key, its grid: `m` has no spare tile row);
  * `tail_mode`, `tail_name`, `tail_check` (the gate).
* `batchfwd.py`: `PackedForward.mx_tail_gate`.
* `model.py`: the gate call in `capture_batch`'s MX block and its log line.
* `fcab.py`: FCAB toggle `noskt`, a lane set captured with `S1ENG_SKX6T=0`, so its mixed graphs run skx6. It is only
  used for in-situ A/B.
* Tools: `skx6t_bench.py` (exact / solo / time / burst), `unit_mx.py` (engine-path unit test: gate, switch, graph
  replay with a moved M1, a broken variant must be caught).

To merge on S1+KF:
1. `patch -p1 < skx6t_vs_s1kf.patch`.
2. Copy `skx6t.cubin` next to `mx.py`, or build it with the line above.
3. Set `S1ENG_SKX6T=1`.

The log must show `lever used: skx6t ... 0 rejected` and `mix 128->192 check: PASS`. On another base the gate re-checks
every key anyway; the kernel is independent of KF.

Not done (possible follow-up): the R-grid padding rows `[r_real, R)` are still computed by every kernel of a mixed
graph, skx6t included. Skipping them inside skx6t would need the real row count on the device (`PackedMeta.mx` has
`[r1, b1]` only). The other MXT lever (MIX_RSTEP 32, kfstack track) halves that padding for all kernels.
