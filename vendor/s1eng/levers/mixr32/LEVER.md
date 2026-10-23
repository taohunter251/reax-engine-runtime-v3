# mixr32: `S1ENG_MIX_RSTEP=32` on S1+KF+skx6t (#131)

## Status in R3 (kai #151 + PROFUSE-v + this line; ruj, 23 Oct 2026)

In R3 the switch is set in the image config (`S1ENG_MIX_RSTEP=32`); nothing else of this lever ships. Earlier screens
of the same line on the later bases (Verda dev VM, gVisor, separate plain starts, ABBA, 30 blocks x 256 items):
* on B2 = tcb2 (#147): idle +0.148 % (CI -0.42..+0.72), loaded +0.045 % (n.s.), exact (b2i MIXR32-B2);
* on kai #151 (= `k151x`, dashboard #155): idle +0.315 % (CI +0.149..+0.481), loaded -0.051 % (CI -0.556..+0.456),
  exact; dashboard #155 3.47 / 3.54 and #156 (the same bytes + a hotkey label) 3.45 / 3.51 vs #151's own three jobs
  3.52 / 3.49, 3.45 / 3.53, 3.45 / 3.58: no resolved effect on the board.
In R3 the new 32-grid mixed graphs also run PROFUSE-v's `ffs_pv5n` in their GDN layers; MX's `mix_check`, the skx6t /
KFX gates and a start-up of every mixed graph against the per-decision graphs (b2i's mb_mix micro, run on the R3 image
before its measurements) cover that. The R3 measurements are in the R3 source bundle's README, not here.

Owner: ruj. Screened by the r32b track (2026-10-21 22:45-23:10 UTC), built and confirmed by the fixb track
(2026-10-22). The switch is S1's own env knob (`mix_rows` in `model.py`, team code), so this lever ships no new engine
code. Ours: the finding that the knob pays once skx6t is in, the in-situ A/B harness used to measure it
(`mixr32_harness_vs_131.patch`, inert without `S1ENG_ARMS`, not part of any shipped image), the separate-start
confirmation and the fixb image. Nothing comes from UID 198 / 177 / 7.

Switch: `S1ENG_MIX_RSTEP`
* `=32`: the 128 -> 192 mixed graphs (MX) are captured on a 32-row R grid.
* `=64` or unset: #131 exactly (S1's default).

**Status (2026-10-22 01:30 UTC):** exact. On the dev VM it is faster: +0.199 % (95 % CI +0.065..+0.333) in separate
plain starts vs #131, and +0.29 % (CI +0.05..+0.54) in situ. The team dashboard did **not** confirm it: job #132 (the
fixb image = #131 + this line) passed 6/6 at idle 3.36 / loaded 3.37, below #131's 3.43 / 3.46. Pooled and adjusted
for the reference, that is -1.87 % (se 0.71) idle and -2.23 % (se 1.34) loaded. Job-to-job drift of that size has
been seen on identical bytes (+1.32 %, #127 -> #129), so this is not a resolved loss either. The team best stays #131.
Re-check on the board (a second session set) before adopting it as a base.

## What it does

MX lets bucket-128 decisions ride in bucket-192 graphs. For each slot count B, `mix_rows` picks the R bins of the mixed
graphs from the bucket-192 packed R values (the ones that have plan picks). It keeps those on a grid of
`S1ENG_MIX_RSTEP` rows, from `max(step, 64 x (Bprev + 1))` rows up to B full-length bucket-192 decisions (the top bin,
always kept). The scheduler takes the smallest captured bin that holds the batch's rows. Every bin is one of the
bucket-192 packed R values (the packed graphs use a 32-row grid up to B 8, `S1ENG_RSTEP=32/64`), so every bin of the
32 grid has its plan picks too.
* 64 grid (#131): `slots/R bins {4: 9, 5: 8, 6: 10, 7: 11, 8: 12, 12: 21}` = 71 mixed graphs per lane.
* 32 grid: `slots/R bins {4: 16, 5: 15, 6: 18, 7: 21, 8: 23, 12: 31}` = 124 mixed graphs per lane. Every 64-grid bin is
  still there (same lower bound, same top bin, 32 | 64).
* Padding per mixed launch: 31.1 -> 15.5 rows. Mixed launches carry ~22 % of the decisions, so the executed rows per
  decision drop by 0.36 % (141.87 -> 141.36; useful rows 138.6 in both).

## Why it pays only with skx6t

The one-grid kernels with 64- or 128-row tiles over `[0, R)` run the same tiles on both grids: a 32-grid bin is R64 or
R64 - 32, and ceil(R / 64) is the same for both. The saved rows only go away where work is charged per m16 sub-tile or
per row:
* skx6t (our MX two-class GEMM without its padded tails) runs only the m16 sub-tiles that hold real rows in the last
  tile of each row segment. Its rows per mixed launch drop by 2.7 % (823.1 -> 801.1).
* The per-row (non-GEMM) work of the mixed graphs shrinks with R.
* Plain skx6 pads each segment to whole 128-row tiles (914.9 / 882.2 rows per mixed launch on the two grids), so on
  S1+KF without skx6t the same switch measured only +0.07 % (kfstack, CI -0.14..+0.27).

## Exactness

No arithmetic changes: the switch only adds graphs at other R bins, built from the same kernels and plan picks.
* MX's start-up `mix_check` compares every row of the mixed graphs with that row's own per-decision graph: `mix
  128->192 check ... PASS (42 graphs, 302 rows, 0 differ)`.
* The skx6t gate (`mx_tail_gate`, per (N, K, R) key vs skx6 itself) covers the new R keys: `lever used: skx6t
  (S1ENG_SKX6T, variant t) at 240 (N, K, R) keys ... 0 rejected` (#131: 160 keys).
* KF 190 keys / 0 rejected, RARN 179, SKR 119, GRP 87, self-check True on all four buckets: unchanged.
* Plain starts with the switch on: c16 2000 + c16r 2000 + c1 128 byte-equal to `/srv/reax/stack1/runs/ref_bundle`, long
  probe 58/58 the same. Checked 5 times: #131 with `-e S1ENG_MIX_RSTEP=32` (r32b r32x) and the 4 fixb starts of the
  final image (fixb fb_2B / 3B / 6B / 7B).

## Measurements (gVisor runsc, Verda RTX PRO 6000 dev VM, official regime: 256-item timed runs at c16, 3 s idle before each)

### In situ, one engine (r32b r32s, 2026-10-21 22:56-23:07 UTC)

Harness image `127.0.0.1:5000/ruj/engine@sha256:1df6f909...` (VM only) = #131 + `mixr32_harness_vs_131.patch`, started
with `S1ENG_MIX_RSTEP=32`. Arm `base {mix_rstep: 64}` makes exactly #131's picks; arm `r32 {}` uses the 32 grid.
abdriver2, 60 blocks, the same 256 items for both arms of a block, seeded random order.

| arm | items/s | vs base | se | 95 % CI | wins | c1 p50 |
|---|---|---|---|---|---|---|
| base (64 grid = #131) | 334.74 | - | - | - | - | 8.46 ms |
| r32 (32 grid) | 335.71 | **+0.29 %** | 0.12 | +0.05..+0.54 | 41/60 | 8.44 ms |

Median +0.28 %, order-balanced +0.29 %, halves +0.35 / +0.24 %, Wilcoxon p 0.009, sign test p 0.006.

### Separate plain starts, ABBA (fixb fb, 2026-10-22 00:13-00:52 UTC)

A = #131 (`e7c8807d`), B = the fixb image (`19ec3a26`, #131 + `ENV S1ENG_MIX_RSTEP=32` + a docs layer); order 1A 2B
3B 4A 5A 6B 7B 8A; no ARMS env; 30 blocks per start, the same items in every start (seed 7373).

| | A = #131 | B = fixb |
|---|---|---|
| items/s (mean of 4 starts) | 336.66 | 337.32 |
| starts | 336.59 / 336.73 / 336.75 / 336.56 | 337.35 / 337.38 / 337.72 / 336.86 |
| serving after | 149.3 s | 159.3 s |
| mixed graphs per lane / skx6t keys | 71 / 160 | 124 / 240 |
| energy per 256-item run | 598.0 J | 596.4 J (-0.29 %, CI -0.74..+0.17) |

* **Primary (start means, item-matched, df 6): +0.199 % (se 0.055, 95 % CI +0.065..+0.333).** With a drift term
  +0.199 % (CI +0.048..+0.349).
* Block level (120 item-matched blocks): +0.199 % (se 0.058), B faster in 78/120. Pairs +0.22 / +0.19 / +0.29 /
  +0.09 %, all 4 above 0.
* Start-to-start sd ~0: the spread of the start means is what run-to-run noise (0.47 % per timed run) gives.

### Team dashboard (gVisor, the scoring-machine type; job #132, 2026-10-22 00:53-01:25 UTC)

`86.38.238.51/ruj/engine:fixb` (`sha256:19ec3a26...`), 3 sessions per phase: pass 6/6, dist 0.000, top-1 100 %,
latency 0.63x.

| phase | #132 fixb | #131 | pooled adjusted d (#132 vs #131) |
|---|---|---|---|
| idle | 3.36 +- 0.02 (3.339 / 3.367 / 3.386; 341.0 items/s, ref 101.4) | 3.43 +- 0.04 (345.6, ref 100.7) | **-1.87 %** (se 0.71) |
| loaded | 3.37 +- 0.04 (3.339 / 3.367 / 3.414; 158.3 items/s, ref 46.4) | 3.46 +- 0.04 (164.1, ref 47.3) | **-2.23 %** (se 1.34) |

`dashpool.py` (plan-2): d = ln(candidate items/s) - beta ln(reference items/s), beta 0.85 idle / 0.66 loaded. The se
covers session noise only, not job-to-job drift.
* For scale: the same bytes moved +1.32 % between jobs (#127 -> #129).
* #131 read +1.62 % over S1+KF on the board, where the VM said +0.32 %.
* #132 reads -0.26 % (se 0.63) vs S1+KF, where the VM predicts about +0.5 %.

So #131 probably drew high and #132 low. A one-vs-one job difference has an se of ~1.3 % once drift is counted, which
puts -1.87 % about 1.6 se below the VM's +0.2 %. Nothing in the change is tuned to the dev VM: the new mixed bins reuse
the bucket-192 packed R values and their existing plan and KF picks, and the NTV memory guard decides before the lanes
are captured. Drift is the likelier cause, but the board result stands as "not confirmed". By the plan-2 rule (do not
adopt an exact lever whose pooled adjusted delta is below -1 %), do not make it the base on this one job.

## Cost

* Start-up +10 s: 53 more mixed graphs per lane to capture, 80 more skx6t gate keys. Serving at ~160 s, well inside
  the validator's 900 s start timeout.
* Device memory +0.2 GiB (lane 0) / +0.4 GiB (lane 1).
* c1 latency unchanged (p50 8.46-8.64 ms in every start of both images).

## For rgh

Optional, pending a second dashboard session set. On S1+KF+skx6t (#131), add `S1ENG_MIX_RSTEP=32` to the ENV
(`env.txt`). Nothing else changes. The log must show:
* `slots/R bins {4: 16, 5: 15, 6: 18, 7: 21, 8: 23, 12: 31}`;
* `lever used: skx6t (S1ENG_SKX6T, variant t) at 240 (N, K, R) keys ... 0 rejected`;
* `mix 128->192 check ... PASS (42 graphs, 302 rows, 0 differ)`.

Do not use it on a base without skx6t, where it is null. With other `S1ENG_MIX_B` slot sets the bins change, and the
skx6t gate covers whatever R keys result.

## Files

* `env.txt`: the env line.
* `mixr32_harness_vs_131.patch`: the in-situ A/B harness on #131's `model.py`. It adds the arms key `mix_rstep`, which
  limits the scheduler's mixed picks to one grid, and the `armrows` row counters. Every line it adds is inert without
  `S1ENG_ARMS`. It is not in any shipped image's code; the fixb image carries a copy under `/opt/s1eng/levers/mixr32/`
  for reference only.
* Notes and raw data: VPS `build/r32b/PROGRESS.md` (screen), `build/fixb/PROGRESS.md` (build, confirmation, dashboard);
  VM `/srv/reax/r32b/runs`, `/srv/reax/fixb/runs`.
