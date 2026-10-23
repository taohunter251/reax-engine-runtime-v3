# S1+KF (ruj, 2026-10-21)

S1 (rgh/s1eng@sha256:86e01fac..., s1stack) + KF, ruj's exact in-CTA split-K GEMM for the N=2560 split-K pins (MLP down,
GDN out_proj, attention o_proj) and attention k|v. Switch `S1ENG_2N2560SPLITKPINSMLPDOWNGDNOUTPROJATTENTIONOPROJATTENTIONKV`
(`=1` built-in table, `=0` / unset = S1 exactly: nothing is loaded).

* `s1kf_vs_s1.patch`: S1's tree -> this tree (`patch -p1` from /opt/s1eng; `kf.cubin` is binary: next to kf.py).
* SKR skips every (K, bucket, R) whose pick is a KF algo (`skr.select`, `is_kf`), as own-final keeps SKR out of MX's
  two-class shapes. MX's mixed graphs keep their two-class skx6 GEMM (mx_setup reads the pins, not the S6 picks).
  RARN's rule keeps today's path at a KF key (KF + add_rmsnorm). GRP's grouped k|v + q launch keeps its sites.
* FCAB harness only: arm `nokf` = base picks + base RARN + base SKR selections; FCAB sets capture the env lane's MX
  mixed graphs too (gated by mix_check), so S1's joins run in every arm.
Full notes and numbers: the team VPS, build/levers_for_rgh/kf/LEVER.md.
