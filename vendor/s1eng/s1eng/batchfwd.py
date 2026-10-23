"""S4: one forward for B decisions of the same bucket (stacked rows, [B, L] right-padded exactly as the reference
pads each decision), bit-identical to B separate reference forwards.

Per op:
  * linear layers: ONE cuBLASLt GEMM over the B*L stacked rows with the algo the reference gets for M = L, selected
    and verified at start-up on this GPU (lt.Pinned); a shape without a verified algo runs per decision through torch
    F.linear at M = L (the reference op itself);
  * row-wise / element-wise fused kernels (RMSNorm, SwiGLU, GDN gate glue, gated norm): rows are independent;
  * conv (per sequence: the causal window never crosses into another decision), q/k norm + RoPE, attention gate:
    batched kernels with one grid slice per decision;
  * fla chunk_gated_delta_rule on [B, L, H, D] and SDPA on [B, H, L, D] (the [1, 1, L, L] mask broadcast): one program
    per (decision, head, block), verified bit-equal to B = 1 by T1/T2;
  * final norm on the B last rows only (row-wise), readout as B separate [1, 2560] @ W^T products (the reference op).
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from . import fla_fused, onorm, rarn, skr
from .fastfwd import FastForward


class BatchForward:
    def __init__(self, ff: FastForward, pinned, ctx):
        assert {"rms", "swiglu", "attnglue", "glue", "gnorm", "conv", "qkn", "fmask"} <= ff.on, ff.on
        self.ff = ff
        self.fz = ff.fused
        self.pinned = pinned
        self.ctx = ctx
        self.ws = None  # workspace of the set being captured (set by the capture code)

    # ------------------------------------------------------------------ linear layers
    def lin(self, mod: torch.nn.Linear, x: torch.Tensor) -> torch.Tensor:
        B, L, K = x.shape
        W = mod.weight
        N = W.shape[0]
        algo = self.pinned.algo.get((N, K, L)) if self.pinned is not None else None
        if B == 1 or algo is None:
            if B == 1:
                return F.linear(x, W)
            return torch.cat([F.linear(x[b:b + 1], W) for b in range(B)])
        out = torch.empty((B, L, N), dtype=x.dtype, device=x.device)
        algo = self.pinned.algo_r.get((N, K, L, B * L), algo)   # S6: faster algo of the same bit class
        self.ctx.run(algo, W, x.view(B * L, K), out.view(B * L, N), self.ws)
        return out

    # ------------------------------------------------------------------ blocks
    def _gdn(self, d, x):
        la = d["la"]
        B, L, _ = x.shape
        fz = self.fz
        mixed_qkv = self.lin(la.in_proj_qkv, x)
        z = self.lin(la.in_proj_z, x)
        b = self.lin(la.in_proj_b, x)
        a = self.lin(la.in_proj_a, x)
        q, k, v = fz.conv_silu_split(mixed_qkv, d["conv_w"], la.key_dim, la.value_dim,
                                     la.num_v_heads // la.num_k_heads, la.head_k_dim, la.head_v_dim)
        beta, g = fz.gdn_gate(b, a, d["negexpA"], la.dt_bias)
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        o, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
                                      use_qk_l2norm_in_kernel=True, cu_seqlens=None)
        o = fz.gated_rmsnorm(o.reshape(-1, la.head_v_dim), z.reshape(-1, la.head_v_dim), la.norm.weight,
                             la.norm.variance_epsilon)
        return self.lin(la.out_proj, o.reshape(B, L, -1))

    def _attn(self, d, x, mask, cos, sin):
        sa = d["sa"]
        B, L, _ = x.shape
        hd = sa.head_dim
        fz = self.fz
        qg = self.lin(sa.q_proj, x).view(B, L, -1, hd * 2)
        kr = self.lin(sa.k_proj, x).view(B, L, -1, hd)
        vr = self.lin(sa.v_proj, x).view(B, L, -1, hd)
        q = fz.q_norm_rope(qg, d["qn_w"], cos, sin, self.ff.eps)
        k, v = fz.kv_norm_rope_rep(kr, vr, d["kn_w"], cos, sin, self.ff.eps, sa.num_key_value_groups)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, scale=sa.scaling, is_causal=False)
        return self.lin(sa.o_proj, fz.attn_gate(o, qg))

    def _mlp(self, d, x):
        mlp = d["mlp"]
        return self.lin(mlp.down_proj, self.fz.silu_mul(self.lin(mlp.gate_proj, x), self.lin(mlp.up_proj, x)))

    # ------------------------------------------------------------------ forward
    def __call__(self, ids: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        ff, fz = self.ff, self.fz
        B, L = ids.shape
        mask, cos, sin = ff._consts(ids[:1])
        h = ff.tm.embed_tokens(ids)
        pending = None
        for d in ff.layers:
            if pending is None:
                x = fz.rmsnorm(h, d["ln1_w"], ff.eps)
            else:
                h, x = fz.add_rmsnorm(h, pending, d["ln1_w"], ff.eps)
            if d["type"] == "linear_attention":
                y = self._gdn(d, x)
            else:
                y = self._attn(d, x, mask, cos, sin)
            h, x = fz.add_rmsnorm(h, y, d["ln2_w"], ff.eps)
            pending = self._mlp(d, x)
        # final norm on the last row of every decision only (row-wise op)
        rows = torch.arange(B, device=ids.device) * L + last
        hl = h.view(B * L, -1).index_select(0, rows).contiguous()
        pl = pending.view(B * L, -1).index_select(0, rows).contiguous()
        _, hn = fz.add_rmsnorm(hl, pl, ff.fn_w, ff.eps)
        sw_t = ff.slot_weight.T
        return torch.cat([hn[b:b + 1] @ sw_t for b in range(B)])


class PackedMeta:
    """Static device inputs of one packed graph (B decision slots, padded length L, R packed GEMM rows). pfx > 0 (E1):
    every decision starts with the pfx template tokens and the R rows hold positions pfx.. only."""

    def __init__(self, B: int, L: int, R: int, device="cuda", pfx: int = 0, mix: int = 0):
        self.B, self.L, self.R, self.pfx = B, L, R, pfx
        # MX (ruj r2 mixbatch): mix = S > 0 -> a mixed graph: slots [0, b1) hold bucket-L decisions (rows [0, r1)),
        # slots [b1, ..) bucket-S decisions (rows [r1, R)); mx = [r1, b1] (int32, staged with the other fields)
        self.S = int(mix)
        l0t = os.environ.get("S1ENG_L0T", "0") == "1"
        # E4 (S1ENG_META_BLOB): the fields are views of one int64 and one int32 device blob (256-byte aligned), so a
        # launch stages them in matching pinned blobs and copies them with two H2D copies instead of eight
        self.blob = os.environ.get("S1ENG_META_BLOB", "0") == "1"
        if self.blob:
            self.lay64, self.n64 = blob_layout([("ids", R), ("last", B)] + ([("l0t_slot", R)] if l0t else []), 32)
            self.lay32, self.n32 = blob_layout([("start", B), ("slen", B), ("row_pos", R), ("pvar", B), ("plen", B)]
                                               + ([("mx", 2)] if self.S else []), 64)
            self.b64 = torch.zeros((self.n64,), dtype=torch.long, device=device)
            self.b32 = torch.zeros((self.n32,), dtype=torch.int32, device=device)
            for name, (o, n) in self.lay64.items():
                setattr(self, name, self.b64[o:o + n])
            for name, (o, n) in self.lay32.items():
                setattr(self, name, self.b32[o:o + n])
            self.slen.fill_(L)
            self.plen.fill_(pfx)
            if not l0t:
                self.l0t_slot = None
            if self.S:
                self.mx_r1, self.mx_b1 = self.mx[0:1], self.mx[1:2]
            return
        self.ids = torch.zeros((R,), dtype=torch.long, device=device)
        self.start = torch.zeros((B,), dtype=torch.int32, device=device)
        self.slen = torch.full((B,), L, dtype=torch.int32, device=device)
        self.row_pos = torch.zeros((R,), dtype=torch.int32, device=device)
        self.last = torch.zeros((B,), dtype=torch.long, device=device)
        # PV (a4): per slot, the prefix variant (cache block) and its length (>= pfx); one variant of length pfx = E1
        self.pvar = torch.zeros((B,), dtype=torch.int32, device=device)
        self.plen = torch.full((B,), pfx, dtype=torch.int32, device=device)
        # L0T (kai): per packed row, the slot of its token in the bucket's layer-0 table (filled by l0t.Table.prepare);
        # allocated only with S1ENG_L0T=1 (flag off: nothing allocated, nothing launched)
        self.l0t_slot = torch.zeros((R,), dtype=torch.long, device=device) if l0t else None
        if self.S:
            self.mx = torch.zeros((2,), dtype=torch.int32, device=device)
            self.mx_r1, self.mx_b1 = self.mx[0:1], self.mx[1:2]


class MixPair:
    """MX (ruj r2 mixbatch): one (D, S) pair of mixed graphs - bucket-S decisions riding in bucket-D packed graphs, the
    D decisions' rows first. cls: (N, K) -> (slice width at D, at S) for every GEMM shape whose pinned bit class
    differs between the buckets (those run the two-class kernel mx.run, M1 = the graph's r1; on the pruned last layer
    (rows = slots) both buckets' own M = B paths, selected per slot); every other GEMM's class is the same in both
    buckets, so D's path is exact for every row. pc: layer -> {name: joint prefix cache} (D's variant blocks, then S's;
    S slots stage pvar + voff). Built by PackedForward.mx_setup, which byte-checks every two-class shape."""

    def __init__(self, D: int, S: int):
        self.D, self.S = int(D), int(S)
        self.cls = {}
        self.pc = {}
        self.voff = 0


def blob_layout(fields, align: int):
    """E4: [(name, length)] -> ({name: (offset, length)}, total) with every offset a multiple of align elements."""
    out, off = {}, 0
    for name, n in fields:
        out[name] = (off, n)
        off += -(-n // align) * align
    return out, off


def packed_layout(lengths: list, B: int, L: int, R: int, pfx: int = 0, plens: list | None = None):
    """Host-side metadata for decisions of the given token counts in a (B, L, R) packed graph (numpy int arrays):
    start, slen, row_pos, last. Unused slots have length 0. pfx > 0 (E1): only positions pfx.. get packed rows
    (start = row of position pfx, slen = the full length; every length must exceed pfx)."""
    import numpy as np

    start = np.zeros(B, dtype=np.int32)
    slen = np.zeros(B, dtype=np.int32)
    last = np.zeros(B, dtype=np.int64)
    row_pos = np.zeros(R, dtype=np.int32)
    r = 0
    for b, n in enumerate(lengths):
        p = pfx if plens is None else plens[b]  # PV: this decision's own (cached) prefix length
        assert n > p >= pfx
        k = n - p
        start[b] = r
        slen[b] = n
        last[b] = r + k - 1
        row_pos[r:r + k] = b * L + np.arange(p, n, dtype=np.int32)
        r += k
    assert r <= R and len(lengths) <= B
    return start, slen, row_pos, last


class PackedForward:
    """S5: B decisions of bucket L with their REAL token rows packed back to back (R GEMM rows, R >= sum of lengths).
    Every GEMM runs once over the R rows with the pinned algo of M = L; row-wise kernels run on the R rows; conv, the
    GDN gate glue and q/k norm + RoPE write fla's / SDPA's padded [B, L] layout (zeros in padding positions); the
    gated norm and the attention gate read the padded outputs back into packed rows (row_pos).

    E1 (meta.pfx > 0): every decision starts with the same pfx template tokens. Their rows are the same in every
    decision (causal model, row-independent per-row ops), so the R rows hold positions pfx.. only, and the four
    kernels that build the padded layouts read positions 0..pfx-1 from pcache[(L, layer)]: the GEMM outputs (qkv,
    b, a; q|gate, k, v) of the prefix rows, recorded once per bucket by an eager pfx = 0 forward with the bucket's
    own pinned algos (record_prefix). fla and SDPA get the same padded inputs bit for bit, every later op runs on the
    real rows only."""

    def __init__(self, bf: BatchForward):
        self.bf = bf
        self.ff = bf.ff
        self.fz = bf.fz
        self.pinned = bf.pinned
        self.ctx = bf.ctx
        self.prune = False  # set by capture_batch when M = 1..Bmax rows verified for every shape
        self.gva = False    # S1ENG_GVA: fla gets q/k with their own 16 heads (native grouped values), no repeat
        self.causal = {}    # L -> True: SDPA on the efficient kernel with is_causal=True (skips fully masked key
        #                     blocks) verified at start-up to give the masked call's bits (check_causal)
        self.pcache = {}    # E1: (L, layer) -> {name: [pfx, N] GEMM output rows of the template prefix}
        #                     PV (a4): [V * pmax[L], N], variant v's rows at v * pmax[L] .. v * pmax[L] + len(v) - 1
        self.pmax = {}      # PV: L -> rows per variant block of pcache (the longest variant's length)
        # FF (rgh): fla_fused.gdn_forward for the GDN layers; S1ENG_FF="BV,num_warps,num_stages" ("0" = fla)
        ff_env = os.environ.get("S1ENG_FF", "0")
        self.ff_fused = ff_env not in ("", "0")
        self.ff_cfg = tuple(int(x) for x in ff_env.split(",")) if self.ff_fused and "," in ff_env else (32, 4, 1)
        self.ff_skip = os.environ.get("S1ENG_FF_SKIP", "1") == "1"   # unused packed slots exit at once
        self.hf = os.environ.get("S1ENG_HF", "0") == "1"   # HF (rgh): conv_silu + gdn_gate in one launch
        # FF split mode: w/u fused into fwd_h (stores h, v_new), fla's parallel chunk_fwd_o computes o
        self.ff_mode = os.environ.get("S1ENG_FF_MODE", "")              # "" fused, "split", "ps"
        self.ff_split = self.ff_mode == "split"
        # FF-PEEL (kai, v4): ps mode's pass S with chunk 0 / the last chunk peeled (fla_fused module docstring); default
        # off (= job 38's ps kernel and launch); capture_batch switches it off unless ps mode passed the start-up check
        self.ff_peel = os.environ.get("S1ENG_FF_PEEL", "0") == "1"
        self.rec = None     # E1 recording: {(layer, name): rows 0..rec_p-1}, filled by an eager pfx = 0 forward
        self.rec_p = 0
        self.l0t = None     # L0T (kai): l0t.Table (S1ENG_L0T=1, set by capture_batch): layer 0's rows by token id
        self.pro = set()    # E4 PRO: buckets whose GDN prologue runs fused.gdn_prologue_packed (check_prologue passed)
        self.pron = set()   # PRON (rgh): PRO buckets whose prologue runs the native pron kernel (check_pron passed)
        self.onorm = onorm.enabled()   # ONORM (rgh): the gated norm as the FFS kernel's epilogue (onorm.py)
        self.profuse = os.environ.get("S1ENG_PROFUSE", "0") == "1"   # PROFUSE-v (ruj r3, onormpv.py): v in ONORM
        self.lr1 = set()    # E4 LR1: buckets whose last layer attends with one query row per slot (check_lr1 passed)
        self._ar = {}       # E4 LR1: B -> int32 arange(B) (the attention gate's row index of the one-row output)
        self.gqa = set()    # E4 GQA: buckets whose attention reads K/V once per KV head (check_gqa passed)
        self.rarn = {}      # RARN: (K, L, R) -> (ngr kernel, S) for the N=2560 split-K GEMMs (out_proj / o_proj /
        #                     down_proj) whose slice fold + residual add + RMSNorm run fused (rarn.select passed)
        self.xa = set()     # XA (ATTN): buckets whose attention layers run xa.attn (computed rows only, native
        #                     kernel + fused gate; check_xa passed)
        self.xa_gtab = None  # XA: the gate's sigmoid table (S1ENG_XA_GTAB=1, default off: the per-element sigmoid is faster)
        self.xa_fq = False   # XA: q norm + RoPE folded into the kernel (xa.attn_q; S1ENG_XA_FQ, set before check_xa)
        # XA3 (kai/rf-xa, xa3.cubin; S1ENG_XA3 / FCAB pick lever xa3): bucket -> variant the captured graphs launch in
        # the XA layers (xa_ver) / XL (xl_ver) instead of xa_fwd2q; xa3_ok / xl3_ok = the buckets whose variant passed
        # the start-up gate (xa.variant_check inside check_xa / fcl.xl_check); xa_want = the variant being gated (None:
        # none); xa_vbad = (kind, bucket) -> why the variant missed there
        self.xa_ver = {}
        self.xl_ver = {}
        self.xa3_ok = {}
        self.xl3_ok = {}
        self.xa_want = None
        self.xa_vbad = {}
        self.xa_vn = {}   # (kind, bucket) -> variant comparisons that passed (finish / NTV need >= 1)
        # PAR (rgh, u198 idea): independent GEMMs of a layer on a side stream beside the main one, forked / joined with
        # events inside the captured lane graph (graph branches): "ba" b|a beside qkv|z, "z" (where qkv and z are
        # separate GEMMs) b|a + z beside qkv / prologue / gated delta rule, "q" k|v beside q_proj (+ q norm/RoPE),
        # "last" the last layer's q_proj (M = B) + q norm/RoPE beside its k|v, "gu" (unfused gate/up without SG) up
        # beside gate. Same kernels on the same inputs, so the same bits. Only while capture_batch sets ws_side (the
        # side branch's own cuBLASLt / NG split-K workspace, one per lane), never while recording a prefix.
        # default = every site (Verda 10-21 idle read: +3.0 % vs h8, exact); S1ENG_PAR="" or "0" turns it off
        self.par = {p for p in os.environ.get("S1ENG_PAR", "z,ba,q,last,gu").split(",") if p}
        self.ws_side = None
        self._sides = {}    # current stream handle -> its side stream
        # FCL BA (ruj v14 fixedcost, fcl.py): buckets whose GDN b|a projection runs as ONE Triton launch (fcl.ba_gemm)
        # instead of the split-K GEMM + reduce (ba_check passed at start-up): L -> mode
        self.ba = {}
        # FCL LT (fcl.py): buckets where the pruned last layer's gathers + LR1 bias run as one launch (lt_check);
        # qkr stays empty here (QKR is not ported onto v15: XA folds the q norm + RoPE)
        self.qkr = set()
        self.lt = set()
        # FCL EMB / RO (fcl.py): buckets whose first gathers (embedding + L0T layer-0 rows) run as one launch
        # (emb_check) / whose readout's B GEMV products + cat run as one call (ro_check): L -> readout mode
        self.emb = set()
        self.ro = {}
        self.ro_rows = None     # ROP (pjs e10, S1ENG_RO_PAD): graph B -> RO's bmm row count (fcl.ro_plan; 0 = per-row)
        self._emb_rows = None   # EMB: layer 0's L0T rows gathered with the embedding (consumed by _gdn)
        # FCL TG (fcl.py): (N, K, bucket) -> mode: the last layer's M = B GEMMs as 16-row-tile kernels (tg_check)
        self.tg = {}
        self.xl = set()    # FCL XL: buckets whose last layer attends through XA's computed-rows kernel (xl_check)
        self.tgs = set()   # FCL TGS: buckets whose last-layer MLP gate + up + SwiGLU at M = B run as one launch (tgs_check)
        # MX (ruj r2 mixbatch): (D, S) -> MixPair of the mixed graphs (mx_setup); _mx = (pair, meta) while a mixed
        # forward runs, _mx_last while its pruned last layer runs (M = B rows = slots); l0t_stack = the L0T tables of
        # every bucket in one arena (mixed graphs gather each row from its own bucket's table by absolute slot)
        self.mxp = {}
        self._mx = None
        self._mx_last = False
        self.l0t_stack = None
        # RN (ruj r2 hotspot, rn.py): buckets whose row norms (add + RMSNorm, RARN fold + add + RMSNorm, plain RMSNorm)
        # / GDN gated norm run through rn.cubin's single-pass kernels (rn.check_norm / check_gnorm passed at start-up);
        # _rn / _rng = the current forward's choice (set in __call__; never while recording a prefix)
        self.rn = set()
        self.rng = set()
        self._rn = False
        self._rng = False
        # SKR (ruj r2 fixedcost2, skr.py): (K, bucket, R) -> (cuBLASLt split-K algo, S) of the N=2560 projections whose
        # slice partials fold straight into add_rmsnorm (skr.select passed at start-up; the lane graphs lose the
        # splitKreduce node in front of each fold: skr.prune)
        self.skr = {}
        # GRP (rgh 10-21): (site "gdn" | "att", L, R) -> grp.Pick whose one-launch grouped GEMM (b|a + qkv[|z]; k|v +
        # q) passed check_grp at start-up (set by capture_batch from the frozen GRP plan); elsewhere the v13b path
        self.grp = {}
        self.par_late = int(os.environ.get("S1ENG_PAR_LATE", "0"))   # test knob (see _gdn)
        # LATE (rgh 10-21): PAR sites whose side branch is issued AFTER the main GEMM's graph node (same fork, same
        # kernels): the block scheduler then dispatches the critical-path GEMM's CTAs first
        self.late = {p for p in os.environ.get("S1ENG_PAR_LATE_SITES", "").split(",") if p}
        self.late_r = {}    # LATE per (L, R) from the GRP plan: (L, R) -> set of sites

    def _par(self, what: str) -> bool:
        return what in self.par and self.ws_side is not None and self.rec is None

    def _side_stream(self, cur):
        st = self._sides.get(cur.cuda_stream)
        while st is None or int(st.cuda_stream) == int(cur.cuda_stream):
            st = torch.cuda.Stream()
        self._sides[cur.cuda_stream] = st
        return st

    def _on_side(self, fn):
        """PAR: run fn() on the current stream's side stream (after the work queued so far) with the side workspace;
        returns (fn's result, join token). The main stream continues at once; _join(token) makes it wait."""
        cur = torch.cuda.current_stream()
        side = self._side_stream(cur)
        side.wait_stream(cur)
        ws = self.bf.ws
        self.bf.ws = self.ws_side
        try:
            with torch.cuda.stream(side):
                out = fn()
        finally:
            self.bf.ws = ws
        return out, (cur, side)

    def _fork(self):
        """PAR (late side): the fork point (current stream, its side stream, event after the work queued so far); the
        side work is then issued after the main stream's next launch (_on_side_from), so that launch's graph node comes
        first."""
        cur = torch.cuda.current_stream()
        ev = torch.cuda.Event()
        ev.record(cur)
        return cur, self._side_stream(cur), ev

    def _on_side_from(self, fork, fn):
        cur, side, ev = fork
        side.wait_event(ev)
        ws = self.bf.ws
        self.bf.ws = self.ws_side
        try:
            with torch.cuda.stream(side):
                out = fn()
        finally:
            self.bf.ws = ws
        return out, (cur, side)

    @staticmethod
    def _join(tok) -> None:
        cur, side = tok
        cur.wait_stream(side)

    def sdpa(self, q, k, v, mask, scale, L):
        if self.causal.get(L):
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, scale=scale, is_causal=True)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=False)

    @torch.inference_mode()
    def check_causal(self, L: int, heads: int, hd: int, scale: float, bmax: int, seed: int = 5) -> bool:
        """Start-up gate of the causal skip at bucket L: random decisions (zeros in padding positions, and the pruned
        layer's one-query-row pattern) through both calls, every B up to bmax; bit equality required everywhere."""
        mask = self.ff.consts[L][0]
        g = torch.Generator(device="cuda")
        g.manual_seed(seed)
        lprev = max([x for x in (0, 128, 192, 256, 320) if x < L])
        for B in sorted({1, 2, 3, 4, 8, bmax}):
            for trial in range(3):
                shp = (B, heads, L, hd)
                q, k, v = (torch.randn(shp, generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16) * 1.5
                           for _ in range(3))
                n = torch.randint(lprev + 1, L + 1, (B,), generator=g, device="cuda")
                pos = torch.arange(L, device="cuda")
                keep = (pos[None, :] < n[:, None]).to(torch.bfloat16)[:, None, :, None]
                q, k, v = q * keep, k * keep, v * keep
                if trial == 2:   # pruned last layer: only the last real row of every decision has a query
                    q = q * (pos[None, :] == (n - 1)[:, None]).to(torch.bfloat16)[:, None, :, None]
                ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale,
                                                     is_causal=False)
                from torch.nn.attention import SDPBackend, sdpa_kernel

                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    got = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, scale=scale,
                                                         is_causal=True)
                if not torch.equal(ref.view(torch.int16), got.view(torch.int16)):
                    return False
        return True

    PRO_EPS = 1e-6      # fla 0.5.2 l2norm_fwd's eps (gdn_forward_ps calls it with the default)

    @torch.inference_mode()
    def check_prologue(self, L: int, bs, layers, pfx: int = 0, pmax: int = 0, seed: int = 47) -> str:
        """E4 PRO start-up gate of bucket L: fused.gdn_prologue_packed against the four-kernel path as kai v4 runs it
        (conv_silu_split_packed + gdn_gate_packed, then fla's l2norm_fwd on the q|k buffer and chunk_local_cumsum(g,
        64, RCP_LN2) as in fla_fused.gdn_forward_ps), bit for bit in q, k, v, beta and g, on random packed layouts:
        B in bs, these layers' conv / gate weights, row strides of the fused qkv|z and b|a blocks, inputs at several
        scales (softplus' t > 20 branch included), and with pfx > 0 random cached prefix rows, prefix lengths in
        pfx..pmax and 3 variant blocks. "" if every byte agrees."""
        from fla.modules.l2norm import l2norm_fwd
        from fla.ops.utils.constant import RCP_LN2
        from fla.ops.utils.cumsum import chunk_local_cumsum

        fz = self.fz
        dev = "cuda"
        g = torch.Generator(device=dev)
        g.manual_seed(seed + L)
        rng = np.random.default_rng(seed + L)
        lprev = max([x for x in (0, 128, 192, 256, 320) if x < L])
        nvar = 3
        pmax = max(pmax, pfx)
        for li, d in enumerate(layers):
            la = d["la"]
            KD, VD = la.key_dim, la.value_dim
            C = 2 * KD + VD
            NH = la.num_v_heads
            for B in bs:
                for trial in range(2):
                    lo = max(lprev + 1, pfx + 1)
                    n = rng.integers(lo, L + 1, B)
                    pl = rng.integers(pfx, pmax + 1, B) if pfx else np.zeros(B, dtype=np.int64)
                    pl = np.minimum(pl, n - 1) if pfx else pl
                    pv = rng.integers(0, nvar, B)
                    rows = (n - pl).astype(np.int64)
                    st = np.concatenate([[0], np.cumsum(rows)[:-1]]).astype(np.int32)
                    R = int(rows.sum()) + 32
                    sc = (1.0, 8.0, 0.02)[(li + B + trial) % 3]
                    full = (torch.randn((R, C + 4096), generator=g, device=dev) * sc).to(torch.bfloat16)
                    mixed = full[:, :C]
                    ba = (torch.randn((R, 2 * NH), generator=g, device=dev) * 4).to(torch.bfloat16)
                    ba[:, NH:NH + 4] += 24.0           # a > 20 - dt_bias: softplus' identity branch
                    b, a = ba[:, :NH], ba[:, NH:]
                    start = torch.from_numpy(st).to(dev)
                    slen = torch.from_numpy(n.astype(np.int32)).to(dev)
                    kw = {}
                    if pfx:
                        cache = (torch.randn((nvar * pmax, C), generator=g, device=dev) * sc).to(torch.bfloat16)
                        cba = (torch.randn((nvar * pmax, 2 * NH), generator=g, device=dev) * 4).to(torch.bfloat16)
                        kw = dict(pfx=pfx, plen=torch.from_numpy(pl.astype(np.int32)).to(dev),
                                  pvar=torch.from_numpy(pv.astype(np.int32)).to(dev), pmax=pmax)
                        cvk = dict(cache=cache, **kw)
                        cgk = dict(cb=cba[:, :NH], ca=cba[:, NH:], **kw)
                    else:
                        cvk, cgk = {}, {}
                    q0, k0, v0 = fz.conv_silu_split_packed(mixed, d["conv_w"], KD, VD, start, slen, B, L, gva=self.gva,
                                                           **cvk)
                    beta0, g0 = fz.gdn_gate_packed(b, a, d["negexpA"], la.dt_bias, start, slen, B, L, **cgk)
                    nq = q0.shape[0]
                    assert k0.data_ptr() == q0.data_ptr() + q0.numel() * q0.element_size()
                    qk = torch.as_strided(q0, (2 * nq,) + tuple(q0.shape[1:]), q0.stride())
                    qk, _ = l2norm_fwd(qk)
                    q0, k0 = qk[:nq], qk[nq:]
                    g0 = chunk_local_cumsum(g0, chunk_size=64, scale=RCP_LN2)
                    pk = dict(cache=cvk.get("cache"), cb=cgk.get("cb"), ca=cgk.get("ca"), **kw)
                    q1, k1, v1, beta1, g1 = fz.gdn_prologue_packed(mixed, d["conv_w"], KD, VD, b, a, d["negexpA"],
                                                                   la.dt_bias, start, slen, B, L, self.PRO_EPS,
                                                                   RCP_LN2, gva=self.gva, **pk)
                    torch.cuda.synchronize()
                    for name, x0, x1, view in (("q", q0, q1, torch.int16), ("k", k0, k1, torch.int16),
                                               ("v", v0, v1, torch.int16), ("beta", beta0, beta1, torch.int16),
                                               ("g", g0, g1, torch.int32)):
                        if x0.shape != x1.shape or not torch.equal(x0.contiguous().view(view), x1.contiguous().view(view)):
                            return f"{name} differs (layer #{li}, B={B}, pfx={pfx}, scale {sc})"
                    if not fla_fused.supported(q1, k1, v1):
                        return "FF does not take these shapes"
            from .gov import GOV

            GOV.tick()
        return ""

    @torch.inference_mode()
    def check_pron(self, L: int, bs, layers, pfx: int = 0, pmax: int = 0, seed: int = 53) -> str:
        """PRON (rgh) start-up gate of bucket L: pron.prologue (native) against fused.gdn_prologue_packed (the PRO
        kernel), bit for bit in q, k, v, beta and g, on random packed layouts as check_prologue builds them (every
        given GDN layer's conv / gate weights, B in bs, two input scales per B, unused slot, pfx > 0 with prefix lengths
        pfx..pmax over 3 variant blocks). "" if every byte agrees."""
        from fla.ops.utils.constant import RCP_LN2

        from . import pron

        fz = self.fz
        dev = "cuda"
        g = torch.Generator(device=dev)
        g.manual_seed(seed + L)
        rng = np.random.default_rng(seed + L)
        lprev = max([x for x in (0, 128, 192, 256, 320) if x < L])
        nvar = 3
        pmax = max(pmax, pfx)
        for li, d in enumerate(layers):
            la = d["la"]
            KD, VD = la.key_dim, la.value_dim
            C = 2 * KD + VD
            NH = la.num_v_heads
            for B in bs:
                for trial in range(2):
                    lo = max(lprev + 1, pfx + 1)
                    n = rng.integers(lo, L + 1, B)
                    pl = rng.integers(pfx, pmax + 1, B) if pfx else np.zeros(B, dtype=np.int64)
                    pl = np.minimum(pl, n - 1) if pfx else pl
                    if B > 2 and trial == 1:
                        n[-1], pl[-1] = 0, 0     # unused packed slot
                    pv = rng.integers(0, nvar, B)
                    rows = np.maximum(n - pl, 0).astype(np.int64)
                    st = np.concatenate([[0], np.cumsum(rows)[:-1]]).astype(np.int32)
                    R = int(rows.sum()) + 32
                    sc = (1.0, 8.0, 0.02)[(li + B + trial) % 3]
                    full = (torch.randn((R, C + 4096), generator=g, device=dev) * sc).to(torch.bfloat16)
                    mixed = full[:, :C]
                    ba = (torch.randn((R, 2 * NH), generator=g, device=dev) * 4).to(torch.bfloat16)
                    ba[:, NH:NH + 4] += 24.0
                    b, a = ba[:, :NH], ba[:, NH:]
                    start = torch.from_numpy(st).to(dev)
                    slen = torch.from_numpy(n.astype(np.int32)).to(dev)
                    pk = {}
                    if pfx:
                        cache = (torch.randn((nvar * pmax, C), generator=g, device=dev) * sc).to(torch.bfloat16)
                        cba = (torch.randn((nvar * pmax, 2 * NH), generator=g, device=dev) * 4).to(torch.bfloat16)
                        pk = dict(cache=cache, cb=cba[:, :NH].contiguous(), ca=cba[:, NH:].contiguous(), pfx=pfx,
                                  plen=torch.from_numpy(pl.astype(np.int32)).to(dev),
                                  pvar=torch.from_numpy(pv.astype(np.int32)).to(dev), pmax=pmax)
                    if not pron.takes(mixed, d["conv_w"], b, a, self.gva, KD, VD, L):
                        return "pron does not take these shapes"
                    r0 = fz.gdn_prologue_packed(mixed, d["conv_w"], KD, VD, b, a, d["negexpA"], la.dt_bias, start,
                                                slen, B, L, self.PRO_EPS, RCP_LN2, gva=self.gva, **pk)
                    r1 = pron.prologue(mixed, d["conv_w"], KD, VD, b, a, d["negexpA"], la.dt_bias, start, slen, B, L,
                                       self.PRO_EPS, RCP_LN2, gva=self.gva, **pk)
                    torch.cuda.synchronize()
                    for name, x0, x1 in zip(("q", "k", "v", "beta", "g"), r0, r1):
                        vw = torch.int32 if x0.dtype == torch.float32 else torch.int16
                        if x0.shape != x1.shape or not torch.equal(x0.contiguous().view(vw), x1.contiguous().view(vw)):
                            return f"{name} differs (layer #{li}, B={B}, pfx={pfx}, scale {sc})"
            from .gov import GOV

            GOV.tick()
        return ""

    @torch.inference_mode()
    def check_pron_domain(self, d) -> str:
        """PRON: the conv + SiLU + l2norm path over all 65536 bf16 conv sums (weights (0, 0, 0, 1): the sum is x_t
        exactly) and the gate over all 65536 bf16 b and a values, native vs the PRO kernel, bit for bit; and the
        conv weights' magnitudes (FHFMA: products of bf16 values exact in f32)."""
        from fla.ops.utils.constant import RCP_LN2

        from . import pron

        la = d["la"]
        KD, VD, NH = la.key_dim, la.value_dim, la.num_v_heads
        C = 2 * KD + VD
        dev = "cuda"
        for dd in self.ff.layers:
            if dd["type"] != "linear_attention":
                continue
            w = dd["conv_w"].float().abs()
            if bool(((w > 0) & (w < 2.0 ** -60)).any()) or not bool(torch.isfinite(w).all()):
                return "conv weights outside the FHFMA exactness range"
        pat = torch.arange(65536, dtype=torch.int32, device=dev).to(torch.int16).view(torch.bfloat16)
        w = torch.zeros((C, 4), dtype=torch.bfloat16, device=dev)
        w[:, 3] = 1.0
        L = 128
        mixed = torch.zeros((L, C), dtype=torch.bfloat16, device=dev)
        mixed.view(-1)[:65536 * 2:2] = pat
        mixed.view(-1)[1:65536 * 2:2] = pat.flip(0)
        z32 = torch.zeros(1, dtype=torch.int32, device=dev)
        for (mx, b, a, ww, LL) in ((mixed, mixed[:, :NH], mixed[:, NH:2 * NH], w, L),
                                   (torch.zeros((2048, C), dtype=torch.bfloat16, device=dev), pat.view(2048, NH),
                                    pat.view(2048, NH).flip(0).contiguous(), d["conv_w"], 2048)):
            sl = torch.full((1,), LL, dtype=torch.int32, device=dev)
            r0 = self.fz.gdn_prologue_packed(mx, ww, KD, VD, b, a, d["negexpA"], la.dt_bias, z32, sl, 1, LL,
                                             self.PRO_EPS, RCP_LN2, gva=self.gva)
            r1 = pron.prologue(mx, ww, KD, VD, b, a, d["negexpA"], la.dt_bias, z32, sl, 1, LL, self.PRO_EPS, RCP_LN2,
                               gva=self.gva)
            torch.cuda.synchronize()
            for name, x0, x1 in zip(("q", "k", "v", "beta", "g"), r0, r1):
                vw = torch.int32 if x0.dtype == torch.float32 else torch.int16
                if not torch.equal(x0.contiguous().view(vw), x1.contiguous().view(vw)):
                    return f"{name} differs on the bf16 domain sweep (L={LL})"
        return ""

    @torch.inference_mode()
    def record_prefix(self, L: int, prefix: list, filler: int = 11) -> None:
        """E1: one eager forward (pfx = 0, B = 1, R = L, the GEMM algos of M = L) of the prefix followed by filler
        tokens; keeps every layer's prefix rows in pcache[(L, layer)].
        PV (a4): ``prefix`` may be a list of variants (the first = the template prefix, every other one longer and
        starting with it): one recording per variant, stacked in blocks of pmax[L] rows (one variant: the E1 layout)."""
        variants = prefix if prefix and isinstance(prefix[0], (list, tuple)) else [prefix]
        pmax = max(len(v) for v in variants)
        stacks: dict = {}
        for v in variants:
            for key, t in self._record_one(L, list(v), filler).items():
                stacks.setdefault(key, []).append(t)
        for (i, name), ts in stacks.items():
            if len(ts) == 1 and ts[0].shape[0] == pmax:
                blk = ts[0]
            else:
                blk = torch.zeros((len(ts) * pmax, ts[0].shape[1]), dtype=ts[0].dtype, device=ts[0].device)
                for vi, t in enumerate(ts):
                    blk[vi * pmax: vi * pmax + t.shape[0]] = t
            self.pcache.setdefault((L, i), {})[name] = blk
        self.pmax[L] = pmax

    def _record_one(self, L: int, prefix: list, filler: int) -> dict:
        P = len(prefix)
        m = PackedMeta(1, L, L, device=self.ff.slot_weight.device)
        ids = torch.full((L,), filler, dtype=torch.long)
        ids[:P] = torch.tensor(prefix, dtype=torch.long)
        st, sl, rp, la = packed_layout([L], 1, L, L)
        m.ids.copy_(ids)
        m.start.copy_(torch.from_numpy(st))
        m.slen.copy_(torch.from_numpy(sl))
        m.row_pos.copy_(torch.from_numpy(rp))
        m.last.copy_(torch.from_numpy(la))
        self.rec, self.rec_p = {}, P
        try:
            self(m)
            if m.ids.is_cuda:
                torch.cuda.synchronize()
            rec = self.rec
        finally:
            self.rec, self.rec_p = None, 0
        return rec

    @torch.inference_mode()
    def mx_setup(self, D: int, S: int, weights: dict, fused: list, seed: int = 101) -> str:
        """MX (ruj r2 mixbatch) start-up gate of the pair (D, S): the pinned bit class of every layer GEMM shape at both
        buckets (cfg_of: slice width = ceil64 slices of splitK, reduction OUTPUT_TYPE; unsplit = K); shapes whose
        classes differ run the two-class kernel, byte-checked here against F.linear on M = D / M = S blocks (two inputs
        incl. x64 outliers, the boundary inside and on a tile edge, repeated) - for the fused blocks whose parts both
        differ, against the parts side by side. b / a: FCL BA's one mode when it is on for both buckets. The joint
        prefix caches need the same variant length at both buckets. "" if the pair is usable (self.mxp[(D, S)])."""
        from . import lt as ltmod
        from . import mx

        if mx.lib() is None:
            return "two-class kernels unavailable (cubin / GPU)"
        mp = MixPair(D, S)
        g = torch.Generator(device="cuda")
        g.manual_seed(seed + D + S)
        ba_same = self.ba.get(D) is not None and self.ba.get(D) == self.ba.get(S)
        for (N, K), W in sorted(weights.items()):
            cl = []
            for L in (D, S):
                a = self.pinned.algo.get((N, K, L))
                if a is None:
                    return f"no pin for {N}x{K}@{L}"
                cfg = ltmod.cfg_of(a)
                split, red = int(cfg.get("splitK", 1)), int(cfg.get("reduction", 0))
                cl.append((K, 0) if split <= 1 else (mx.chunk_of(K, split), red))
            if cl[0] == cl[1]:
                continue
            if N in (32, 64) and ba_same:   # b / a: FCL BA runs every row in its one (shared) mode
                continue
            if any(c[1] not in (0, 4) for c in cl):
                return f"{N}x{K}: class {cl} (only OUTPUT_TYPE split-K or unsplit reproduced)"
            mp.cls[(N, K)] = (cl[0][0], cl[1][0])
        # the two-class kernel on every such shape (and fused blocks of two such parts), byte for byte
        checks = [(N, K, W, None) for (N, K), W in weights.items() if (N, K) in mp.cls]
        for blk, n1 in fused:
            c1 = mp.cls.get((n1, blk.shape[1]))
            if c1 is not None and blk.shape[0] == 2 * n1:
                checks.append((blk.shape[0], blk.shape[1], blk, n1))
        for N, K, W, n1 in checks:
            c = mp.cls[(n1 or N, K)]
            for R1, R2 in ((5 * (D - 40) - 37, 2 * (S - 40) - 11), (128 * 3, 128 * 2), (77, 300), (0, 200), (200, 0)):
                R = R1 + R2
                if R <= 0:
                    continue
                x = torch.randn((R, K), generator=g, device="cuda")
                x[::7] *= 64
                x = x.to(torch.bfloat16)
                ref = []
                for (L, lo, hi) in ((D, 0, R1), (S, R1, R)):
                    if hi <= lo:
                        continue
                    xs = x[lo:hi]
                    nb = -(-(hi - lo) // L)
                    xp = torch.zeros((nb * L, K), dtype=x.dtype, device="cuda")
                    xp[:hi - lo] = xs
                    parts = [W[:n1], W[n1:]] if n1 else [W]
                    ref.append(torch.cat([torch.cat([torch.nn.functional.linear(xp[b * L:(b + 1) * L].unsqueeze(0),
                                                                                Wp).squeeze(0)
                                                     for b in range(nb)]) for Wp in parts], 1)[:hi - lo])
                ref = torch.cat(ref)
                m1 = torch.tensor([R1], dtype=torch.int32, device="cuda")
                name = mx.kernel_for(N, K, R)
                for rep_ in range(2):
                    out = torch.full((R, N), -3.0, dtype=torch.bfloat16, device="cuda")
                    mx.run(name, W, x, out, c[0], c[1], m1=m1)
                    if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                        nd = int((out.view(torch.int16) != ref.view(torch.int16)).sum())
                        return f"two-class GEMM {N}x{K} ({name}, rows {R1}+{R2}): {nd} values differ"
        if self.pcache:
            # S1 (MX x NTV): D's variant blocks may be wider than S's (NTV buckets: pmax 45 with the nonce-token blocks,
            # hotspot's 44 elsewhere) and may lack the q|gate rows (NTV drops them where XA never reads them). The mixed
            # graph runs D's path with D's pmax, so the joint cache takes D's keys, D's blocks unchanged, then S's blocks
            # re-laid at D's pmax rows (each variant row the same bytes at the same position in its block); D's own
            # pcache becomes a view of the joint tensor's first rows, so D's (large, NTV) blocks are not held twice.
            pd, ps = self.pmax.get(D), self.pmax.get(S)
            if pd is None or ps is None or ps > pd:
                return f"prefix variant blocks differ ({pd} vs {ps} rows)"
            for (L, i), c in self.pcache.items():
                if L != D:
                    continue
                cs = self.pcache.get((S, i))
                if cs is None or not set(c) <= set(cs):
                    return f"prefix cache of layer {i} missing at bucket {S}"
            for (L, i), c in list(self.pcache.items()):
                if L != D:
                    continue
                cs = self.pcache[(S, i)]
                jc = {}
                for k in list(c):
                    td, ts = c[k], cs[k]
                    nd_, ns_ = td.shape[0] // pd, ts.shape[0] // ps
                    if td.shape[0] != nd_ * pd or ts.shape[0] != ns_ * ps or td.shape[1:] != ts.shape[1:]:
                        return f"prefix cache {k} of layer {i}: block layout {tuple(td.shape)} / {tuple(ts.shape)}"
                    j = torch.zeros(((nd_ + ns_) * pd,) + tuple(td.shape[1:]), dtype=td.dtype, device=td.device)
                    j[:nd_ * pd].copy_(td)
                    j[nd_ * pd:].view(ns_, pd, *td.shape[1:])[:, :ps].copy_(ts.view(ns_, ps, *ts.shape[1:]))
                    jc[k] = j
                    c[k] = j[:nd_ * pd]   # D's pcache: a view of the joint rows (same bytes); the old block is freed
                    mp.voff = nd_
                    del td
                mp.pc[i] = jc
        torch.cuda.synchronize()
        self.mxp[(D, S)] = mp
        return ""

    @torch.inference_mode()
    def mx_tail_gate(self, D: int, S: int, weights: dict, fused: list, rs: list, seed: int = 303):
        """skx6t (ruj): start-up byte gate of the tail-skip two-class kernel (mx.tail_mode()) for the pair (D, S): every
        two-class shape of mx_setup (and the fused blocks it checked) at every R of the pair's mixed graphs, the
        variant against skx6 itself (mx.tail_check: all rows, all bits, M1 at every residue mod 16 / tile edges / both
        ends, two inputs). A key that passes runs the variant (mx.TAIL), any other key keeps skx6.
        -> (keys passed, keys rejected, variant) or None (switch off / no pair)"""
        from . import mx

        v = mx.tail_mode()
        mp = self.mxp.get((D, S))
        if not v or mp is None or mx.lib() is None:
            return None
        g = torch.Generator(device="cuda")
        g.manual_seed(seed + D + S)
        checks = [(N, K, W, mp.cls[(N, K)]) for (N, K), W in sorted(weights.items(), key=lambda kv: kv[0])
                  if (N, K) in mp.cls]
        for blk, n1 in fused:   # pair(): a fused block whose parts share both classes runs as one two-class launch
            c1 = mp.cls.get((n1, blk.shape[1]))
            if c1 is not None and blk.shape[0] == 2 * n1:
                checks.append((blk.shape[0], blk.shape[1], blk, c1))
        ok, bad = 0, []
        for N, K, W, c in checks:
            for R in sorted(set(rs)):
                name = mx.kernel_for(N, K, R)
                key = (name, N, K, R, int(c[0]), int(c[1]))   # the classes too: two pairs share (N, K, R) keys
                if key in mx.TAIL:
                    continue
                nd = mx.tail_check(name, W, c[0], c[1], R, v, g)
                if nd == 0:
                    mx.TAIL[key] = mx.tail_name(name, v)
                    ok += 1
                else:
                    bad.append(f"{N}x{K}@R{R} ({nd})")
        torch.cuda.synchronize()
        return ok, bad, v

    @torch.inference_mode()
    def mx_kfx_gate(self, D: int, S: int, weights: dict, fused: list, rs: list, seed: int = 505):
        """KFX (kai/kx-k2 #6): start-up byte gate of KF's mainloop for the two-class GEMMs (mx.kfx_spec()) for the pair
        (D, S): every two-class shape of mx_setup (and the fused blocks it checked) whose N the switch names, at every R
        of the pair's mixed graphs, the kfx kernel against skx6 and the skx6t variant the key runs (mx.kfx_check: all
        rows, all bits, M1 at every residue mod 16 / every 32-row edge from both ends, four inputs incl. both classes'
        order witnesses, each launch twice). A key that passes runs kfx (mx.KFX), any other key keeps skx6t / skx6.
        -> (keys passed, keys rejected, kfx kernel) or None (switch off / no pair / kfx kernels not loaded)"""
        from . import mx

        sp = mx.kfx_spec()
        mp = self.mxp.get((D, S))
        L = mx.lib() if sp is not None and mp is not None else None
        if L is None or not getattr(L, "xfn", None):
            return None
        kn = mx.kfx_name()
        g = torch.Generator(device="cuda")
        g.manual_seed(seed + D + S)
        checks = [(N, K, W, mp.cls[(N, K)]) for (N, K), W in sorted(weights.items(), key=lambda kv: kv[0])
                  if (N, K) in mp.cls]
        for blk, n1 in fused:   # pair(): a fused block whose parts share both classes runs as one two-class launch
            c1 = mp.cls.get((n1, blk.shape[1]))
            if c1 is not None and blk.shape[0] == 2 * n1:
                checks.append((blk.shape[0], blk.shape[1], blk, c1))
        ok, bad, added = 0, [], []
        try:
            for N, K, W, c in checks:
                if (sp and N not in sp) or N % mx.XBN:   # the switch names other sites / no 128-column tiles
                    continue
                for R in sorted(set(rs)):
                    name = mx.kernel_for(N, K, R)
                    key = (name, N, K, R, int(c[0]), int(c[1]))   # the classes too: two pairs share (N, K, R) keys
                    if key in mx.KFX:
                        continue
                    nd = mx.kfx_check(name, W, c[0], c[1], R, g, kn)
                    if nd == 0:
                        mx.KFX[key] = kn
                        added.append(key)
                        ok += 1
                    else:
                        bad.append(f"{N}x{K}@R{R} ({nd})")
            torch.cuda.synchronize()
        except BaseException:
            for key in added:   # an error drops this pair's keys only: an earlier pair's passed keys stay in use
                mx.KFX.pop(key, None)
            raise
        return ok, bad, kn

    def mx_stack_l0t(self) -> int:
        """MX: every bucket's L0T table rows moved into one arena (the books' tensors become row views of it, contents
        copied: inserts, gathers and the host maps are unchanged) -> {L: row offset of its book}; a mixed graph stages
        absolute slots (own book's slot + its offset). Must run before any graph captures a table pointer."""
        if self.l0t is None or not self.l0t.books:
            return 0
        bks = sorted(self.l0t.books.items())
        q0, b0 = bks[0][1].qkvz, bks[0][1].ba
        nq_all = sum(bk.qkvz.shape[0] for _, bk in bks)
        qa = torch.zeros((nq_all, q0.shape[1]), dtype=q0.dtype, device=q0.device)
        ba = torch.zeros((nq_all, b0.shape[1]), dtype=b0.dtype, device=b0.device)
        off = 0
        self.l0t_off = {}
        for L, bk in bks:
            n = bk.qkvz.shape[0]
            qa[off:off + n].copy_(bk.qkvz)
            ba[off:off + n].copy_(bk.ba)
            bk.qkvz, bk.ba = qa[off:off + n], ba[off:off + n]
            self.l0t_off[L] = off
            off += n
        torch.cuda.synchronize()
        self.l0t_stack = (qa, ba, bks[0][1].nq, bks[0][1].nb)
        return off

    def available(self, L: int) -> bool:
        return all(self.pinned.algo.get(k) is not None for k in self.pinned.base_keys(L))

    def gemm(self, W: torch.Tensor, x: torch.Tensor, L: int) -> torch.Tensor:
        if self._mx is not None:
            mp, m = self._mx
            c = mp.cls.get((W.shape[0], x.shape[1]))
            if c is not None:   # MX: the buckets' classes differ for this shape: each row in its own bucket's class
                return self._mx_gemm(mp, m, W, x, c)
        return self._gemm1(W, x, L)

    def _gemm1(self, W: torch.Tensor, x: torch.Tensor, L: int) -> torch.Tensor:
        R, K = x.shape
        N = W.shape[0]
        if R <= 16 and self.tg and self.rec is None:   # FCL TG: the last layer's M = B GEMMs (tg_check at start-up)
            mode = self.tg.get((N, K, L))
            if mode is not None:
                from . import fcl

                return fcl.tg_gemm(x, W, mode)
        out = torch.empty((R, N), dtype=x.dtype, device=x.device)
        algo = self.pinned.algo_r.get((N, K, L, R), self.pinned.algo[(N, K, L)])   # S6: same bit class, faster
        self.ctx.run(algo, W, x, out, self.bf.ws)
        return out

    def lin(self, mod, x: torch.Tensor, L: int) -> torch.Tensor:
        return self.gemm(mod.weight, x, L)

    def _mx_gemm(self, mp, m, W: torch.Tensor, x: torch.Tensor, c) -> torch.Tensor:
        """MX: x @ W^T with the D decisions' rows in bucket D's bit class and the S decisions' rows in bucket S's. The
        pruned last layer (rows = slots, M = B): both buckets' own M = B paths (TG / pins, start-up verified) on all B
        rows, then slot b < b1 takes D's row (a select: copies). Else the two-class kernel with M1 = r1 (device)."""
        R, K = x.shape
        if self._mx_last and R == m.B:
            yd = self._gemm1(W, x, mp.D)
            ys = self._gemm1(W, x, mp.S)
            sel = self._ar[m.B] < m.mx_b1
            return torch.where(sel.unsqueeze(1), yd, ys)
        from . import mx

        N = W.shape[0]
        out = torch.empty((R, N), dtype=x.dtype, device=x.device)
        mx.run(mx.kernel_for(N, K, R), W, x, out, c[0], c[1], m1=m.mx_r1)
        return out

    def lin_res(self, mod, x: torch.Tensor, L: int):
        """RARN: a GEMM whose output only feeds add_norm: deferred to add_norm where rarn.select verified the fused
        kernel for (K, bucket, R), else the GEMM output (today's path). v14: the deferred GEMM runs on the main stream
        inside add_norm, after every PAR side branch of its layer has joined (each PAR site joins before returning)."""
        v = self.rarn.get((x.shape[1], L, x.shape[0])) if self.rarn else None
        if v is not None and self._mx is not None and (mod.weight.shape[0], x.shape[1]) in self._mx[0].cls:
            v = None   # MX: a two-class shape: the GEMM output + add_rmsnorm (RARN folds one class only)
        if v is None or mod.weight.shape[0] != rarn.N_OUT:
            if (self.skr and self.rec is None and mod.weight.shape[0] == skr.N_OUT   # SKR (r2 fixedcost2)
                    and (self._mx is None or (mod.weight.shape[0], x.shape[1]) not in self._mx[0].cls)):
                # own-final: in an MX mixed graph a two-class shape keeps MX's two-class GEMM (SKR folds one class)
                s = self.skr.get((x.shape[1], L, x.shape[0]))
                if s is not None:
                    return skr.Deferred(mod.weight, x, s[0], s[1], L)
            return self.lin(mod, x, L)
        return rarn.Deferred(mod.weight, x, v[0], v[1], L)

    def add_norm(self, h: torch.Tensor, y, w: torch.Tensor, eps: float):
        """fused.add_rmsnorm(h, y, w); for a RARN-deferred y the fused kernels of rarn.apply (same bits: rarn
        module docstring)."""
        if isinstance(y, skr.Deferred):   # SKR: the pick's Kernel2, then the fold from its workspace partials
            return skr.apply(self.ctx, y.algo, y.W, y.x, h, w, eps, y.S, self.bf.ws)
        if isinstance(y, rarn.Deferred):
            if self._rn:   # RN: the same ngr GEMM, the fold + add + RMSNorm in rn.cubin (check_norm)
                from . import rn

                return rn.rarn_apply(y.name, y.W, y.x, h, w, eps, y.S, self.bf.ws)
            return rarn.apply(y.name, y.W, y.x, h, w, eps, y.S, self.bf.ws)
        if self._rn:
            from . import rn

            if rn.rows_ok(h, y, w):
                return rn.add_rmsnorm(h, y, w, eps)
        return self.fz.add_rmsnorm(h, y, w, eps)

    def norm(self, x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        """fused.rmsnorm(x, w); RN's single-pass kernel where the gate passed"""
        if self._rn:
            from . import rn

            if rn.rows_ok(x, None, w):
                return rn.rmsnorm(x, w, eps)
        return self.fz.rmsnorm(x, w, eps)

    def materialize(self, y) -> torch.Tensor:
        """the GEMM output of a RARN-deferred y (today's GEMM), else y"""
        return self.gemm(y.W, y.x, y.L) if isinstance(y, (rarn.Deferred, skr.Deferred)) else y

    def pair(self, d, gname: str, mod1, mod2, x: torch.Tensor, L: int):
        """Both projections of a fusable pair: one GEMM over the fused block [W1; W2] when start-up selection chose it
        for this (bucket, R) (same bit class as the parts, verified bit-exact), else the two GEMMs. Returns column views."""
        blk = d["fused"].get(gname)
        R = x.shape[0]
        if self._mx is not None:
            mp, m = self._mx
            K = x.shape[1]
            c1, c2 = mp.cls.get((mod1.weight.shape[0], K)), mp.cls.get((mod2.weight.shape[0], K))
            if c1 is not None or c2 is not None:
                # MX: a part's class differs between the buckets: both parts in the same classes -> one two-class
                # launch over the fused block (each output column is its part's own arithmetic), else the parts
                if blk is not None and c1 == c2 and not (self._mx_last and R == m.B):
                    y = self._mx_gemm(mp, m, blk, x, c1)
                    n1 = mod1.weight.shape[0]
                    return y[:, :n1], y[:, n1:]
                return self.lin(mod1, x, L), self.lin(mod2, x, L)
        if blk is not None and self.pinned.fuse_r.get((blk.shape[0], blk.shape[1], L, R), False):
            y = self.gemm(blk, x, L)
            n1 = mod1.weight.shape[0]
            return y[:, :n1], y[:, n1:]
        return self.lin(mod1, x, L), self.lin(mod2, x, L)

    def _ba_pair(self, d, la, x: torch.Tensor, L: int):
        """The GDN layer's (b, a): FCL BA's one-launch kernel where its start-up gate passed (column views of one
        [R, 64] output, the layout of pair()'s fused path), else pair(). Never while recording a prefix."""
        mode = self.ba.get(L)
        if mode is not None and self._mx is not None and self.ba.get(self._mx[0].S) != mode:
            mode = None   # MX: BA reproduces both buckets only with the same mode (mx_setup checks the pins then)
        if mode is not None and self.rec is None:
            from . import fcl

            y = fcl.ba_gemm(x, fcl.ba_weight(d), mode)
            n1 = la.in_proj_b.weight.shape[0]
            return y[:, :n1], y[:, n1:]
        return self.pair(d, "b_a", la.in_proj_b, la.in_proj_a, x, L)

    def _late(self, site: str, L: int, R: int) -> bool:
        """LATE: this PAR site issues its side branch after the main GEMM's node at (bucket, R) (env or GRP plan)."""
        return site in self.late or site in self.late_r.get((L, R), ())

    def _grp_at(self, site: str, L: int, R: int):
        """GRP: the start-up-checked grouped launch at (site, bucket, R), None for the v13b path (also while recording
        a template prefix)."""
        if not self.grp or self.rec is not None or self._mx is not None:   # MX: mixed graphs keep the per-GEMM path
            return None
        return self.grp.get((site, L, R))

    def _grp_gdn(self, d, x, g, L: int):
        """GRP GDN site: returns ((mixed, z, b, a), z join token or None). main 'qkvz': one launch writes the fused
        qkv|z block's output (column views, as pair()'s fused GEMM) and b|a; main 'qkv': one launch writes qkv and b|a,
        z runs as before (PAR: on the side stream from here, joined at the gated norm; else after)."""
        from . import grp as grpmod

        la = d["la"]
        R = x.shape[0]
        ba = d["fused"]["b_a"]
        y2 = torch.empty((R, ba.shape[0]), dtype=x.dtype, device=x.device)
        ztok = None
        n1 = la.in_proj_qkv.weight.shape[0]
        if g.main == "qkvz":
            blk = d["fused"]["qkv_z"]
            y = torch.empty((R, blk.shape[0]), dtype=x.dtype, device=x.device)
            grpmod.run(g, blk, x, y, ba, y2)
            mixed, z = y[:, :n1], y[:, n1:]
        else:
            # PAR z: 'qkv' z forked before the grouped kernel; 'qkv2' forked at the same point but its node issued
            # after the grouped kernel's; 'qkv3' z waits for the grouped kernel (runs beside the GDN chain only)
            late = g.main == "qkv2" and self._par("z")
            fk = self._fork() if late else None
            if self._par("z") and g.main == "qkv":
                z, ztok = self._on_side(lambda: self.lin(la.in_proj_z, x, L))
            mixed = torch.empty((R, n1), dtype=x.dtype, device=x.device)
            grpmod.run(g, la.in_proj_qkv.weight, x, mixed, ba, y2)
            if late:
                z, ztok = self._on_side_from(fk, lambda: self.lin(la.in_proj_z, x, L))
            elif g.main == "qkv3" and self._par("z"):
                z, ztok = self._on_side(lambda: self.lin(la.in_proj_z, x, L))
            elif ztok is None:
                z = self.lin(la.in_proj_z, x, L)
        nb = la.in_proj_b.weight.shape[0]
        return (mixed, z, y2[:, :nb], y2[:, nb:]), ztok

    def _grp_att(self, d, x, g):
        """GRP attention site: one launch writes q_proj's output and the fused k|v block's (column views)."""
        from . import grp as grpmod

        sa = d["sa"]
        R = x.shape[0]
        kv = d["fused"]["k_v"]
        Wq = sa.q_proj.weight
        q2 = torch.empty((R, Wq.shape[0]), dtype=x.dtype, device=x.device)
        y2 = torch.empty((R, kv.shape[0]), dtype=x.dtype, device=x.device)
        grpmod.run(g, Wq, x, q2, kv, y2)
        nk = sa.k_proj.weight.shape[0]
        return q2, y2[:, :nk], y2[:, nk:]

    @torch.inference_mode()
    def check_grp(self, L: int, picks: dict, gdn_layers, att_layers, seed: int = 71) -> dict:
        """GRP start-up gate of bucket L. picks: {(site, R): grp.Pick} (chunk2 set). For every pick, on the given
        layers' weights and inputs at three kinds (per-block scales 1..3, x8, 1 % outliers x32 with 5 % zero rows):
        the grouped launch's every output byte vs (1) the selected v13b path at that R (pair() / lin() with the bucket's
        S6 picks) and (2) the per-decision reference F.linear of each projection on M = L blocks. Returns {(site, R):
        reason} for the picks that failed (the others are bit-equal)."""
        from . import grp as grpmod
        from .gov import GOV

        bad = {}
        if not picks:
            return bad
        g = torch.Generator(device="cuda")
        g.manual_seed(seed + L)
        K = None
        for site in ("gdn", "att"):
            rs = sorted(R for (s_, R) in picks if s_ == site)
            if not rs:
                continue
            layers = gdn_layers if site == "gdn" else att_layers
            for li, d in enumerate(layers):
                if site == "gdn":
                    la = d["la"]
                    mods = [la.in_proj_qkv, la.in_proj_z, la.in_proj_b, la.in_proj_a]
                else:
                    sa = d["sa"]
                    mods = [sa.q_proj, sa.k_proj, sa.v_proj]
                K = mods[0].weight.shape[1]
                nb = -(-max(rs) // L)
                for kind in range(3):
                    x = torch.randn((nb * L, K), generator=g, device="cuda")
                    if kind == 0:
                        x = x * (1 + torch.arange(nb * L, device="cuda").div(L, rounding_mode="floor") % 3)[:, None]
                    elif kind == 1:
                        x = x * 8
                    else:
                        x = torch.where(torch.rand((nb * L, K), generator=g, device="cuda") < 0.01, x * 32, x * 0.5)
                        x[torch.rand((nb * L,), generator=g, device="cuda") < 0.05] = 0
                    x = x.to(torch.bfloat16)
                    ref = [torch.cat([F.linear(x[b * L:(b + 1) * L].unsqueeze(0), mo.weight).squeeze(0)
                                      for b in range(nb)]) for mo in mods]
                    for R in rs:
                        p = picks[(site, R)]
                        if (site, R) in bad:
                            continue
                        xr = x[:R].contiguous()
                        try:
                            if site == "gdn":
                                (o1, o2, o3, o4), tok = self._grp_gdn(d, xr, p, L)
                                if tok is not None:
                                    self._join(tok)
                                outs = [o1, o2, o3, o4]
                                s1, s2 = self.pair(d, "qkv_z", la.in_proj_qkv, la.in_proj_z, xr, L)
                                s3, s4 = self.pair(d, "b_a", la.in_proj_b, la.in_proj_a, xr, L)
                                std = [s1, s2, s3, s4]
                            else:
                                outs = list(self._grp_att(d, xr, p))
                                s3, s4 = self.pair(d, "k_v", sa.k_proj, sa.v_proj, xr, L)
                                std = [self.lin(sa.q_proj, xr, L), s3, s4]
                        except Exception as exc:  # noqa: BLE001 - a failing pick only loses itself
                            bad[(site, R)] = f"{type(exc).__name__}: {exc}"
                            continue
                        for j, (o, sd, rf) in enumerate(zip(outs, std, ref)):
                            o = o.contiguous().view(torch.int16)
                            if not torch.equal(o, sd.contiguous().view(torch.int16)):
                                bad[(site, R)] = f"output {j} differs from the selected path (layer #{li}, input {kind})"
                                break
                            if not torch.equal(o, rf[:R].view(torch.int16)):
                                bad[(site, R)] = f"output {j} differs from per-decision F.linear (layer #{li}, input {kind})"
                                break
                    GOV.tick()
        torch.cuda.synchronize()
        return bad

    def _l0t_fused(self, d, gname: str, R: int, L: int) -> bool:
        """L0T (kai): pair()'s choice at this (bucket, R): the fused GEMM (column views of one output) or the parts."""
        blk = d["fused"].get(gname)
        return blk is not None and bool(self.pinned.fuse_r.get((blk.shape[0], blk.shape[1], L, R), False))

    def _l0t_rows(self, d, m: PackedMeta):
        """L0T (kai): layer 0's (qkv, z), (b, a) gathered from the token table in the layout pair() gives here.
        MX: from the stacked tables (every row's slot is absolute: its own bucket's table), as Table.rows gathers."""
        fq, fb = self._l0t_fused(d, "qkv_z", m.R, m.L), self._l0t_fused(d, "b_a", m.R, m.L)
        if self._mx is not None:
            qt, bt, nq, nb = self.l0t_stack
            out = []
            for t, n1, fused in ((qt, nq, fq), (bt, nb, fb)):
                if fused:
                    y = t.index_select(0, m.l0t_slot)
                    out.append((y[:, :n1], y[:, n1:]))
                else:
                    out.append((t[:, :n1].index_select(0, m.l0t_slot), t[:, n1:].index_select(0, m.l0t_slot)))
            return out[0], out[1]
        return self.l0t.rows(m.L, m.l0t_slot, fq, fb)

    def _pc(self, L: int, i: int) -> dict:
        """the prefix rows of layer i for this forward: bucket L's pcache, or MX's joint cache of the pair"""
        return self._mx[0].pc[i] if self._mx is not None else self.pcache[(L, i)]

    def _record(self, i: int, **rows):
        """E1: keep the template prefix's rows (positions 0..rec_p-1 of the one decision of a recording forward)."""
        for name, t in rows.items():
            self.rec[(i, name)] = t[:self.rec_p].clone(memory_format=torch.contiguous_format)

    def _profuse(self, L: int, B: int, z, la, m: PackedMeta, mixed, d, c) -> bool:
        """PROFUSE-v (ruj r3) for this packed graph: bucket L passed onormpv.check and ONORM itself would run here
        (_gdn's ONORM conditions: FF-PEEL ps with slen skips, onorm.use(L, B), the norm's operand layout, S1ENG_ONORM_OFF
        unset; the native kkt_solve and the FFS library loaded), with operand layouts the kernel takes"""
        from . import ffs, onorm, onormpv

        return (self.onorm and self.ff_peel and self.ff_skip and onormpv.use(L, B)
                and os.environ.get("S1ENG_ONORM_OFF", "0") != "1" and fla_fused.KKT2["nat"] is not None
                and ffs.lib() is not None and onorm.supported(z, la.norm.weight, m.start, m.plen)
                and onormpv.takes(mixed, d["conv_w"], c.get("mixed"), m.pfx))

    def _gdn(self, d, x, m: PackedMeta, i: int):
        la = d["la"]
        fz, L, B = self.fz, m.L, m.B
        ztok = None
        g = self._grp_at("gdn", L, m.R) if x is not None else None
        if self.l0t is not None and x is None:   # L0T (kai): layer 0, rows from the token table (same bytes)
            rows, self._emb_rows = self._emb_rows, None   # FCL EMB: already gathered with the embedding
            (mixed, z), (b, a) = rows if rows is not None else self._l0t_rows(d, m)
        elif g is not None:   # GRP: b|a + qkv (z beside it on the PAR side stream) or b|a + qkv|z in one launch
            (mixed, z, b, a), ztok = self._grp_gdn(d, x, g, L)
        elif self._par("z") and not self._l0t_fused(d, "qkv_z", m.R, L):
            # PAR: qkv and z are separate GEMMs at this (bucket, R): b|a then z on the side stream; the prologue waits
            # for b|a only, z joins at the gated norm (it overlaps qkv, the prologue and the gated delta rule)
            def side():
                ba = self._ba_pair(d, la, x, L)
                ev = torch.cuda.Event()
                ev.record()
                return ba, ev, self.lin(la.in_proj_z, x, L)

            if self.par_late == 1 or self._late("z", L, m.R):   # LATE: qkv's node first, then the side's b|a + z
                fk = self._fork()
                mixed = self.lin(la.in_proj_qkv, x, L)
                ((b, a), ev_ba, z), ztok = self._on_side_from(fk, side)
            elif self.par_late == 2:   # (test knob) b|a on the side beside qkv, z on the side after qkv
                (b, a), tok_ba = self._on_side(lambda: self.pair(d, "b_a", la.in_proj_b, la.in_proj_a, x, L))
                mixed = self.lin(la.in_proj_qkv, x, L)
                self._join(tok_ba)
                z, ztok = self._on_side(lambda: self.lin(la.in_proj_z, x, L))
                ev_ba = torch.cuda.Event()
                ev_ba.record()
            else:
                ((b, a), ev_ba, z), ztok = self._on_side(side)
                mixed = self.lin(la.in_proj_qkv, x, L)
            torch.cuda.current_stream().wait_event(ev_ba)
        elif self._par("ba"):   # PAR: b|a on the side stream beside qkv|z
            if self._late("ba", L, m.R):   # LATE: the main GEMM's graph node first, the side's b|a after it (same fork)
                fk = self._fork()
                mixed, z = self.pair(d, "qkv_z", la.in_proj_qkv, la.in_proj_z, x, L)
                (b, a), tok = self._on_side_from(fk, lambda: self._ba_pair(d, la, x, L))
            else:
                (b, a), tok = self._on_side(lambda: self._ba_pair(d, la, x, L))
                mixed, z = self.pair(d, "qkv_z", la.in_proj_qkv, la.in_proj_z, x, L)
            self._join(tok)
        else:
            mixed, z = self.pair(d, "qkv_z", la.in_proj_qkv, la.in_proj_z, x, L)
            b, a = self._ba_pair(d, la, x, L)
        if self.rec is not None:
            self._record(i, mixed=mixed, b=b, a=a)
        c = self._pc(L, i) if m.pfx else {}
        pv = dict(plen=m.plen, pvar=m.pvar, pmax=self.pmax.get(L, m.pfx)) if m.pfx else {}
        pro = L in self.pro and self.rec is None and self.ff_fused and self.ff_mode == "ps"
        if pro and self.profuse and L in self.pron and self._profuse(L, B, z, la, m, mixed, d, c):
            # PROFUSE-v (ruj r3, onormpv.py): ONORM's kernel computes v itself from the raw qkv rows (PRON's arithmetic),
            # so the native prologue writes q / k only; the same bits as PRON + KKT2 + ONORM (onormpv.check at start-up)
            from fla.ops.utils.constant import RCP_LN2

            from . import onormpv, pron

            q, k, _, beta, g = pron.prologue(mixed, d["conv_w"], la.key_dim, la.value_dim, b, a, d["negexpA"],
                                             la.dt_bias, m.start, m.slen, B, L, self.PRO_EPS, RCP_LN2, gva=self.gva,
                                             cache=c.get("mixed"), cb=c.get("b"), ca=c.get("a"), pfx=m.pfx,
                                             qk_only=True, **pv)
            if ztok is not None:   # the kernel reads z (as ONORM's does)
                self._join(ztok)
                ztok = None
            o = onormpv.forward(q, k, g, beta, m.slen, z, la.norm.weight, la.norm.variance_epsilon, m.start, m.plen,
                                mixed, d["conv_w"], c.get("mixed"), m.pfx, m.pvar, pv.get("pmax", 0))
            return self.lin_res(la.out_proj, o.view(m.R, -1), L)
        if pro:   # E4 PRO: conv + l2norm(q, k) + gate + chunk cumsum of g in one launch (check_prologue at start-up)
            from fla.ops.utils.constant import RCP_LN2

            fn = fz.gdn_prologue_packed
            if L in self.pron:   # PRON (rgh): the native kernel, same bits (check_pron at start-up)
                from . import pron

                fn = pron.prologue
            q, k, v, beta, g = fn(mixed, d["conv_w"], la.key_dim, la.value_dim, b, a, d["negexpA"], la.dt_bias,
                                  m.start, m.slen, B, L, self.PRO_EPS, RCP_LN2, gva=self.gva, cache=c.get("mixed"),
                                  cb=c.get("b"), ca=c.get("a"), pfx=m.pfx, **pv)
        elif self.hf:   # HF (rgh): conv + gate in one launch (same bodies, same bits)
            q, k, v, beta, g = fz.conv_gate_packed(mixed, d["conv_w"], la.key_dim, la.value_dim, b, a, d["negexpA"],
                                                   la.dt_bias, m.start, m.slen, B, L, gva=self.gva,
                                                   cache=c.get("mixed"), cb=c.get("b"), ca=c.get("a"), pfx=m.pfx,
                                                   **pv)
        else:
            q, k, v = fz.conv_silu_split_packed(mixed, d["conv_w"], la.key_dim, la.value_dim, m.start, m.slen, B, L,
                                                gva=self.gva, cache=c.get("mixed"), pfx=m.pfx, **pv)
            beta, g = fz.gdn_gate_packed(b, a, d["negexpA"], la.dt_bias, m.start, m.slen, B, L, cb=c.get("b"),
                                         ca=c.get("a"), pfx=m.pfx, **pv)
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        if self.ff_fused and fla_fused.supported(q, k, v):
            # FF (rgh): recompute_w_u + fwd_h + fwd_o in one kernel, same arithmetic (start-up self-check vs fla)
            fwd = fla_fused.gdn_forward_ps_peel if self.ff_peel and self.ff_mode == "ps" else \
                fla_fused.forward_of(self.ff_mode)
            kw = {"pre": True} if pro else {}
            if pro and self.onorm and self.ff_peel and self.ff_skip and onorm.use(L, B) and \
                    onorm.supported(z, la.norm.weight, m.start, m.plen) and \
                    os.environ.get("S1ENG_ONORM_OFF", "0") != "1":   # r2 hotspot: FCAB arm noonorm
                # ONORM (rgh): the gated norm as FFS's epilogue (o comes back as the packed norm output); the kernel
                # reads z, so a PAR "z" side branch joins before it (z still overlaps qkv and the prologue)
                kw["norm"] = (z, la.norm.weight, la.norm.variance_epsilon, m.start, m.plen)
                if ztok is not None:
                    self._join(ztok)
                    ztok = None
            o = fwd(q, k, v, g, beta, BV=self.ff_cfg[0], num_warps=self.ff_cfg[1], num_stages=self.ff_cfg[2],
                    slen=m.slen if self.ff_skip else None, **kw)
            if o.dim() == 2:   # ONORM ran: o = the packed gated-norm output [R * 32, 128] (RARN: lin_res as below)
                return self.lin_res(la.out_proj, o.view(m.R, -1), L)
        else:
            if pro:   # PRO's q, k are normalised and g cumulated: only FF ps takes them (check_prologue tests this)
                raise RuntimeError("PRO outputs need the FF ps path")
            o, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
                                          use_qk_l2norm_in_kernel=True, cu_seqlens=None)
        if ztok is not None:
            self._join(ztok)
        if self._rng:   # RN: the gated norm in rn.cubin (check_gnorm)
            from . import rn

            o = rn.gated_rmsnorm_packed(o, z, la.norm.weight, la.norm.variance_epsilon, m.row_pos, la.num_v_heads)
        else:
            o = fz.gated_rmsnorm_packed(o, z, la.norm.weight, la.norm.variance_epsilon, m.row_pos, la.num_v_heads)
        return self.lin_res(la.out_proj, o.view(m.R, -1), L)

    def _attn(self, d, x, m: PackedMeta, mask, cos, sin, i: int):
        sa = d["sa"]
        fz, L, B = self.fz, m.L, m.B
        hd = sa.head_dim
        c = self._pc(L, i) if m.pfx else {}
        pv = dict(plen=m.plen, pvar=m.pvar, pmax=self.pmax.get(L, m.pfx)) if m.pfx else {}
        # XA (ATTN): query rows of the computed positions only, native attention (the reference's masked SDPA bits) with
        # the output gate fused (and with S1ENG_XA_FQ the q norm + RoPE); K/V once per KV head (check_xa)
        xa_on = L in self.xa and self.rec is None
        q = qx = None
        g = self._grp_at("att", L, m.R)
        if g is not None:
            # GRP: k|v + q_proj in one launch (same bytes as the PAR / plain path, check_grp); then the q norm/RoPE the
            # path needs (none with XA_FQ: the XA kernel folds it)
            qg, kr, vr = self._grp_att(d, x, g)
            qg = qg.view(m.R, -1, hd * 2)
            if not xa_on:
                q = fz.q_norm_rope_packed(qg, d["qn_w"], cos, sin, self.ff.eps, m.start, m.slen, B, L,
                                          cache=c.get("qg"), pfx=m.pfx, **pv)
            elif not self.xa_fq:
                qx = fz.q_norm_rope_rows(qg, d["qn_w"], cos, sin, self.ff.eps, m.row_pos, L)
        elif self._par("q"):
            # PAR: k|v on the side stream beside q_proj + the q norm/RoPE this path needs (with XA_FQ none: the XA kernel
            # folds it, so only q_proj runs beside k|v)
            if self._late("q", L, m.R):   # LATE: q_proj's node first, then the side's k|v (same fork)
                fk = self._fork()
                qg = self.lin(sa.q_proj, x, L).view(m.R, -1, hd * 2)
                (kr, vr), tok = self._on_side_from(fk, lambda: self.pair(d, "k_v", sa.k_proj, sa.v_proj, x, L))
            else:
                (kr, vr), tok = self._on_side(lambda: self.pair(d, "k_v", sa.k_proj, sa.v_proj, x, L))
                qg = self.lin(sa.q_proj, x, L).view(m.R, -1, hd * 2)
            if not xa_on:
                q = fz.q_norm_rope_packed(qg, d["qn_w"], cos, sin, self.ff.eps, m.start, m.slen, B, L,
                                          cache=c.get("qg"), pfx=m.pfx, **pv)
            elif not self.xa_fq:
                qx = fz.q_norm_rope_rows(qg, d["qn_w"], cos, sin, self.ff.eps, m.row_pos, L)
            self._join(tok)
        else:
            q2 = self.lin(sa.q_proj, x, L)
            qg = q2.view(m.R, -1, hd * 2)
            kr, vr = self.pair(d, "k_v", sa.k_proj, sa.v_proj, x, L)
            if self.rec is not None:
                self._record(i, qg=q2, k=kr, v=vr)
            if not xa_on:
                q = fz.q_norm_rope_packed(qg, d["qn_w"], cos, sin, self.ff.eps, m.start, m.slen, B, L,
                                          cache=c.get("qg"), pfx=m.pfx, **pv)
        if xa_on:
            from . import xa

            k4, v4 = fz.kv_norm_rope_rep_packed(kr, vr, d["kn_w"], cos, sin, self.ff.eps, 1, m.start, m.slen, B, L,
                                                ck=c.get("k"), cv=c.get("v"), pfx=m.pfx, **pv)
            if self.xa_fq:   # XA3: the bucket's gated variant (xa_ver, read at capture) or xa_fwd2q
                o = xa.attn_q(qg, d["qn_w"], cos, sin, self.ff.eps, k4, v4, m.start, m.slen, m.plen, L, sa.scaling,
                              gtab=self.xa_gtab, ver=self.xa_ver.get(L))
            else:
                if qx is None:
                    qx = fz.q_norm_rope_rows(qg, d["qn_w"], cos, sin, self.ff.eps, m.row_pos, L)
                o = xa.attn(qx, k4, v4, qg, m.start, m.slen, m.plen, L, sa.scaling, gtab=self.xa_gtab)
            return self.lin_res(sa.o_proj, o, L)
        if L in self.gqa and self.rec is None:   # E4 GQA: K/V once per KV head, SDPA over KV-head groups
            k4, v4 = fz.kv_norm_rope_rep_packed(kr, vr, d["kn_w"], cos, sin, self.ff.eps, 1, m.start, m.slen, B, L,
                                                ck=c.get("k"), cv=c.get("v"), pfx=m.pfx, **pv)
            o_g = self.sdpa_grouped(q, k4, v4, mask, sa.scaling, L)
            return self.lin_res(sa.o_proj, fz.attn_gate_grouped(o_g, qg, m.row_pos, L, k4.shape[1]), L)
        k, v = fz.kv_norm_rope_rep_packed(kr, vr, d["kn_w"], cos, sin, self.ff.eps, sa.num_key_value_groups,
                                          m.start, m.slen, B, L, ck=c.get("k"), cv=c.get("v"), pfx=m.pfx, **pv)
        o = self.sdpa(q, k, v, mask, sa.scaling, L)
        return self.lin_res(sa.o_proj, fz.attn_gate_packed(o, qg, m.row_pos, L), L)

    def sdpa_grouped(self, q, k4, v4, mask, scale, L, bias=None, bias_rep=None):
        """E4 GQA: self.sdpa (bias=None) or sdpa_last's bias call over KV-head groups: q [B, H, Lq, D] viewed as
        [B * G, H / G, Lq, D], k4 / v4 [B, G, L, D] as [B * G, H / G, L, D] with a stride-0 head (no repeated copy);
        -> [B * G, H / G, Lq, D] (the same values: check_gqa)."""
        B, H, Lq, D = q.shape
        G = k4.shape[1]
        qv = q.view(B * G, H // G, Lq, D)
        kk = k4.reshape(B * G, 1, L, D).expand(B * G, H // G, L, D)
        vv = v4.reshape(B * G, 1, L, D).expand(B * G, H // G, L, D)
        if bias is None and bias_rep is None:
            return self.sdpa(qv, kk, vv, mask, scale, L)
        from torch.nn.attention import SDPBackend, sdpa_kernel

        # FCL LT: bias_rep = the bias already repeated per KV group (fcl.lt_gather), the same bytes and layout
        am = bias_rep if bias_rep is not None else bias.repeat_interleave(G, dim=0)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            return F.scaled_dot_product_attention(qv, kk, vv, attn_mask=am, dropout_p=0.0, scale=scale,
                                                  is_causal=False)

    @torch.inference_mode()
    def check_gqa(self, L: int, d, cos, sin, bmax: int, seed: int = 11) -> str:
        """E4 GQA start-up gate of bucket L: kv_norm_rope_rep_packed with rep 1 == every REP-th head of the repeated
        output; sdpa_grouped == self.sdpa on the repeated K/V (full queries, the pruned layer's one-row pattern, and
        with LR1 on the one-row bias call), bit for bit, at B up to bmax with zeros past n. "" if every byte agrees."""
        sa = d["sa"]
        fz = self.fz
        mask = self.ff.consts[L][0]
        hd = sa.head_dim
        rep = sa.num_key_value_groups
        heads = sa.q_proj.weight.shape[0] // (2 * hd)
        hk = heads // rep
        g = torch.Generator(device="cuda")
        g.manual_seed(seed + L)
        lprev = max([x for x in (0, 128, 192, 256, 320) if x < L])
        ar = torch.arange(L, device="cuda")
        for B in sorted({1, 2, 3, 4, 5, 8, bmax}):
            if B > bmax:
                continue
            n = torch.randint(lprev + 1, L + 1, (B,), generator=g, device="cuda")
            slen = n.to(torch.int32)
            start = torch.cat([torch.zeros(1, dtype=torch.long, device="cuda"), n.cumsum(0)[:-1]]).to(torch.int32)
            R = int(n.sum())
            kr = (torch.randn((R, hk * hd), generator=g, device="cuda") * 2).to(torch.bfloat16)
            vr = (torch.randn((R, hk * hd), generator=g, device="cuda") * 2).to(torch.bfloat16)
            k_rep, v_rep = fz.kv_norm_rope_rep_packed(kr, vr, d["kn_w"], cos, sin, self.ff.eps, rep, start, slen, B, L)
            k4, v4 = fz.kv_norm_rope_rep_packed(kr, vr, d["kn_w"], cos, sin, self.ff.eps, 1, start, slen, B, L)
            if not (torch.equal(k_rep[:, ::rep].view(torch.int16), k4.view(torch.int16))
                    and torch.equal(v_rep[:, ::rep].view(torch.int16), v4.view(torch.int16))):
                return f"rep-1 K/V differ from the repeated heads (B={B})"
            keep = (ar[None, :] < n[:, None]).to(torch.bfloat16)[:, None, :, None]
            for trial in range(2):
                q = (torch.randn((B, heads, L, hd), generator=g, device="cuda") * 1.5).to(torch.bfloat16) * keep
                if trial == 1:   # the pruned layer: one query row per decision
                    q = q * (ar[None, :] == (n - 1)[:, None]).to(torch.bfloat16)[:, None, :, None]
                want = self.sdpa(q, k_rep, v_rep, mask, sa.scaling, L).reshape(B, hk, rep, L, hd)
                got = self.sdpa_grouped(q, k4, v4, mask, sa.scaling, L).reshape(B, hk, rep, L, hd)
                if not torch.equal(want.contiguous().view(torch.int16), got.contiguous().view(torch.int16)):
                    return f"grouped attention differs (B={B}, trial {trial})"
            if L in self.lr1:
                q1 = (torch.randn((B, heads, 1, hd), generator=g, device="cuda") * 1.5).to(torch.bfloat16)
                want = self.sdpa_last(q1, k_rep, v_rep, mask, sa.scaling, slen, B, L).reshape(B, hk, rep, 1, hd)
                bias = mask[0, 0].index_select(0, (slen.long() - 1).clamp_(min=0)).view(B, 1, 1, L)
                got = self.sdpa_grouped(q1, k4, v4, mask, sa.scaling, L, bias=bias).reshape(B, hk, rep, 1, hd)
                if not torch.equal(want.contiguous().view(torch.int16), got.contiguous().view(torch.int16)):
                    return f"grouped one-row attention differs (B={B})"
        return ""

    @torch.inference_mode()
    def check_xa(self, L: int, layers, cos, sin, bmax: int, pfx: int = 0, pmax: int = 0, seed: int = 17) -> str:
        """XA (ATTN) start-up gate of bucket L: xa.attn (after q_norm_rope_rows and the rep-1 K/V) against the
        reference chain - q_norm_rope_packed, repeated K/V, the masked SDPA call of the reference (the bucket's [L, L]
        mask, PyTorch's dispatch) and attn_gate_packed - byte for byte on random packed decisions: these layers'
        norm weights, B up to bmax with an unused slot, pfx > 0 with cached prefix rows of 3 variants (lengths
        pfx..pmax); plus raw q / K / V at several scales straight into the kernel and every bf16 gate value (65536).
        Rows past the used ones must come back zero. "" if every byte agrees."""
        from . import xa

        fz = self.fz
        mask = self.ff.consts[L][0]
        sa = layers[0]["sa"]
        hd = sa.head_dim
        rep = sa.num_key_value_groups
        heads = sa.q_proj.weight.shape[0] // (2 * hd)
        hk = heads // rep
        if not xa.supported(heads, hk, hd, L):
            return "kernel unavailable or unsupported shape"
        dev = "cuda"
        g = torch.Generator(device=dev)
        g.manual_seed(seed + L + 1000 * pfx)
        rng = np.random.default_rng(seed + L + 1000 * pfx)
        lprev = max([x for x in (0, 128, 192, 256, 320) if x < L])
        nvar = 3
        pmax = max(pmax, pfx)
        eps = self.ff.eps

        def layout(B, lo, hi, unused):
            n = rng.integers(lo, hi + 1, B)
            pl = np.minimum(rng.integers(pfx, pmax + 1, B), n - 1) if pfx else np.zeros(B, dtype=np.int64)
            if unused and B > 1:
                n[-1] = 0
                pl[-1] = pfx
            rows = np.where(n > 0, n - pl, 0)
            st = np.concatenate([[0], np.cumsum(rows)[:-1]]).astype(np.int32)
            st = np.where(n > 0, st, 0).astype(np.int32)
            Rr = int(rows.sum())
            R = Rr + int(rng.integers(0, 40))
            rp = np.zeros(R, dtype=np.int32)
            for bb in range(B):
                if n[bb] > 0:
                    rp[st[bb]:st[bb] + rows[bb]] = bb * L + np.arange(pl[bb], n[bb], dtype=np.int32)
            t = lambda a: torch.from_numpy(np.ascontiguousarray(a).astype(np.int32)).to(dev)
            pv = rng.integers(0, nvar, B)
            return n, pl, Rr, R, t(st), t(n), t(pl), t(pv), t(rp)

        def same(ref, got, Rr, what):
            if not torch.equal(ref[:Rr].contiguous().view(torch.int16), got[:Rr].contiguous().view(torch.int16)):
                nd = int((ref[:Rr].view(torch.int16) != got[:Rr].view(torch.int16)).sum())
                return f"{what}: {nd} values differ"
            if got.shape[0] > Rr and bool((got[Rr:].view(torch.int16) != 0).any()):
                return f"{what}: rows past the used ones not zero"
            return ""

        def run(qg, k4, v4, start, slen, plen, row_pos, d, qx=None):
            if self.xa_fq:
                o = xa.attn_q(qg, d["qn_w"], cos, sin, eps, k4, v4, start, slen, plen, L, sa.scaling, gtab=self.xa_gtab)
                # XA3 (kai): the variant being gated (xa_want) on the same inputs, byte for byte against xa_fwd2q's
                # output (all rows); a miss turns only the variant off for this bucket
                xa.variant_check(self, "xa", L, o, lambda v, vo: xa.attn_q(qg, d["qn_w"], cos, sin, eps, k4, v4, start,
                                                                        slen, plen, L, sa.scaling, gtab=self.xa_gtab,
                                                                        ver=v, out=vo))
                return o
            if qx is None:
                qx = fz.q_norm_rope_rows(qg, d["qn_w"], cos, sin, eps, row_pos, L)
            return xa.attn(qx, k4, v4, qg, start, slen, plen, L, sa.scaling, gtab=self.xa_gtab)

        bs = sorted({b for b in (1, 2, 3, 5, 8, bmax) if b <= bmax})
        for li, d in enumerate(layers):
            # the first two layers on every B; the others (their q / k norm weights) on B = bmax
            for B in (bs if li < 2 else [bmax]):
                for trial in range(2):
                    lo = max(lprev + 1, pfx + 1)
                    n, pl, Rr, R, start, slen, plen, pvar, row_pos = layout(B, lo, L, trial == 1)
                    sc = (1.0, 6.0, 0.05)[(li + B + trial) % 3]
                    qg = (torch.randn((R, heads, 2 * hd), generator=g, device=dev) * sc).to(torch.bfloat16)
                    kr = (torch.randn((R, hk * hd), generator=g, device=dev) * sc).to(torch.bfloat16)
                    vr = (torch.randn((R, hk * hd), generator=g, device=dev) * 2).to(torch.bfloat16)
                    kw, ckw = {}, {}
                    if pfx:
                        cq = (torch.randn((nvar * pmax, heads * 2 * hd), generator=g, device=dev) * sc).to(torch.bfloat16)
                        ck = (torch.randn((nvar * pmax, hk * hd), generator=g, device=dev) * sc).to(torch.bfloat16)
                        cv = (torch.randn((nvar * pmax, hk * hd), generator=g, device=dev) * 2).to(torch.bfloat16)
                        kw = dict(pfx=pfx, plen=plen, pvar=pvar, pmax=pmax)
                        ckw = dict(ck=ck, cv=cv, **kw)
                        q_pad = fz.q_norm_rope_packed(qg, d["qn_w"], cos, sin, eps, start, slen, B, L, cache=cq, **kw)
                    else:
                        q_pad = fz.q_norm_rope_packed(qg, d["qn_w"], cos, sin, eps, start, slen, B, L)
                    k_rep, v_rep = fz.kv_norm_rope_rep_packed(kr, vr, d["kn_w"], cos, sin, eps, rep, start, slen, B, L,
                                                              **ckw)
                    o = F.scaled_dot_product_attention(q_pad, k_rep, v_rep, attn_mask=mask, dropout_p=0.0,
                                                       scale=sa.scaling, is_causal=False)
                    ref = fz.attn_gate_packed(o, qg, row_pos, L)
                    k4, v4 = fz.kv_norm_rope_rep_packed(kr, vr, d["kn_w"], cos, sin, eps, 1, start, slen, B, L, **ckw)
                    got = run(qg, k4, v4, start, slen, plen, row_pos, d)
                    why = same(ref, got, Rr, f"layer #{li}, B={B}, trial {trial}, pfx={pfx}")
                    if why:
                        return why
        # raw q / K / V straight into the kernel (logit scales from tiny to saturated), against the masked call
        # (FQ: raw q_proj rows, q = their norm + RoPE with layer 0's weights)
        d0 = layers[0]
        for B in bs:
            for sc in (0.3, 1.0, 3.0, 8.0):
                n, pl, Rr, R, start, slen, plen, pvar, row_pos = layout(B, max(lprev + 1, pfx + 1), L, B > 2)
                qg = (torch.randn((R, heads, 2 * hd), generator=g, device=dev) * 3).to(torch.bfloat16)
                if self.xa_fq:
                    qg[:, :, :hd] = (torch.randn((R, heads, hd), generator=g, device=dev) * sc).to(torch.bfloat16)
                    qx = fz.q_norm_rope_rows(qg, d0["qn_w"], cos, sin, eps, row_pos, L)
                else:
                    qx = (torch.randn((R, heads, hd), generator=g, device=dev) * sc).to(torch.bfloat16)
                keep = (torch.arange(L, device=dev)[None, :] < slen[:, None].long()).to(torch.bfloat16)[:, None, :, None]
                k4 = (torch.randn((B, hk, L, hd), generator=g, device=dev) * sc).to(torch.bfloat16) * keep
                v4 = (torch.randn((B, hk, L, hd), generator=g, device=dev) * 2).to(torch.bfloat16) * keep
                q_pad = torch.zeros((B, heads, L, hd), dtype=torch.bfloat16, device=dev)
                rr = torch.arange(Rr, device=dev)
                rpl = row_pos[:Rr].long()
                q_pad[rpl // L, :, rpl % L] = qx[rr]
                o = F.scaled_dot_product_attention(q_pad, k4.repeat_interleave(rep, 1), v4.repeat_interleave(rep, 1),
                                                   attn_mask=mask, dropout_p=0.0, scale=sa.scaling, is_causal=False)
                ref = fz.attn_gate_packed(o, qg, row_pos, L)
                got = run(qg, k4, v4, start, slen, plen if pfx else None, row_pos, d0, qx=qx)
                why = same(ref, got, Rr, f"raw inputs B={B}, scale {sc}")
                if why:
                    return why
        # every bf16 gate value: 16 one-token decisions (o = V[0] = 1 exactly), gate = all 65536 bit patterns
        B = 16
        start = torch.arange(B, dtype=torch.int32, device=dev)
        slen = torch.ones(B, dtype=torch.int32, device=dev)
        row_pos = (torch.arange(B, dtype=torch.int32, device=dev) * L).contiguous()
        k4 = torch.zeros((B, hk, L, hd), dtype=torch.bfloat16, device=dev)
        k4[:, :, 0] = (torch.randn((B, hk, hd), generator=g, device=dev)).to(torch.bfloat16)
        v4 = torch.zeros_like(k4)
        v4[:, :, 0] = 1.0
        qg = torch.zeros((B, heads, 2, hd), dtype=torch.int16, device=dev)
        qg[:, :, 1] = torch.arange(65536, dtype=torch.int32, device=dev).to(torch.int16).view(B, heads, hd)
        qg = qg.view(torch.bfloat16).view(B, heads, 2 * hd)
        if self.xa_fq:
            qg[:, :, :hd] = (torch.randn((B, heads, hd), generator=g, device=dev)).to(torch.bfloat16)
            qx = fz.q_norm_rope_rows(qg, d0["qn_w"], cos, sin, eps, row_pos, L)
        else:
            qx = (torch.randn((B, heads, hd), generator=g, device=dev)).to(torch.bfloat16)
        q_pad = torch.zeros((B, heads, L, hd), dtype=torch.bfloat16, device=dev)
        q_pad[:, :, 0] = qx
        o = F.scaled_dot_product_attention(q_pad, k4.repeat_interleave(rep, 1), v4.repeat_interleave(rep, 1),
                                           attn_mask=mask, dropout_p=0.0, scale=sa.scaling, is_causal=False)
        ref = fz.attn_gate_packed(o, qg, row_pos, L)
        got = run(qg.contiguous(), k4, v4, start, slen, None, row_pos, d0, qx=qx)
        torch.cuda.synchronize()
        nan = torch.isnan(ref)
        if not bool(torch.isnan(got)[nan].all()):
            return "gate values: NaN pattern differs"
        if not torch.equal(ref.view(torch.int16)[~nan], got.view(torch.int16)[~nan]):
            return f"gate values: {int((ref.view(torch.int16) != got.view(torch.int16))[~nan].sum())} differ"
        if self.xa_want and self.xa_fq and ("xa", L) not in self.xa_vbad:
            # XA3 (kai): the variant against xa_fwd2q on outlier and order-witness inputs, every B (after every check
            # above, so their random draws are unchanged)
            for B in bs:
                for kind in ("outlier", "witness"):
                    n, pl, Rr, R, start, slen, plen, pvar, row_pos = layout(B, max(lprev + 1, pfx + 1), L, B > 2)
                    qg, k4, v4 = xa.witness_inputs(kind, g, R, B, L, heads, hk, hd, slen, device=dev)
                    pw = plen if pfx else None
                    base = xa.attn_q(qg, d0["qn_w"], cos, sin, eps, k4, v4, start, slen, pw, L, sa.scaling,
                                     gtab=self.xa_gtab)
                    xa.variant_check(self, "xa", L, base, lambda v, vo: xa.attn_q(qg, d0["qn_w"], cos, sin, eps, k4,
                                                                               v4, start, slen, pw, L, sa.scaling,
                                                                               gtab=self.xa_gtab, ver=v, out=vo))
        return ""

    def _attn_last(self, d, x, xl, m: PackedMeta, mask, cos, sin, pos_last, i: int, lt_bias=None, plo=None):
        """Last decoder layer, attention type: k/v for every row, q / attention gate / o_proj for the B last rows only
        (the other query positions are zero; each query row of SDPA depends only on itself, all keys and values)."""
        sa = d["sa"]
        fz, L, B = self.fz, m.L, m.B
        hd = sa.head_dim
        lr1 = L in self.lr1 and self.rec is None
        # FCL XL: XA's computed-rows kernel with one row per slot (q norm + RoPE, attention and gate in one launch)
        xlon = lr1 and L in self.xl and L in self.gqa
        tok = q1 = None
        late_last = self._late("last", L, m.R)
        if self._par("last"):   # PAR: q_proj (M = B) + the one-row q norm/RoPE on the side stream beside k|v
            def side():
                g = self.lin(sa.q_proj, xl, L).view(B, -1, hd * 2)
                return g, (fz.q_norm_rope_last(g, d["qn_w"], cos, sin, self.ff.eps, m.slen, B) if lr1 and not xlon
                           else None)

            if late_last:   # LATE: k|v's node first, then the side's q_proj (same fork)
                fk = self._fork()
                kr, vr = self.pair(d, "k_v", sa.k_proj, sa.v_proj, x, L)
                (qg, q1), tok = self._on_side_from(fk, side)
            else:
                (qg, q1), tok = self._on_side(side)
        if not (tok is not None and late_last):
            kr, vr = self.pair(d, "k_v", sa.k_proj, sa.v_proj, x, L)
        if self.rec is not None:
            self._record(i, k=kr, v=vr)
        c = self._pc(L, i) if m.pfx else {}
        pv = dict(plen=m.plen, pvar=m.pvar, pmax=self.pmax.get(L, m.pfx)) if m.pfx else {}
        if tok is None:
            qg = self.lin(sa.q_proj, xl, L).view(B, -1, hd * 2)
        gqa = L in self.gqa and self.rec is None
        k, v = fz.kv_norm_rope_rep_packed(kr, vr, d["kn_w"], cos, sin, self.ff.eps, 1 if gqa else
                                          sa.num_key_value_groups, m.start, m.slen, B, L, ck=c.get("k"),
                                          cv=c.get("v"), pfx=m.pfx, **pv)
        if tok is not None:
            self._join(tok)
        if xlon:
            from . import xa

            if plo is None:
                plo = (m.slen - 1).clamp(min=0)
            o = xa.attn_q(qg, d["qn_w"], cos, sin, self.ff.eps, k, v, self._ar[B], m.slen, plo, L, sa.scaling,
                          gtab=self.xa_gtab, ver=self.xl_ver.get(L))   # XA3: the XL bucket's gated variant or 2q
            return self.lin(sa.o_proj, o, L)
        if lr1:
            # E4 LR1: only each slot's query row n - 1 attends (the mask's row n - 1 as an additive bias, the efficient
            # kernel as the causal call uses): one row per slot instead of B x L rows (check_lr1 at start-up)
            if q1 is None:
                q1 = fz.q_norm_rope_last(qg, d["qn_w"], cos, sin, self.ff.eps, m.slen, B)
            if gqa:   # E4 GQA: K/V once per KV head (check_gqa)
                if lt_bias is not None:   # FCL LT: the repeated bias rows came with the gathers (one launch)
                    o_g = self.sdpa_grouped(q1, k, v, mask, sa.scaling, L, bias_rep=lt_bias)
                else:
                    bias = mask[0, 0].index_select(0, (m.slen.long() - 1).clamp_(min=0)).view(B, 1, 1, L)
                    o_g = self.sdpa_grouped(q1, k, v, mask, sa.scaling, L, bias=bias)
                return self.lin(sa.o_proj, fz.attn_gate_grouped(o_g, qg, self._ar[B], 1, k.shape[1]), L)
            o1 = self.sdpa_last(q1, k, v, mask, sa.scaling, m.slen, B, L)
            return self.lin(sa.o_proj, fz.attn_gate_packed(o1, qg, self._ar[B], 1), L)
        q = fz.q_norm_rope_packed(qg, d["qn_w"], cos, sin, self.ff.eps, m.start, m.slen, B, L, only_last=True)
        if gqa:
            o_g = self.sdpa_grouped(q, k, v, mask, sa.scaling, L)
            return self.lin(sa.o_proj, fz.attn_gate_grouped(o_g, qg, pos_last, L, k.shape[1]), L)
        o = self.sdpa(q, k, v, mask, sa.scaling, L)
        return self.lin(sa.o_proj, fz.attn_gate_packed(o, qg, pos_last, L), L)

    def sdpa_last(self, q1, k, v, mask, scale, slen, B: int, L: int):
        """E4 LR1: SDPA of one query row per slot ([B, H, 1, D]) with the additive bias row slen - 1 of the bucket's
        mask, on the efficient kernel -> [B, H, 1, D]."""
        from torch.nn.attention import SDPBackend, sdpa_kernel

        bias = mask[0, 0].index_select(0, (slen.long() - 1).clamp_(min=0)).view(B, 1, 1, L)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            return F.scaled_dot_product_attention(q1, k, v, attn_mask=bias, dropout_p=0.0, scale=scale, is_causal=False)

    @torch.inference_mode()
    def check_lr1(self, L: int, d, cos, sin, bmax: int, seed: int = 7) -> str:
        """E4 LR1 start-up gate of bucket L: (1) q_norm_rope_last == row n - 1 of q_norm_rope_packed(only_last=True)
        bit for bit; (2) sdpa_last's one row == row n - 1 of the call the pruned layer makes today (self.sdpa: the
        causal efficient call or the masked call) and of the masked call, bit for bit; random decisions (zeros past n)
        at several B up to bmax, unused slots included. "" if every byte agrees."""
        sa = d["sa"]
        fz = self.fz
        mask = self.ff.consts[L][0]
        hd = sa.head_dim
        heads = sa.q_proj.weight.shape[0] // (2 * hd)
        g = torch.Generator(device="cuda")
        g.manual_seed(seed + L)
        lprev = max([x for x in (0, 128, 192, 256, 320) if x < L])
        for B in sorted({1, 2, 3, 4, 5, 8, bmax}):
            if B > bmax:
                continue
            self._ar.setdefault(B, torch.arange(B, dtype=torch.int32, device="cuda"))
            for trial in range(3):
                n = torch.randint(lprev + 1, L + 1, (B,), generator=g, device="cuda")
                if trial == 2 and B > 1:
                    n[-1] = 0                                     # an unused slot
                slen = n.to(torch.int32)
                qg = (torch.randn((B, heads, 2 * hd), generator=g, device="cuda") * 2).to(torch.bfloat16)
                qa = fz.q_norm_rope_packed(qg, d["qn_w"], cos, sin, self.ff.eps, slen, slen, B, L, only_last=True)
                q1 = fz.q_norm_rope_last(qg, d["qn_w"], cos, sin, self.ff.eps, slen, B)
                pos = (n - 1).clamp(min=0)
                ref_q = qa[torch.arange(B, device="cuda"), :, pos]          # [B, H, D]
                if not torch.equal(ref_q.view(torch.int16), q1[:, :, 0].view(torch.int16)):
                    return f"q row differs (B={B}, trial {trial})"
                shp = (B, heads, L, hd)
                keep = (torch.arange(L, device="cuda")[None, :] < n[:, None]).to(torch.bfloat16)[:, None, :, None]
                k, v = ((torch.randn(shp, generator=g, device="cuda") * 1.5).to(torch.bfloat16) * keep for _ in range(2))
                o_today = self.sdpa(qa, k, v, mask, sa.scaling, L)
                o_mask = F.scaled_dot_product_attention(qa, k, v, attn_mask=mask, dropout_p=0.0, scale=sa.scaling,
                                                        is_causal=False)
                o1 = self.sdpa_last(q1, k, v, mask, sa.scaling, slen, B, L)
                real = n > 0
                for name, o in (("today's call", o_today), ("the masked call", o_mask)):
                    want = o[torch.arange(B, device="cuda"), :, pos][real]
                    if not torch.equal(want.view(torch.int16), o1[:, :, 0][real].view(torch.int16)):
                        return f"attention row differs from {name} (B={B}, trial {trial})"
        return ""

    def _mlp(self, d, x, L, defer: bool = False):
        mlp = d["mlp"]
        down = self.lin_res if defer else self.lin   # RARN: the main loop's down_proj may fold into the next add_norm
        if x.shape[0] <= 16 and L in self.tgs and self.rec is None:   # FCL TGS: gate + up + SwiGLU in one launch
            from . import fcl

            return down(mlp.down_proj, fcl.tgs_mlp(x, mlp.gate_proj.weight, mlp.up_proj.weight), L)
        sw = getattr(self.pinned, "swiglu_r", {}).get((L, x.shape[0]))
        blk = d["fused"].get("gate_up")
        if sw is not None and blk is not None:   # SG (rgh): gate|up GEMM + SwiGLU in one kernel (start-up verified)
            from . import ng

            h = torch.empty((x.shape[0], blk.shape[0] // 2), dtype=x.dtype, device=x.device)
            return down(mlp.down_proj, ng.run_swiglu(sw, blk, x, h), L)
        if self._par("gu") and not self._l0t_fused(d, "gate_up", x.shape[0], L):   # PAR: up beside gate
            if self._late("gu", L, x.shape[0]):   # LATE: gate's node first, then the side's up (same fork)
                fk = self._fork()
                gt = self.lin(mlp.gate_proj, x, L)
                up, tok = self._on_side_from(fk, lambda: self.lin(mlp.up_proj, x, L))
            else:
                up, tok = self._on_side(lambda: self.lin(mlp.up_proj, x, L))
                gt = self.lin(mlp.gate_proj, x, L)
            self._join(tok)
        else:
            gt, up = self.pair(d, "gate_up", mlp.gate_proj, mlp.up_proj, x, L)
        return down(mlp.down_proj, self.fz.silu_mul(gt, up), L)

    def __call__(self, m: PackedMeta) -> torch.Tensor:
        if getattr(m, "S", 0):   # MX: a mixed graph of the pair (m.L, m.S)
            self._mx, self._mx_last = (self.mxp[(m.L, m.S)], m), False
            try:
                return self._fwd(m)
            finally:
                self._mx, self._mx_last = None, False
        return self._fwd(m)

    def _fwd(self, m: PackedMeta) -> torch.Tensor:
        ff, fz = self.ff, self.fz
        L = m.L
        mask, cos, sin = ff.consts[L]                     # per-length constants (computed at set-0 capture)
        # L0T (kai): layer 0's norm + input projections come from the token table (never while recording a prefix)
        l0t = self.l0t is not None and self.rec is None and L in self.l0t.books
        self._emb_rows = None
        from . import rn as _rnm   # RN (rn.py): this forward's row norms / gated norm in rn.cubin (the env's kernel
        #                           kind / hpw, each gated at start-up; FCAB arms may capture with another one)
        # (MX: never in a mixed graph - RN's start-up gate covers the per-bucket packed R values only)
        self._rn = L in self.rn and self.rec is None and self._mx is None and _rnm.usable("norm")
        self._rng = L in self.rng and self.rec is None and self._mx is None and _rnm.usable("gnorm")
        if l0t and L in self.emb:   # FCL EMB: embedding + layer 0's table rows in one launch (emb_check)
            from . import fcl

            d0, bk = ff.layers[0], self.l0t.books[L]
            qt, bt, nq, nb = self.l0t_stack if self._mx is not None else (bk.qkvz, bk.ba, bk.nq, bk.nb)   # MX: stacked
            h, self._emb_rows = fcl.emb_gather(ff.tm.embed_tokens.weight, m.ids, m.l0t_slot, qt, nq, bt, nb,
                                               self._l0t_fused(d0, "qkv_z", m.R, L),
                                               self._l0t_fused(d0, "b_a", m.R, L))
        else:
            h = ff.tm.embed_tokens(m.ids)                   # [R, 2560]
        pending = None
        nl = len(ff.layers)
        prune = self.prune and ff.layers[-1]["type"] != "linear_attention"
        for i, d in enumerate(ff.layers):
            if pending is None:
                x = None if l0t else self.norm(h, d["ln1_w"], ff.eps)
            else:
                h, x = self.add_norm(h, pending, d["ln1_w"], ff.eps)
            if prune and i == nl - 1:
                # last layer: only the B last rows continue (row-wise ops on gathered rows; GEMMs at M = B with the
                # bucket's pinned algo, verified row-invariant at start-up)
                self._mx_last = self._mx is not None   # MX: the M = B GEMMs' rows are slots (boundary b1)
                lt = L in self.lt and self.rec is None
                if lt:   # FCL LT: xl, hl, pos_last and the LR1 + GQA bias rows in one launch (lt_check)
                    from . import fcl

                    sa = d["sa"]
                    xlon = L in self.xl and L in self.lr1 and L in self.gqa   # XL needs no bias rows
                    G = sa.k_proj.weight.shape[0] // sa.head_dim if (L in self.lr1 and L in self.gqa and
                                                                      not xlon) else 0
                    xl, hl0, pos_last, ltb, plo = fcl.lt_gather(x, h, m.row_pos, m.last, m.slen, mask, m.B, L, G,
                                                                with_plo=True)
                else:
                    xl = x.index_select(0, m.last).contiguous()
                    pos_last = m.row_pos.index_select(0, m.last).contiguous()
                    ltb = plo = None
                y = self._attn_last(d, x, xl, m, mask, cos, sin, pos_last, i, lt_bias=ltb, plo=plo)
                hl = hl0 if lt else h.index_select(0, m.last).contiguous()
                hl, xl = self.add_norm(hl, y, d["ln2_w"], ff.eps)
                pl = self._mlp(d, xl, L)
                break
            if d["type"] == "linear_attention":
                y = self._gdn(d, x, m, i)
            else:
                y = self._attn(d, x, m, mask, cos, sin, i)
            h, x = self.add_norm(h, y, d["ln2_w"], ff.eps)
            pending = self._mlp(d, x, L, defer=i < nl - 1)
        else:
            hl = h.index_select(0, m.last).contiguous()
            pl = self.materialize(pending).index_select(0, m.last).contiguous()
        _, hn = self.add_norm(hl, pl, ff.fn_w, ff.eps)
        sw_t = ff.slot_weight.T
        ro = self.ro.get(L) if self.rec is None else None
        if ro is not None:   # FCL RO: the B readout products + cat as one call (ro_check)
            from . import fcl

            if self.ro_rows is not None:   # ROP (pjs e10): the row count ro_plan verified for this B (0: per-row)
                return fcl.readout_rows(hn, sw_t, self.ro_rows.get(m.B, 0))
            return fcl.readout(hn, sw_t, ro)
        return torch.cat([hn[b:b + 1] @ sw_t for b in range(m.B)])
