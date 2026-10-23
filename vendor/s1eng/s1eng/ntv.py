"""NTV (kai, S1ENG_NTV=1, default off): nonce-token prefix variants, so the first tokens of the validator's record id
leave the packed rows too.

The validator puts a fresh record id first in every timed evidence value (engrunner._perturb_state: secrets.token_hex(4)
as {"ref": nonce, **state}, {"ref": nonce, "entries": list} or "[ref {nonce}] {state}"). v15's PV (a4) caches the
template prefix (40 tokens) and the two family heads that end before the id: ' {"ref": "' (44) and ' "[ref' (42). The
tokenizer splits hex into single digits and short letter runs (and, in the string family, a lone ' ' before a digit),
so the next one or two tokens come from a small set: ' {"ref": "3' or ' "[ref 3' cover a sixteenth of their family
each. NTV adds the K (S1ENG_NTV_K, default 32) trie nodes past the family heads with the most expected saved tokens
as further cached variants: on held-out validator prompts (s1items + token_hex(4), plumb-4b's tokenizer; 3 x 20000,
spike/ntv_gain.py) 1.07 tokens per decision (1.068-1.072) leave the packed rows at K = 32, 0.76 % of the ~140 rows v15
computes per decision, within 0.01 of the best 32 nodes picked on the held-out prompts themselves.

Derivation (variants): only the template and a fixed enumeration of hex ids (nonces(): sha256 of "ntv<i>", the first 8
hex digits; no request data), encoded through the engine's own prompt builder at start-up: SAMPLES ids per family,
each family's evidence as v15's prefix_variants writes it, weighted 2 : 1 (dict and list states : strings; the
validator's s1items has 4 dict / list state makers and 2 string ones). Every sample's base is the longest variant it
starts with (and outgrows); its nodes are its first base + 1 .. base + DEPTH tokens (each leaves at least one token).
select() takes the K nodes greedily by marginal saved tokens (a deeper node only adds what its selected ancestor does
not already save); ties go to the shorter node, then the smaller ids. Deterministic: the same tokenizer gives the same
variants on every start-up. Like every variant, a node costs speed if it is wrong, never bits: a request uses a
variant only if its own ids start with it.

Exactness: a variant's rows are recorded by PackedForward.record_prefix (the same eager forward as v15's three) into
the same block layout (pmax rows per variant); check_prefix samples variants at random, so cover() runs every variant
block at least once per bucket against the per-decision graph, bit for bit, after check_prefix passed. A failure of
either (or of record_prefix) with the nonce-token variants turns NTV off for every bucket first (Run.fallback: v15's
own variants recorded again for the buckets done so far and checked as v15's loop does, PRO retry included); only a
failure of v15's own variants then turns a bucket's prefix off, v15's path. A variant reaches the kernels only as its
slot's (pvar, plen), which v15 already varies (40 / 42 / 44): conv, gate and the PRO prologue read cached rows pvar *
pmax + pos for pos < plen; so does v16's PRON (pron.cubin, the PRO prologue's bits natively: pfx, pmax, B as runtime
int arguments, pvar / plen per slot from device memory, row pvar * pmax in 64 bits; no shared memory, a grid of
conv + gate blocks independent of pmax and of the variant count) on the GDN cache rows (qkv, b, a), never on the
attention q|gate rows NTV drops; XA, GQA and the causal SDPA take the padded keys 0 .. n-1 (cached k / v below plen),
queries for computed rows only; LR1 one query row per slot; L0T maps packed rows only. pmax (a Triton constexpr)
becomes 45.

On hotspot (ruj's fc8 FCL levers, fcl.py = fc9's; no PKV): every FCL path is off while a prefix is recorded
(record_prefix sets pf.rec: BA, TG, TGS, LT and RO check it, XL via LR1, EMB via L0T), so every recorded variant row
comes from v16's path, the bytes the FCL gates prove their kernels equal to. No FCL gate reads pcache or a prefix
layout (each compares its kernel with the path it replaces on synthetic rows), so trim, recheck and NTV off leave
them as verified; BA / LT / XL / TG / TGS run inside check_prefix and cover (eager forwards, pf.rec None) with every
variant's (pvar, plen); XL takes the last layer's padded k / v (cached rows below plen) and one computed row per slot;
EMB / RO (set up after the recording) read no prefix state. hotspot's LL (S1ENG_LEAN) only changes how BatchLoop enqueues a
packed meta-blob launch (driver-API H2D copies, graph launch, D2H copy, event record); the blobs it copies are
stage_blob's, pvar / plen included (each slot's pick_variant answer, the Index's with NTV), the same bytes as the torch
path's copies; it has no start-up gate and never runs while capture_batch records or checks a prefix.

hotspot's ONORM (rgh v17, onorm.py: the GDN gated norm as the FFS kernel's epilogue, on by default) runs only where
PRO runs (batchfwd.py _gdn: its dispatch needs ``pro``, which needs pf.rec None), so never while a prefix is recorded;
record_prefix's forward is B = 1 anyway (below onorm.bmin) and records the qkv / b / a rows upstream of FFS, so no
recorded row passes through it. It reads no pcache, pvar or pmax: its inputs are FFS's (the prologue's q, k, v, g,
beta, A and slen) plus z, the norm weight and the per-slot start / plen of the packed layout, which say where a slot's
computed rows go (positions plen .. slen - 1 to rows start ..). NTV changes only which plen a slot has (its variant's
length, 43-45 tokens for the nonce-token blocks beside hotspot's 40 / 42 / 44). onorm.check proves the kernel on
synthetic layouts with plen 40 / 42 / 44, 0 / 1 / 63 / 64 / 65 and 0, before prepare (model.py: right after PRON,
scratch freed); with NTV the variant plens reach it end to end in check_prefix (every packed B, random variants) and
cover() (the largest packed B, every variant block once), both against the per-decision graph, so every nonce-token
block runs through the fused norm bit-checked before serving. onorm.ON is fixed at start-up and its dispatch also
needs L in pf.pro: where recheck() or the fallback drops PRO for a bucket, ONORM is off with it (FFS + the Triton
norm, hotspot's own path for a non-PRO bucket); where the fallback restores PRO, ONORM comes back exactly as
onorm.check verified it (prefix-independent). ruj's RN (rn.py) is off in hotspot (S1ENG_RN=0); its dispatch also
needs pf.rec None.

Runtime: pick_variant takes an Index (the variants as a list, plus a token trie past the template): one list compare
of the template and at most DEPTH + the longest head's extra tokens dict steps instead of a scan over every variant.
Index.pick returns exactly the scan's answer (the longest variant ids starts with and outgrows; tests/test_ntv.py
against the brute-force scan on generated prompts).

Memory: each variant costs pmax rows per bucket of every layer's cached GEMM outputs (row_bytes: GDN qkv|b|a, attention
q|gate|k|v; the pruned last layer k|v; ~531 KiB per row on plumb-4b, ~419 KiB with XA, which never reads the prefix's
q|gate rows, so NTV drops them for its XA buckets). At K = 32 on plumb-4b: 35 variants x 45 rows, 0.63 GiB per XA
bucket (0.79 GiB without XA).

ntvl (kai/rv14-ntvl, d75: with serving models resident the old guard kept 1 of 32 at 20 GiB left and 0 at 16):
- Buckets: only S1ENG_NTV_BUCKETS (default "192,256", ~95 % of the decisions; "all" = every prefix bucket, kai/rv14's
  NTV) cache the nonce-token blocks: 1.26 GiB at K = 32 instead of 2.52. The other prefix buckets record hotspot's own 3
  variants (pmax 44, q|gate rows kept: hotspot's layout). The variant list stays global with the nonce-token variants
  last, so every bucket shares hotspot's indices 0-2 and an NTV bucket's pvar 3.. are the nonce-token blocks; each bucket's
  list is variants_of(...)[:n] (bucket_variants: n = all in an NTV bucket, common() elsewhere). capture_batch records
  and checks bucket L's list (Run.variants), self_check samples it, and BatchLoop's Index picks per bucket
  (Index(lim={bucket: n}, bucket=bucket_of); its plain-list fallback is cut to the common rows), so a decision's
  (pvar, plen) always names a block its bucket's pcache holds (tests/test_ntv_startup.py, every scenario). cover() runs
  every block of every NTV bucket; a bucket without NTV blocks takes hotspot's check and hotspot's path on a failure (NTV stays
  on). PRO / PRON / XA (run at start-up at the global pmax 45) run again at each bucket's runtime pmax where it differs.
- Reserve: fit() keeps the k variants whose NTV-bucket blocks leave hotspot's own estimate free: S1ENG_MEM_MARGIN_GIB +
  lanes x (S1ENG_LANE_EST_GIB + S1ENG_PAR_EXTRA_GIB with PAR) + S1ENG_NTV_RESERVE_GIB (default 0: no NTV reserve of
  its own; was 1.0), and whose recording peak (one bucket's recordings and block, q|gate included) stays out of
  S1ENG_MEM_MARGIN_GIB. The other buckets' hotspot blocks (~0.05 GiB each) are hotspot's own cost and are not charged.
- Value order: setup keeps select's expected saved tokens per decision of each variant (frequency x extra tokens);
  trim orders by it times the NTV buckets' share of the decisions (SHARES, d75) and keeps the best k, logging k and the
  expected saved tokens per decision kept. At 9.35 GiB free (d75, 20 GiB left) and reserve 8.00 GiB: all 32.

Never fatal: setup(), prepare() (trim, recheck), Run.check and summary() catch every exception; any failure logs
"lever not used: ntv (<why>)" and the engine goes on as hotspot (off(): model.prefix_variants = v15's own list, so
pcache and BatchLoop share v15's variant indices; BatchLoop keeps the plain list and v15's scan if Index fails).
pmax: capture_batch's PRO, PRON (v16) and XA blocks check the prefix layout with the variants setup() made (pmax 45 at
K = 32); where trim, NTV's buckets or NTV off change a bucket's runtime pmax, recheck() runs those checks again at it
(tests/test_ntv_startup.py). cover() runs the packed forward itself, so PRON's buckets run pron.prologue on every
variant block there.

Importing this module never imports torch (tests/test_ntv.py); cover() imports it when called.
"""
from __future__ import annotations

import collections
import hashlib
import os
import time

import numpy as np

K_DEFAULT = 32       # S1ENG_NTV_K
DEPTH = 2            # nonce tokens past a family head
SAMPLES = 1024       # enumerated ids per family
FAMILIES = ((2, lambda n: {"ref": n, "a": 1}), (1, lambda n: f"[ref {n}] some text"))   # (weight, evidence)
BUCKETS_DEFAULT = "192,256"   # S1ENG_NTV_BUCKETS
# share of the validator's decisions per bucket (d75, the timed load on plumb-4b: 128 ~4 %, 192 ~57-59 %, 256 ~37-39 %,
# 320 ~0.5 %): weights a variant's expected saving by the buckets that cache it (trim's value order and its log line)
SHARES = {128: 0.04, 192: 0.58, 256: 0.38, 320: 0.005}


def on(env=os.environ) -> bool:
    """S1ENG_NTV="1" (model.py / main.py: _ntv())."""
    return env.get("S1ENG_NTV", "") == "1"


def cap_k(env=os.environ) -> int:
    """S1ENG_NTV_K: at most this many nonce-token variants (default 32; 0 = none)."""
    return max(0, int(env.get("S1ENG_NTV_K", str(K_DEFAULT))))


def reserve_gib(env=os.environ) -> float:
    """S1ENG_NTV_RESERVE_GIB: memory NTV leaves free beyond hotspot's own estimates (default 0: trim reserves hotspot's margin +
    lanes x (lane + PAR) and nothing of its own)."""
    return float(env.get("S1ENG_NTV_RESERVE_GIB", "0"))


def ntv_buckets(env=os.environ):
    """S1ENG_NTV_BUCKETS: the buckets whose pcache holds the nonce-token variant blocks (default "192,256": ~95 % of the
    decisions); "all" = every prefix bucket (None); "" = none (NTV not used). Sorted, without repeats."""
    v = env.get("S1ENG_NTV_BUCKETS", BUCKETS_DEFAULT).strip()
    if v == "all":
        return None
    return tuple(sorted({int(x) for x in v.split(",") if x.strip()}))


def pick_buckets(buckets: list, env=os.environ) -> list:
    """The prefix buckets (sorted) that cache the nonce-token variants."""
    want = ntv_buckets(env)
    return [L for L in sorted(buckets) if want is None or L in want]


def nonces(n: int) -> list:
    """The fixed enumeration of record ids: 8 hex digits of sha256("ntv<i>") (uniform, like token_hex(4))."""
    return [hashlib.sha256(f"ntv{i}".encode()).hexdigest()[:8] for i in range(n)]


def longest(ids: list, have: list) -> int:
    """Length of the longest of ``have`` that ids starts with and outgrows (pick_variant's rule), or -1."""
    best = -1
    for v in have:
        k = len(v)
        if len(ids) > k > best and ids[:k] == v:
            best = k
    return best


def select(samples: list, have: list, k: int, depth: int = DEPTH, values: list | None = None) -> list:
    """Greedy top-k trie nodes. samples: [(integer weight, ids)]; have: the variants already cached (lists of ids).
    -> up to k new variants (lists of ids), best first. values (a list): appended per node, its expected saved tokens
    per sample (frequency x extra tokens: the marginal weighted saving over the samples' total weight), which the
    greedy order never increases."""
    have = [list(v) for v in have]
    weight, best, through = [], [], {}
    total = sum(int(w) for w, _ in samples) or 1
    for w, ids in samples:
        ids = list(ids)
        b = longest(ids, have)
        if b < 0:
            continue
        si = len(weight)
        weight.append(int(w))
        best.append(b)
        for d in range(1, depth + 1):
            if len(ids) > b + d:
                through.setdefault(tuple(ids[:b + d]), []).append(si)
    out = []
    for _ in range(k):
        top, tg = None, 0
        for node, sis in through.items():
            g = sum(weight[s] * (len(node) - best[s]) for s in sis if best[s] < len(node))
            if g > tg or (g == tg and g > 0 and (len(node), node) < (len(top), top)):
                top, tg = node, g
        if top is None:
            break
        out.append(list(top))
        if values is not None:
            values.append(tg / total)
        for s in through.pop(top):
            best[s] = max(best[s], len(top))
    return out


def variants(encode, have: list, k: int, n: int = SAMPLES, depth: int = DEPTH, values: list | None = None) -> list:
    """The nonce-token variants for this tokenizer: encode = Encoder.encode_full; have = the template prefix and v15's
    variants. values: select's (expected saved tokens per decision of each, the families weighted 2 : 1)."""
    from .prompt import prompt_text

    ids = nonces(n)
    samples = [(w, encode(prompt_text(mk(x), "Pick one.", ["a: first", "b: second"]))) for w, mk in FAMILIES for x in ids]
    return select(samples, have, k, depth, values=values)


class Index(list):
    """The cached variants (index 0 = the template prefix, every other one starts with it; numpy or list rows), plus a
    trie of their tokens past the template. pick(ids) = pick_variant's scan: the index of the longest variant ids
    starts with and outgrows, or -1.

    ntvl: lim = {bucket: n} (the first n rows are the variants that bucket's pcache holds; the nonce-token ones are
    last, so every bucket shares hotspot's indices) and bucket = model.bucket_of: pick(ids) is the scan over
    rows[:lim[bucket_of(len(ids))]] (a bucket not in lim: the rows every bucket caches, common = min(lim)), so a
    decision never gets a variant its bucket does not cache. Without lim: every row, as before."""

    def __init__(self, rows, lim: dict | None = None, bucket=None):
        super().__init__(rows)
        self.lim = {int(L): int(n) for L, n in lim.items()} if lim else None
        if self.lim is not None:
            if bucket is None or not all(1 <= n <= len(self) for n in self.lim.values()):
                raise ValueError(f"bad per-bucket variant counts {self.lim} for {len(self)} variants")
        self.common = min(self.lim.values()) if self.lim else len(self)
        self.bucket = bucket
        self.t0 = [int(x) for x in self[0]] if len(self) else []
        self.p0 = len(self.t0)
        self.root: dict = {}
        self.depth = 0
        for vi, v in enumerate(self):
            v = [int(x) for x in v]
            if v[:self.p0] != self.t0:
                raise ValueError(f"variant {vi} does not start with the template prefix")
            node = self.root
            tail = v[self.p0:]
            for j, t in enumerate(tail):
                ent = node.setdefault(t, [{}, -1])
                if j == len(tail) - 1 and ent[1] < 0:   # the first of equal variants, as the scan
                    ent[1] = vi
                node = ent[0]
            self.depth = max(self.depth, len(tail))

    def pick(self, ids) -> int:
        n, p0 = len(ids), self.p0
        if not len(self) or n <= p0:
            return -1
        seg = ids[:p0 + self.depth]
        seg = seg.tolist() if hasattr(seg, "tolist") else list(seg)
        if seg[:p0] != self.t0:
            return -1
        cap = len(self) if self.lim is None else self.lim.get(self.bucket(n), self.common)
        best, node = 0, self.root
        for pos in range(p0, min(len(seg), n - 1)):   # a variant must leave at least one token
            ent = node.get(seg[pos])
            if ent is None:
                break
            if 0 <= ent[1] < cap:
                best = ent[1]
            node = ent[0]
        return best


def row_bytes(layers, prune: bool, drop_qg: bool, elem: int = 2) -> int:
    """Bytes of one cached prefix row per bucket: what record_prefix keeps of every layer (GDN: in_proj_qkv, b, a;
    attention: q_proj (q|gate, not kept with drop_qg), k, v; the pruned last attention layer: k, v)."""
    n = 0
    last = len(layers) - 1
    for i, d in enumerate(layers):
        if d["type"] == "linear_attention":
            la = d["la"]
            n += sum(m.weight.shape[0] for m in (la.in_proj_qkv, la.in_proj_b, la.in_proj_a))
        else:
            sa = d["sa"]
            n += sa.k_proj.weight.shape[0] + sa.v_proj.weight.shape[0]
            if not drop_qg and not (prune and i == last):
                n += sa.q_proj.weight.shape[0]
    return n * elem


def block_bytes(lens: list, rb: int) -> int:
    """pcache bytes of one bucket: every variant in a block of pmax rows."""
    return len(lens) * max(lens) * rb if lens else 0


def _cost(base_lens: list, extra_lens: list, rbs: list, k: int) -> int:
    return sum(block_bytes(base_lens + extra_lens[:k], rb) for rb in rbs)


def _peak(base_lens: list, extra_lens: list, peak_rb: int, k: int) -> int:
    return 2 * block_bytes(base_lens + extra_lens[:k], peak_rb)


def fit(base_lens: list, extra_lens: list, rbs: list, free: int, reserve: int, peak_rb: int = 0,
        margin: int = 0) -> int:
    """The most of extra_lens (in order) whose recording leaves ``reserve`` bytes free: base_lens = v15's variants,
    rbs = row bytes of every prefix bucket, free = free device bytes before the recording. peak_rb: row bytes as
    recorded (q|gate included); record_prefix holds one bucket's recordings and its stacked block at once, a peak of 2
    such blocks above what is kept, which must leave ``margin`` free."""
    k = len(extra_lens)
    while k > 0 and (free - _cost(base_lens, extra_lens, rbs, k) < reserve or
                     free - _cost(base_lens, extra_lens, rbs, k) - _peak(base_lens, extra_lens, peak_rb, k) < margin):
        k -= 1
    return k


def fit_why(base_lens: list, extra_lens: list, rbs: list, free: int, reserve: int, k: int, peak_rb: int = 0,
            margin: int = 0) -> str:
    """Why fit() stopped at k (k < len(extra_lens)): what variant k + 1 would have left free."""
    left = free - _cost(base_lens, extra_lens, rbs, k + 1)
    if left < reserve:
        return f"variant {k + 1} would leave {left / 2**30:.2f} GiB free < reserve {reserve / 2**30:.2f} GiB"
    left -= _peak(base_lens, extra_lens, peak_rb, k + 1)
    return f"variant {k + 1}'s recording peak would leave {left / 2**30:.2f} GiB free < margin {margin / 2**30:.2f} GiB"


def setup(model, encode, log=print, env=os.environ) -> list:
    """main.py, after PV: model.prefix_variants = v15's + the nonce-token variants (model.ntv_base = how many are
    v15's, for trim; model.ntv_v15 = v15's own list, which off() puts back). Returns the new ones. Never raises: any
    exception logs "lever not used: ntv (<ExcType>: <msg>)" and leaves v15's variants (main.py's guard catches what
    could still escape, e.g. the import)."""
    pv = getattr(model, "prefix_variants", None)
    try:
        return _setup(model, pv, encode, log, env)
    except Exception as exc:  # noqa: BLE001 - NTV must never stop the engine from serving
        model.prefix_variants = pv
        for name in STATE:
            if hasattr(model, name):
                delattr(model, name)
        log(f"lever not used: ntv ({type(exc).__name__}: {exc})")
        return []


def _setup(model, pv, encode, log, env) -> list:
    if not pv or not getattr(model, "prefix_ids", None):
        log("lever not used: ntv (no prefix variants to extend: S1ENG_PREFIX / S1ENG_PREFIX_VARIANTS off)")
        return []
    t0 = time.time()
    k = cap_k(env)
    have = [list(model.prefix_ids)] + [list(v) for v in pv]
    vals: list = []
    extra = variants(encode, have, k, values=vals)
    model.prefix_variants = [list(v) for v in pv] + extra
    model.ntv_base, model.ntv_v15, model.ntv_value = len(pv), pv, vals
    lens = dict(sorted(collections.Counter(len(v) for v in extra).items()))
    log(f"NTV: {len(extra)} nonce-token prefix variants (cap {k}; tokens: count {lens}) from {SAMPLES} hex ids x "
        f"{len(FAMILIES)} evidence families in {time.time() - t0:.2f}s; {sum(vals):.3f} expected saved tokens per "
        f"decision at {len(extra)}")
    return extra


STATE = ("ntv_base", "ntv_v15", "ntv_value", "ntv_buckets", "ntv_common")   # NTV's attributes on the model


def active(model) -> bool:
    """setup() added the nonce-token variants and nothing has turned NTV off since."""
    return getattr(model, "ntv_base", None) is not None


def off(model, why: str = "", log=print) -> None:
    """NTV off: model.prefix_variants = v15's own list again (the one setup() extended); logs "lever not used: ntv
    (<why>)" when why is given."""
    if active(model):
        v15 = getattr(model, "ntv_v15", None)
        model.prefix_variants = v15 if v15 is not None else list(model.prefix_variants)[:model.ntv_base]
        for name in STATE:
            if hasattr(model, name):
                delattr(model, name)
    if why:
        log(f"lever not used: ntv ({why})")


def trim(model, pf, prefix: list, buckets: list, lanes_n: int, log=print, env=os.environ, free=None) -> int:
    """capture_batch, before the prefix recording (pf.xa known). buckets: the prefix buckets; the nonce-token variants
    go to those of S1ENG_NTV_BUCKETS only (model.ntv_buckets), the others record hotspot's own variants as hotspot does.
    Value order: the nonce-token variants by expected saved tokens per decision (setup's values, frequency x extra
    tokens, times the NTV buckets' share of the decisions, SHARES; equal values keep select's order); keeps the best k
    whose NTV-bucket blocks leave hotspot's reserve free (fit: S1ENG_MEM_MARGIN_GIB + lanes x (S1ENG_LANE_EST_GIB +
    S1ENG_PAR_EXTRA_GIB with PAR) + S1ENG_NTV_RESERVE_GIB, default 0) and whose recording peak stays out of the margin.
    The other buckets' blocks are hotspot's own (3 variants, as without NTV) and are not charged. Logs k and the expected
    saved tokens per decision kept. Returns k."""
    nb = getattr(model, "ntv_base", None)
    if nb is None or not buckets:
        return 0
    nbk = pick_buckets(buckets, env)
    others = [L for L in sorted(buckets) if L not in nbk]
    if not nbk:
        want = ntv_buckets(env)
        model.prefix_variants = list(model.prefix_variants)[:nb]
        log(f"lever not used: ntv (NTV memory guard: no prefix bucket among S1ENG_NTV_BUCKETS {list(want or [])} "
            f"(prefix buckets {sorted(buckets)}))")
        return 0
    pv = list(model.prefix_variants)
    P = len(prefix)
    base = [P] + [len(v) for v in pv[:nb] if len(v) > P and list(v[:P]) == list(prefix)]
    extra = pv[nb:]
    vals = list(getattr(model, "ntv_value", None) or [])
    if len(vals) != len(extra):
        vals = [0.0] * len(extra)
    order = sorted(range(len(extra)), key=lambda i: -vals[i])   # stable: equal values keep select's order
    extra, vals = [extra[i] for i in order], [vals[i] for i in order]
    share = sum(SHARES.get(L, 0.0) for L in nbk)
    xa = getattr(pf, "xa", None) or set()
    rbs = [row_bytes(pf.ff.layers, bool(getattr(pf, "prune", False)), L in xa) for L in nbk]
    if free is None:
        import torch

        free = torch.cuda.mem_get_info()[0]
    par = bool(getattr(pf, "par", None))

    def gib(name, default):
        return float(env.get(name, default))
    margin = gib("S1ENG_MEM_MARGIN_GIB", "2.0")
    res = margin + lanes_n * (gib("S1ENG_LANE_EST_GIB", "1.5") + (gib("S1ENG_PAR_EXTRA_GIB", "1.5") if par else 0.0))
    res += reserve_gib(env)
    elens = [len(v) for v in extra]
    peak_rb = row_bytes(pf.ff.layers, bool(getattr(pf, "prune", False)), False)
    k = fit(base, elens, rbs, free, int(res * 2**30), peak_rb=peak_rb, margin=int(margin * 2**30))
    note = ""
    if k < K_DEFAULT:   # q_s1v2.sh's smoke wants k >= 24 on an -ntv image: say why it is short
        note = (fit_why(base, elens, rbs, free, int(res * 2**30), k, peak_rb=peak_rb, margin=int(margin * 2**30))
                if k < len(extra) else f"{len(extra)} derived, S1ENG_NTV_K {cap_k(env)}")
        note = f" ({k} < {K_DEFAULT}: {note})"
    model.prefix_variants = pv[:nb] + extra[:k]
    model.ntv_value = vals[:k]
    if k:
        model.ntv_buckets = tuple(nbk)
    lens = base + [len(v) for v in extra[:k]]
    total = sum(block_bytes(lens, rb) for rb in rbs)
    more = total - sum(block_bytes(base, rb) for rb in rbs)
    msg = (f"NTV memory guard: {k}/{len(extra)} nonce-token variants kept{note} for buckets {nbk} "
           f"({share * sum(vals[:k]):.3f} of {share * sum(vals):.3f} expected saved tokens per decision): prefix cache "
           f"{total / 2**30:.2f} GiB (+{more / 2**30:.2f} GiB), {free / 2**30:.2f} GiB free, reserve {res:.2f} GiB "
           f"(lanes {lanes_n}{' with PAR' if par else ''})"
           + (f"; hotspot's {len(base)} variants for buckets {others}" if others else ""))
    log(f"lever not used: ntv ({msg})" if not k else msg)
    return k


def common(model, prefix: list, variants_of) -> int:
    """How many of variants_of(model, prefix) every bucket caches: all of them but the nonce-token ones, which are
    last (hotspot's indices are the same in every bucket)."""
    allv = variants_of(model, prefix)
    nb = getattr(model, "ntv_base", None)
    if nb is None:
        return len(allv)
    ext = {tuple(int(x) for x in v) for v in list(model.prefix_variants)[nb:]}
    isx = [tuple(int(x) for x in v) in ext for v in allv]
    n = isx.count(False)
    if any(isx[:n]):
        raise ValueError("the nonce-token variants are not the last cached variants")
    return n


def bucket_variants(model, prefix: list, L: int, variants_of) -> list:
    """The variants bucket L's pcache holds (record_prefix's list, in pvar order): variants_of(model, prefix) in an NTV
    bucket (model.ntv_buckets; None or NTV off: every bucket), its first common() entries (hotspot's) elsewhere."""
    allv = variants_of(model, prefix)
    nbk = getattr(model, "ntv_buckets", None)
    if not active(model) or nbk is None or L in nbk:
        return allv
    return allv[:common(model, prefix, variants_of)]


def limits(model, prefix: list, buckets, variants_of):
    """{bucket: how many variants its pcache holds} for Index, or None (NTV off or every bucket alike)."""
    if not active(model) or getattr(model, "ntv_buckets", None) is None:
        return None
    return {L: len(bucket_variants(model, prefix, L, variants_of)) for L in buckets}


def index(model, rows, prefix: list, buckets, variants_of, bucket_of) -> "Index":
    """BatchLoop's variants (rows = hotspot's list of numpy rows, variants_of's order): an Index restricted per bucket to
    the variants its pcache holds."""
    lim = limits(model, list(prefix), buckets, variants_of)
    if lim is not None and len(rows) != len(variants_of(model, list(prefix))):
        raise ValueError(f"{len(rows)} variant rows for {len(variants_of(model, list(prefix)))} cached variants")
    return Index(rows, lim=lim, bucket=bucket_of if lim is not None else None)


def pmax_of(variants: list) -> int:
    """Rows per variant block in pcache (record_prefix): the longest cached variant."""
    return max(len(v) for v in variants)


def prepare(model, pf, prefix: list, pfx_of: dict, pack: dict, prow: dict, prow0: dict, lanes_n: int, variants_of,
            check_prefix, log=print, env=os.environ, free=None):
    """capture_batch, right before hotspot's prefix recording (after PRO / PRON / ONORM / XA / RARN and hotspot's BA / LT /
    XL / TG / TGS gates, so pf.xa and pf.pron are known and the ONORM / FCL gates' scratch is freed): trim(); any
    exception there, or no variant kept, turns NTV off (v15's own variant list, logged "lever not used: ntv (...)").
    capture_batch's PRO, PRON and XA blocks ran their prefix checks with the variants as setup() left them (pmax 45 at K = 32); if the runtime
    pmax differs now, recheck() runs them again at it. Returns the recording's Run (falsy once NTV is off), or None: NTV
    not used, hotspot's recording. Never raises."""
    if not active(model):
        return None
    try:
        return _prepare(model, pf, prefix, pfx_of, pack, prow, prow0, lanes_n, variants_of, check_prefix, log, env,
                        free)
    except Exception as exc:  # noqa: BLE001 - past trim's own guard: v15's variants, as for any other failure
        off(model, f"{type(exc).__name__}: {exc}", log)
        return None


def _prepare(model, pf, prefix, pfx_of, pack, prow, prow0, lanes_n, variants_of, check_prefix, log, env, free):
    pm0 = pmax_of(variants_of(model, prefix))   # what capture_batch's PRO / PRON / XA checks ran at, every bucket
    try:
        if not trim(model, pf, prefix, sorted(pfx_of), lanes_n, log=log, env=env, free=free):
            off(model)   # trim logged "lever not used: ntv (NTV memory guard: 0/...)"
        else:
            model.ntv_common = common(model, prefix, variants_of)
    except Exception as exc:  # noqa: BLE001 - NTV must never stop the engine from serving
        off(model, f"{type(exc).__name__}: {exc}", log)
    pm = {L: pmax_of(bucket_variants(model, prefix, L, variants_of)) for L in sorted(pfx_of)}
    if any(p != pm0 for p in pm.values()):
        recheck(pf, pack, pfx_of, pm, pm0, log=log, env=env)
    if not active(model):
        return None
    return Run(model, pf, prefix, pfx_of, pack, prow, prow0, variants_of, check_prefix, pm, log, env)


def recheck(pf, pack: dict, pfx_of: dict, pmax, was, log=print, env=os.environ) -> None:
    """v16's PRO, PRON and XA start-up checks of the prefix buckets again, at the runtime pmax (capture_batch ran them
    at ``was``, before trim / NTV off), exactly as capture_batch's PRO / PRON / XA blocks run them. pmax / was: one
    value for every bucket or {bucket: pmax} (ntvl: an NTV bucket's pmax is the nonce-token list's, the others' hotspot's);
    only the buckets whose pmax differs from ``was`` run again (an int pmax: every prefix bucket). The same layers, B
    values and arguments; a bucket keeps or gets the lever iff both its checks pass (a member already passed the
    pmax-independent pfx = 0 check, which is deterministic: only its prefix check runs again). PRON (rgh's native
    prologue, S1ENG_PRON unset = on) runs inside PRO's branch only: it is re-checked on the PRO buckets as just
    re-checked, leaves a bucket that lost PRO, and a PRO bucket that is not a member gets it only after PRON's bf16
    domain sweep (once) passes too, as in capture_batch. Never raises."""
    pro = env.get("S1ENG_PRO", "0") == "1" and getattr(pf, "ff_fused", False) and getattr(pf, "ff_mode", "") == "ps"
    pron = pro and env.get("S1ENG_PRON", "1") == "1" and getattr(pf, "pron", None) is not None
    xa = env.get("S1ENG_XA", "0") == "1"
    pm_of = (lambda L: pmax.get(L, was_of(L))) if isinstance(pmax, dict) else (lambda L: pmax)
    was_of = (lambda L: was.get(L, 0)) if isinstance(was, dict) else (lambda L: was)
    todo = [L for L in sorted(pfx_of) if not isinstance(pmax, dict) or pm_of(L) != was_of(L)]
    if not (pro or xa) or not todo:
        return
    t1 = time.time()
    notes = []

    def run(kind, members, L, full, pfx):
        try:
            why = ("" if L in members else full()) or pfx()
        except Exception as exc:  # noqa: BLE001 - the lever must never take the batch lanes down with it
            why = f"{type(exc).__name__}: {exc}"
        if why:
            notes.append(f"{kind} {L}: {why}")
            members.discard(L)
        else:
            members.add(L)
    try:
        ff = pf.ff
        if pro:   # capture_batch's PRO block
            gdn = [d for d in ff.layers if d["type"] == "linear_attention"]
            layers = [gdn[0], gdn[len(gdn) // 2], gdn[-1]]
            for L in todo:
                bs = sorted({b for b in (1, 2, 5, max(pack[L])) if b <= max(pack[L])})
                pp = pfx_of[L]
                run("pro", pf.pro, L, lambda: pf.check_prologue(L, bs, layers),
                    lambda: pf.check_prologue(L, bs, layers, pfx=pp, pmax=max(pm_of(L), pp)))
        if pron:   # capture_batch's PRON block (v16), on the PRO buckets as re-checked above
            gdn = [d for d in ff.layers if d["type"] == "linear_attention"]
            for L in [L for L in todo if L in pf.pron and L not in pf.pro]:
                pf.pron.discard(L)
            dom = None   # PRON's bf16 domain sweep: run once, only for a PRO bucket that is not a member yet
            for L in [L for L in todo if L in pf.pro]:
                if L not in pf.pron and dom is None:
                    try:
                        from . import pron as pronmod

                        dom = "no pron kernel for this GPU" if pronmod.lib() is None else pf.check_pron_domain(gdn[0])
                    except Exception as exc:  # noqa: BLE001
                        dom = f"{type(exc).__name__}: {exc}"
                if L not in pf.pron and dom:
                    notes.append(f"pron {L}: {dom}")
                    continue
                bs = sorted({b for b in (1, 2, 5, max(pack[L])) if b <= max(pack[L])})
                pp = pfx_of[L]
                run("pron", pf.pron, L, lambda: pf.check_pron(L, bs, gdn[::2]),
                    lambda: pf.check_pron(L, bs, gdn[::2], pfx=pp, pmax=max(pm_of(L), pp)))
        if getattr(pf, "profuse", False):   # PROFUSE-v (ruj r3, onormpv.py): its prefix check again at the runtime pmax
            # for its buckets (the pfx = 0 check is pmax-independent); a bucket that lost PRON or PRO leaves it
            from . import onormpv
            from . import onorm as onm

            gdn = [d for d in ff.layers if d["type"] == "linear_attention"]
            for L in [L for L in todo if L in onormpv.ON]:
                if L not in pf.pron or L not in pf.pro:
                    onormpv.ON.discard(L)
                    notes.append(f"profuse {L}: PRON / PRO off")
                    continue
                bs = sorted(b for b in pack[L] if b >= onm.bmin(L))
                pp = pfx_of[L]
                run("profuse", onormpv.ON, L, lambda: "",
                    lambda: onormpv.check(pf, L, bs, gdn[::2], pfx=pp, pmax=max(pm_of(L), pp)))
        if xa:   # capture_batch's XA block
            das = [d for d in ff.layers if d["type"] != "linear_attention"]
            mid = len(das) // 2
            layers = [das[0], das[mid]] + [d for i, d in enumerate(das) if i not in (0, mid)]
            want3 = getattr(pf, "xa_want", None)
            if want3:
                from . import xa as xamod
            for L in todo:
                _m, cos, sin = ff.consts[L]
                pp = pfx_of[L]
                n3 = xamod.vcount(pf, "xa", L) if want3 else 0
                run("xa", pf.xa, L, lambda: pf.check_xa(L, layers, cos, sin, max(pack[L])),
                    lambda: pf.check_xa(L, layers, cos, sin, max(pack[L]), pfx=pp, pmax=max(pm_of(L), pp)))
                if want3 and L in pf.xa:   # XA3: the re-check must compare the variant again (>= 1 more pass)
                    xamod.require_new(pf, "xa", L, n3)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"re-check: {type(exc).__name__}: {exc}")
    _cuda_sync()
    pron_now = sorted(getattr(pf, "pron", None) or [])
    at = {L: pm_of(L) for L in todo} if isinstance(pmax, dict) else pmax
    if isinstance(at, dict) and todo == sorted(pfx_of) and len(set(at.values())) == 1:
        at = at[todo[0]]   # one pmax for every prefix bucket
    if isinstance(was, dict) and len(set(was_of(L) for L in todo)) == 1:
        was = was_of(todo[0])
    pvf = ""
    if getattr(pf, "profuse", False):   # PROFUSE-v (ruj r3): its buckets after the re-check
        from . import onormpv

        pvf = f", PROFUSE buckets {sorted(onormpv.ON)}"
    log(f"NTV: PRO / PRON / XA prefix checks re-run at pmax {at} (start-up ran them at {was}) in "
        f"{time.time() - t1:.1f}s: PRO buckets {sorted(pf.pro)}, PRON buckets {pron_now}, XA buckets {sorted(pf.xa)}"
        + pvf + (f"; lever not used for {'; '.join(notes)}" if notes else ""))


def _cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - housekeeping only
        pass


class Run:
    """capture_batch's prefix recording with the nonce-token variants (falsy once NTV is off).

    check(L, why) after bucket L's record_prefix / check_prefix (why: their failure, "" if check_prefix passed): with
    why = "", cover() runs every variant block once; any failure turns NTV off for every bucket (fallback) before
    anything else is given up: v15's own variants, the buckets recorded so far and L recorded again and checked as
    v15's loop does (check_prefix, then without PRO), the PRO / PRON / XA checks again at v15's pmax first. Every
    bucket's pcache then holds the one global variant list BatchLoop takes (prefix_variants_of), and only a failure of
    v15's own variants turns a bucket's prefix off (v15's path: L's by capture_batch, an earlier one's here)."""

    def __init__(self, model, pf, prefix, pfx_of, pack, prow, prow0, variants_of, check_prefix, pmax, log, env):
        self.model, self.pf, self.prefix = model, pf, prefix
        self.pfx_of, self.pack, self.prow, self.prow0 = pfx_of, pack, prow, prow0
        self.variants_of, self.check_prefix = variants_of, check_prefix
        self.pmax, self.log, self.env = pmax, log, env   # pmax: {bucket: runtime pmax} (or one int for every bucket)
        self.pro0 = set(pf.pro)   # PRO buckets as the recording starts (v15's PRO retry starts from these)
        self.done: list = []      # NTV buckets recorded and checked with the nonce-token variants
        self.on = True

    def __bool__(self) -> bool:
        return self.on

    def caches(self, L: int) -> bool:
        """Bucket L's pcache gets the nonce-token variant blocks (while NTV is on)."""
        nbk = getattr(self.model, "ntv_buckets", None)
        return self.on and active(self.model) and (nbk is None or L in nbk)

    def variants(self, L: int) -> list:
        """The variants bucket L records and checks now (hotspot's everywhere once NTV is off)."""
        return bucket_variants(self.model, self.prefix, L, self.variants_of)

    def check(self, L: int, why: str) -> str:
        """"" (bucket L keeps its prefix) or the failure that turns L's prefix off (v15's path in capture_batch). A
        bucket without the nonce-token blocks recorded hotspot's own variants: its result is hotspot's (no cover, no NTV
        off)."""
        if not self.caches(L):
            return why
        try:
            if not why:
                why = cover(self.model, self.pf, L, self.prow[L], self.prefix, self.variants(L))
                if not why:
                    self.done.append(L)
                    return ""
            return self.fallback(L, why)
        except Exception as exc:  # noqa: BLE001 - never the lanes
            self.on = False
            off(self.model, f"{type(exc).__name__}: {exc}", self.log)
            return f"{type(exc).__name__}: {exc}"

    def fallback(self, L: int, why: str) -> str:
        m, pf, prefix = self.model, self.pf, self.prefix
        self.on = False
        off(m, f"variant check: bucket {L}: {why}", self.log)
        for L2 in self.done + [L]:   # PRO as the recording started, for the buckets recorded again only (a bucket
            # without NTV blocks keeps what hotspot's PRO retry left it)
            if L2 in self.pro0:
                pf.pro.add(L2)
            else:
                pf.pro.discard(L2)
        pm = pmax_of(self.variants_of(m, prefix))
        was = self.pmax if isinstance(self.pmax, dict) else {L2: self.pmax for L2 in self.pfx_of}
        if any(was.get(L2, pm) != pm for L2 in self.pfx_of):
            recheck(pf, self.pack, self.pfx_of, {L2: pm for L2 in self.pfx_of}, was, log=self.log, env=self.env)
        variants = self.variants_of(m, prefix)
        out, kept = why, []
        for L2 in self.done + [L]:   # v15's loop body, with v15's variants
            for key in [k for k in pf.pcache if k[0] == L2]:
                del pf.pcache[key]
            try:
                pf.record_prefix(L2, variants if len(variants) > 1 else prefix)
                w = self.check_prefix(m, pf, L2, self.prow[L2], prefix, variants=variants)
                if w and L2 in pf.pro:
                    self.log(f"lever not used: pro for bucket {L2} (template prefix check with it: {w})")
                    pf.pro.discard(L2)
                    w = self.check_prefix(m, pf, L2, self.prow[L2], prefix, variants=variants)
            except Exception as exc:  # noqa: BLE001 - the bucket keeps the S5 layout
                w = f"{type(exc).__name__}: {exc}"
            if L2 == L:
                out = w
            elif w:
                self.log(f"template prefix OFF for bucket {L2} (S5 layout): {w}")
                del self.pfx_of[L2]
                self.prow[L2] = self.prow0[L2]
                for key in [k for k in pf.pcache if k[0] == L2]:
                    del pf.pcache[key]
            if not w:
                kept.append(L2)
        self.log(f"NTV off: v15's {len(variants)} prefix variants (pmax {pm}) recorded again and checked for buckets "
                 f"{kept}")
        return out


def summary_line(n_variants: int, n_ntv: int, lens: list, buckets: list, nbytes: int, xa_buckets: list, others=(),
                 n_other: int = 0, tokens=None) -> str:
    """The start-up line (q_s1v2.sh TAGLINES "ntv:NTV on: (2[4-9]|3[0-9]) nonce-token": at least 24 kept, which the
    idle smoke box has the memory for; a run with serving models resident may trim, its memory guard line says how
    many and why). buckets: the NTV buckets (n_variants each); others: the prefix buckets with hotspot's n_other."""
    if not buckets:
        return "lever not used: ntv (no bucket kept the template prefix)"
    if not n_ntv:
        return f"lever not used: ntv (no nonce-token variant kept; {n_variants} prefix variants per bucket)"
    return (f"NTV on: {n_ntv} nonce-token prefix variants ({n_variants} per bucket, {min(lens)}-{max(lens)} tokens) "
            f"for buckets {list(buckets)}" + (f", hotspot's {n_other} for {list(others)}" if others else "")
            + f"; prefix cache {nbytes / 2**30:.2f} GiB"
            + (f" (XA buckets without q|gate rows: {list(xa_buckets)})" if xa_buckets else "")
            + (f"; {tokens:.3f} expected saved tokens per decision" if tokens is not None else ""))


def summary(model, pf, prefix: list, pfx_of: dict) -> str:
    """summary_line for capture_batch, after the prefix recording: what pcache holds for the buckets that kept it.
    Never raises (a log line must not stop the engine; without its "NTV on:" line the smoke fails the image)."""
    try:
        return _summary(model, pf, prefix, pfx_of)
    except Exception as exc:  # noqa: BLE001
        return f"NTV: no summary line ({type(exc).__name__}: {exc})"


def _summary(model, pf, prefix: list, pfx_of: dict) -> str:
    nb = getattr(model, "ntv_base", None) or 0
    pv = list(getattr(model, "prefix_variants", None) or [])
    ntv = pv[nb:]
    n = 1 + len([v for v in pv[:nb] if len(v) > len(prefix)]) + len(ntv)
    nbytes = sum(t.numel() * t.element_size() for key, d in pf.pcache.items() if key[0] in pfx_of for t in d.values())
    want = getattr(model, "ntv_buckets", None)
    nbk = [L for L in sorted(pfx_of) if want is None or L in want]
    others = [L for L in sorted(pfx_of) if L not in nbk]
    xa = [L for L in nbk if L in (getattr(pf, "xa", None) or set())]
    vals = getattr(model, "ntv_value", None)
    tokens = sum(SHARES.get(L, 0.0) for L in nbk) * sum(vals) if vals is not None else None
    return summary_line(n, len(ntv), [len(v) for v in ntv], nbk, nbytes, xa, others=others, n_other=n - len(ntv),
                        tokens=tokens)


def cover(model, pf, L: int, prow_l: dict, prefix: list, variants: list, seed: int = 94) -> str:
    """Every variant block of bucket L at least once: groups of the largest packed B, slot j of a group the next
    variant, each a random-tail decision of the bucket, against the per-decision graph of set 0 (check_prefix's
    method). "" if every logit is bit-equal, else the first difference."""
    import torch

    from .batchfwd import PackedMeta, packed_layout
    from .gov import GOV
    from .model import GRAPH_LENGTHS

    try:
        with torch.inference_mode():
            P = len(prefix)
            g = torch.Generator().manual_seed(seed + L)
            vocab = model.hf.config.vocab_size
            lprev = max([x for x in GRAPH_LENGTHS if x < L], default=0)
            heads = [np.asarray(v, dtype=np.int64) for v in variants]
            graph, s_ids, s_last, out1 = model.graph_sets[0][L]
            B = max(prow_l)
            for c0 in range(0, len(heads), B):
                vis = list(range(c0, min(c0 + B, len(heads))))
                seqs = []
                for vi in vis:
                    head = heads[vi]
                    n = int(torch.randint(max(lprev + 1, len(head) + 1), L + 1, (1,), generator=g))
                    tail = torch.randint(10, min(vocab, 150000), (n - len(head),), generator=g).numpy()
                    seqs.append(np.concatenate([head, tail]))
                ref = []
                for ids in seqs:
                    s_ids.zero_()
                    s_ids[0, :len(ids)] = torch.from_numpy(ids).to(s_ids.device)
                    s_last.fill_(len(ids) - 1)
                    graph.replay()
                    ref.append(out1[0].clone())
                pls = [len(heads[vi]) for vi in vis]
                T = sum(len(x) - pl for x, pl in zip(seqs, pls))
                R = next((r for r in prow_l[B] if r >= T), None)
                if R is None:
                    return f"no R bin for B={B} T={T}"
                m = PackedMeta(B, L, R, device=s_ids.device, pfx=P)
                st, sl, rp, la = packed_layout([len(x) for x in seqs], B, L, R, P, plens=pls)
                ids = torch.zeros((R,), dtype=torch.long)
                ids[:T] = torch.from_numpy(np.concatenate([x[pl:] for x, pl in zip(seqs, pls)]))
                m.ids.copy_(ids)
                m.start.copy_(torch.from_numpy(st))
                m.slen.copy_(torch.from_numpy(sl))
                m.row_pos.copy_(torch.from_numpy(rp))
                m.last.copy_(torch.from_numpy(la))
                m.pvar[:len(vis)].copy_(torch.tensor(vis, dtype=torch.int32))
                m.plen[:len(vis)].copy_(torch.tensor(pls, dtype=torch.int32))
                got = pf(m)
                if got.is_cuda:
                    torch.cuda.synchronize()
                GOV.tick()
                for r, vi in enumerate(vis):
                    if not torch.equal(got[r].view(torch.int16), ref[r].view(torch.int16)):
                        return f"B={B} R={R}: variant {vi} (length {pls[r]}) differs from the per-decision graph"
    except Exception as exc:  # noqa: BLE001 - a failure turns the bucket's prefix off, never the lanes
        return f"{type(exc).__name__}: {exc}"
    return ""
