"""SEL2b (kai): a per-SKU table of exact cuBLASLt configurations, offered to S6 tuning (lt.tune) as extra candidates.

Ported from kai's engfast (team/sel2 18bbae0: engfast/gemm_table.py, ltenum.py unstable_class / split_class and
LtEnumApi.build). On kai's engine the table's picks (mostly cuBLASLt algorithm id 67: other numerical flags than the
id-21 pins, byte-equal on every input checked: 166/166 cells, 20,010/20,010 oracle) saved 4.5 % of GEMM time and gave
+2.45 % throughput. cuBLASLt can run ~12,500 byte-equal tile x custom-option variants per shape, too many to time at
start-up, and its heuristic never offers the fast ones: they are measured offline (spike/s1_table.py) and shipped.

Here a table pick is NOT an override. It joins the S6 pool of its (N, K, bucket) next to the heuristic algos, their
split-forced clones and NG3, and goes through every S6 step (byte check against the per-decision F.linear, pair
timing, the 2 % margin over the pin, the re-check at every R it serves). Before that it must also pass ``prove``: the
scale-8 input and the outlier input (1 % of entries x 64), the pair twice (a pass that fails after an earlier one
passed = "unstable"). Only configurations in the pin's split-K class are offered; the unstable class (an attribute
left UNDEFINED) is dropped on load. SG and the fused-or-parts decision read S6's times, so they are judged against the
faster GEMM without any change.

S1ENG_GEMM_TABLE: "" or "0" = off (the default: the base job's behaviour, nothing here runs); "1" = the built-in table for
this GPU model (gemm_table_s1_<sku>.json, else kai's engfast table gemm_table_<sku>.json, whose keys only partly match
s1eng's row grid: its configurations then serve as anchor candidates); anything else = a table file. The table is used
only on the GPU model and with the cuBLAS wheel it was measured with; any skip logs "lever not used: gemm_table (...)".
S1ENG_GEMM_TABLE_MAX (default 32): table configurations offered per (N, K, bucket).

In lt.tune a table configuration enters an anchor's contest only if it is the table's pick at an R of that group (an
engfast table, measured on another row grid: every table configuration of the shape when none is) and it fits the
workspace (ws, and ws_b with pair timing) and reproduces the per-decision bytes at every R of the group (fits_group).
The per-R refinement (and so a table pick at a single R) only runs for N >= S1ENG_TUNE_REFINE_MIN_N (2048): for the
1024, 32 and b_a 64 shapes table picks count only through the anchors. Any exception in the table path logs "lever not
used: gemm_table (...)" and S6 goes on without the table (disabled), keeping the batch lanes.

Format (schema 1, the same as engfast's)::

    {"schema": 1, "gpu": "<torch.cuda.get_device_name()>", "meta": {"cublas": "13.1.1.3", ...},
     "entries": [{"kind": "plain", "n_out": 8192, "k": 2560, "L": 192, "m": 768, "cfg": {"id": 67, ...},
                  "ms": 0.0995, "c1_ms": 0.1399},
                 {"kind": "merged", "widths": [9216, 9216], "n_out": 18432, ...}]}

s1eng key: (N, K, bucket L, packed rows R) with N the fused width for a merged block (S1ENG_FUSE: qkv_z 12288,
gate_up 18432, k_v 2048, b_a 64); "m" is R. ``c1_ms``: the time of the configuration the engine picks without the
table (s1eng's own pick for an s1 table, C1's best for an engfast table); entries less than MIN_GAIN faster than it
are not offered.

Importing this module never imports torch or loads cuBLASLt.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import json
import os
from importlib import metadata
from typing import Callable, Iterable, Sequence

SCHEMA = 1
KINDS = ("plain", "merged")
CONFIG_FIELDS = ("id", "tile", "stages", "splitk", "reduction", "swizzle", "custom", "inner_shape", "cluster_shape")
# table cfg field -> lt.CFG name (lt.cfg_key order: algoId, tile, splitK, reduction, swizzle, custom, stages, inner,
# cluster)
LT_NAMES = {"id": "algoId", "tile": "tile", "splitk": "splitK", "reduction": "reduction", "swizzle": "swizzle",
            "custom": "custom", "stages": "stages", "inner_shape": "inner", "cluster_shape": "cluster"}
LT_ORDER = ("algoId", "tile", "splitK", "reduction", "swizzle", "custom", "stages", "inner", "cluster")
TILE_UNDEFINED = STAGES_UNDEFINED = 0
REDUCTION_NONE = 0
MIN_GAIN = 0.02  # an entry is offered only if its recorded ms beats c1_ms by this share (2 %: above the noise)
DEFAULT_MAX = 32  # S1ENG_GEMM_TABLE_MAX: table configurations offered per (N, K, bucket)
PROVE_REPEATS = 2  # passes over the extra byte checks (scale 8, outlier)
SEED = 104729  # generator of the extra check inputs (its own: S6's inputs stay the ones the base job draws)
PROVENANCE = ("cublas", "driver", "torch", "cuda", "rule", "grid")
CUBLAS_WHEELS = ("nvidia-cublas", "nvidia-cublas-cu12")
SKUS = (("RTX PRO 6000", "pro6000"),)
S1_GRID = "s1v2"  # meta["grid"] of a table measured on s1eng's own (N, K, L, R) grid (spike/s1_table.py)
HERE = os.path.dirname(os.path.abspath(__file__))
# cuBLASLt types of s1eng's GEMMs (lt.py): COMPUTE_32F, scale 32F, bf16 A / B / C / D
LT_TYPES = (68, 0, 14, 14, 14, 14)


# ----------------------------------------------------------------------------------------------- configurations
def unstable_class(cfg: dict) -> str | None:
    """SEL2b exclusion rule (ltenum.unstable_class): a tile or stages attribute left UNDEFINED. All 18 unstable
    candidates of SEL2-quick were such configs (id 24, stages UNDEFINED): equal to the pin on two inputs, unequal on the
    next two. A conservative exclusion, not a proof that the others are stable: the repeated byte checks remain the
    guard."""
    if int(cfg.get("tile", TILE_UNDEFINED)) == TILE_UNDEFINED or \
            int(cfg.get("stages", STAGES_UNDEFINED)) == STAGES_UNDEFINED:
        return "undefined_attr"
    return None


def split_class(cfg: dict) -> tuple[int, int]:
    """(split-K, reduction) of a table cfg; every unsplit configuration is (1, NONE)."""
    s = int(cfg.get("splitk", 1) or 1)
    return (1, REDUCTION_NONE) if s <= 1 else (s, int(cfg.get("reduction", 0)))


def pin_class(rc: dict) -> tuple[int, int]:
    """The same for an lt.cfg_of dict (the pin's)."""
    return split_class({"splitk": rc.get("splitK", 1), "reduction": rc.get("reduction", 0)})


def lt_key(cfg: dict) -> tuple:
    """lt.cfg_key of the algo a table cfg builds (the attributes it sets, in lt.CFG order)."""
    by = {LT_NAMES[f]: int(cfg[f]) for f in CONFIG_FIELDS}
    return tuple(by[n] for n in LT_ORDER)


def config_key(cfg: dict) -> tuple:
    return tuple(int(cfg[f]) for f in CONFIG_FIELDS)


# ----------------------------------------------------------------------------------------------- entries / tables
def _pos_int(e: dict, name: str) -> int:
    v = e.get(name)
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        raise ValueError(f"table entry {name} must be a positive int, got {v!r}")
    return v


def _ms(e: dict, name: str, optional: bool = False) -> None:
    v = e.get(name)
    if v is None and optional:
        return
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0:
        raise ValueError(f"table entry {name} must be a positive number, got {v!r}")


def validate_entry(e: dict) -> dict:
    if not isinstance(e, dict) or e.get("kind") not in KINDS:
        raise ValueError(f"table entry kind must be one of {KINDS}, got {e.get('kind') if isinstance(e, dict) else e!r}")
    for name in ("n_out", "k", "L", "m"):
        _pos_int(e, name)
    if e["kind"] == "merged":
        w = e.get("widths")
        if not isinstance(w, list) or len(w) < 2 or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0
                                                         for x in w) or sum(w) != e["n_out"]:
            raise ValueError(f"merged table entry widths must be >= 2 positive ints summing to n_out, got {w!r}")
    cfg = e.get("cfg")
    if not isinstance(cfg, dict) or set(cfg) != set(CONFIG_FIELDS) or any(
            isinstance(cfg[f], bool) or not isinstance(cfg[f], int) or cfg[f] < 0 for f in CONFIG_FIELDS):
        raise ValueError(f"table entry cfg must have exactly the int attributes {CONFIG_FIELDS}, got {cfg!r}")
    _ms(e, "ms")
    _ms(e, "c1_ms", optional=True)
    return e


def make_entry(kind: str, k: int, L: int, m: int, cfg: dict, ms: float, c1_ms: float | None, n_out: int | None = None,
               widths: Sequence[int] | None = None, **extra) -> dict:
    e: dict = {"kind": kind}
    if kind == "merged":
        e["widths"] = [int(w) for w in (widths or ())]
        e["n_out"] = sum(e["widths"])
    else:
        e["n_out"] = int(n_out) if n_out is not None else 0
    e.update(k=int(k), L=int(L), m=int(m))
    if not isinstance(cfg, dict) or set(cfg) < set(CONFIG_FIELDS):
        raise ValueError(f"table entry cfg must name every attribute {CONFIG_FIELDS}, got {cfg!r}")
    e["cfg"] = {f: int(cfg[f]) for f in CONFIG_FIELDS}
    e["ms"] = float(ms)
    e["c1_ms"] = None if c1_ms is None else float(c1_ms)
    e.update(extra)
    return validate_entry(e)


def entry_key(e: dict) -> tuple:
    """s1eng's key: (N, K, L, R); N = the fused width of a merged entry."""
    return (int(e["n_out"]), int(e["k"]), int(e["L"]), int(e["m"]))


def has_gain(entry: dict, min_gain: float = MIN_GAIN) -> bool:
    """Recorded at least ``min_gain`` faster than the engine's own pick at its key (or nothing to compare with)."""
    if entry.get("c1_ms") is None:
        return True
    c1 = float(entry["c1_ms"])
    return (c1 - float(entry["ms"])) >= min_gain * c1 - 1e-12


class Table:
    """Entries by s1eng key (N, K, L, R). ``excluded``: entries dropped on load (unstable class)."""

    def __init__(self, gpu: str, entries: Iterable[dict], meta: dict | None = None, excluded: int = 0) -> None:
        self.gpu, self.meta, self.excluded = gpu, dict(meta or {}), excluded
        self._by: dict[tuple, dict] = {}
        for e in entries:
            key = entry_key(validate_entry(e))
            if key in self._by:
                raise ValueError(f"duplicate table entry {key}")
            self._by[key] = e

    def __len__(self) -> int:
        return len(self._by)

    @property
    def entries(self) -> list[dict]:
        return [self._by[k] for k in sorted(self._by)]

    def pick(self, N: int, K: int, L: int, R: int, min_gain: float = MIN_GAIN) -> dict | None:
        """The entry at (N, K, L, R) if its recorded gain is at least ``min_gain``."""
        e = self._by.get((int(N), int(K), int(L), int(R)))
        return e if e is not None and has_gain(e, min_gain) else None

    def picks(self, N: int, K: int, L: int, min_gain: float = MIN_GAIN) -> dict[int, dict]:
        """R -> entry for every R of (N, K, L) with a recorded gain."""
        return {k[3]: e for k, e in sorted(self._by.items()) if k[:3] == (int(N), int(K), int(L))
                and has_gain(e, min_gain)}

    def configs(self, N: int, K: int, L: int, cls: tuple[int, int] | None = None, limit: int | None = DEFAULT_MAX,
                min_gain: float = MIN_GAIN) -> list[tuple[dict, list[int]]]:
        """Distinct cfgs of (N, K, L) with the R values they are the pick for, in the split class ``cls`` (None: any):
        the cfg serving the most R first, then the larger recorded gain, then cfg order; at most ``limit``."""
        by: dict[tuple, list] = {}
        for R, e in self.picks(N, K, L, min_gain).items():
            if cls is not None and split_class(e["cfg"]) != tuple(cls):
                continue
            ck = config_key(e["cfg"])
            ent = by.setdefault(ck, [e["cfg"], [], 0.0])
            ent[1].append(R)
            if e.get("c1_ms"):
                ent[2] = max(ent[2], 1.0 - float(e["ms"]) / float(e["c1_ms"]))
        order = sorted(by.items(), key=lambda kv: (-len(kv[1][1]), -kv[1][2], kv[0]))
        out = [(dict(v[0]), sorted(v[1])) for _, v in order]
        return out if limit is None else out[:max(0, int(limit))]


def from_dict(d: dict) -> Table:
    if not isinstance(d, dict) or d.get("schema") != SCHEMA:
        raise ValueError(f"table schema must be {SCHEMA}, got {d.get('schema') if isinstance(d, dict) else d!r}")
    if not isinstance(d.get("gpu"), str) or not d["gpu"]:
        raise ValueError("table gpu must be the GPU name it was measured on")
    entries = d.get("entries")
    if not isinstance(entries, list):
        raise ValueError("table entries must be a list")
    keep, dropped = [], 0
    for e in entries:
        validate_entry(e)
        if unstable_class(e["cfg"]):
            dropped += 1
            continue
        keep.append(e)
    return Table(d["gpu"], keep, d.get("meta") or {}, excluded=dropped)


def load(path: str) -> Table:
    with open(path) as f:
        try:
            d = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"table JSON: {exc}") from None
    return from_dict(d)


def dump(path: str, gpu: str, entries: Iterable[dict], meta: dict | None = None) -> Table:
    t = Table(gpu, entries, meta)
    with open(path, "w") as f:
        json.dump({"schema": SCHEMA, "gpu": t.gpu, "meta": t.meta, "entries": t.entries}, f, indent=1)
        f.write("\n")
    return t


def merge(tables: Sequence[Table]) -> Table:
    """One table from several runs (e.g. one per bucket) on one GPU model: per key the entry with the smaller ms. The
    runs must agree on every PROVENANCE key but the driver (kept as a sorted list)."""
    if not tables:
        raise ValueError("nothing to merge")
    gpu = tables[0].gpu
    if any(t.gpu != gpu for t in tables):
        raise ValueError(f"tables from different GPUs: {sorted({t.gpu for t in tables})}")
    meta: dict = {}
    for key in PROVENANCE:
        values = {json.dumps(t.meta.get(key), sort_keys=True) for t in tables}
        if key == "driver" and len(values) > 1:
            drv = set()
            for t in tables:
                v = t.meta.get(key)
                drv.update(v if isinstance(v, list) else ([] if v is None else [v]))
            meta[key] = sorted(drv)
            continue
        if len(values) > 1:
            raise ValueError(f"tables with different {key}: {sorted(values)}")
        if tables[0].meta.get(key) is not None:
            meta[key] = tables[0].meta[key]
    created = [t.meta["created_utc"] for t in tables if isinstance(t.meta.get("created_utc"), str)]
    if created:
        meta["created_utc"] = max(created)
    meta["merged_from"] = [t.meta for t in tables]
    best: dict[tuple, dict] = {}
    for t in tables:
        for e in t.entries:
            key = entry_key(e)
            if key not in best or e["ms"] < best[key]["ms"]:
                best[key] = e
    return Table(gpu, best.values(), meta, excluded=sum(t.excluded for t in tables))


# ----------------------------------------------------------------------------------------------- start-up guards
def sku_for(gpu: str) -> str | None:
    return next((sku for name, sku in SKUS if name in gpu), None)


def builtin_path(gpu: str, root: str | None = None) -> str | None:
    """s1eng's own table for this GPU model, else kai's engfast table (anchor candidates only), else None."""
    sku = sku_for(gpu)
    if sku is None:
        return None
    for name in (f"gemm_table_s1_{sku}.json", f"gemm_table_{sku}.json"):
        path = os.path.join(HERE if root is None else root, name)
        if os.path.isfile(path):
            return path
    return None


def runtime_cublas() -> str | None:
    for name in CUBLAS_WHEELS:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def enabled(environ) -> bool:
    return environ.get("S1ENG_GEMM_TABLE", "") not in ("", "0")


_LOOKUP = object()


def lt_version_str(v) -> str | None:
    """cublasLtGetVersion() (major * 10000 + minor * 100 + patch) as "major.minor.patch"; None if unknown."""
    if v is None:
        return None
    v = int(v)
    return f"{v // 10000}.{v // 100 % 100}.{v % 100}"


def loaded_lt_version(lib) -> int | None:
    """The version the loaded libcublasLt reports (lt.lt), or None."""
    try:
        fn = lib.cublasLtGetVersion
        fn.argtypes, fn.restype = [], C.c_size_t
        return int(fn())
    except Exception:  # noqa: BLE001 - an old or odd library: the wheel check alone decides
        return None


def from_env(environ, gpu: str, log=print, cublas=_LOOKUP, root: str | None = None, loaded=None) -> Table | None:
    """The table for this start, or None: off (no log), or a skip ("lever not used: gemm_table (<reason>)"): no
    built-in table for this GPU model, a file that cannot be read, another GPU model, another cuBLAS wheel (unknown on
    either side is a skip too: the same attributes can name another kernel in another cuBLASLt), or a loaded
    libcublasLt (``loaded``: its cublasLtGetVersion(), None = unknown, not checked) whose major.minor.patch is not the
    wheel's (the wheel metadata and the library lt.py loads can differ)."""
    if not enabled(environ):
        return None
    src = environ["S1ENG_GEMM_TABLE"]

    def skip(reason: str) -> None:
        log(f"lever not used: gemm_table ({reason})")

    path = builtin_path(gpu, root) if src == "1" else src
    if path is None:
        skip(f"no built-in table for GPU {gpu!r}")
        return None
    try:
        table = load(path)
    except (OSError, ValueError) as exc:
        skip(f"{path}: {type(exc).__name__}: {exc}")
        return None
    if table.gpu != gpu:
        skip(f"{path} was measured on {table.gpu!r}, this GPU is {gpu!r}")
        return None
    runtime = runtime_cublas() if cublas is _LOOKUP else cublas
    measured = table.meta.get("cublas")
    if runtime is None or measured is None or str(measured) != str(runtime):
        skip(f"{path} was measured with cuBLAS {measured or 'unknown'}, this process has cuBLAS {runtime or 'unknown'}")
        return None
    lv = lt_version_str(loaded)
    if lv is not None and lv != ".".join(str(runtime).split(".")[:3]):
        skip(f"the loaded libcublasLt {lv} is not the cuBLAS wheel {runtime} the table was measured with")
        return None
    try:
        table.max = int(environ.get("S1ENG_GEMM_TABLE_MAX", str(DEFAULT_MAX)))
    except ValueError:
        skip(f"S1ENG_GEMM_TABLE_MAX must be an int, not {environ.get('S1ENG_GEMM_TABLE_MAX')!r}")
        return None
    table.path = path
    log(f"gemm_table: {path} ({len(table)} entries, {table.excluded} excluded; grid {table.meta.get('grid', 'engfast')}, "
        f"cuBLAS {runtime}, loaded libcublasLt {lv or 'unknown'}, max {table.max} configs per (N, K, bucket))")
    return table


# ----------------------------------------------------------------------------------------------- S6 rules (pure)
def prove(checks: Sequence[Callable[[], bool]], repeats: int = PROVE_REPEATS) -> str:
    """Extra byte checks of a table configuration: every check, the whole sequence ``repeats`` times. "" (passed),
    "unequal" / "unequal<i>" (the first pass failed at check i), "unstable" (a later pass failed after the first
    passed), "error" (a call raised)."""
    for rep in range(max(1, repeats)):
        for i, check in enumerate(checks):
            try:
                same = bool(check())
            except Exception:  # noqa: BLE001 - the matmul refused it: not usable here
                return "error"
            if not same:
                if rep > 0:
                    return "unstable"
                return "unequal" if i == 0 else f"unequal{i + 1}"
    return ""


def group_keys(table_rs: dict, rs: Iterable[int], fallback_all: bool = True) -> set:
    """Table-only candidates timed at an anchor whose group serves ``rs``: those that are the table's pick at some R
    of the group. When none is: all of them if ``fallback_all`` (a table measured on another row grid: kai's engfast
    table), else none (an s1v2 table has no pick there because s1eng's own pick was fastest or the pick failed)."""
    rs = set(rs)
    hit = {k for k, picked in table_rs.items() if rs & set(picked)}
    return hit if hit or not fallback_all else set(table_rs)


def anchor_fallback(table: Table) -> bool:
    """group_keys' fallback_all for this table: True unless it was measured on s1eng's own grid."""
    return table.meta.get("grid") != S1_GRID


def ws_cap(ws_bytes: int, ws_b_bytes: int | None) -> int:
    """The workspace a table configuration may need: S6 runs it on ``ws`` and, with pair timing, on ``ws_b``."""
    return ws_bytes if ws_b_bytes is None else min(ws_bytes, ws_b_bytes)


def fits_group(rs: Iterable[int], need: Callable[[int], int | None], eq: Callable[[int], bool], cap: int) -> str:
    """Before a table configuration enters an anchor's contest: at every R of the group its workspace (``need(R)``,
    None = AlgoCheck refused) fits ``cap`` and its output equals the per-decision reference (``eq(R)``). So a table
    winner never fails S6's re-check at another R (which would put the whole group back on the pin). "" or
    "<reason>@R"."""
    for r in rs:
        n = need(r)
        if n is None:
            return f"algocheck@{r}"
        if n > cap:
            return f"workspace@{r}"
        try:
            same = bool(eq(r))
        except Exception:  # noqa: BLE001 - the matmul refused it at this R
            return f"error@{r}"
        if not same:
            return f"unequal@{r}"
    return ""


def disabled(table: Table, reason: str) -> Table:
    """An empty stand-in after the table path raised: S6 goes on with its own candidates only (no table configuration
    is offered from then on), and the reason is logged and kept."""
    d = Table(table.gpu, [], table.meta, excluded=table.excluded)
    d.max = getattr(table, "max", DEFAULT_MAX)
    d.path = getattr(table, "path", None)
    d.failed = reason
    return d


def selection_digest(algo_r: dict, fuse_r: dict, swiglu_r: dict) -> str:
    """12 hex digits naming one start-up's timing-based picks: S6 (algo_r: (N, K, L, R) -> lt.cfg_key), fused or parts
    (fuse_r) and SG (swiglu_r: (L, R) -> kernel). Start-ups with the same digest run the same GEMM configurations."""
    def sg(v):
        return [getattr(v, "name", repr(v)), getattr(v, "drop", None)]
    doc = {"algo_r": sorted([list(k), list(v)] for k, v in algo_r.items()),
           "fuse_r": sorted([list(k), bool(v)] for k, v in fuse_r.items()),
           "swiglu_r": sorted([list(k), sg(v)] for k, v in swiglu_r.items())}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()[:12]


def with_pick(top: list, pick, exact) -> list:
    """Refinement candidates at one R: the anchor's top (plus the pin) and the table's pick for that R, if it passed
    every check (it is in ``exact``) and is not there yet."""
    if pick is None or pick in top or pick not in exact:
        return list(top)
    return list(top) + [pick]


def kind_of(key, kref, kinds: dict) -> str:
    """table / ng / heur / pin for a winner's key."""
    if key == kref:
        return "pin"
    return kinds.get(key, "heur")


def winners_line(shape: str, per_r: dict) -> str:
    """One log line per (N, K, bucket): 'R96:table R128:ng ...' (R ascending)."""
    return f"  S6 winners {shape}: " + " ".join(f"R{r}:{per_r[r]}" for r in sorted(per_r))


def tally(kinds: Iterable[str]) -> dict:
    out = {"table": 0, "ng": 0, "heur": 0, "pin": 0}
    for k in kinds:
        out[k] = out.get(k, 0) + 1
    return out


# ----------------------------------------------------------------------------------------------- cuBLASLt (GPU side)
def build(ltmod, handle, cfg: dict):
    """The cuBLASLt algo of a table cfg (ltenum.LtEnumApi.build): cublasLtMatmulAlgoInit for its id with s1eng's GEMM
    types, then every other attribute set in lt.CFG order; None if cuBLASLt refuses one. ``ltmod``: s1eng.lt;
    ``handle``: an Lt context's handle."""
    lib = ltmod.lt
    if getattr(lib.cublasLtMatmulAlgoInit, "argtypes", None) is None:
        lib.cublasLtMatmulAlgoInit.argtypes = [C.c_void_p] + [C.c_int] * 6 + [C.c_int, C.POINTER(ltmod.Algo)]
        lib.cublasLtMatmulAlgoInit.restype = C.c_int
    algo = ltmod.Algo()
    if lib.cublasLtMatmulAlgoInit(handle, *LT_TYPES, int(cfg["id"]), C.byref(algo)) != 0:
        return None
    by = {LT_NAMES[f]: int(cfg[f]) for f in CONFIG_FIELDS}
    for name in LT_ORDER[1:]:   # the attribute order of ltenum.LtEnumApi.build (ltpin.CONFIG_ATTRS)
        attr, ctype = ltmod.CFG[name]
        v = ctype(by[name])
        if lib.cublasLtMatmulAlgoConfigSetAttribute(C.byref(algo), attr, C.byref(v), C.sizeof(v)) != 0:
            return None
    return algo
