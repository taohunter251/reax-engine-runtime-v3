from contextlib import contextmanager
import copy
import ctypes
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
IMAGE = 'ghcr.io/johnkell9801-hub/s1eng@sha256:f3f647041d0c91ced167603e27df0f5ca518c5c9c7f6ee6f746567e1ed8ae606'

def load_public(directory):
    name = 'reax_public'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, directory/'__init__.py',
                                                  submodule_search_locations=[str(directory)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    for module in ('ng', 'lt', 'plan', 'fused', 'batchfwd', 'kprio'):
        importlib.import_module(name+'.'+module)
    return package


def clone_algorithm(value, package):
    if value is None:
        return None
    if hasattr(value, 'name') and hasattr(value, 'split'):
        return package.ng.NgAlgo(value.name, value.drop, value.chunk, value.split, value.red)
    result = package.lt.Algo()
    assert ctypes.sizeof(result) == ctypes.sizeof(value)
    ctypes.memmove(ctypes.byref(result), ctypes.byref(value), ctypes.sizeof(result))
    return result


class FullCore:
    def __init__(self, gpu, assets):
        self.original = gpu.m.pf
        directory = Path(__file__).resolve().parents[1]/'vendor'/'s1eng'
        self.package = package = load_public(directory)
        self.source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in directory.iterdir() if p.is_file()}
        self.context = context = package.lt.Lt()
        pinned = package.lt.Pinned(context)
        for key, value in vars(self.original.pinned).items():
            if key == 'ctx':
                continue
            if key in ('algo', 'algo_r'):
                value = {shape: clone_algorithm(algorithm, package) for shape, algorithm in value.items()}
            elif isinstance(value, dict):
                value = dict(value)
            setattr(pinned, key, value)
        pinned.ctx = context
        pinned._ws = torch.empty(max(pinned.ws_bytes, 64 << 20), device='cuda', dtype=torch.uint8)
        pinned._gen = torch.Generator(device='cuda').manual_seed(1259)
        weights, fused_info = {}, {}
        for index, layer in enumerate(self.original.ff.layers):
            for kind in ('mlp', 'la', 'sa'):
                if kind not in layer:
                    continue
                for module in layer[kind].modules():
                    weight = getattr(module, 'weight', None)
                    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                        weights.setdefault(tuple(weight.shape), []).append(weight)
            for name, weight in layer['fused'].items():
                weights.setdefault(tuple(weight.shape), []).append(weight)
                if name == 'gate_up':
                    entry = fused_info.setdefault(tuple(weight.shape), ('gate_up', len(weight)//2, len(weight)//2, []))
                    entry[3].append((index, weight))
        rows = {}
        for lane in gpu.lanes:
            for (length, _), entries in lane.packed.items():
                rows.setdefault(length, set()).update(entry[0] for entry in entries)
        groups = lambda length: {row: [row] for row in sorted(rows.get(length, set()))}
        plan = package.plan.load('1', log=lambda value: print('FULL_CORE '+value, flush=True))
        assert plan is not None, 'Published plan must match this Server GPU and dependency stack'
        package.plan.apply_s6(plan, pinned, groups, weights,
                              log=lambda value: print('FULL_CORE '+value, flush=True))
        pinned.fuse_r = {(n, k, length, count): bool(value) for n, k, length, count, value in plan['fuse_r']}
        pinned.swiglu_r = {}
        package.plan.apply_sg(plan, pinned, package.fused, fused_info, groups, sorted(rows),
                              log=lambda value: print('FULL_CORE '+value, flush=True))
        core_bf = copy.copy(self.original.bf)
        core_bf.ctx, core_bf.pinned, core_bf.fz = context, pinned, package.fused
        core = package.batchfwd.PackedForward(core_bf)
        for key, value in vars(self.original).items():
            if key not in ('bf', 'ctx', 'pinned', 'fz') and not callable(value):
                setattr(core, key, dict(value) if key in ('_ar', '_sides') else value)
        core.fz, core.ctx, core.pinned = package.fused, context, pinned
        core.pron = set(assets.manifest['full_public_core']['pron_buckets'])
        core.xa = set(assets.manifest['full_public_core']['attention_buckets'])
        core.xa_fq, core.xa_gtab = True, None
        core.par, core._sides = {'ba', 'q', 'last', 'gu', 'z'}, {}
        core.ws_side = None
        self.core, self.workspaces, self.side_workspaces = core, {}, {}
        self.assets = assets
        self.native_kkt = importlib.import_module(package.__name__+'.kkt2n').Kkt2n()
        importlib.import_module(package.__name__+'.fla_fused').KKT2['nat'] = self.native_kkt
        self.gemm_marks = package.kprio.GEMM_MARKS
        core.rarn = package.rarn.select(pinned, context, weights, rows,
                                       core.ff.layers[0]['ln2_w'], core.ff.eps,
                                       package.rarn.rule_from_env(),
                                       log=lambda value: print('FULL_CORE '+value, flush=True))
        pinned._ws = None
        self.plan = {'image': IMAGE, 'source_sha256': self.source_hashes,
                     's6_picks': len(pinned.algo_r), 'swiglu_picks': len(pinned.swiglu_r),
                     'rarn_picks': len(core.rarn), 'parallel_sites': sorted(core.par),
                     'pron_buckets': sorted(core.pron), 'attention_buckets': sorted(core.xa),
                     'workspace_bytes': pinned.ws_bytes,
                     'attribution': 'Public UID164 reference implementation and binaries; no independent authorship claim.',
                     'original_UID56_plans_and_captures_modified': False}
        print('FULL_CORE_READY '+json.dumps({k: v for k, v in self.plan.items() if k != 'source_sha256'}), flush=True)

    @contextmanager
    def workspace(self):
        source = self.original.bf.ws
        key = source.data_ptr()
        count = max(source.numel(), self.core.pinned.ws_bytes, 1 << 20)
        if key not in self.workspaces:
            self.workspaces[key] = torch.empty(count, device='cuda', dtype=torch.uint8)
            self.side_workspaces[key] = torch.empty(count, device='cuda', dtype=torch.uint8)
        previous = self.core.bf.ws, self.core.ws_side
        self.core.bf.ws, self.core.ws_side = self.workspaces[key], self.side_workspaces[key]
        try:
            yield self.core
        finally:
            self.core.bf.ws, self.core.ws_side = previous

    def __call__(self, meta):
        with self.workspace() as core:
            return core(meta)


class GateUp:
    def __init__(self, full, first=12):
        self.full, self.first = full, first
        self.retained = {'indices': {id(layer): i for i,layer in enumerate(full.core.ff.layers)},
                         'choices': full.assets.payload['gate_up_routes'], 'native': full.assets.gate_up}
        self.bank = full.assets.gate_up_banks

    def mlp(self, layer, x, length, approximate=False, defer=False):
        core, retained = self.full.core, self.retained
        index = retained['indices'][id(layer)]
        if not approximate or index < self.first or index not in self.bank:
            return core._mlp(layer, x, length, defer=defer)
        choices = retained['choices']
        key = min((key for key in choices if key[0] == 128),
                  key=lambda key: (key[1] != length, abs(key[2]-len(x))))
        route = choices[key]
        hidden = retained['native'].project(x, self.bank[index], route['kernel'], route['drop'])
        return (core.lin_res if defer else core.lin)(layer['mlp'].down_proj, hidden, length)


class Checkpoint:
    def __init__(self, full, first=12, start=12):
        self.full, self.start, self.gup = full, start, GateUp(full, first)
        self.gemm_marks = full.gemm_marks

    def prefix(self, meta):
        core = self.full.core
        ff, z = core.ff, core.fz
        mask, cos, sin = ff.consts[meta.L]
        h = ff.tm.embed_tokens(meta.ids)
        pending = None
        lookup = core.l0t is not None and core.rec is None and meta.L in core.l0t.books
        for index in range(self.start):
            layer = ff.layers[index]
            if pending is None:
                x = None if lookup else z.rmsnorm(h, layer['ln1_w'], ff.eps)
            else:
                h, x = core.add_norm(h, pending, layer['ln1_w'], ff.eps)
            mixed = (core._gdn(layer, x, meta, index) if layer['type'] == 'linear_attention' else
                     core._attn(layer, x, meta, mask, cos, sin, index))
            h, x = core.add_norm(h, mixed, layer['ln2_w'], ff.eps)
            pending = self.gup.mlp(layer, x, meta.L, False, True)
        return core.add_norm(h, pending, ff.layers[self.start]['ln1_w'], ff.eps)

    def _suffix(self, meta, h, x, approximate=False):
        core = self.full.core
        ff, z = core.ff, core.fz
        mask, cos, sin = ff.consts[meta.L]
        for index in range(self.start, 32):
            layer = ff.layers[index]
            if index > self.start:
                h, x = core.add_norm(h, pending, layer['ln1_w'], ff.eps)
            if index == 31:
                selected = x.index_select(0, meta.last).contiguous()
                positions = meta.row_pos.index_select(0, meta.last).contiguous()
                mixed = core._attn_last(layer, x, selected, meta, mask, cos, sin, positions, index)
                final = h.index_select(0, meta.last).contiguous()
                final, selected = z.add_rmsnorm(final, mixed, layer['ln2_w'], ff.eps)
                pending = self.gup.mlp(layer, selected, meta.L, approximate, False)
                _, normalized = z.add_rmsnorm(final, pending, ff.fn_w, ff.eps)
                return torch.cat([normalized[b:b+1]@ff.slot_weight.T for b in range(meta.B)])
            mixed = (core._gdn(layer, x, meta, index) if layer['type'] == 'linear_attention' else
                     core._attn(layer, x, meta, mask, cos, sin, index))
            h, x = core.add_norm(h, mixed, layer['ln2_w'], ff.eps)
            pending = self.gup.mlp(layer, x, meta.L, approximate, True)

    def suffix(self, meta, h, x, approximate=False):
        with self.full.workspace():
            return self._suffix(meta, h, x, approximate)

    def __call__(self, meta, approximate=False):
        with self.full.workspace():
            checkpoint = self.prefix(meta)
            return self._suffix(meta, *checkpoint, approximate), checkpoint

    def approximate(self, meta):
        return self(meta, True)[0]

    def exact(self, meta):
        return self(meta, False)[0]

