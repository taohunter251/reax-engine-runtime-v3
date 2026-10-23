"""Restore the last qualified local model from its saved construction record."""
import copy
import ast
import ctypes as C
import hashlib
import importlib.util
import json
import pickle
import sys
import statistics
import types
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path('/opt/reax-v3/private')
CHECKPOINT = ROOT / 'output/maintenance-export-before-validator-benchmark-r1'


def load_file(path, tag):
    path = Path(path)
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(tag, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[tag] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path.parent))


def main(ctx, output):
    import s1eng
    import s1eng.lt
    sys.modules['reax_public'] = s1eng
    sys.modules['reax_public.lt'] = s1eng.lt
    from reaxv2.driver import NativeInt8

    data = pickle.loads((CHECKPOINT / 'restore.pkl').read_bytes())
    def restore_native(spec, tag):
        source = spec['source']
        if Path(source).name == 'wide_warp_hybrid.py':
            width = len(next(iter(spec['attrs']['variants'].values())))
            family = {4: 'model-preaddressed-GUP-r1', 5: 'model-preaddressed-GUP-tail-bands-r2'}[width]
            source = ROOT / 'snapshots' / family / 'wide_warp_hybrid.py'
        if Path(source).name.endswith('_screen.py'):
            tree = ast.parse(Path(source).read_text())
            tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef)
                         and n.name == spec['class_name']]
            module = types.ModuleType('restore_e502_' + tag)
            module.__dict__.update(torch=torch, statistics=statistics)
            exec(compile(tree, str(source), 'exec'), module.__dict__)
        else:
            module = load_file(source, 'restore_e502_' + tag)
        value = object.__new__(getattr(module, spec['class_name']))
        if 'binary' in spec:
            binary = Path(spec['binary'])
            assert hashlib.sha256(binary.read_bytes()).hexdigest() == spec['sha256']
            NativeInt8.__init__(value, binary)
        for key, item in spec['attrs'].items():
            if key == 'natives':
                item = {k: restore_native(v, tag + '_' + k) for k, v in item.items()}
            setattr(value, key, item)
        return value

    geometry_source = Path(data['geometry_source'])
    geometry_tree = ast.parse(geometry_source.read_text())
    geometry_tree.body = [node for node in geometry_tree.body
                          if isinstance(node, ast.ClassDef) and node.name == data['geometry_class']]
    geometry_module = types.ModuleType('restore_e502_geometry')
    geometry_module.__dict__.update(statistics=statistics)
    exec(compile(geometry_tree, str(geometry_source), 'exec'), geometry_module.__dict__)
    geometry = object.__new__(getattr(geometry_module, data['geometry_class']))
    geometry.original = ctx.assets.gate_up
    geometry.routes = data['routes']
    geometry.by_route = data['by_route']
    geometry.native = {k: restore_native(v, k) for k, v in data['natives'].items()}
    inner = copy.copy(ctx.model)
    inner.gup = copy.copy(ctx.model.gup)
    inner.gup.retained = dict(ctx.model.gup.retained, native=geometry)
    inner.gup.mlp = type(inner.gup).mlp.__get__(inner.gup, type(inner.gup))
    inner.gemm_marks = data['gemm_marks']
    ctx.retained['exact_pipeline_holdout'] = dict(model=inner)
    ctx.retained['preaddressed_sliced_packed'] = dict(
        native=restore_native(data['sliced'], 'sliced'), operators=data['sliced_operators'])

    from exact_local_base import model as base_model
    model = base_model(ctx)
    metadata = pickle.loads((CHECKPOINT / 'retained-construction-metadata.pkl').read_bytes())
    index = {}
    def index_nodes(value):
        if isinstance(value, dict):
            if 'object' in value:
                index[value['object']] = value
            for child in value.values():
                index_nodes(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                index_nodes(child)
    index_nodes(metadata)

    def ref(value):
        return index[value['object_reference']] if isinstance(value, dict) and 'object_reference' in value else value

    def plain(value):
        value = ref(value)
        if not isinstance(value, dict):
            return value
        if 'dict' in value:
            return {plain(k): plain(v) for k, v in value['dict']}
        if 'values' in value:
            items = [plain(v) for v in value['values']]
            return tuple(items) if value['sequence_type'] == 'tuple' else items
        raise ValueError('Expected CPU configuration, found ' + str(value)[:300])

    outer = ref(dict(metadata['n64_extended_holdout']['dict'])['model'])
    chain = []
    node = outer
    while 'inner' in node.get('attrs', {}):
        chain.append(node)
        node = ref(node['attrs']['inner'])
    small_tensors = torch.load(CHECKPOINT / 'small-configuration-tensors.pt', weights_only=True)
    constructions = []
    for number, node in enumerate(reversed(chain)):
        native_record = ref(node['attrs']['native'])
        module = load_file(native_record['source'], 'restore_e502_native_' + str(number))
        native = object.__new__(module.Native)
        NativeInt8.__init__(native, CHECKPOINT / (native_record['binary_sha256'] + '.cubin'))
        for key, record in native_record.get('attrs', {}).items():
            value = ref(record)
            if key == 'configs':
                native.configs = {}
            elif isinstance(value, dict) and 'tensor' in value:
                setattr(native, key, small_tensors[value['tensor']].to('cuda'))
            else:
                setattr(native, key, plain(value))
        binary_hash = hashlib.sha256(native.image.raw[:-1]).hexdigest()
        assert binary_hash == native_record['binary_sha256'], (node['type'], binary_hash)
        kind = node['type']
        if kind == 'exact_local_extensions.Candidate':
            from exact_local_extensions import Candidate
        elif kind == 'prefix_table_model.Candidate':
            from prefix_table_model import Candidate
            fn = C.c_void_p()
            native.check(native.cuda.cuModuleGetFunction(C.byref(fn), native.module, b'initialize_silu_table'))
            native.check(native.cuda.cuLaunchKernel(fn, 256, 1, 1, 256, 1, 1, 0,
                         C.c_void_p(torch.cuda.current_stream().cuda_stream), None, None))
            torch.cuda.synchronize()
            assert native.domain_check()['all_bits_equal']
        elif kind == 'n64_parts_model.Candidate':
            from n64_parts_model import Candidate
        else:
            raise ValueError(kind)
        wrapper = object.__new__(Candidate)
        wrapper.inner, wrapper.native = model, native
        wrapper.gemm_marks = plain(node['attrs']['gemm_marks'])
        if 'routes' in node['attrs']:
            wrapper.routes = plain(node['attrs']['routes'])
        if kind == 'n64_parts_model.Candidate':
            wrapper.force_pairs = node['attrs']['force_pairs']
            wrapper.usage = defaultdict(int)
        model = wrapper
        constructions.append(dict(kind=kind, binary_sha256=binary_hash,
                                  routes=len(getattr(wrapper, 'routes', {}))))

    ctx.retained['deploy_model'] = model
    return model
