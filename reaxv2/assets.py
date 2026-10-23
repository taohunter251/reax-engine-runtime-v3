"""Load the frozen, inference-only parameter representations."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch

from .driver import NativeInt8
from .gate_up import Native as GateUpNative
from .sliced import Native as SlicedNative
from .rotation import Native as RotationNative


class Assets:
    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = json.loads((directory/'manifest.json').read_text())
        def verified(entry):
            path = directory/entry['path']
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != entry['sha256']:
                raise ValueError('Inference asset checksum failed: '+entry['path'])
            return path
        path = verified(self.manifest['parameters'])
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if payload['schema'] != 1 or payload['checkpoint_start'] != 12:
            raise ValueError('Unsupported frozen projection schema')
        self.payload = payload
        def banks(entries):
            return {key: SimpleNamespace(**{
                name: value.to(device='cuda') if isinstance(value,torch.Tensor) else value
                for name,value in fields.items()}) for key,fields in entries.items()}
        self.gate_up_banks = banks(payload.pop('gate_up'))
        self.projections = banks(payload.pop('projections'))
        self.gate_up = GateUpNative(verified(self.manifest['gate_up_kernels']))
        self.sliced = SlicedNative(verified(self.manifest['sliced_projection_kernels']))
        self.rotated = {}
        for threads, config in self.manifest['rotated_projection_kernels'].items():
            # Each wrapper shares matrix handles but selects its own prepare shape.
            verified(config['matrix'])
            prepare = NativeInt8(verified(config['prepare']))
            self.rotated[int(threads)] = RotationNative(self.sliced,prepare,int(threads))

    def bind(self, core):
        self.policy = Policy(core, self)


class Policy:
    def __init__(self, core, assets):
        self.assets = assets
        self.indices = {id(layer):i for i,layer in enumerate(core.ff.layers)}
        self.source = {'chunk_for': assets.payload['chunk_for']}
        self.weights, self.by_pointer = {}, {}
        for index, layer in enumerate(core.ff.layers):
            if not 12 <= index < 31:
                continue
            self.weights[index,'down'] = layer['mlp'].down_proj.weight
            self.by_pointer[layer['mlp'].down_proj.weight.data_ptr()] = index,'down'
            if layer['type'] == 'linear_attention':
                self.weights[index,'gdn_input'] = layer['fused']['qkv_z']
                self.weights[index,'gdn_output'] = layer['la'].out_proj.weight
                self.by_pointer[layer['la'].out_proj.weight.data_ptr()] = index,'gdn_output'

    def project(self, index, family, x, length):
        data, assets = self.assets.payload, self.assets
        chunk = data['chunk_for'].get((index,family,length))
        if chunk is None:
            return None
        projection = assets.projections[index,family,chunk]
        if family == 'gdn_input':
            table = data['projection_routes']
            choices = [key for key in table if key[0] == family and key[3] == 128]
            key = min(choices,key=lambda key:(key[1] != length,abs(key[2]-len(x))))
            route = table[key]
            return assets.sliced.project(x,projection,route['kernel'],route['drop'])
        table = data['rotation_routes']
        choices = [key for key in table if key[0] == family and key[3] == data['rotation_cap']]
        key = min(choices,key=lambda key:(key[1] != length,abs(key[2]-len(x))))
        route = table[key]
        threads = min(route['threads'],projection.block)
        return assets.rotated[threads].project(x,projection,route['kernel'],route['drop'])
