"""Rebuild the unchanged-precision local component checkpoint from immutable artifacts."""
from pathlib import Path
import copy,importlib.util,sys
from exact_addressed_projection_model import ProjectionGeometry,router
from split_gated_projection import SplitRouter

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def model(ctx):
 retained=ctx.retained.get('exact_local_base')
 if retained:return retained['model']
 original=ctx.retained['exact_pipeline_holdout']['model'];result=copy.copy(original);result.gup=copy.copy(original.gup)
 geometry=copy.copy(original.gup.retained['native']);tables={};root=Path('/opt/reax-v3/private/snapshots')
 for family,binary in (('bulk','addressed_gate_up.cubin'),('tail','band_gate_up.cubin')):
  path=root/f'model-addressed-{family}-activation-table-r1';module=load('local_exact_table_'+family,path/'wide_warp_hybrid.py');tables[family]=module.Native(path/binary)
 extended=object.__new__(type(geometry.native['extended']));extended.__dict__.update(geometry.native['extended'].__dict__);extended.natives=tables
 geometry.native=dict(tables,extended=extended)
 result.gup.retained=dict(original.gup.retained,native=geometry);result.gup.mlp=type(result.gup).mlp.__get__(result.gup,type(result.gup))
 projections=ProjectionGeometry(ctx.retained['preaddressed_sliced_packed']);result.router=SplitRouter(router(ctx.model.router,projections))
 result.gemm_marks=(*result.gemm_marks,'addressed_','scheduled_','band_k','prefix_single_','local_fold_')
 ctx.retained['exact_local_base']=dict(model=result,tables=tables,geometry=geometry,projections=projections)
 return result
