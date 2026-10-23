"""Substitute the exact BF16 activation table while retaining frozen GEMM plans."""
from contextlib import contextmanager
class Candidate:
 def __init__(self,inner,native):self.inner=inner;self.native=native;self.gemm_marks=(*inner.gemm_marks,'prefix_table_')
 def __getattr__(self,name):return getattr(object.__getattribute__(self,'inner'),name)
 @contextmanager
 def hooks(self):
  core=self.full.core;old=core._mlp
  def mlp(layer,x,length,defer=False):
   algo=core.pinned.swiglu_r.get((length,len(x)));name=self.native.aliases.get(getattr(algo,'name',None));block=layer['fused'].get('gate_up')
   if name is None or block is None:return old(layer,x,length,defer=defer)
   hidden=self.native.product(x,block,name,algo.drop)
   return (core.lin_res if defer else core.lin)(layer['mlp'].down_proj,hidden,length)
  core._mlp=mlp
  try:yield
  finally:core._mlp=old
 def __call__(self,meta,approximate=False):
  with self.hooks():return self.inner(meta,approximate)
 def suffix(self,meta,h,x,approximate=False):
  with self.hooks():return self.inner.suffix(meta,h,x,approximate)
