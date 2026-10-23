"""Attach only profitable, original-arithmetic projection schedules to a local model."""
from contextlib import contextmanager
import statistics
from local_fold_projection import plan
class Candidate:
 def __init__(self,inner,state):
  self.inner=inner;self.native=state['native'];self.routes={};groups={}
  for row in state['operators']:
   if not row['all_bits_equal']:continue
   cfg=row['plan'];key=tuple(row['shape']),row['length'],row['rows'],cfg['parts'],cfg['chunk'],cfg['mode'],row['kernel'];groups.setdefault(key,[]).append(row)
  for key,rows in groups.items():
   hot=statistics.median(r['speedup'] for r in rows);cold=statistics.median(r['cold_speedup'] for r in rows)
   if min(hot,cold)<1.05:continue
   schedule=key[:-1];value=dict(kernel=key[-1],hot=hot,cold=cold)
   if schedule not in self.routes or min(hot,cold)>min(self.routes[schedule]['hot'],self.routes[schedule]['cold']):self.routes[schedule]=value
  self.gemm_marks=(*inner.gemm_marks,'local_fold_','wave_fold_')
 def __getattr__(self,name):return getattr(object.__getattribute__(self,'inner'),name)
 @contextmanager
 def hooks(self):
  core=self.inner.full.core;old=core.gemm
  def gemm(w,x,L):
   shape=tuple(w.shape)
   if shape not in ((2048,2560),):return old(w,x,L)
   cfg=plan(core,shape,L,len(x));key=shape,L,len(x),cfg['parts'],cfg['chunk'],cfg['mode'];route=self.routes.get(key)
   return old(w,x,L) if route is None else self.native.product(x,w,cfg['chunk'],route['kernel'])
  core.gemm=gemm
  try:yield
  finally:core.gemm=old
 def __call__(self,meta,approximate=False):
  with self.hooks():return self.inner(meta,approximate)
 def suffix(self,meta,h,x,approximate=False):return self.inner.suffix(meta,h,x,approximate)
