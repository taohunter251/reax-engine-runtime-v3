"""Original BF16 per-slice rounding and ordered fold with no global partial tensor."""
from pathlib import Path
import ctypes as C
import json
import torch
from reaxv2.driver import NativeInt8
class Native(NativeInt8):
 def __init__(self):
  super().__init__(Path(__file__).with_name('local_fold.cubin'));self.variants=json.loads(Path(__file__).with_name('fold_shapes.json').read_text())
 def product(self,x,w,chunk,name):
  m,k=x.shape;n=len(w);bm,bn,threads=self.variants[name]
  assert x.is_contiguous() and w.is_contiguous() and k%64==0 and chunk%64==0 and n%bn==0
  if name not in self.functions:
   fn=C.c_void_p();self.check(self.cuda.cuModuleGetFunction(C.byref(fn),self.module,name.encode()));ptr=C.c_void_p();size=C.c_size_t();shared=C.c_uint();self.check(self.cuda.cuModuleGetGlobal_v2(C.byref(ptr),C.byref(size),self.module,(name+'_required_smem').encode()));self.check(self.cuda.cuMemcpyDtoH_v2(C.byref(shared),ptr,4));self.check(self.cuda.cuFuncSetAttribute(fn,8,shared.value));self.functions[name]=fn,shared.value
  fn,shared=self.functions[name];out=torch.empty((m,n),device=x.device,dtype=x.dtype);args=[C.c_void_p(t.data_ptr()) for t in (x,w,out)]+[C.c_int(v) for v in (m,n,k,chunk)];params=(C.c_void_p*len(args))(*[C.cast(C.pointer(v),C.c_void_p) for v in args]);self.check(self.cuda.cuLaunchKernel(fn,((m+bm-1)//bm)*(n//bn),1,1,threads,1,1,shared,C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None));return out

def plan(core,shape,L,rows):
 a=core.pinned.algo_r.get((*shape,L,rows),core.pinned.algo[(*shape,L)])
 if hasattr(a,'split'):return dict(parts=max(1,a.split),chunk=a.chunk if a.split>1 else shape[1],mode=a.red)
 from reax_public.lt import cfg_of
 c=cfg_of(a);p=max(1,c.get('splitK',1));return dict(parts=p,chunk=((shape[1]+p-1)//p+63)//64*64,mode=c.get('reduction',0))
