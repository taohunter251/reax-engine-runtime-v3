"""Preserve each origin's BF16 split/round/fold contract in a mixed cohort."""
import ctypes as C
from pathlib import Path
import torch,triton
import triton.language as tl
from reaxv2.driver import NativeInt8

@triton.jit
def fold(parts,split,out,M:tl.constexpr,N:tl.constexpr,P0:tl.constexpr,P1:tl.constexpr,B:tl.constexpr):
 idx=tl.program_id(0)*B+tl.arange(0,B);valid=idx<M*N;row=idx//N
 border=tl.load(split);count=tl.where(row<border,P0,P1)
 value=tl.load(parts+idx,mask=valid,other=0.).to(tl.float32)
 for p in tl.static_range(1,max(P0,P1)):
  active=p<count
  term=tl.load(parts+p*M*N+idx,mask=valid&active,other=0.).to(tl.float32)
  value=tl.where(active,value+term,value)
 tl.store(out+idx,value.to(tl.bfloat16),mask=valid)

def reduction(core,shape,L,rows):
 a=core.pinned.algo_r.get((*shape,L,rows),core.pinned.algo[(*shape,L)])
 cfg=core.pinned.ctx.cfg_of(a) if hasattr(core.pinned.ctx,'cfg_of') else None
 if hasattr(a,'split'):
  p=max(1,a.split);mode=a.red;chunk=a.chunk
 else:
  import reax_public.lt as lt
  cfg=lt.cfg_of(a);p=max(1,cfg.get('splitK',1));mode=cfg.get('reduction',0);chunk=((shape[1]+p-1)//p+63)//64*64
 assert mode in (0,4),(shape,L,rows,mode)
 if p==1:chunk=shape[1]
 return dict(parts=p,chunk=chunk,mode=mode)

class Native(NativeInt8):
 shapes={'origin_m64n128':(64,128,128),'origin_m128n128':(128,128,256),'origin_m64n256':(64,256,256),'origin_m32n32':(32,32,64)}
 def __init__(self):super().__init__(Path(__file__).with_name('mixed_origin_bf16.cubin'))
 def partials(self,x,w,split,left,right,name='origin_m64n128'):
  m,k=x.shape;n=len(w);bm,bn,threads=self.shapes[name];p0,p1=left['parts'],right['parts']
  assert n%bn==0 and k%64==0 and x.is_contiguous() and w.is_contiguous()
  if name not in self.functions:
   f=C.c_void_p();self.check(self.cuda.cuModuleGetFunction(C.byref(f),self.module,name.encode()));ptr=C.c_void_p();size=C.c_size_t();sm=C.c_uint()
   self.check(self.cuda.cuModuleGetGlobal_v2(C.byref(ptr),C.byref(size),self.module,(name+'_required_smem').encode()));self.check(self.cuda.cuMemcpyDtoH_v2(C.byref(sm),ptr,4));self.check(self.cuda.cuFuncSetAttribute(f,8,sm.value));self.functions[name]=f,sm.value
  fn,sm=self.functions[name];out=torch.empty((max(p0,p1),m,n),device=x.device,dtype=torch.bfloat16)
  args=[C.c_void_p(t.data_ptr()) for t in (x,w,split,out)]+[C.c_int(v) for v in (m,n,k,left['chunk'],right['chunk'],p0,p1)];params=(C.c_void_p*len(args))(*[C.cast(C.pointer(v),C.c_void_p) for v in args])
  grid=((m+bm-1)//bm+1)*(n//bn)
  self.check(self.cuda.cuLaunchKernel(fn,grid,1,max(p0,p1),threads,1,1,sm,C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None))
  return out
 def project(self,x,w,split,left,right,name='origin_m64n128'):
  parts=self.partials(x,w,split,left,right,name);m,n=parts.shape[1:];out=torch.empty((m,n),device=x.device,dtype=x.dtype)
  fold[(triton.cdiv(m*n,512),)](parts,split,out,M=m,N=n,P0=left['parts'],P1=right['parts'],B=512,num_warps=4,enable_fp_fusion=False)
  return out
