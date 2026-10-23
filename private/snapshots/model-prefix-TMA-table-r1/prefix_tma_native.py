"""BF16 model prefix matrices loaded by TMA, preserving original MMA order."""
from pathlib import Path
import ctypes as C,json,math
import torch
from reaxv2.driver import NativeInt8
class TensorMap:
 def __init__(self,cuda,tensor,rows):
  assert tensor.is_contiguous() and tensor.dtype==torch.bfloat16
  height,width=tensor.shape;assert width%64==0 and tensor.data_ptr()%16==0
  self.storage=C.create_string_buffer(128+63);self.pointer=C.c_void_p((C.addressof(self.storage)+63)&~63)
  NativeInt8.check(cuda.cuTensorMapEncodeTiled(self.pointer,C.c_int(9),C.c_uint(2),C.c_void_p(tensor.data_ptr()),(C.c_uint64*2)(width,height),(C.c_uint64*1)(width*2),(C.c_uint32*2)(64,rows),(C.c_uint32*2)(1,1),C.c_int(0),C.c_int(3),C.c_int(0),C.c_int(0)))
class Native(NativeInt8):
 def __init__(self):
  super().__init__(Path(__file__).with_name('prefix_tma.cubin'));self.variants=json.loads(Path(__file__).with_name('tma-shapes.json').read_text());self.weights={}
 def weight(self,w,bn,wn):
  key=w.data_ptr(),bn,wn
  if key not in self.weights:
   assert not torch.cuda.is_current_stream_capturing(),'Prewarm packed weights before capture'
   n,k=w.shape;half=bn//2
   packed=w.view(2,n//bn,wn,half//wn,k).permute(1,2,0,3,4).reshape(n,k).contiguous();self.weights[key]=packed,TensorMap(self.cuda,packed,bn)
  return self.weights[key]
 def product(self,x,w,name):
  bm,bn,wm,wn,st=self.variants[name];m,k=x.shape;n=len(w);assert k==2560 and n%bn==0
  _,map_w=self.weight(w,bn,wn);map_a=TensorMap(self.cuda,x,bm);shared=max(st*(bm+bn)*64*2+st*8,bm*(bn+8)*2)
  if name not in self.functions:
   fn=C.c_void_p();self.check(self.cuda.cuModuleGetFunction(C.byref(fn),self.module,name.encode()));self.check(self.cuda.cuFuncSetAttribute(fn,8,shared));self.functions[name]=fn
  out=torch.empty((m,n//2),device=x.device,dtype=x.dtype);args=[C.c_void_p(out.data_ptr()),C.c_int(m),C.c_int(n)];params=(C.c_void_p*5)(map_a.pointer,map_w.pointer,*[C.cast(C.pointer(a),C.c_void_p) for a in args]);self.check(self.cuda.cuLaunchKernel(self.functions[name],math.ceil(m/bm),n//bn,1,wm*wn*32,1,1,shared,C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None));return out
