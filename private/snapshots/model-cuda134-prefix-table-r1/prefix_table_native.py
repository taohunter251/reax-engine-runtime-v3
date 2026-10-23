"""Original prefix layouts with an exhaustive BF16 activation function table."""
from pathlib import Path
import ctypes as C,json
import torch
from bf16_prefix_axis import Native as Parent
class Native(Parent):
 def __init__(self):
  super().__init__();self.aliases=json.loads(Path(__file__).with_name('original-aliases.json').read_text());fn=C.c_void_p();self.check(self.cuda.cuModuleGetFunction(C.byref(fn),self.module,b'initialize_silu_table'));self.check(self.cuda.cuLaunchKernel(fn,256,1,1,256,1,1,0,C.c_void_p(torch.cuda.current_stream().cuda_stream),None,None));torch.cuda.synchronize()
 def domain_check(self):
  a=torch.empty(65536,device='cuda',dtype=torch.bfloat16);b=torch.empty_like(a);fn=C.c_void_p();self.check(self.cuda.cuModuleGetFunction(C.byref(fn),self.module,b'check_silu_domain'));args=[C.c_void_p(t.data_ptr()) for t in (a,b)];params=(C.c_void_p*2)(*[C.cast(C.pointer(v),C.c_void_p) for v in args]);self.check(self.cuda.cuLaunchKernel(fn,256,1,1,256,1,1,0,C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None));torch.cuda.synchronize();return dict(cases=65536,all_bits_equal=torch.equal(a.view(torch.int16),b.view(torch.int16)))
