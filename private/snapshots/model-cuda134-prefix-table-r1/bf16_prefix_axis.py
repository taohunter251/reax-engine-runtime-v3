"""Launch fixed-K exact BF16 prefix tiles without changing the model weights."""
from pathlib import Path
import ctypes as C
import json,math
import torch
from reaxv2.driver import NativeInt8

class Native(NativeInt8):
    def __init__(self):
        super().__init__(Path(__file__).with_name('prefix_axis.cubin'))
        self.variants=json.loads(Path(__file__).with_name('prefix_shapes.json').read_text())
    def product(self,x,w,name,drop=0):
        bm,bn,sm,sn,threads,kind=self.variants[name]
        m,k=x.shape;n=len(w)
        assert k==2560 and x.is_contiguous() and w.is_contiguous() and n%bn==0 and n%sn==0
        if name not in self.functions:
            fn=C.c_void_p();self.check(self.cuda.cuModuleGetFunction(C.byref(fn),self.module,name.encode()))
            ptr=C.c_void_p();size=C.c_size_t();shared=C.c_uint()
            self.check(self.cuda.cuModuleGetGlobal_v2(C.byref(ptr),C.byref(size),self.module,(name+'_required_smem').encode()))
            self.check(self.cuda.cuMemcpyDtoH_v2(C.byref(shared),ptr,4));self.check(self.cuda.cuFuncSetAttribute(fn,8,shared.value));self.functions[name]=fn,shared.value
        fn,shared=self.functions[name];mb=max(0,m//bm-drop)*bm
        blocks=(mb//bm)*(n//bn)+math.ceil((m-mb)/sm)*(n//sn)
        out=torch.empty((m,n//2 if kind=='swiglu' else n),device=x.device,dtype=x.dtype)
        args=[C.c_void_p(t.data_ptr()) for t in (x,w,out)]+[C.c_int(v) for v in (m,n,k,mb)]
        params=(C.c_void_p*len(args))(*[C.cast(C.pointer(v),C.c_void_p) for v in args])
        self.check(self.cuda.cuLaunchKernel(fn,blocks,1,1,threads,1,1,shared,C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None))
        return out
