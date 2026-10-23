"""N64 control matrices, fixed independent K64 slices and the unchanged FP32 fold."""
from pathlib import Path
import ctypes as C
import torch, triton
from mixed_origin_bf16 import fold
from reaxv2.driver import NativeInt8


class Native(NativeInt8):
    variants={'n64_parts_0':(32,32,64),'n64_parts_1':(32,64,128),'n64_parts_2':(64,64,128),
              'n64_parts_3':(32,64,128),'n64_parts_4':(64,64,128),'n64_parts_5':(32,32,64)}
    def __init__(self):
        super().__init__(Path(__file__).with_name('n64_parts.cubin'))
        self.split=torch.zeros(1,device='cuda',dtype=torch.int32)

    def project(self,x,w,config,name,workspace_key=None):
        assert x.is_contiguous() and w.is_contiguous() and tuple(w.shape)==(64,2560)
        assert config['mode']==4 and config['chunk']==64 and config['parts'] in (40,41)
        m=len(x);count=config['parts'];bm,bn,threads=self.variants[name]
        if name not in self.functions:
            fn=C.c_void_p();self.check(self.cuda.cuModuleGetFunction(C.byref(fn),self.module,name.encode()))
            pointer=C.c_void_p();size=C.c_size_t();shared=C.c_uint()
            self.check(self.cuda.cuModuleGetGlobal_v2(C.byref(pointer),C.byref(size),self.module,(name+'_required_smem').encode()))
            self.check(self.cuda.cuMemcpyDtoH_v2(C.byref(shared),pointer,4));self.check(self.cuda.cuFuncSetAttribute(fn,8,shared.value))
            self.functions[name]=fn,shared.value
        fn,shared=self.functions[name]
        parts=torch.empty((count,m,64),device=x.device,dtype=x.dtype);out=torch.empty((m,64),device=x.device,dtype=x.dtype)
        values=[C.c_void_p(t.data_ptr()) for t in (x,w,parts)]+[C.c_int(m)]
        args=(C.c_void_p*len(values))(*[C.cast(C.pointer(value),C.c_void_p) for value in values])
        self.check(self.cuda.cuLaunchKernel(fn,((m+bm-1)//bm)*(64//bn),1,count,threads,1,1,shared,
            C.c_void_p(torch.cuda.current_stream().cuda_stream),args,None))
        fold[(triton.cdiv(m*64,512),)](parts,self.split,out,M=m,N=64,P0=count,P1=count,B=512,num_warps=4,enable_fp_fusion=False)
        return out
