"""Launch geometry-specialized tiles on the original frozen parameter banks."""
import ctypes as C
import json
import math
from pathlib import Path
import torch
from reaxv2.gate_up import Native as Original


class Native(Original):
    def __init__(self,binary):
        super().__init__(binary)
        self.initialize=C.c_void_p()
        self.check(self.cuda.cuModuleGetFunction(C.byref(self.initialize),self.module,b'initialize_silu_table'))
        self.check(self.cuda.cuLaunchKernel(self.initialize,256,1,1,256,1,1,0,C.c_void_p(torch.cuda.current_stream().cuda_stream),None,None))
        torch.cuda.synchronize()
        self.variants=json.loads(Path(__file__).with_name('wide_warp_shapes.json').read_text())

    def product(self,x,w,xh,wh,sa,sw,bias,name,drop=0):
        m,k=x.shape;n=len(w);kh=xh.shape[1]
        assert k==2432 and kh==128 and n==18432
        bm,small,bn,threads,sn=self.variants[name]
        if name not in self.functions:
            fn=C.c_void_p();self.check(self.cuda.cuModuleGetFunction(C.byref(fn),self.module,name.encode()))
            ptr=C.c_void_p();size=C.c_size_t();shared=C.c_uint()
            self.check(self.cuda.cuModuleGetGlobal_v2(C.byref(ptr),C.byref(size),self.module,(name+'_required_smem').encode()))
            self.check(self.cuda.cuMemcpyDtoH_v2(C.byref(shared),ptr,4))
            self.check(self.cuda.cuFuncSetAttribute(fn,8,shared.value))
            self.functions[name]=fn,shared.value
        fn,shared=self.functions[name]
        mb=max(0,m//bm-drop)*bm
        blocks=(mb//bm)*(n//bn)+math.ceil((m-mb)/small)*(n//sn)
        out=torch.empty((m,n//2),device=x.device,dtype=torch.bfloat16)
        args=[C.c_void_p(t.data_ptr()) for t in (x,w,xh,wh,sa,sw,bias,out)]
        args.extend(C.c_int(v) for v in (m,n,k,kh,mb))
        params=(C.c_void_p*len(args))(*[C.cast(C.pointer(v),C.c_void_p) for v in args])
        self.check(self.cuda.cuLaunchKernel(fn,blocks,1,1,threads,1,1,shared,
                                          C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None))
        return out

    def domain_check(self):
        direct=torch.empty(65536,device='cuda',dtype=torch.bfloat16);table=torch.empty_like(direct)
        fn=C.c_void_p();self.check(self.cuda.cuModuleGetFunction(C.byref(fn),self.module,b'check_silu_domain'))
        args=[C.c_void_p(t.data_ptr()) for t in (direct,table)]
        prm=(C.c_void_p*2)(*[C.cast(C.pointer(v),C.c_void_p) for v in args])
        self.check(self.cuda.cuLaunchKernel(fn,256,1,1,256,1,1,0,C.c_void_p(torch.cuda.current_stream().cuda_stream),prm,None))
        torch.cuda.synchronize()
        assert torch.equal(direct.view(torch.int16),table.view(torch.int16))
        return {'BF16_input_patterns':65536,'bit_identical':True}
