"""Launch exact grouped projection schedules with the original prepare and fold."""
import ctypes as C
import json,math
from pathlib import Path
import torch,triton
from reaxv2.driver import NativeInt8
from reaxv2.quantize import reduce_kernel

class Native(NativeInt8):
    def __init__(self):
        super().__init__(Path(__file__).with_name('addressed_sliced.cubin'))
        self.variants=json.loads(Path(__file__).with_name('sliced_shapes.json').read_text())

    def project(self,x,p,name,prepare,drop=0):
        low,high,scales=prepare(x,p)
        m,n=len(x),len(p.low_weight);bm,sm,bn,threads,bk=self.variants[name]
        assert p.group%(2*bk)==0 and (not p.hp or p.hp%bk==0) and n%bn==0
        if name not in self.functions:
            fn=C.c_void_p();self.check(self.cuda.cuModuleGetFunction(C.byref(fn),self.module,name.encode()))
            ptr=C.c_void_p();size=C.c_size_t();shared=C.c_uint()
            self.check(self.cuda.cuModuleGetGlobal_v2(C.byref(ptr),C.byref(size),self.module,(name+'_required_smem').encode()))
            self.check(self.cuda.cuMemcpyDtoH_v2(C.byref(shared),ptr,4))
            self.check(self.cuda.cuFuncSetAttribute(fn,8,shared.value));self.functions[name]=fn,shared.value
        fn,shared=self.functions[name];mb=max(0,m//bm-drop)*bm
        blocks=(mb//bm+math.ceil((m-mb)/sm))*(n//bn)
        partial=torch.empty((p.parts,m,n),device=x.device,dtype=x.dtype)
        args=[C.c_void_p(t.data_ptr()) for t in (low,p.low_weight,high,p.high_weight,scales,p.scales,p.bias,partial)]
        args.extend(C.c_int(v) for v in (m,n,p.parts*p.group,p.parts*p.hp,p.group,p.hp,mb))
        params=(C.c_void_p*len(args))(*[C.cast(C.pointer(v),C.c_void_p) for v in args])
        self.check(self.cuda.cuLaunchKernel(fn,blocks,1,p.parts,threads,1,1,shared,C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None))
        if p.parts==1:return partial[0]
        result=torch.empty((m,n),device=x.device,dtype=x.dtype)
        reduce_kernel[(triton.cdiv(m*n,512),)](partial,result,TOTAL=m*n,P=p.parts,B=512,num_warps=4,enable_fp_fusion=False)
        return result
