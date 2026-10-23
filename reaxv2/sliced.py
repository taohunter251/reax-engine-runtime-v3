import ctypes as C
import math
import torch
import triton
from .driver import NativeInt8
from .quantize import prepare_kernel, reduce_kernel
CONFIGS={'slice_m128n128': (128, 32, 128, 256), 'slice_m64n128': (64, 32, 128, 256), 'slice_m64n128w4': (64, 32, 128, 128), 'slice_m128n256': (128, 32, 256, 256), 'slice_m128n128k64': (128, 32, 128, 256)}

class Native(NativeInt8):
    def prepare(self,x,p):
        low=torch.empty((len(x),p.parts*p.group),device=x.device,dtype=torch.int8)
        high=torch.empty((len(x),p.parts*p.hp),device=x.device,dtype=x.dtype)
        scales=torch.empty((p.parts,len(x)),device=x.device)
        prepare_kernel[(len(x),p.parts)](x,p.low_columns,p.high_columns,p.centers,p.smoothing,
            low,high,scales,M=len(x),K=p.width,G=p.group,HP=p.hp,P=p.parts,
            B=triton.next_power_of_2(p.group),BH=triton.next_power_of_2(max(1,p.hp)),
            num_warps=8,enable_fp_fusion=False)
        return low,high,scales

    def project(self,x,p,name,drop=0,reference=False):
        low,high,scales=self.prepare(x,p)
        m,n=len(x),len(p.low_weight)
        partials=torch.empty((p.parts,m,n),device=x.device,dtype=x.dtype)
        bm,sm,bn,threads=CONFIGS[name]
        assert n%bn==0
        if name not in self.functions:
            handle=C.c_void_p()
            self.check(self.cuda.cuModuleGetFunction(C.byref(handle),self.module,name.encode()))
            pointer=C.c_void_p();size=C.c_size_t();required=C.c_uint()
            self.check(self.cuda.cuModuleGetGlobal_v2(C.byref(pointer),C.byref(size),self.module,
                                                    (name+'_required_smem').encode()))
            self.check(self.cuda.cuMemcpyDtoH_v2(C.byref(required),pointer,4))
            self.check(self.cuda.cuFuncSetAttribute(handle,8,required.value))
            self.functions[name]=handle,required.value
        function,shared=self.functions[name]
        mb=max(0,m//bm-drop)*bm
        blocks=(mb//bm+math.ceil((m-mb)/sm))*(n//bn)
        args=[C.c_void_p(t.data_ptr()) for t in (low,p.low_weight,high,p.high_weight,scales,p.scales,p.bias,partials)]
        args.extend(C.c_int(v) for v in (m,n,p.parts*p.group,p.parts*p.hp,p.group,p.hp,mb))
        parameters=(C.c_void_p*len(args))(*[C.cast(C.pointer(value),C.c_void_p) for value in args])
        self.check(self.cuda.cuLaunchKernel(function,blocks,1,p.parts,threads,1,1,shared,
            C.c_void_p(torch.cuda.current_stream().cuda_stream),parameters,None))
        if p.parts==1:return partials[0]
        out=torch.empty((m,n),device=x.device,dtype=x.dtype)
        reduce_kernel[(triton.cdiv(m*n,512),)](partials,out,TOTAL=m*n,P=p.parts,B=512,
                                             num_warps=4,enable_fp_fusion=False)
        return out

