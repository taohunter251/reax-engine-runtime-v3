import ctypes as C
import math
import torch
import triton
from .quantize import _prepare
from .driver import NativeInt8

CONFIGS={'hybrid8_b128n256_s64n256': (128, 64, 256, 98304), 'hybrid8_b64n256_s32n256': (64, 32, 256, 81920), 'hybrid8_b128n128_s32n128': (128, 32, 128, 98304), 'hybrid8_b128n256k32_s32n256k32': (128, 32, 256, 73728), 'hybrid8_b64n128k64_s32n128k64': (64, 32, 128, 49152), 'hybrid8_b64n128k32_s32n128k32': (64, 32, 128, 36864), 'hybrid8_b64n128k32_min2': (64, 32, 128, 36864), 'hybrid8_b32n128k32w4_min3': (32, 16, 128, 30720)}

class Native(NativeInt8):
    def product(self,x,w,xh,wh,sa,sw,bias,name,drop=0):
        m,k=x.shape;n=len(w);kh=xh.shape[1]
        bm,small,bn,shared=CONFIGS[name]
        assert k%128==0 and kh%64==0 and n%bn==0
        if name not in self.functions:
            handle=C.c_void_p();self.check(self.cuda.cuModuleGetFunction(C.byref(handle),self.module,name.encode()))
            self.check(self.cuda.cuFuncSetAttribute(handle,8,shared));self.functions[name]=handle
        mb=max(0,m//bm-drop)*bm
        blocks=(mb//bm+math.ceil((m-mb)/small))*(n//bn)
        out=torch.empty((m,n//2),device=x.device,dtype=torch.bfloat16)
        args=[C.c_void_p(t.data_ptr()) for t in (x,w,xh,wh,sa,sw,bias,out)]
        args.extend(C.c_int(v) for v in (m,n,k,kh,mb))
        parameters=(C.c_void_p*len(args))(*[C.cast(C.pointer(v),C.c_void_p) for v in args])
        threads=128 if 'w4' in name else 256
        self.check(self.cuda.cuLaunchKernel(self.functions[name],blocks,1,1,threads,1,1,shared,
             C.c_void_p(torch.cuda.current_stream().cuda_stream),parameters,None))
        return out

    def project(self,x,p,name,drop=0):
        m=len(x);kl=len(p.low_columns);kh=len(p.high_columns)
        low=torch.empty((m,kl),device=x.device,dtype=torch.int8)
        high=torch.empty((m,kh),device=x.device,dtype=torch.bfloat16)
        scales=torch.empty(m,device=x.device)
        _prepare[(m,)](x,p.low_columns,p.high_columns,p.smoothing,p.centers,low,high,scales,
             K=p.width,KL=kl,KH=kh,BL=triton.next_power_of_2(kl),BH=triton.next_power_of_2(kh),
             CENTER=p.centered,num_warps=8,enable_fp_fusion=False)
        return self.product(low,p.low_weight,high,p.high_weight,scales,p.scales,p.bias,name,drop)

