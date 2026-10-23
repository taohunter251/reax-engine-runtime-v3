"""Descriptor kernels with the exhaustive exact BF16 SiLU function table."""
from pathlib import Path
import ctypes as C,json
import torch
from reaxv2.driver import NativeInt8
from prefix_tma_native import Native as Original


class Native(Original):
    def __init__(self):
        NativeInt8.__init__(self,Path(__file__).with_name('prefix_tma.cubin'))
        self.variants=json.loads(Path(__file__).with_name('tma-shapes.json').read_text())
        self.weights={}
        fn=C.c_void_p();self.check(self.cuda.cuModuleGetFunction(C.byref(fn),self.module,b'initialize_silu_table'))
        self.check(self.cuda.cuLaunchKernel(fn,256,1,1,256,1,1,0,C.c_void_p(torch.cuda.current_stream().cuda_stream),None,None))
        torch.cuda.synchronize()

    def domain_check(self):
        a=torch.empty(65536,device='cuda',dtype=torch.bfloat16);b=torch.empty_like(a)
        fn=C.c_void_p();self.check(self.cuda.cuModuleGetFunction(C.byref(fn),self.module,b'check_silu_domain'))
        args=[C.c_void_p(x.data_ptr()) for x in (a,b)]
        params=(C.c_void_p*2)(*[C.cast(C.pointer(v),C.c_void_p) for v in args])
        self.check(self.cuda.cuLaunchKernel(fn,256,1,1,256,1,1,0,C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None))
        torch.cuda.synchronize()
        equal=torch.equal(a.view(torch.int16),b.view(torch.int16))
        assert equal,'Full BF16 SiLU domain differs'
        return dict(patterns=65536,all_bits_equal=equal)
