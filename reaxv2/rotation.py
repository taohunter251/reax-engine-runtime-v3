import ctypes as C
import torch
from .sliced import Native as SlicedNative

class Native(SlicedNative):
    def __init__(self,matrices,prep,threads=128):
        self.__dict__.update(matrices.__dict__)
        self.prep=prep;self.threads=threads
    def kernel_name(self,p):
        return f'rotation_g{p.group}_b{p.block}_t{self.threads}'

    def project(self,x,p,name,drop=0,reference=False):
        return super().project(x,p,selected_kernel(name,p),drop,reference=reference)

    def prepare(self,x,p):
        m=len(x);name=self.kernel_name(p)
        low=torch.empty((m,p.parts*p.group),device=x.device,dtype=torch.int8)
        high=torch.empty((m,0),device=x.device,dtype=x.dtype)
        scales=torch.empty((p.parts,m),device=x.device)
        loader=self.prep
        if name not in loader.functions:
            function=C.c_void_p()
            loader.check(loader.cuda.cuModuleGetFunction(C.byref(function),loader.module,name.encode()))
            loader.check(loader.cuda.cuFuncSetAttribute(function,8,4*p.group))
            loader.functions[name]=function
        args=[C.c_void_p(t.data_ptr()) for t in (x,p.centers,low,scales)]
        args.extend(C.c_int(v) for v in (m,p.width,p.chunk,p.parts))
        parameters=(C.c_void_p*len(args))(*[C.cast(C.pointer(value),C.c_void_p) for value in args])
        loader.check(loader.cuda.cuLaunchKernel(loader.functions[name],m,p.parts,1,self.threads,1,1,4*p.group,
            C.c_void_p(torch.cuda.current_stream().cuda_stream),parameters,None))
        return low,high,scales


def selected_kernel(name,projection):
    allowed=('slice_m128n128','slice_m64n128','slice_m64n128w4','slice_m128n256','slice_m128n128k64')
    if name not in allowed:raise ValueError(f'Missing inner-loop contract for {name}')
    # Each INT8 MMA pipeline step consumes twice the named BF16 depth.
    depth=64 if name.endswith('k64') else 32
    if projection.group%(2*depth)==0 and (not projection.hp or projection.hp%depth==0):return name
    if name.endswith('k64') and projection.group%64==0 and (not projection.hp or projection.hp%32==0):
        return name[:-3]
    raise ValueError(f'Unsupported projection inner-loop alignment: {name}, group={projection.group}, high={projection.hp}')

