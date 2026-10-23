"""Expose the exact INT8 z branch while this model computes recurrent state."""
from contextlib import contextmanager
import copy
import ctypes as C
import math
import sys
import torch
from reaxv2.projection import Router
from reaxv2.sliced import Native as Sliced
from reaxv2.sliced import CONFIGS


class Prepared(Sliced):
    def __init__(self,original):
        self.__dict__.update(original.__dict__)
        self.ready=None

    def prepare(self,x,p):
        assert self.ready is not None
        return self.ready

    def project_ready(self,x,p,ready,name,drop,out=None):
        assert self.ready is None
        if out is not None:
            assert p.parts==1
            m,n=len(x),len(p.low_weight)
            bm,small,bn,threads=CONFIGS[name]
            # Warmup obtains the unchanged CUDA function and its shared-memory size.
            if name not in self.functions:
                self.project_ready(x,p,ready,name,drop)
            fn,shared=self.functions[name]
            mb=max(0,m//bm-drop)*bm
            blocks=(mb//bm+math.ceil((m-mb)/small))*(n//bn)
            low,high,scales=ready
            args=[C.c_void_p(t.data_ptr()) for t in (low,p.low_weight,high,p.high_weight,scales,p.scales,p.bias,out)]
            args.extend(C.c_int(v) for v in (m,n,p.group,p.hp,p.group,p.hp,mb))
            params=(C.c_void_p*len(args))(*[C.cast(C.pointer(v),C.c_void_p) for v in args])
            self.check(self.cuda.cuLaunchKernel(fn,blocks,1,1,threads,1,1,shared,
                       C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None))
            return out
        self.ready=ready
        try:return Sliced.project(self,x,p,name,drop)
        finally:self.ready=None


class SplitRouter(Router):
    def __init__(self,original,z_phase='projection'):
        self.__dict__.update(original.__dict__)
        self.matrix=Prepared(self.policy.assets.sliced)
        self.banks={}
        self.coverage={}
        self.z_phase=z_phase
        self.launch_boundaries={}
        if z_phase!='projection':assert self.core.ff_mode=='ps'

    def split_bank(self,p,begin,end):
        key=id(p),begin,end
        if key not in self.banks:
            b=copy.copy(p)
            b.low_weight=p.low_weight[begin:end]
            b.high_weight=p.high_weight[begin:end]
            b.scales=p.scales[:,begin:end].contiguous()
            b.bias=p.bias[:,begin:end].contiguous()
            self.banks[key]=b
        return self.banks[key]

    @contextmanager
    def context(self):
        core=self.core
        original_gdn,original_lin,original_fused=core._gdn,core.lin,core._l0t_fused
        with super().context():
            routed_gdn=core._gdn
            def gated(layer,x,meta,index):
                data=self.policy.assets.payload
                chunk=data['chunk_for'].get((index,'gdn_input',meta.L))
                if chunk is None or not core._par('z'):
                    return routed_gdn(layer,x,meta,index)
                p=self.policy.assets.projections[index,'gdn_input',chunk]
                assert p.parts==1
                ready=self.policy.assets.sliced.prepare(x,p)
                qkv=layer['la'].in_proj_qkv
                z=layer['la'].in_proj_z
                count=len(qkv.weight)
                table=data['projection_routes']
                choices=[k for k in table if k[0]=='gdn_input' and k[3]==128]
                route=table[min(choices,key=lambda k:(k[1]!=meta.L,abs(k[2]-len(x))))]
                late={}
                module=sys.modules[type(core).__module__].fla_fused
                saved_recurrence={}
                old_kkt=module.KKT2['nat']
                def launch_z(boundary):
                    if not late or 'token' in late:return
                    self.launch_boundaries[boundary]=self.launch_boundaries.get(boundary,0)+1
                    _,late['token']=core._on_side(lambda:self.matrix.project_ready(
                        x,late['bank'],ready,route['kernel'],route['drop'],out=late['out']))
                def linear(module,values,length):
                    if module is qkv:
                        b=self.split_bank(p,0,count)
                    elif module is z:
                        b=self.split_bank(p,count,len(p.low_weight))
                        if self.z_phase!='projection':
                            target=torch.empty((len(values),len(b.low_weight)),device=values.device,dtype=values.dtype)
                            late.update(bank=b,out=target)
                            return target
                    else:return original_lin(module,values,length)
                    return self.matrix.project_ready(values,b,ready,route['kernel'],route['drop'])
                def fused(d,name,rows,length):
                    if d is layer and name=='qkv_z':return False
                    return original_fused(d,name,rows,length)
                core.lin,core._l0t_fused=linear,fused
                def recurrent(fn):
                    def call(*args,**kwargs):
                        if self.z_phase=='recurrence':launch_z('before-recurrence')
                        result=fn(*args,**kwargs)
                        launch_z('fallback-after-recurrence')
                        if 'token' in late:core._join(late['token'])
                        return result
                    return call
                if self.z_phase!='projection':
                    for name in (('gdn_forward_ps_peel',) if core.ff_peel else ('gdn_forward_ps',)):
                        saved_recurrence[name]=getattr(module,name)
                        setattr(module,name,recurrent(saved_recurrence[name]))
                    if self.z_phase=='state' and old_kkt is not None:
                        class StateBoundary:
                            def takes(self,*args,**kwargs):return old_kkt.takes(*args,**kwargs)
                            def __call__(self,*args,**kwargs):
                                result=old_kkt(*args,**kwargs)
                                launch_z('after-native-kkt')
                                return result
                        module.KKT2['nat']=StateBoundary()
                key=index,meta.L,len(x)
                self.coverage[key]=self.coverage.get(key,0)+1
                try:return original_gdn(layer,x,meta,index)
                finally:
                    core.lin,core._l0t_fused=original_lin,original_fused
                    module.KKT2['nat']=old_kkt
                    for name,value in saved_recurrence.items():setattr(module,name,value)
            core._gdn=gated
            try:yield
            finally:core._gdn=routed_gdn
