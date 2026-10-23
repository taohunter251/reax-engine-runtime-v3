"""Use proven profitable N64 tiles while preserving every original reduction plan."""
from collections import defaultdict
from contextlib import contextmanager
from mixed_origin_bf16 import reduction


class Candidate:
    def __init__(self,inner,state,force_pairs=False):
        self.inner,self.native=inner,state['native']
        self.force_pairs=force_pairs
        self.usage=defaultdict(int)
        self.routes={}
        groups=defaultdict(list)
        for row in state['operators']:
            if not row['all_bits_equal']:
                continue
            cfg=row['original_reduction']
            groups[row['length'],row['rows'],cfg['parts'],cfg['chunk'],cfg['mode'],row['kernel']].append(row)
        for key,rows in groups.items():
            if len(rows)<4:
                continue
            score=min(min(r['speedup'],r['cold_speedup']) for r in rows)
            if score<1.05:
                continue
            route=key[:-1]
            if route not in self.routes or score>self.routes[route]['score']:
                self.routes[route]=dict(kernel=key[-1],score=score)
        self.gemm_marks=(*inner.gemm_marks,'n64_parts_')

    def __getattr__(self,name):
        return getattr(object.__getattribute__(self,'inner'),name)

    @contextmanager
    def hooks(self):
        core=self.full.core
        old,old_pair=core.gemm,core.pair
        def gemm(weight,x,length):
            if tuple(weight.shape)!=(64,2560) or length not in (192,256):
                return old(weight,x,length)
            config=reduction(core,(64,2560),length,len(x))
            route=self.routes.get((length,len(x),config['parts'],config['chunk'],config['mode']))
            if route is not None:self.usage['matrix',length,len(x)]+=1
            return old(weight,x,length) if route is None else self.native.project(x,weight,config,route['kernel'])
        def pair(layer,name,left,right,x,length):
            weight=layer['fused'].get(name)
            if not self.force_pairs or name!='b_a' or x is None or weight is None or tuple(weight.shape)!=(64,2560) or length not in (192,256):
                return old_pair(layer,name,left,right,x,length)
            config=reduction(core,(64,2560),length,len(x))
            route=self.routes.get((length,len(x),config['parts'],config['chunk'],config['mode']))
            if route is None:return old_pair(layer,name,left,right,x,length)
            fused=core.pinned.fuse_r.get((64,2560,length,len(x)),False)
            self.usage['pair_previously_fused' if fused else 'pair_previously_separate',length,len(x)]+=1
            value=self.native.project(x,weight,config,route['kernel'])
            return value[:,:len(left.weight)],value[:,len(left.weight):]
        core.gemm=gemm
        core.pair=pair
        try:
            yield
        finally:
            core.gemm=old
            core.pair=old_pair

    def __call__(self,meta,approximate=False):
        with self.hooks():
            return self.inner(meta,approximate)

    def suffix(self,meta,h,x,approximate=False):
        with self.hooks():
            return self.inner.suffix(meta,h,x,approximate)
