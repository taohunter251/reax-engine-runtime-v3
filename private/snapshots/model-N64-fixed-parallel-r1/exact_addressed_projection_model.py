"""Route only frozen projection products to exact preaddressed schedules."""
import copy,statistics
from reaxv2.sliced import Native as Sliced

class ProjectionGeometry:
    def __init__(self,state):
        self.native=state['native'];self.routes={};grouped={}
        for row in state['operators']:
            key=row['family'],row['group'],row['parts'],row['hp'],row['rows'],row['kernel'],row['drop']
            grouped.setdefault(key,[]).append(row)
        for (family,group,parts,hp,rows,kernel,drop),data in grouped.items():
            hot=statistics.median(d['speedup'] for d in data)
            cold=statistics.median(d['cold_speedup'] for d in data)
            if min(hot,cold)<1.025:continue
            key=family,group,parts,hp,rows
            candidate=dict(kernel=kernel,drop=drop,hot=hot,cold=cold)
            if key not in self.routes or min(hot,cold)>min(self.routes[key]['hot'],self.routes[key]['cold']):self.routes[key]=candidate
    def pick(self,x,p):
        family={9216:'down',4096:'gdn_output',2560:'gdn_input'}.get(p.width)
        if family is None or len(p.low_weight)!=(12288 if family=='gdn_input' else 2560):return None
        choices=[k for k in self.routes if k[:4]==(family,p.group,p.parts,p.hp)]
        if not choices:return None
        return self.routes[min(choices,key=lambda k:abs(k[-1]-len(x)))]

class Rotated:
    def __init__(self,original,geometry):self.original,self.geometry=original,geometry
    def prepare(self,x,p):return self.original.prepare(x,p)
    def project(self,x,p,name,drop=0,reference=False):
        route=self.geometry.pick(x,p)
        if route is None:return self.original.project(x,p,name,drop,reference)
        return self.geometry.native.project(x,p,route['kernel'],self.original.prepare,route['drop'])

class Grouped(Sliced):
    def __init__(self,original,geometry):
        self.__dict__.update(original.__dict__)
        self.geometry=geometry
    def project(self,x,p,name,drop=0,reference=False):
        route=self.geometry.pick(x,p)
        if route is None:return super().project(x,p,name,drop,reference)
        return self.geometry.native.project(x,p,route['kernel'],self.prepare,route['drop'])

def router(original,geometry):
    result=copy.copy(original);policy=copy.copy(original.policy)
    assets=copy.copy(policy.assets)
    assets.rotated={k:Rotated(v,geometry) for k,v in assets.rotated.items()}
    assets.sliced=Grouped(assets.sliced,geometry)
    policy.assets=assets
    policy.project=type(policy).project.__get__(policy,type(policy))
    result.policy=policy
    return result
