"""Check real grouped model products with hot and cold weights before promotion."""
import json,statistics
import torch
from addressed_sliced import Native
from pipeline_screen import captured,timing

def main(ctx,output):
    native=ctx.retained['preaddressed_sliced']['native'];rows=[]
    samples=torch.load('/opt/reax-v3/private/output/precision-allocation-r3/projection-inputs.pt',weights_only=True)
    samples[str((20,'gdn_input'))]=torch.load('/opt/reax-v3/private/output/model-preaddressed-sliced-projections-r3/gdn-input.pt',weights_only=True)
    evict=torch.zeros(128<<20,device='cuda',dtype=torch.float32)
    for family in ('down','gdn_output','gdn_input'):
        sample=samples[str((20,family))];x=sample['x'].cuda().repeat((3,1)).contiguous()
        for length in (128,192,256,320):
            chunk=ctx.assets.payload['chunk_for'].get((20,family,length))
            if chunk is None:continue
            p=ctx.assets.projections[20,family,chunk]
            packed_rows=sorted({entry[0] for lane in ctx.gpu.lanes for (L,B),entries in lane.packed.items() if L==length for entry in entries})
            for m in packed_rows:
                a=x[:min(m,len(x))].contiguous();m=len(a)
                if family=='gdn_input':
                    table=ctx.assets.payload['projection_routes'];cap=128;prepare=ctx.assets.sliced.prepare
                    key=min((k for k in table if k[0]==family and k[3]==cap),key=lambda k:(k[1]!=length,abs(k[2]-m)))
                    route=table[key];original=ctx.assets.sliced
                else:
                    table=ctx.assets.payload['rotation_routes'];cap=ctx.assets.payload['rotation_cap']
                    key=min((k for k in table if k[0]==family and k[3]==cap),key=lambda k:(k[1]!=length,abs(k[2]-m)))
                    route=table[key];original=ctx.assets.rotated[min(route['threads'],p.block)];prepare=original.prepare
                bg,want=captured(lambda:original.project(a,p,route['kernel'],route['drop']));want=want.clone()
                for name,(_,_,_,_,bk) in native.variants.items():
                    if p.group%(2*bk) or (p.hp and p.hp%bk):continue
                    for drop in (0,1):
                        cg,got=captured(lambda:native.project(a,p,name,prepare,drop))
                        exact=torch.equal(got.view(torch.int16),want.view(torch.int16))
                        first=native.project(a[:16].contiguous(),p,name,prepare,0)
                        invariant=torch.equal(first.view(torch.int16),want[:16].view(torch.int16))
                        rec=dict(family=family,length=length,rows=m,group=p.group,parts=p.parts,hp=p.hp,kernel=name,drop=drop,all_bits_equal=exact,first16_invariant=invariant,base_route=route)
                        if not (exact and invariant):torch.save(dict(input=a.cpu(),expected=want.cpu(),actual=got.cpu(),first=first.cpu(),config=rec),output/f'failure-{family}-{length}-{m}-{name}-{drop}.pt')
                        assert exact and invariant,rec
                        times={'base':[],'candidate':[]};cold={'base':[],'candidate':[]}
                        for arm in ('base','candidate','candidate','base'):
                            times[arm].append(timing(bg if arm=='base' else cg,repeats=6))
                            evict.add_(1.);start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                            start.record();(bg if arm=='base' else cg).replay();end.record();end.synchronize()
                            cold[arm].append(start.elapsed_time(end))
                        ms={k:statistics.median(v) for k,v in times.items()};cms={k:statistics.median(v) for k,v in cold.items()}
                        rows.append(dict(**rec,milliseconds=ms,speedup=ms['base']/ms['candidate'],cold_milliseconds=cms,cold_speedup=cms['base']/cms['candidate']))
                (output/'operators.json').write_text(json.dumps(rows,indent=2))
    ctx.retained['preaddressed_sliced_packed']=dict(native=native,operators=rows,source=str(output))
    return dict(cases=len(rows),all_controls_bit_equal=True,scope='Same projection banks, codes, BF16 partial rounding and fold. No serving change.')
