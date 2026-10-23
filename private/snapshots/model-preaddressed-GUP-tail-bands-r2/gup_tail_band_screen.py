"""Compare exact dynamic tail bands at the row counts the packed evaluator uses."""
from pathlib import Path
import json,statistics
import torch
from pipeline_screen import captured,timing
from wide_warp_hybrid import Native


def main(ctx,output):
    native=Native(Path(__file__).with_name('band_gate_up.cubin'));rows=[]
    evict=torch.zeros(128<<20,device='cuda',dtype=torch.float32)
    sample=torch.load('/opt/reax-v3/private/output/precision-allocation-r3/layer-20-activation.pt',weights_only=True)
    x=sample['x'].cuda();length=sample['length'];p=ctx.assets.gate_up_banks[20]
    choices=ctx.assets.payload['gate_up_routes']
    for m in range(384,min(len(x),1024)+1,32):
        a=x[:m].contiguous();route=choices[min((k for k in choices if k[0]==128),key=lambda k:(k[1]!=length,abs(k[2]-m)))]
        base,expected=captured(lambda:ctx.assets.gate_up.project(a,p,route['kernel'],route['drop']));expected=expected.clone()
        for name,(_,sm,_,_,_) in native.variants.items():
            for drop in (0,1):
                tail=m-max(0,m//128-drop)*128
                if not tail or not (sm-64<tail<=sm):continue
                candidate,actual=captured(lambda:native.project(a,p,name,drop))
                equal=torch.equal(actual.view(torch.int16),expected.view(torch.int16))
                first=native.project(a[:16].contiguous(),p,name,0)
                invariant=torch.equal(first.view(torch.int16),expected[:16].view(torch.int16))
                record=dict(rows=m,kernel=name,drop=drop,all_output_bits_equal=equal,first16_batch_invariant=invariant,
                    base_route=route)
                if not (equal and invariant):torch.save(dict(input=a.cpu(),expected=expected.cpu(),actual=actual.cpu(),first=first.cpu(),config=record),output/f'failure-{m}-{name}-{drop}.pt')
                assert equal and invariant,record
                times={'base':[],'candidate':[]}
                for arm in ('base','candidate','candidate','base'):
                    times[arm].append(timing(base if arm=='base' else candidate,repeats=12))
                warm={k:statistics.median(v) for k,v in times.items()}
                cold={'base':[],'candidate':[]}
                for arm in ('base','candidate','candidate','base'):
                    evict.add_(1.)
                    start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    start.record();(base if arm=='base' else candidate).replay();end.record();end.synchronize()
                    cold[arm].append(start.elapsed_time(end))
                cms={k:statistics.median(v) for k,v in cold.items()}
                rows.append(dict(**record,milliseconds=warm,speedup=warm['base']/warm['candidate'],cold_milliseconds=cms,cold_speedup=cms['base']/cms['candidate']))
        (output/'operators.json').write_text(json.dumps(rows,indent=2))
    ctx.retained['gup_tail_bands']=dict(native=native,operators=rows,source=str(output))
    return dict(cases=len(rows),all_controls_bit_equal=True,scope='Unchanged frozen quantization; row-tail operator comparison only.')
