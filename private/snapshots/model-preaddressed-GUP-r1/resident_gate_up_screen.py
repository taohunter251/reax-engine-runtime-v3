"""Check frozen GUP bytes before evaluating alternate occupancy/layout costs."""
import json
from pathlib import Path
import statistics
import torch
from pipeline_screen import captured,timing
from wide_warp_hybrid import Native


def main(ctx,output):
    native=Native(Path(__file__).with_name('addressed_gate_up.cubin'));results=[]
    evict=torch.zeros(128<<20,device='cuda',dtype=torch.float32)
    for index in (12,20,30):
        sample=torch.load(f'/opt/reax-v3/private/output/precision-allocation-r3/layer-{index}-activation.pt',weights_only=True)
        x=sample['x'].to('cuda');length=sample['length'];p=ctx.assets.gate_up_banks[index]
        routes=ctx.assets.payload['gate_up_routes']
        for size in (32,128,512,640,768,896,1024):
            a=x[:min(size,len(x))].contiguous()
            route=routes[min((k for k in routes if k[0]==128),key=lambda k:(k[1]!=length,abs(k[2]-len(a))))]
            base,expected=captured(lambda:ctx.assets.gate_up.project(a,p,route['kernel'],route['drop']))
            expected=expected.clone()
            for name in native.variants:
                for drop in (0,1):
                    graph,got=captured(lambda:native.project(a,p,name,drop))
                    exact=torch.equal(got.view(torch.int16),expected.view(torch.int16))
                    first=native.project(a[:16].contiguous(),p,name,0)
                    invariant=torch.equal(first.view(torch.int16),expected[:16].view(torch.int16))
                    row=dict(layer=index,rows=len(a),kernel=name,drop=drop,all_frozen_output_bits_equal=exact,first16_batch_invariant=invariant)
                    if not (exact and invariant):
                        torch.save(dict(input=a.cpu(),expected=expected.cpu(),actual=got.cpu(),first=first.cpu(),config=row),output/f'failure-{index}-{len(a)}-{name}-{drop}.pt')
                    assert exact and invariant,row
                    times={'base':[],'candidate':[]}
                    for _ in range(3):
                        for arm in ('base','candidate','candidate','base'):
                            times[arm].append(timing(base if arm=='base' else graph,repeats=8))
                    ms={k:statistics.median(v) for k,v in times.items()}
                    cold={'base':[],'candidate':[]}
                    for arm in ('base','candidate','candidate','base'):
                        evict.add_(1.)
                        begin,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        begin.record();(base if arm=='base' else graph).replay();end.record();end.synchronize()
                        cold[arm].append(begin.elapsed_time(end))
                    cold_ms={k:statistics.median(v) for k,v in cold.items()}
                    results.append(dict(**row,milliseconds=ms,speedup=ms['base']/ms['candidate'],
                        cold_milliseconds=cold_ms,cold_speedup=cold_ms['base']/cold_ms['candidate']))
            (output/'operators.json').write_text(json.dumps(results,indent=2))
    ctx.retained['preaddressed_gate_up']=dict(native=native,operators=results,source=str(output))
    return dict(cases=len(results),all_controls_bit_equal=True,scope='Frozen model banks and arithmetic. Layout/residency operator comparison, no serving benchmark.')
