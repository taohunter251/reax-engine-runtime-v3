"""Compare ordered TMA/table products with the current exact prefix path."""
from pathlib import Path
import json,statistics
import torch
from probe_utils import make_entries
from exact_extension_full_screen import dedicated
from ordered_tma_table_native import Native


def main(ctx,output):
    import reax_subnet.engrunner as runner
    from reax_subnet.engvalidator import TRACKS
    native=Native();domain=native.domain_check()
    native.weights=ctx.retained['BF16_prefix_TMA']['native'].weights
    core=ctx.full.core;old=core._mlp;samples={}
    table=ctx.retained['n64_extended_holdout']['model'].inner.native
    def observe(layer,x,length,defer=False):
        index=next(i for i,d in enumerate(core.ff.layers) if d is layer)
        if index in (4,8,10,11) and length in (192,256):
            key=index,length
            if key not in samples or len(x)>len(samples[key]):samples[key]=x.clone()
        return old(layer,x,length,defer)
    docs=[d.body() for d in runner.fresh_decisions(512,TRACKS['s1-fast'].pool)]
    (output/'requests.json').write_text(json.dumps(docs))
    core._mlp=observe
    try:
        for part,entry in make_entries(ctx.encoder,ctx.gpu,docs,count=8):
            ctx.model(entry[2],True)
            if len(samples)==8 and min(map(len,samples.values()))>=1152:break
    finally:core._mlp=old
    assert len(samples)==8,len(samples)
    stream=dedicated(ctx)
    def capture(call):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):call()
        stream.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):value=call()
        graph.replay();torch.cuda.synchronize();return graph,value
    def timing(graph,count):
        start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        for _ in range(count):graph.replay()
        end.record();end.synchronize();return start.elapsed_time(end)/count
    eviction=torch.zeros(64<<20,device='cuda',dtype=torch.float32)
    records=[]
    for (index,length),x in samples.items():
        layer=core.ff.layers[index];weight=layer['fused']['gate_up']
        source=f'source-{index}-{length}.pt'
        torch.save(dict(input=x.cpu(),layer=index,length=length,weight='Frozen original BF16 fused gate_up'),output/source)
        for rows in sorted({128,*range(512,min(len(x),1536)+1,32)}):
            if rows>len(x):continue
            a=x[:rows].contiguous();algorithm=core.pinned.swiglu_r.get((length,rows))
            alias=table.aliases.get(getattr(algorithm,'name',None))
            if alias is None:continue
            base,expected=capture(lambda:table.product(a,weight,alias,algorithm.drop));expected=expected.clone()
            for name in native.variants:
                graph,actual=capture(lambda:native.product(a,weight,name))
                first=native.product(a[:16].contiguous(),weight,name)
                equal=torch.equal(expected.view(torch.int16),actual.view(torch.int16))
                invariant=torch.equal(expected[:16].view(torch.int16),first.view(torch.int16))
                record=dict(layer=index,length=length,rows=rows,kernel=name,all_bits_equal=equal,first16_invariant=invariant)
                if equal and invariant:
                    hot,cold=dict(base=[],candidate=[]),dict(base=[],candidate=[])
                    for arm in ('base','candidate','candidate','base'):
                        selected=base if arm=='base' else graph
                        hot[arm].append(timing(selected,8));eviction.add_(1);cold[arm].append(timing(selected,1))
                    equal=torch.equal(expected.view(torch.int16),actual.view(torch.int16))
                    record['after_replay_bits_equal']=equal
                    ms={arm:statistics.median(v) for arm,v in hot.items()};cms={arm:statistics.median(v) for arm,v in cold.items()}
                    record.update(milliseconds=ms,cold_milliseconds=cms,speedup=ms['base']/ms['candidate'],cold_speedup=cms['base']/cms['candidate'])
                if not equal or not invariant:
                    torch.save(dict(source=source,record=record,expected=expected.cpu(),actual=actual.cpu(),first16=first.cpu()),
                        output/f'failure-{index}-{length}-{rows}-{name}.pt')
                records.append(record)
            (output/'operators.json').write_text(json.dumps(records,indent=2))
    result=dict(cases=len(records),all_bits_equal=all(r['all_bits_equal'] and r['first16_invariant'] and r.get('after_replay_bits_equal',False) for r in records),domain=domain,
        scope='Ordered TMA plus exact BF16 SiLU, compared with current prefix table at all observed packed32 shapes and four model layers. No serving or official qualification claim.')
    ctx.retained['ordered_TMA_table']=dict(native=native,operators=records,samples=samples,result=result,source=str(output))
    return result
