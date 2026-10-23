"""Check one live integer pipeline against group restarts and the frozen operator."""
import hashlib,json,statistics,subprocess
from pathlib import Path
import torch
import torch.nn.functional as F
from grouped_gate_up import bank,Native as Restarted
from pipelined_gate_up import Native


def captured(call):
    for _ in range(2):call()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=call()
    g.replay();torch.cuda.synchronize()
    return g,out


def timing(graph,repeats=16):
    a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)];a.record()
    for _ in range(repeats):graph.replay()
    b.record();b.synchronize()
    return a.elapsed_time(b)/repeats


def main(ctx,output):
    root=Path(__file__).resolve().parent;binary=output/'pipeline.cubin'
    command=['/usr/local/cuda-13.0/bin/nvcc','-O3','-arch=sm_120a','--cubin','-Xptxas=-v',str(root/'src/pipelined_grouped8.cu'),'-o',str(binary)]
    with (output/'compiler.log').open('wb') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0,(output/'compiler.log').read_text()[-5000:]
    (output/'build.json').write_text(json.dumps({'command':command,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()},indent=2))
    native=Native(binary);restart=Restarted(Path('/opt/reax-v3/private/src/grouped_hybrid8.cubin'))
    rows=[];cases=0
    for index in (12,20,30):
        sample=torch.load(f'/opt/reax-v3/private/output/precision-allocation-r3/layer-{index}-activation.pt',weights_only=True)
        x=sample['x'].to('cuda');L=sample['length'];old=ctx.assets.gate_up_banks[index];weight=ctx.full.core.ff.layers[index]['fused']['gate_up']
        representations={p:bank(weight,old,p) for p in (1,2,4)}
        routes=ctx.assets.payload['gate_up_routes']
        for M in (128,512,768,1024,len(x)):
            a=x[:M].contiguous();route=routes[min((k for k in routes if k[0]==128),key=lambda k:(k[1]!=L,abs(k[2]-len(a))))]
            base=lambda:ctx.assets.gate_up.project(a,old,route['kernel'],route['drop'])
            base_graph,base_out=captured(base);wanted=base_out.clone()
            gates=F.linear(a,weight);g,u=gates.chunk(2,-1);reference=(F.silu(g)*u).float()
            original_mse=float(((wanted.float()-reference)**2).mean())
            for parts,p in representations.items():
                sequential=restart.project(a,p,'grouped_b64n128_s32n128',0)
                for name in native.configs:
                    call=lambda name=name,p=p:native.project(a,p,name,0)
                    actual=call();equal=torch.equal(actual,sequential)
                    assert equal,('pipeline differs from identical quantization',index,M,parts,name,float((actual.float()-sequential.float()).abs().max()))
                    if parts==1 and not torch.equal(actual,wanted):
                        torch.save({'actual':actual.cpu(),'frozen':wanted.cpu(),'input':a.cpu()},output/'one-group-failure.pt')
                        raise AssertionError(('one-group frozen control differs',index,M,name,float((actual.float()-wanted.float()).abs().max())))
                    small=native.project(a[:16].contiguous(),p,name,0)
                    assert torch.equal(small,actual[:16]),'batch-shape rounding changed'
                    graph,out=captured(call);times={'base':[],'candidate':[]}
                    for _ in range(2):
                        for arm in ('base','candidate','candidate','base'):
                            times[arm].append(timing(base_graph if arm=='base' else graph))
                    medians={key:statistics.median(v) for key,v in times.items()}
                    mse=float(((actual.float()-reference)**2).mean());row={'layer':index,'rows':M,'parts':parts,'group':p.group,'kernel':name,'equal_restarted_grouped_quantization':equal,'batch_first16_exact':True,'milliseconds':medians,'operator_speedup':medians['base']/medians['candidate'],'mse_ratio':mse/original_mse}
                    rows.append(row);cases+=1
                    if cases%21==0:
                        (output/'results.json').write_text(json.dumps(rows,indent=2));print('PIPELINE_PROGRESS '+json.dumps({'cases':cases,'layer':index,'rows':M}),flush=True)
            (output/'results.json').write_text(json.dumps(rows,indent=2))
    assert cases==315,cases
    best=[]
    for index in (12,20,30):
        for M in sorted({r['rows'] for r in rows}):
            for parts in (1,2,4):best.append(max((r for r in rows if r['layer']==index and r['rows']==M and r['parts']==parts),key=lambda r:r['operator_speedup']))
    (output/'best.json').write_text(json.dumps(best,indent=2))
    return {'cases':cases,'all_bitwise_grouped_controls_passed':True,'all_single_group_frozen_controls_passed':True,'scope':'Actual activation operator screen. Best routes are timing selection only; quantization grouping must remain fixed across serving batch sizes.'}
