"""Independent calibration of protected INT8 gate/up in the canonical prefix.

No answers choose channels or weights. Statistics come from a separate fresh
unlabeled split; all other frozen v2 projections keep their representations.
"""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import json,statistics
import numpy as np
import torch
from reaxv2.projection import MergedCheckpoint
from probe_utils import make_entries
from pipeline_screen import captured,timing


def bank(weight,stats,protected):
    mean=(stats['sum']/stats['count']).float()
    energy=(stats['square']/stats['count']-mean.double().square()).clamp_min(1.e-12).float().sqrt()
    wf=weight.float();importance=energy*wf.square().sum(0).sqrt()
    high=importance.topk(protected).indices.sort().values
    keep=torch.ones(weight.shape[1],device=weight.device,dtype=torch.bool);keep[high]=False
    low=keep.nonzero().flatten();values=wf[:,low]
    centers=mean[low].contiguous();amax=stats['maximum'][low]+centers.abs()
    proposed=(amax.clamp_min(1.e-8)/values.abs().amax(0).clamp_min(1.e-8)).sqrt()
    smoothing=2.**proposed.log2().round().clamp(-8,8)
    scaled=values*smoothing;scales=scaled.abs().amax(1).clamp_min(1.e-12)/127.
    codes=(scaled/scales[:,None]).round().clamp(-127,127).to(torch.int8)
    bias=(centers.double()[None,:]@values.double().T).flatten().float()
    return SimpleNamespace(width=weight.shape[1],centered=True,
        low_columns=low.to(torch.int32),high_columns=high.to(torch.int32),
        low_weight=codes.contiguous(),high_weight=weight[:,high].contiguous(),
        smoothing=smoothing.contiguous(),centers=centers,scales=scales.contiguous(),bias=bias.contiguous())


def main(ctx,output):
    import reax_subnet.engrunner as er
    from reax_subnet.engvalidator import TRACKS
    from reax_subnet.engcomp import check_correctness
    from s1eng.server import Engine
    from s1eng.prompt import options_of
    params=TRACKS['s1-fast'];core=ctx.full.core;indices={id(d):i for i,d in enumerate(core.ff.layers)}
    calibration=[d.body() for d in er.fresh_decisions(128,params.pool)]
    (output/'calibration-requests.json').write_text(json.dumps(calibration));stats={};real_rows=0
    old=core._mlp
    def observe(layer,x,L,defer=False):
        index=indices[id(layer)]
        if 4<=index<12:
            v=x[:real_rows].float();row=stats.setdefault(index,{'count':0,'sum':torch.zeros(x.shape[1],device=x.device,dtype=torch.float64),'square':torch.zeros(x.shape[1],device=x.device,dtype=torch.float64),'maximum':torch.zeros(x.shape[1],device=x.device)})
            row['count']+=len(v);row['sum']+=v.double().sum(0);row['square']+=v.double().square().sum(0);row['maximum']=torch.maximum(row['maximum'],v.abs().amax(0))
        return old(layer,x,L,defer=defer)
    core._mlp=observe
    try:
        for _,entry in make_entries(ctx.encoder,ctx.gpu,calibration,count=8):
            real_rows=int((entry[2].slen-entry[2].plen).sum());ctx.model(entry[2],False)
    finally:core._mlp=old
    banks={};selection={}
    for index,row in stats.items():
        for protected in (256,512):
            p=bank(core.ff.layers[index]['fused']['gate_up'],row,protected);banks[index,protected]=p
            selection[str((index,protected))]=p.high_columns.cpu().tolist()
    torch.save({i:{k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in s.items()} for i,s in stats.items()},output/'calibration-statistics.pt')
    (output/'channel-selection.json').write_text(json.dumps(selection))
    recipes={'v2-fast':None,'start10-protect512':(10,512),'start8-protect512':(8,512),'start8-protect256':(8,256),'start4-protect512':(4,512)}
    models={'v2-fast':ctx.model}
    for name,recipe in recipes.items():
        if recipe is None:continue
        first,protected=recipe;m=MergedCheckpoint(ctx.full,('down','gdn_input','gdn_output'),first=first,start=first)
        m.gup.bank=dict(ctx.assets.gate_up_banks)
        for index in range(first,12):m.gup.bank[index]=banks[index,protected]
        models[name]=m
    cases=json.loads(Path('/workspace/reax-precision-search/output/v2-image-peers-server-r1/failure-cases.json').read_text())
    regressions=list({c['request']['id']:c['request'] for c in cases if c['arm']=='ours'}.values())
    fresh=[d.body() for d in er.fresh_decisions(2048,params.pool)]
    assert not ({d['id'] for d in calibration}&{d['id'] for d in fresh})
    (output/'requests.json').write_text(json.dumps({'fresh':fresh,'regressions':regressions}));engine=Engine.__new__(Engine);engine.temperature=2.07
    def probability(doc,row):
        q=next(iter(doc['questions'].values()));return list(engine._answer(q['type'],options_of(q),row)['probabilities'].values())
    summary={};all_logits={}
    for split,documents in (('fresh',fresh),('regressions',regressions)):
        raw={};refs={};answers={name:{} for name in recipes};byid={d['id']:d for d in documents}
        for step,(part,entry) in enumerate(make_entries(ctx.encoder,ctx.gpu,documents,count=8)):
            expected=ctx.model(entry[2],False)[0].float().cpu().numpy()
            for b,(doc,_) in enumerate(part):refs[doc['id']]=probability(doc,expected[b]);raw[doc['id']]={'reference':expected[b].tolist()}
            for name,m in models.items():
                if step==0:
                    canonical=m(entry[2],False)[0].float().cpu().numpy();assert np.array_equal(canonical,expected),('checkpoint moved canonical bits',name)
                got=m(entry[2],True)[0].float().cpu().numpy()
                for b,(doc,_) in enumerate(part):raw[doc['id']][name]=got[b].tolist();answers[name][doc['id']]=probability(doc,got[b])
            if step%16==0:print('EARLIER_GATE_UP_PROGRESS '+json.dumps({'split':split,'items':len(refs)}),flush=True)
        rows=[]
        for name in recipes:
            checks=[];errors=[]
            for ident in refs:
                if np.argmax(answers[name][ident])!=np.argmax(refs[ident]):errors.append({'request':byid[ident],'reference_probabilities':refs[ident],'candidate_probabilities':answers[name][ident],'logits':raw[ident]})
            for margin in (.125,.25,.375):
                count=0;wrong=0;corrected={}
                for ident in refs:
                    scores=np.sort(raw[ident][name][:len(refs[ident])]);guard=len(scores)>1 and scores[-1]-scores[-2]<=margin
                    count+=int(guard);corrected[ident]=refs[ident] if guard else answers[name][ident];wrong+=int(np.argmax(corrected[ident])!=np.argmax(refs[ident]))
                check=check_correctness(corrected,refs,params);checks.append({'margin':margin,'exact_rechecks':count,'post_recheck_top_errors':wrong,'quality_ok':check.ok,'mean_distance':check.mean_distance,'max_distance':check.max_distance})
            rows.append({'recipe':name,'items':len(refs),'raw_top_errors':len(errors),'guards':checks})
            (output/(split+'-failures-'+name+'.json')).write_text(json.dumps(errors))
        summary[split]=rows;(output/'summary.json').write_text(json.dumps(summary,indent=2));(output/(split+'-raw.json')).write_text(json.dumps(raw));(output/(split+'-probabilities.json')).write_text(json.dumps({'reference':refs,'candidates':answers}))
        print('EARLIER_GATE_UP_SUMMARY '+json.dumps({'split':split,'recipes':rows}),flush=True)
    costs=[]
    for i,(_,entry) in enumerate(make_entries(ctx.encoder,ctx.gpu,fresh[:64],count=5)):
        if i>=6:break
        base,_=captured(lambda:ctx.model(entry[2],True)[0])
        for name,m in models.items():
            if name=='v2-fast':continue
            candidate,_=captured(lambda:m(entry[2],True)[0]);samples={'base':[],'candidate':[]}
            for _ in range(3):
                for arm in ('base','candidate','candidate','base'):samples[arm].append(timing(base if arm=='base' else candidate,repeats=4))
            medians={arm:statistics.median(x) for arm,x in samples.items()};costs.append({'recipe':name,'L':entry[2].L,'B':entry[2].B,'rows':entry[0],'milliseconds':medians,'full_graph_speedup':medians['base']/medians['candidate']})
        (output/'graph-costs.json').write_text(json.dumps(costs,indent=2))
    ctx.retained['earlier_gate_up']={'banks':banks,'models':models,'summary':summary,'recipes':recipes,'costs':costs,'source':str(output)}
    return {'fresh_unlabeled_calibration':128,'fresh_independent_eval':2048,'recipes':len(recipes),'scope':'Full model early gate/up quantization plus canonical repair screen; no serving gain claimed.'}
