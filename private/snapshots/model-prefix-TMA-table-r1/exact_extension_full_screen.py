"""Whole-model exact controls and paired GPU costs of local component changes."""
from contextlib import contextmanager
import copy,ctypes as C,gc,json,statistics
import torch
from probe_utils import make_entries
from exact_local_base import model as base_model
from exact_local_extensions import Candidate

def dedicated(ctx):
 handle=C.c_void_p();assert C.CDLL('libcuda.so.1').cuStreamCreateWithPriority(C.byref(handle),C.c_uint(1),C.c_int(0))==0
 stream=torch.cuda.ExternalStream(handle.value,device=0);ctx.retained.setdefault('research_external_streams',[]).append((handle,stream));return stream

def capture(ctx,model,meta,slot):
 core=ctx.full.core;old=ctx.full.workspace
 @contextmanager
 def workspace():
  saved=core.bf.ws,core.ws_side;core.bf.ws,core.ws_side=slot['workspace']
  try:yield core
  finally:core.bf.ws,core.ws_side=saved
 ctx.full.workspace=workspace;stream=slot['stream'];core._sides[stream.cuda_stream]=slot['side'];stream.wait_stream(torch.cuda.current_stream())
 try:
  with torch.cuda.stream(stream):model(meta,True)
  stream.synchronize();g=torch.cuda.CUDAGraph(keep_graph=True)
  with torch.cuda.graph(g,pool=slot['pool'],stream=stream):value=model(meta,True)[0]
  from s1eng import kprio
  marks=kprio.GEMM_MARKS
  try:
   kprio.GEMM_MARKS=(*marks,*model.gemm_marks);kprio.apply(g,kprio.mode());g.instantiate()
  finally:kprio.GEMM_MARKS=marks
  g.replay();torch.cuda.synchronize();return g,value
 finally:ctx.full.workspace=old

def paired(graphs,streams,repeats=6):
 b,e=[torch.cuda.Event(enable_timing=True) for _ in range(2)];done=[torch.cuda.Event() for _ in graphs];b.record()
 for stream in streams:stream.wait_event(b)
 for _ in range(repeats):
  for g,stream in zip(graphs,streams):
   with torch.cuda.stream(stream):g.replay()
 for event,stream in zip(done,streams):event.record(stream);torch.cuda.current_stream().wait_event(event)
 e.record();e.synchronize();return b.elapsed_time(e)/repeats

def main(ctx,output):
 import reax_subnet.engrunner as er
 from reax_subnet.engvalidator import TRACKS
 component=base_model(ctx);extension=Candidate(component,ctx.retained['local_fold_projection']);models=dict(frozen=ctx.model,component=component,extension=extension)
 (output/'routes.json').write_text(json.dumps({str(k):v for k,v in extension.routes.items()},indent=2))
 docs=[d.body() for d in er.fresh_decisions(512,TRACKS['s1-fast'].pool)];(output/'requests.json').write_text(json.dumps(docs))
 keys=list(ctx.full.workspaces)[-2:];assert len(keys)==2
 slots=[dict(workspace=(ctx.full.workspaces[k],ctx.full.side_workspaces[k]),stream=dedicated(ctx),side=dedicated(ctx)) for k in keys];streams=[dedicated(ctx),dedicated(ctx)];controls=[];costs=[];raw=[]
 for count in (1,5,8):
  batches={192:[],256:[]}
  for part,entry in make_entries(ctx.encoder,ctx.gpu,docs,count=count):
   L=entry[2].L
   if L not in batches or len(batches[L])>=4:continue
   meta=copy.copy(entry[2])
   for key,value in vars(meta).items():
    if isinstance(value,torch.Tensor):setattr(meta,key,value.clone())
   batches[L].append((part,meta))
   if all(len(v)==4 for v in batches.values()):break
  for L,batch in batches.items():
   for case in range(0,len(batch),2):
    chosen=batch[case:case+2]
    if len(chosen)<2:continue
    graphs={};values={};expected=[]
    for side,(part,meta) in enumerate(chosen):
     want=ctx.model(meta,True)[0].clone();expected.append(want)
     for arm,model in models.items():
      graph,value=capture(ctx,model,meta,dict(slots[side],pool=torch.cuda.graph_pool_handle()));graphs[arm,side]=graph;values[arm,side]=value;eq=torch.equal(value.view(torch.int16),want.view(torch.int16));rec=dict(length=L,count=count,case=case,side=side,rows=len(meta.ids),arm=arm,all_bits_equal=eq);controls.append(rec)
      if not eq:torch.save(dict(requests=[d for d,_ in part],metadata={k:v.cpu() for k,v in vars(meta).items() if isinstance(v,torch.Tensor)},expected=want.cpu(),actual=value.cpu(),record=rec),output/f'failure-{L}-{count}-{case}-{side}-{arm}.pt')
      assert eq,rec
     raw.append(dict(requests=[d for d,_ in part],reference=want.cpu(),component=values['component',side].cpu(),extension=values['extension',side].cpu()))
    for arm in ('component','extension'):
     samples={'frozen':[],'candidate':[]}
     for label in ('frozen','candidate','candidate','frozen'):
      selected='frozen' if label=='frozen' else arm;samples[label].append(paired([graphs[selected,i] for i in (0,1)],streams))
      for i in (0,1):assert torch.equal(values[selected,i].view(torch.int16),expected[i].view(torch.int16))
     ms={k:statistics.median(v) for k,v in samples.items()};costs.append(dict(length=L,count=count,rows=[len(meta.ids) for _,meta in chosen],case=case,arm=arm,milliseconds=ms,speedup=ms['frozen']/ms['candidate']))
    (output/'costs.json').write_text(json.dumps(costs,indent=2));(output/'controls.json').write_text(json.dumps(controls,indent=2));del graphs,values,expected;gc.collect()
 torch.save(raw,output/'raw-controls.pt');ctx.retained['exact_extension_full']=dict(models=models,controls=controls,costs=costs,source=str(output))
 return dict(controls=len(controls),all_bits_equal=True,costs=costs,scope='Paired GPU graphs; no serving test until material whole-model gain is supported.')
