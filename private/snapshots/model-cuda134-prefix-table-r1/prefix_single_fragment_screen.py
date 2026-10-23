"""Real model prefix operands and exact BF16 operator checks before model use."""
import hashlib,json,statistics,ctypes as C
import torch
from probe_utils import make_entries
from bf16_prefix_axis import Native

def captured(call):
 call();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):out=call()
 g.replay();torch.cuda.synchronize();return g,out

def timing(g,repeats=6):
 a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)];a.record()
 for _ in range(repeats):g.replay()
 b.record();b.synchronize();return a.elapsed_time(b)/repeats

def main(ctx,output):
 import reax_subnet.engrunner as er
 from reax_subnet.engvalidator import TRACKS
 native=Native();core=ctx.full.core;samples={};old=core._mlp
 handles=[]
 for _ in range(2):
  handle=C.c_void_p();assert C.CDLL('libcuda.so.1').cuStreamCreateWithPriority(C.byref(handle),C.c_uint(1),C.c_int(0))==0;handles.append((handle,torch.cuda.ExternalStream(handle.value,device=0)))
 stream,side=[v[1] for v in handles];ctx.retained.setdefault('research_external_streams',[]).extend(handles);core._sides[stream.cuda_stream]=side
 def captured(call):
  stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):call()
  stream.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g,stream=stream):out=call()
  g.replay();torch.cuda.synchronize();return g,out
 def observe(layer,x,L,defer=False):
  i=next((i for i,d in enumerate(core.ff.layers) if d is layer),None)
  if i in (4,8) and L in (192,256) and (i,L) not in samples:samples[i,L]=x.clone()
  return old(layer,x,L,defer)
 docs=[d.body() for d in er.fresh_decisions(128,TRACKS['s1-fast'].pool)];(output/'requests.json').write_text(json.dumps(docs));core._mlp=observe
 try:
  for part,entry in make_entries(ctx.encoder,ctx.gpu,docs,count=8):
   ctx.model(entry[2],True)
   if len(samples)==4:break
 finally:core._mlp=old
 assert samples;results=[];evict=torch.zeros(64<<20,device='cuda',dtype=torch.float32)
 for index,((layer_index,L),x) in enumerate(samples.items()):
  layer=core.ff.layers[layer_index];source=f'source-{index}.pt';torch.save(dict(input=x.cpu(),layer=layer_index,length=L,weights='Pinned BF16 model layer.fused.gate_up/qkv_z, unchanged'),output/source)
  for count in (128,512,640,768,896,1024):
   if count>len(x):continue
   a=x[:count].contiguous()
   def gate_up():
    linear=core.lin
    def lin(module,values,length):return values if module is layer['mlp'].down_proj else linear(module,values,length)
    core.lin=lin
    try:return old(layer,a,L,defer=False)
    finally:core.lin=linear
   for kind,w,call in (('swiglu',layer['fused']['gate_up'],gate_up),('plain',layer['fused']['qkv_z'],lambda:core.gemm(layer['fused']['qkv_z'],a,L))):
    if kind=='plain' and not core._l0t_fused(layer,'qkv_z',len(a),L):continue
    with ctx.full.workspace():bg,wanted=captured(call)
    wanted=wanted.clone()
    for name,(_,_,_,_,_,which) in native.variants.items():
     if which!=kind:continue
     cg,value=captured(lambda:native.product(a,w,name));equal=torch.equal(value.view(torch.int16),wanted.view(torch.int16));record=dict(layer=layer_index,length=L,rows=count,kind=kind,kernel=name,all_bits_equal=equal)
     if not equal:torch.save(dict(source_file=source,source_rows=count,weight_family=kind,expected=wanted.cpu(),actual=value.cpu(),record=record),output/f'failure-{index}-{count}-{name}.pt')
     else:
      hot={'base':[],'candidate':[]};cold={'base':[],'candidate':[]}
      for arm in ('base','candidate','candidate','base'):
       selected=bg if arm=='base' else cg;hot[arm].append(timing(selected));evict.add_(1);b,e=[torch.cuda.Event(enable_timing=True) for _ in range(2)];b.record();selected.replay();e.record();e.synchronize();cold[arm].append(b.elapsed_time(e))
      ms={k:statistics.median(v) for k,v in hot.items()};cms={k:statistics.median(v) for k,v in cold.items()};record.update(milliseconds=ms,cold_milliseconds=cms,speedup=ms['base']/ms['candidate'],cold_speedup=cms['base']/cms['candidate'])
     results.append(record)
    (output/'operators.json').write_text(json.dumps(results,indent=2))
 ctx.retained['BF16_single_fragment']=dict(native=native,operators=results,samples=samples,source=str(output))
 return dict(cases=len(results),all_bits_equal=all(r['all_bits_equal'] for r in results),scope='Same pinned BF16 prefix arithmetic and parameter values; no precision change or serving benchmark.')
