"""Full BF16-domain proof and real prefix SwiGLU cost under unchanged tile routes."""
import json,statistics,ctypes as C,hashlib
import torch
from probe_utils import make_entries
from prefix_table_native import Native
from prefix_single_fragment_screen import timing,captured

def main(ctx,output):
 import reax_subnet.engrunner as er
 from reax_subnet.engvalidator import TRACKS
 native=Native();domain=native.domain_check();assert domain['all_bits_equal'];core=ctx.full.core;samples={};old=core._mlp
 def values(owner):
  ptr=C.c_void_p();size=C.c_size_t();owner.check(owner.cuda.cuModuleGetGlobal_v2(C.byref(ptr),C.byref(size),owner.module,b'prefix_silu_values'));assert size.value==131072;raw=(C.c_ubyte*size.value)();owner.check(owner.cuda.cuMemcpyDtoH_v2(raw,ptr,size.value));return torch.frombuffer(bytearray(raw),dtype=torch.int16).clone()
 current=values(native);previous=values(ctx.retained['prefix_activation_table']['native']);bits=torch.arange(65536,dtype=torch.int32).to(torch.int16);wanted=torch.nn.functional.silu(bits.to('cuda').view(torch.bfloat16)).cpu().view(torch.int16);finite=(bits.to(torch.int32)&0x7f80)!=0x7f80
 comparison=dict(cases=65536,finite_cases=int(finite.sum()),new_vs_previous_differences=int((current!=previous).sum()),new_vs_torch_finite_differences=int(((current!=wanted)&finite).sum()),previous_vs_torch_finite_differences=int(((previous!=wanted)&finite).sum()),new_vs_torch_all_differences=int((current!=wanted).sum()))
 torch.save(dict(input_bits=bits,previous=previous,current=current,torch_reference=wanted,finite_mask=finite),output/'exhaustive-torch-silu-control.pt');(output/'torch-domain-control.json').write_text(json.dumps(comparison,indent=2));assert comparison['new_vs_torch_finite_differences']==0,comparison;domain.update(torch_reference=comparison)

 def observe(layer,x,L,defer=False):
  index=next(i for i,d in enumerate(core.ff.layers) if d is layer)
  if index in (4,8) and L in (192,256) and (index,L) not in samples:samples[index,L]=x.clone()
  return old(layer,x,L,defer)
 docs=[d.body() for d in er.fresh_decisions(128,TRACKS['s1-fast'].pool)];(output/'requests.json').write_text(json.dumps(docs));core._mlp=observe
 try:
  for part,entry in make_entries(ctx.encoder,ctx.gpu,docs,count=8):
   ctx.model(entry[2],True)
   if len(samples)==4:break
 finally:core._mlp=old
 rows=[];evict=torch.zeros(64<<20,device='cuda',dtype=torch.float32)
 for index,((layer_index,L),x) in enumerate(samples.items()):
  layer=core.ff.layers[layer_index];w=layer['fused']['gate_up'];torch.save(dict(input=x.cpu(),layer=layer_index,length=L),output/f'source-{index}.pt')
  for count in (128,384,512,640,768,896,1024):
   if count>len(x):continue
   a=x[:count].contiguous();algo=core.pinned.swiglu_r.get((L,count));key=getattr(algo,'name',None)
   if key not in native.aliases:continue
   name=native.aliases[key];drop=algo.drop
   def original():
    linear=core.lin
    def lin(mod,values,length):return values if mod is layer['mlp'].down_proj else linear(mod,values,length)
    core.lin=lin
    try:return old(layer,a,L,defer=False)
    finally:core.lin=linear
   with ctx.full.workspace():bg,want=captured(original)
   want=want.clone();cg,value=captured(lambda:native.product(a,w,name,drop));eq=torch.equal(value.view(torch.int16),want.view(torch.int16));rec=dict(layer=layer_index,length=L,rows=count,original=key,kernel=name,drop=drop,all_bits_equal=eq)
   if not eq:torch.save(dict(source_file=f'source-{index}.pt',expected=want.cpu(),actual=value.cpu(),record=rec),output/f'failure-{index}-{count}.pt')
   else:
    hot={'base':[],'candidate':[]};cold={'base':[],'candidate':[]}
    for arm in ('base','candidate','candidate','base'):
     graph=bg if arm=='base' else cg;hot[arm].append(timing(graph));evict.add_(1);b,e=[torch.cuda.Event(enable_timing=True) for _ in range(2)];b.record();graph.replay();e.record();e.synchronize();cold[arm].append(b.elapsed_time(e))
    ms={k:statistics.median(v) for k,v in hot.items()};cms={k:statistics.median(v) for k,v in cold.items()};rec.update(milliseconds=ms,cold_milliseconds=cms,speedup=ms['base']/ms['candidate'],cold_speedup=cms['base']/cms['candidate'])
   rows.append(rec)
  (output/'operators.json').write_text(json.dumps(rows,indent=2))
 ctx.retained['cuda134_prefix_table']=dict(native=native,domain=domain,operators=rows,source=str(output))
 return dict(domain=domain,cases=len(rows),all_bits_equal=all(r['all_bits_equal'] for r in rows),scope='Same frozen prefix SwiGLU tile and BF16 operations; GPU-generated exhaustive table, no approximate nonlinearities.')
