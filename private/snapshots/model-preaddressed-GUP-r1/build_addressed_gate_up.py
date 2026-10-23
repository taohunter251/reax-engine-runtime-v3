"""Port the attributed NG9B loader/schedule to frozen protected INT8 products."""
from pathlib import Path
import hashlib,json,shutil

r=Path(__file__).resolve().parent
out=r/'snapshots/model-preaddressed-GUP-r1';out.mkdir()
parent=Path('/root/workspace/reax-engine-v2/vendor/s1eng/ngemm9b.cu')
base=parent.read_text().split('// plain epilogue:')[0]
base=base.replace('template <class C, int FOLD = 0>','template <class C, int FOLD = 0, bool ADD = false>')
start=base.index('#pragma unroll\n    for (int i = 0; i < MT; ++i)',base.index('DEVI void mainloop'))
end=base.index('    Loader<C> ld;',start)
base=base[:start]+'    if constexpr (!ADD) {\n'+base[start:end]+'    }\n'+base[end:]
integer=base.replace('namespace ng9b {','namespace addressed_integer {')
begin=integer.index('DEVI void mma16816(');end=integer.index('DEVI void named_sync',begin)
integer=integer[:begin]+'''DEVI void mma16816(float* d,const uint32_t* a,uint32_t b0,uint32_t b1) {
 uint32_t c0=__float_as_uint(d[0]),c1=__float_as_uint(d[1]);
 uint32_t c2=__float_as_uint(d[2]),c3=__float_as_uint(d[3]);
 asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
  : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3)
  : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
 d[0]=__uint_as_float(c0);d[1]=__uint_as_float(c1);
 d[2]=__uint_as_float(c2);d[3]=__uint_as_float(c3);
}
'''+integer[end:]
exact=base.replace('namespace ng9b {','namespace addressed_exact {')
(out/'addressed_tiles.cuh').write_text(integer+'\n}\n'+exact+'\n}\n')
source=(r/'src/register_hybrid8.cu').read_text().split('using H128256=')[0]
source=source.replace('#include "register_hybrid_tiles.cuh"','#include "addressed_tiles.cuh"').replace('exact_tiles::swiglu1','addressed_exact::swiglu1')
old='''integer_tiles::mainloop<C,0>(reinterpret_cast<const __nv_bfloat16*>(A),
      reinterpret_cast<const __nv_bfloat16*>(W),M,K/2,K/2,m0,o0,0,K/2,K/2,
      smem,acc,unused,N/2+o0);'''
new='''addressed_integer::mainloop<C>(reinterpret_cast<const __nv_bfloat16*>(A),
      reinterpret_cast<const __nv_bfloat16*>(W),M,K/2,K/2,m0,o0,N/2+o0,smem,acc);'''
assert old in source
source=source.replace(old,new)
old='exact_tiles::mainloop<C,0,true>(AH,WH,M,KH,KH,m0,o0,0,KH,KH,smem,acc,unused,N/2+o0);'
assert old in source
source=source.replace(old,'addressed_exact::mainloop<C,0,true>(AH,WH,M,KH,KH,m0,o0,N/2+o0,smem,acc);')
source+='''
template<int BM,int BN,int BK,int WM,int WN,int ST>
struct Pipe : addressed_integer::Cfg<BM,BN,BK,WM,WN,ST,1> {
 using Base=addressed_integer::Cfg<BM,BN,BK,WM,WN,ST,1>;
 static constexpr int SMEM=Base::SMEM_PIPE;
};
#define ADDRESSED(NAME,BIG,SMALL,T,R) \\
extern "C" { __device__ __constant__ unsigned int NAME##_required_smem=BIG::SMEM>SMALL::SMEM?BIG::SMEM:SMALL::SMEM; } \\
extern "C" __global__ __launch_bounds__(T,R) void NAME( \\
 const int8_t* A,const int8_t* W,const __nv_bfloat16* AH,const __nv_bfloat16* WH, \\
 const float* sa,const float* sw,const float* bias,__nv_bfloat16* out, \\
 int M,int N,int K,int KH,int mb) { \\
 extern __shared__ __align__(128) unsigned char memory[]; \\
 hybrid_hetero<BIG,SMALL>(A,W,AH,WH,sa,sw,bias,out,M,18432,2432,128,mb,reinterpret_cast<__nv_bfloat16*>(memory)); }
'''
specs=[(128,256,32,2,4,3,1),(128,256,32,4,2,3,1),
       (128,256,64,2,4,2,1),(128,256,64,4,2,2,1),
       (64,256,32,2,4,3,1),(64,256,64,2,4,2,1),
       (128,128,32,4,2,3,2),(128,128,64,4,2,3,1),
       (128,192,32,4,3,3,1),(64,192,32,2,3,2,2)]
variants={}
for i,(bm,bn,bk,wm,wn,st,res) in enumerate(specs):
    name=f'addressed_{i}_m{bm}n{bn}k{bk}'
    swm,swn=(1,wm*wn) if (bn//(wm*wn))%16==0 else (2,wm*wn//2)
    assert bn%swn==0 and (bn//swn)%16==0
    source+=f'using B{i}=Pipe<{bm},{bn},{bk},{wm},{wn},{st}>;\n'
    source+=f'using S{i}=Pipe<32,{bn},32,{swm},{swn},2>;\n'
    source+=f'ADDRESSED({name},B{i},S{i},{wm*wn*32},{res})\n'
    variants[name]=[bm,32,bn,wm*wn*32]
(out/'addressed_gate_up.cu').write_text(source)
(out/'wide_warp_shapes.json').write_text(json.dumps(variants,indent=2))
shutil.copy2(r/'wide_warp_hybrid.py',out/'wide_warp_hybrid.py')
screen=(r/'snapshots/model-wave-balanced-GUP-r1/resident_gate_up_screen.py').read_text()
screen=screen.replace('wave_gate_up.cubin','addressed_gate_up.cubin').replace("ctx.retained['wave_balanced_gate_up']","ctx.retained['preaddressed_gate_up']")
(out/'resident_gate_up_screen.py').write_text(screen)
for file in ('pipeline_screen.py','probe_utils.py'):shutil.copy2(r/file,out/file)
(out/'attribution.json').write_text(json.dumps(dict(parent='Public UID56 NG9B mainloop plus prior attributed UID56/164 hybrid conversion.',
    ng9b_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
    local_change='Use precomputed loader offsets and serpentine instruction order for frozen INT8/BF16 banks; preserve original rounding and columns.'),indent=2))
shutil.copy2(__file__,out/Path(__file__).name)
print(out)
