"""Use the public NG9B address schedule for unchanged grouped INT8 projections."""
from pathlib import Path
import hashlib,json,shutil

r=Path(__file__).resolve().parent
out=r/'snapshots/model-preaddressed-sliced-projections-r1';out.mkdir()
parent=r/'snapshots/model-preaddressed-GUP-r1/addressed_tiles.cuh'
header=parent.read_text()
header=header.replace('float (*tot)[C::NT][4] = nullptr, int chunk = 0)',
                      'float (*tot)[C::NT][4] = nullptr, int chunk = 0, int ldb = 0)')
header=header.replace('ld.init(A, B, m0, n0, n1, M, K, lda);','ld.init(A, B, m0, n0, n1, M, ldb ? ldb : K, lda);')
original=Path('/root/workspace/reax-engine-v2/vendor/s1eng/ngemm9b.cu').read_text()
store=original[original.index('// plain epilogue:'):original.index('// SwiGLU register epilogue:')]
header+='\nnamespace addressed_exact {\n'+store+'\n}\n'
(out/'addressed_sliced_tiles.cuh').write_text(header)
source=Path('/root/workspace/reax-engine-v2/kernel-source/src/sliced_linear8.cu').read_text().split('using L128N128=')[0]
source=source.replace('#include "hybrid8_tiles.cuh"','#include "addressed_sliced_tiles.cuh"')
begin=source.index('    integer_tiles::mainloop<C,0>');end=source.index('    const int warp=',begin)
source=source[:begin]+'''    addressed_integer::mainloop<C>(reinterpret_cast<const __nv_bfloat16*>(a)+part*group/2,
        reinterpret_cast<const __nv_bfloat16*>(w)+part*group/2,M,group/2,KL/2,row,col,-1,
        shared,acc,nullptr,0,KL/2);
'''+source[end:]
begin=source.index('    if(hp>0)');end=source.index('    exact_tiles::store_tile',begin)
source=source[:begin]+'''    if(hp>0)
        addressed_exact::mainloop<C,0,true>(ah+part*hp,wh+part*hp,M,hp,KH,row,col,-1,
                                          shared,acc,nullptr,0,KH);
'''+source[end:]
source=source.replace('exact_tiles::store_tile','addressed_exact::store_tile')
source+='''
#define SLICED(NAME,BIG,SMALL,T,R) \\
extern "C" { __device__ __constant__ unsigned int NAME##_required_smem=BIG::SMEM>SMALL::SMEM?BIG::SMEM:SMALL::SMEM; } \\
extern "C" __global__ __launch_bounds__(T,R) void NAME( \\
 const int8_t* a,const int8_t* w,const __nv_bfloat16* ah,const __nv_bfloat16* wh, \\
 const float* sa,const float* sw,const float* bias,__nv_bfloat16* out, \\
 int M,int N,int KL,int KH,int group,int hp,int mb) { \\
 extern __shared__ __align__(128) unsigned char shared[]; \\
 sliced_dispatch<BIG,SMALL>(a,w,ah,wh,sa,sw,bias,out,M,N,KL,KH,group,hp,mb, \\
                           reinterpret_cast<__nv_bfloat16*>(shared)); }
'''
specs=[(128,128,32,4,2,3,2),(128,128,64,4,2,3,1),
       (64,128,32,2,4,3,2),(64,128,32,2,2,3,3),
       (128,256,32,2,4,3,1),(128,256,64,2,4,2,1),
       (64,256,32,2,4,3,1),(64,256,64,2,4,2,1)]
variants={}
for i,(bm,bn,bk,wm,wn,st,res) in enumerate(specs):
    name=f'scheduled_{i}_m{bm}n{bn}k{bk}'
    swm,swn=(1,wm*wn) if (bn//(wm*wn))%16==0 else (2,wm*wn//2)
    source+=f'using B{i}=addressed_integer::Cfg<{bm},{bn},{bk},{wm},{wn},{st},0>;\n'
    source+=f'using S{i}=addressed_integer::Cfg<32,{bn},32,{swm},{swn},2,0>;\n'
    source+=f'SLICED({name},B{i},S{i},{wm*wn*32},{res})\n'
    variants[name]=[bm,32,bn,wm*wn*32,bk]
(out/'addressed_sliced.cu').write_text(source)
(out/'sliced_shapes.json').write_text(json.dumps(variants,indent=2))
for file in ('addressed_sliced.py','addressed_sliced_screen.py','pipeline_screen.py','probe_utils.py'):shutil.copy2(r/file,out/file)
(out/'attribution.json').write_text(json.dumps(dict(parent='Public UID56 NG9B loader plus prior attributed sliced_linear8.',
    parent_header_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
    local_change='Separate loop-depth and matrix-stride for grouped INT8 products, retaining original BF16 partials and ordered fold.'),indent=2))
shutil.copy2(__file__,out/Path(__file__).name)
print(out)
