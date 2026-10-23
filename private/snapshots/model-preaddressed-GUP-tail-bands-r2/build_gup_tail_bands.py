"""Keep frozen arithmetic while matching paired GUP tails to 32-row buckets."""
from pathlib import Path
import hashlib,json,shutil

r=Path(__file__).resolve().parent
parent=r/'snapshots/model-preaddressed-GUP-r1'
out=r/'snapshots/model-preaddressed-GUP-tail-bands-r2';out.mkdir()
shutil.copy2(parent/'addressed_tiles.cuh',out/'addressed_tiles.cuh')
source=(parent/'addressed_gate_up.cu').read_text().split('template<int BM,int BN,int BK,int WM,int WN,int ST>')[0]
source+='''
template<int BM,int BN,int BK,int WM,int WN,int ST>
struct Pipe : addressed_integer::Cfg<BM,BN,BK,WM,WN,ST,1> {
 using Base=addressed_integer::Cfg<BM,BN,BK,WM,WN,ST,1>;
 static constexpr int SMEM=Base::SMEM_PIPE;
};
#define BAND(NAME,BIG,SMALL) \\
extern "C" { __device__ __constant__ unsigned int NAME##_required_smem=BIG::SMEM>SMALL::SMEM?BIG::SMEM:SMALL::SMEM; } \\
extern "C" __global__ __launch_bounds__(256,1) void NAME( \\
 const int8_t* A,const int8_t* W,const __nv_bfloat16* AH,const __nv_bfloat16* WH, \\
 const float* sa,const float* sw,const float* bias,__nv_bfloat16* out, \\
 int M,int N,int K,int KH,int mb) { \\
 extern __shared__ __align__(128) unsigned char memory[]; \\
 hybrid_hetero<BIG,SMALL>(A,W,AH,WH,sa,sw,bias,out,M,18432,2432,128,mb,reinterpret_cast<__nv_bfloat16*>(memory)); }
'''
tails=[(64,256,32,2,4,3),(64,256,64,2,4,2),
       (96,256,32,2,4,3),(96,256,64,2,4,2),
       (160,128,32,2,4,3),(192,128,32,2,4,3),(224,128,32,2,4,3),
       (160,128,64,2,4,2),(192,128,64,2,4,2),(224,128,64,2,4,2)]
variants={}
for bk,st in ((32,3),(64,2)):
    for sm,sn,sk,swm,swn,ss in tails:
        name=f'band_k{bk}_t{sm}n{sn}k{sk}'
        source+=f'using B_{name}=Pipe<128,256,{bk},2,4,{st}>;\n'
        source+=f'using S_{name}=Pipe<{sm},{sn},{sk},{swm},{swn},{ss}>;\n'
        source+=f'BAND({name},B_{name},S_{name})\n'
        variants[name]=[128,sm,256,256,sn]
(out/'band_gate_up.cu').write_text(source)
(out/'wide_warp_shapes.json').write_text(json.dumps(variants,indent=2))
wrapper=(r/'wide_warp_hybrid.py').read_text()
wrapper=wrapper.replace('bm,small,bn,threads=self.variants[name]','bm,small,bn,threads,sn=self.variants[name]')
wrapper=wrapper.replace('blocks=(mb//bm+math.ceil((m-mb)/small))*(n//bn)','blocks=(mb//bm)*(n//bn)+math.ceil((m-mb)/small)*(n//sn)')
(out/'wide_warp_hybrid.py').write_text(wrapper)
for file in ('gup_tail_band_screen.py','pipeline_screen.py','probe_utils.py'):shutil.copy2(r/file,out/file)
(out/'attribution.json').write_text(json.dumps(dict(parent='Attributed preaddressed-GUP-r1 derived from public UID56/164.',
    parent_source_sha256=hashlib.sha256((parent/'addressed_gate_up.cu').read_bytes()).hexdigest(),
    local_change='Protected-INT8 paired gates with64/96/160/192/224-row tail bands and independent tail column width.'),indent=2))
shutil.copy2(__file__,out/Path(__file__).name)
print(out)
