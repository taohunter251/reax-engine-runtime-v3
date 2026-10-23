"""Apply an exhaustive BF16 SiLU table to unchanged prefix tensor-core schedules."""
from pathlib import Path
import hashlib,json,shutil
r=Path(__file__).resolve().parent;out=r/'snapshots/model-exact-prefix-activation-table-r1';out.mkdir()
parent=r/'references/uid176-kernels-20261023/ngemm9b.cu';s=parent.read_text().split('}  // namespace ng9b')[0]+'}\n'
s=s.replace('#include <stdint.h>','#include <stdint.h>\n__device__ __nv_bfloat16 prefix_silu_values[65536];')
s=s.replace('float s = __bfloat162float(__float2bfloat16_rn(silu_f32(__bfloat162float(g))));','float s = __bfloat162float(__ldg(prefix_silu_values+__bfloat16_as_ushort(g)));')
s+='''
#define PREFIX(NAME,BIG,SMALL) \\
extern "C" {__device__ __constant__ unsigned int NAME##_required_smem=BIG::SMEM>SMALL::SMEM?BIG::SMEM:SMALL::SMEM;} \\
extern "C" __global__ __launch_bounds__(256,1) void NAME(const __nv_bfloat16* A,const __nv_bfloat16* W,__nv_bfloat16* out,int M,int N,int K,int mb) { \\
 extern __shared__ __align__(128) unsigned char smem[]; \\
 ng9b::hetero<BIG,SMALL,1>(A,W,out,M,N,2560,2560,N/2,mb,reinterpret_cast<__nv_bfloat16*>(smem)); }
extern "C" __global__ void initialize_silu_table() {unsigned int b=blockIdx.x*blockDim.x+threadIdx.x;if(b<65536)prefix_silu_values[b]=__float2bfloat16_rn(ng9b::silu_f32(__bfloat162float(__ushort_as_bfloat16((unsigned short)b))));}
extern "C" __global__ void check_silu_domain(__nv_bfloat16* a,__nv_bfloat16* b) {unsigned int bits=blockIdx.x*blockDim.x+threadIdx.x;if(bits<65536){a[bits]=__float2bfloat16_rn(ng9b::silu_f32(__bfloat162float(__ushort_as_bfloat16((unsigned short)bits))));b[bits]=prefix_silu_values[bits];}}
'''
# Original frozen NG9B/NG9T tile geometries, not reselected variants.
specs=[('b128x256k64_s64x256k64',128,256,64,2,4,2,64,256,64,2,4,2),('b128x256k64_s32x256k64',128,256,64,2,4,2,32,256,64,1,8,2),('b128x256k64_s64x256',128,256,64,2,4,2,64,256,32,2,4,4),('b128x256k64_s32x256',128,256,64,2,4,2,32,256,32,1,8,4),('b128x128k64_s64x128k64',128,128,64,2,4,3,64,128,64,2,4,3),('b128x128k64_s32x128',128,128,64,2,4,3,32,128,32,1,8,4)]
for tail in (160,192,224):specs.append((f'b128x256k64_t{tail}x128k64',128,256,64,2,4,2,tail,128,64,2,4,2))
specs.append(('b128x256k64_s96x256k64',128,256,64,2,4,2,96,256,64,2,4,2));variants={};aliases={}
for i,(original,bm,bn,bk,wm,wn,st,sm,sn,sk,swm,swn,ss) in enumerate(specs):
 name=f'prefix_table_{i}';s+=f'using B{i}=ng9b::Cfg<{bm},{bn},{bk},{wm},{wn},{st},1>;\nusing S{i}=ng9b::Cfg<{sm},{sn},{sk},{swm},{swn},{ss},1>;\nPREFIX({name},B{i},S{i})\n';variants[name]=[bm,bn,sm,sn,256,'swiglu'];aliases['nsw9_'+original]=name
(out/'prefix_axis.cu').write_text(s);(out/'prefix_shapes.json').write_text(json.dumps(variants,indent=2));(out/'original-aliases.json').write_text(json.dumps(aliases,indent=2));shutil.copy2(r/'bf16_prefix_axis.py',out/'bf16_prefix_axis.py')
(out/'attribution.json').write_text(json.dumps(dict(parent='Public MIT UID176 NG9B/NG9T geometries',source_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),change='Original frozen BF16 prefix kernel layouts, same ascending k16 reductions. BF16 SiLU output from exhaustive65536-value GPU initialized exact function table, not approximate activation.'),indent=2));shutil.copy2(parent.parent/'LICENSE',out/'LICENSE');shutil.copy2(__file__,out/Path(__file__).name)
print(out)
