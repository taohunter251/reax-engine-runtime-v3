"""Add an exhaustive BF16 activation function to the exact addressed pipeline."""
from pathlib import Path
import hashlib,json,shutil
r=Path(__file__).resolve().parent
extra='''
extern "C" __global__ void initialize_silu_table() {
 unsigned int bits=blockIdx.x*blockDim.x+threadIdx.x;
 if(bits<65536)model_silu_values[bits]=__float2bfloat16_rn(addressed_exact::silu_f32(__bfloat162float(__ushort_as_bfloat16((unsigned short)bits))));
}
extern "C" __global__ void check_silu_domain(__nv_bfloat16* direct,__nv_bfloat16* table) {
 unsigned int bits=blockIdx.x*blockDim.x+threadIdx.x;
 if(bits<65536) {
  auto g=__ushort_as_bfloat16((unsigned short)bits);
  direct[bits]=__float2bfloat16_rn(addressed_exact::silu_f32(__bfloat162float(g)));
  table[bits]=model_silu_values[bits];
 }
}
'''
for family,parent,binary in (('bulk','model-preaddressed-GUP-r1','addressed_gate_up'),('tail','model-preaddressed-GUP-tail-bands-r2','band_gate_up')):
 out=r/f'snapshots/model-addressed-{family}-activation-table-r1';out.mkdir()
 previous=r/'snapshots'/parent
 header=(previous/'addressed_tiles.cuh').read_text()
 old='''    float s = __bfloat162float(__float2bfloat16_rn(silu_f32(__bfloat162float(g))));'''
 assert header.count(old)==2,header.count(old)
 header=header.replace(old,'    float s = __bfloat162float(__ldg(model_silu_values + __bfloat16_as_ushort(g)));')
 (out/'addressed_tiles.cuh').write_text(header)
 source=(previous/(binary+'.cu')).read_text()
 source=source.replace('#include "addressed_tiles.cuh"','#include <cuda_bf16.h>\n__device__ __nv_bfloat16 model_silu_values[65536];\n#include "addressed_tiles.cuh"')
 (out/(binary+'.cu')).write_text(source+extra)
 shutil.copy2(previous/'wide_warp_shapes.json',out/'wide_warp_shapes.json')
 wrapper=(previous/'wide_warp_hybrid.py').read_text().replace("super().__init__(binary)","super().__init__(binary)\n        self.initialize=C.c_void_p()\n        self.check(self.cuda.cuModuleGetFunction(C.byref(self.initialize),self.module,b'initialize_silu_table'))\n        self.check(self.cuda.cuLaunchKernel(self.initialize,256,1,1,256,1,1,0,C.c_void_p(torch.cuda.current_stream().cuda_stream),None,None))\n        torch.cuda.synchronize()")
 domain=(r/'snapshots/model-hybrid-BF16-activation-table-r2/activation_table_native.py').read_text().split('    def domain_check(self):')[1]
 wrapper+='\n    def domain_check(self):'+domain
 (out/'wide_warp_hybrid.py').write_text(wrapper)
 (out/'attribution.json').write_text(json.dumps(dict(parent=parent,source_sha256=hashlib.sha256((previous/(binary+'.cu')).read_bytes()).hexdigest(),change='Replace BF16-rounded SiLU evaluation with exhaustive, GPU-generated exact BF16 table; all weights, MMA groups and matrix layouts unchanged.')))
 shutil.copy2(__file__,out/Path(__file__).name)
 print(out,binary+'.cu')
