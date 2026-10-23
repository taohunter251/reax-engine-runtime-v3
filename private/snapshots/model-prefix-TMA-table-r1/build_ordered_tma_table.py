"""Combine ordered descriptor loads and the already exact BF16 SiLU table."""
from pathlib import Path
import json,shutil

root=Path(__file__).resolve().parent
base=root/'snapshots/model-prefix-TMA-both-r1'
out=root/'snapshots/model-prefix-TMA-table-r1'
out.mkdir()
header=(base/'prefix_tiles.cuh').read_text()
needle='__float2bfloat16_rn(silu_f32(__bfloat162float(g)))'
assert header.count(needle)==1
header=header.replace(needle,'__ldg(model_silu_values + __bfloat16_as_ushort(g))')
(out/'prefix_tiles.cuh').write_text(header)
source=(base/'prefix_tma.cu').read_text()
source=source.replace('#include "prefix_tiles.cuh"','#include <cuda_bf16.h>\n__device__ __nv_bfloat16 model_silu_values[65536];\n#include "prefix_tiles.cuh"')
shapes=json.loads((base/'tma-shapes.json').read_text())
for index,(bm,bn,wm,wn,st) in enumerate(((128,128,2,4,2),(64,128,1,4,2),(64,128,1,4,3),(32,128,1,4,2)),start=6):
    name=f'prefix_bf16_tma_{index}'
    source+=f'using C{index}=ng9b::Cfg<{bm},{bn},64,{wm},{wn},{st},1>;\nTMA_PREFIX({name},C{index},{wm*wn*32})\n'
    shapes[name]=[bm,bn,wm,wn,st]
source+='''
extern "C" __global__ void initialize_silu_table() {
    const unsigned int bits=blockIdx.x*blockDim.x+threadIdx.x;
    if(bits<65536) model_silu_values[bits]=__float2bfloat16_rn(ng9b::silu_f32(__bfloat162float(__ushort_as_bfloat16(bits))));
}
extern "C" __global__ void check_silu_domain(__nv_bfloat16* a,__nv_bfloat16* b) {
    const unsigned int bits=blockIdx.x*blockDim.x+threadIdx.x;
    if(bits<65536) {
        a[bits]=__float2bfloat16_rn(ng9b::silu_f32(__bfloat162float(__ushort_as_bfloat16(bits))));
        b[bits]=model_silu_values[bits];
    }
}
'''
(out/'prefix_tma.cu').write_text(source)
(out/'tma-shapes.json').write_text(json.dumps(shapes,indent=2))
for name in ('LICENSE','prefix_tma_native.py'):shutil.copy2(base/name,out/name)
(out/'attribution.json').write_text(json.dumps(dict(parent='Public UID176 NG9B MIT exact MMA/rounding; local ordered TMA transport and complete BF16-domain SiLU table',
    changes='Both explicit reuse ordering controls, same ascendingK16dot, exactBF16SiLU lookup plus four model-shape geometries. No precision or weight change.',
    scope='Local only; needs isolated 65536domain and actual operator/control proof before any fullmodel qualification'),indent=2))
shutil.copy2(__file__,out/Path(__file__).name)
print(out)
