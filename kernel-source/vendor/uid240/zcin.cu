// zcin.cu - ZC-IN (kai/kx-host, mining r2): a lean packed launch's two input blobs copied by one tiny kernel on the
// lane stream instead of two cuMemcpyHtoDAsync (copy engine). The kernel reads the slot's pinned staging blobs
// through UVA (device-accessible pinned host memory; zcin.py checks the device pointer at start-up) and writes the
// same bytes to the graph's device blobs meta.b64 / meta.b32. Data movement only: graphs, kernels and logits are
// unchanged.
//
// zcin_copy2(d0, s0, n0, d1, s1, n1): blockIdx.y = 0 copies n0 bytes s0 -> d0 (meta.b64), blockIdx.y = 1 copies
// n1 bytes s1 -> d1 (meta.b32); gridDim.x CTAs of 256 threads stride over the bytes. 16-byte vectors when both
// pointers and the byte count are 16-byte aligned (always, for PackedMeta blobs: 256-byte padded), bytes otherwise.
// Host reads use ld.global.cv (__ldcv: no cached copy of the host line is reused, every launch fetches the bytes the
// host wrote before it).
//
// Build (nvcc 13.0 V13.0.88, compile only, in this directory; md5 in KERNELS.md):
//   nvcc -cubin -arch=sm_120a -O3 -Xptxas -v -o zcin.cubin zcin.cu
#include <stdint.h>

extern "C" __global__ void __launch_bounds__(256)
zcin_copy2(uint8_t* d0, const uint8_t* s0, unsigned long long n0,
           uint8_t* d1, const uint8_t* s1, unsigned long long n1)
{
    uint8_t* d = blockIdx.y ? d1 : d0;
    const uint8_t* s = blockIdx.y ? s1 : s0;
    const unsigned long long n = blockIdx.y ? n1 : n0;
    const unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    if (((((uintptr_t)d) | ((uintptr_t)s) | n) & 15ull) == 0) {
        const uint4* s4 = reinterpret_cast<const uint4*>(s);
        uint4* d4 = reinterpret_cast<uint4*>(d);
        const unsigned long long n4 = n >> 4;
        for (unsigned long long i = tid; i < n4; i += stride)
            d4[i] = __ldcv(s4 + i);
    } else {
        for (unsigned long long i = tid; i < n; i += stride)
            d[i] = __ldcv(s + i);
    }
}
