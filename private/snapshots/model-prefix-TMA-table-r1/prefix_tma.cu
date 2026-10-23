#include <cuda.h>
#include <cuda_bf16.h>
__device__ __nv_bfloat16 model_silu_values[65536];
#include "prefix_tiles.cuh"
namespace prefix_tma {
DEVI void init_barrier(uint32_t address) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(address) : "memory");
}
DEVI void expect_bytes(uint32_t address, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(address), "r"(bytes) : "memory");
}
DEVI void wait_phase(uint32_t address, int phase) {
    uint32_t ready;
    do {
        asm volatile("{ .reg .pred done;\n"
                     "mbarrier.try_wait.parity.shared::cta.b64 done, [%1], %2;\n"
                     "selp.u32 %0, 1, 0, done; }"
                     : "=r"(ready) : "r"(address), "r"(phase) : "memory");
    } while (!ready);
}
DEVI void load_tile(uint32_t target, const CUtensorMap* map, int column, int row, uint32_t barrier) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :: "r"(target), "l"(map), "r"(column), "r"(row), "r"(barrier) : "memory");
}

template<class C, bool PIPE = true>
DEVI void dot(const CUtensorMap* map_a, const CUtensorMap* map_w,
              int row, int column, int begin, int count, __nv_bfloat16* shared,
              float (&acc)[C::MT][C::NT][4]) {
    static_assert(C::BK == 64, "128-byte BF16 K tile");
    constexpr int ST = C::STAGES, BK = C::BK, KS = C::KS;
    constexpr int AB = C::A_STAGE * 2, WB = C::B_STAGE * 2;
    uint32_t a0 = ng9b::smem_u32(shared);
    uint32_t w0 = a0 + ST * AB;
    uint32_t bar0 = w0 + ST * WB;
    if (threadIdx.x == 0) {
        for (int stage = 0; stage < ST; ++stage) init_barrier(bar0 + stage * 8);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    auto issue = [&](int step) {
        if (threadIdx.x == 0 && step < count) {
            int stage = step % ST;
            expect_bytes(bar0 + stage * 8, AB + WB);
            load_tile(a0 + stage * AB, map_a, begin + step * 64, row, bar0 + stage * 8);
            load_tile(w0 + stage * WB, map_w, begin + step * 64, column, bar0 + stage * 8);
        }
    };
    for (int stage = 0; stage < ST; ++stage) issue(stage);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int ar = wm * C::WTM + lane % 16;
    const int br = wn * C::WTN + (lane / 16) * 8 + lane % 8;
    uint32_t ao[KS], bo[KS];
    #pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
        ao[ks] = (ar * BK + (((ks * 2 + lane / 16) ^ (ar & 7)) * 8)) * 2;
        bo[ks] = (br * BK + (((ks * 2 + (lane / 8) % 2) ^ (br & 7)) * 8)) * 2;
    }
    #pragma unroll
    for (int i = 0; i < C::MT; ++i)
        #pragma unroll
        for (int j = 0; j < C::NT; ++j)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
    uint32_t af[PIPE ? 2 : 1][C::MT][4], wf[PIPE ? 2 : 1][C::NT][2];
    auto fragments = [&](int buffer, int stage, int ks) {
        #pragma unroll
        for (int i = 0; i < C::MT; ++i)
            ng9b::ldsm_x4(a0 + stage * AB + ao[ks] + i * 16 * BK * 2,
                af[buffer][i][0], af[buffer][i][1], af[buffer][i][2], af[buffer][i][3]);
        #pragma unroll
        for (int j = 0; j < C::NT; j += 2)
            ng9b::ldsm_x4(w0 + stage * WB + bo[ks] + j * 8 * BK * 2,
                wf[buffer][j][0], wf[buffer][j][1], wf[buffer][j + 1][0], wf[buffer][j + 1][1]);
    };
    wait_phase(bar0,0);
    fragments(0,0,0);
    for (int step=0;step<count;++step) {
        int stage=step%ST;
        #pragma unroll
        for (int ks=0;ks<KS;++ks) {
            if (ks+1<KS) fragments((ks+1)&1,stage,ks+1);
            #pragma unroll
            for (int i=0;i<C::MT;++i)
                #pragma unroll
                for (int jj=0;jj<C::NT;++jj) {
                    int j=(i&1)?C::NT-1-jj:jj;
                    ng9b::mma16816(acc[i][j],af[ks&1][i],wf[ks&1][j][0],wf[ks&1][j][1]);
                }
            if (ks+1==KS) {
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncthreads();
                issue(step+ST);
                if (step+1<count) {
                    const int next=(step+1)%ST;
                    wait_phase(bar0+next*8,((step+1)/ST)&1);
                    fragments(0,next,0);
                }
            }
        }
    }
    __syncthreads();
}

}

#define TMA_PREFIX(NAME,C,T) \
extern "C" __global__ __launch_bounds__(T,1) void NAME(const __grid_constant__ CUtensorMap a,const __grid_constant__ CUtensorMap w,__nv_bfloat16* out,int M,int N) { \
 extern __shared__ __align__(1024) unsigned char smem[]; \
 float acc[C::MT][C::NT][4];int row=blockIdx.x*C::BM,col=blockIdx.y*C::BN/2; \
 prefix_tma::dot<C,true>(&a,&w,row,2*col,0,40,reinterpret_cast<__nv_bfloat16*>(smem),acc); \
 ng9b::store_swiglu<C>(acc,out,M,N/2,row,col); }
using C0=ng9b::Cfg<128,256,64,2,4,2,1>;
TMA_PREFIX(prefix_bf16_tma_0,C0,256)
using C1=ng9b::Cfg<64,256,64,2,4,2,1>;
TMA_PREFIX(prefix_bf16_tma_1,C1,256)
using C2=ng9b::Cfg<128,128,64,2,4,3,1>;
TMA_PREFIX(prefix_bf16_tma_2,C2,256)
using C3=ng9b::Cfg<64,128,64,2,2,2,1>;
TMA_PREFIX(prefix_bf16_tma_3,C3,128)
using C4=ng9b::Cfg<128,128,64,2,2,2,1>;
TMA_PREFIX(prefix_bf16_tma_4,C4,128)
using C5=ng9b::Cfg<96,128,64,3,2,2,1>;
TMA_PREFIX(prefix_bf16_tma_5,C5,192)
using C6=ng9b::Cfg<128,128,64,2,4,2,1>;
TMA_PREFIX(prefix_bf16_tma_6,C6,256)
using C7=ng9b::Cfg<64,128,64,1,4,2,1>;
TMA_PREFIX(prefix_bf16_tma_7,C7,128)
using C8=ng9b::Cfg<64,128,64,1,4,3,1>;
TMA_PREFIX(prefix_bf16_tma_8,C8,128)
using C9=ng9b::Cfg<32,128,64,1,4,2,1>;
TMA_PREFIX(prefix_bf16_tma_9,C9,128)

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
