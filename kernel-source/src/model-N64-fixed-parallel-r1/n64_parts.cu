#include "mixed_origin_tiles.cuh"

template<class C>
DEVI void slices(const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* parts,
                 int M,unsigned char* bytes) {
    constexpr int N=64,K=2560;
    const int tiles=(M+C::BM-1)/C::BM;
    const int row=(blockIdx.x%tiles)*C::BM,col=(blockIdx.x/tiles)*C::BN;
    const int p=blockIdx.z,begin=p*64,depth=min(64,K-begin);
    __nv_bfloat16* output=parts+(size_t)p*M*N;
    if(depth<=0) {
        for(int i=threadIdx.x;i<C::BM*C::BN;i+=C::THREADS) {
            const int r=row+i/C::BN,c=col+i%C::BN;
            if(r<M)output[(size_t)r*N+c]=__float2bfloat16_rn(0.f);
        }
        return;
    }
    float accumulator[C::MT][C::NT][4];
    auto shared=reinterpret_cast<__nv_bfloat16*>(bytes);
    addressed_exact::mainloop<C>(x+begin,w+begin,M,depth,K,row,col,-1,shared,accumulator,nullptr,0,K);
    addressed_exact::store_tile<C>(accumulator,output,M,N,row,col,shared);
}

#define KERNEL(NAME,C,T) \
extern "C" {__device__ __constant__ unsigned int NAME##_required_smem=C::SMEM;} \
extern "C" __global__ __launch_bounds__(T,2) void NAME(const __nv_bfloat16* x,const __nv_bfloat16* w, \
    __nv_bfloat16* parts,int M) {extern __shared__ __align__(128) unsigned char bytes[];slices<C>(x,w,parts,M,bytes);}
using C0=addressed_exact::Cfg<32,32,32,1,2,2,0>;
using C1=addressed_exact::Cfg<32,64,32,1,4,2,0>;
using C2=addressed_exact::Cfg<64,64,32,2,2,2,0>;
using C3=addressed_exact::Cfg<32,64,64,1,4,2,0>;
using C4=addressed_exact::Cfg<64,64,64,2,2,2,0>;
using C5=addressed_exact::Cfg<32,32,64,1,2,2,0>;
KERNEL(n64_parts_0,C0,64)
KERNEL(n64_parts_1,C1,128)
KERNEL(n64_parts_2,C2,128)
KERNEL(n64_parts_3,C3,128)
KERNEL(n64_parts_4,C4,128)
KERNEL(n64_parts_5,C5,64)
