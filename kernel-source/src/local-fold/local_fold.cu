// Public UID56 NG9B arithmetic, local short-row split reduction geometry.
#include "local_fold_tiles.cuh"
#define LOCAL(NAME,C,T) \
extern "C" {__device__ __constant__ unsigned int NAME##_required_smem=C::SMEM;} \
extern "C" __global__ __launch_bounds__(T,1) void NAME(const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* y,int M,int N,int K,int chunk) { \
 extern __shared__ __align__(128) unsigned char bytes[]; \
 auto shared=reinterpret_cast<__nv_bfloat16*>(bytes); \
 int tiles=(M+C::BM-1)/C::BM,row=(blockIdx.x%tiles)*C::BM,col=(blockIdx.x/tiles)*C::BN; \
 float partial[C::MT][C::NT][4],total[C::MT][C::NT][4]; \
 ng9b::mainloop<C,4>(x,w,M,K,K,row,col,-1,shared,partial,total,chunk); \
 ng9b::store_tile<C>(total,y,M,N,row,col,shared); }
using C0=ng9b::Cfg<64,128,64,4,4,3,2>;
LOCAL(wave_fold_0,C0,512)
using C1=ng9b::Cfg<96,128,64,3,4,2,2>;
LOCAL(wave_fold_1,C1,384)
using C2=ng9b::Cfg<96,128,64,3,4,3,2>;
LOCAL(wave_fold_2,C2,384)
using C3=ng9b::Cfg<96,128,64,6,2,2,2>;
LOCAL(wave_fold_3,C3,384)
using C4=ng9b::Cfg<96,64,64,3,2,2,2>;
LOCAL(wave_fold_4,C4,192)
using C5=ng9b::Cfg<160,64,64,5,2,2,2>;
LOCAL(wave_fold_5,C5,320)
using C6=ng9b::Cfg<192,64,64,6,2,2,2>;
LOCAL(wave_fold_6,C6,384)
using C7=ng9b::Cfg<192,64,64,12,1,2,2>;
LOCAL(wave_fold_7,C7,384)
using C8=ng9b::Cfg<224,64,64,7,2,2,2>;
LOCAL(wave_fold_8,C8,448)
using C9=ng9b::Cfg<112,128,64,7,2,2,2>;
LOCAL(wave_fold_9,C9,448)
using C10=ng9b::Cfg<128,128,64,4,4,2,2>;
LOCAL(wave_fold_10,C10,512)
