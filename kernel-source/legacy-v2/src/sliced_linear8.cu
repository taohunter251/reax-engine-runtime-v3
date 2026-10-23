// Canonical K slices provide quantization groups without extra reduction stages.
#include "hybrid8_tiles.cuh"

template<class C>
DEVI void sliced_tile(const int8_t* a,const int8_t* w,const __nv_bfloat16* ah,
                     const __nv_bfloat16* wh,const float* sa,const float* sw,const float* bias,
                     __nv_bfloat16* out,int M,int N,int KL,int KH,int group,int hp,
                     int row,int col,__nv_bfloat16* shared) {
    const int part=blockIdx.z;
    float acc[C::MT][C::NT][4],unused[1][1][4];
    integer_tiles::mainloop<C,0>(reinterpret_cast<const __nv_bfloat16*>(a),
        reinterpret_cast<const __nv_bfloat16*>(w),M,KL/2,KL/2,row,col,
        part*group/2,(part+1)*group/2,KL/2,shared,acc,unused);
    const int warp=threadIdx.x/32,lane=threadIdx.x%32;
    const int wm=warp/C::WN,wn=warp%C::WN;
    #pragma unroll
    for(int i=0;i<C::MT;++i)
        #pragma unroll
        for(int j=0;j<C::NT;++j)
            #pragma unroll
            for(int q=0;q<4;++q) {
                const int r=row+wm*C::WTM+i*16+lane/4+(q>=2?8:0);
                const int c=col+wn*C::WTN+j*8+(lane%4)*2+(q&1);
                const float value=__int2float_rn(__float_as_int(acc[i][j][q]));
                const float scale_a=r<M?sa[part*M+r]:0.f;
                acc[i][j][q]=__fadd_rn(__fmul_rn(__fmul_rn(value,scale_a),sw[part*N+c]),bias[part*N+c]);
            }
    if(hp>0)
        exact_tiles::mainloop<C,0,true>(ah,wh,M,KH,KH,row,col,part*hp,(part+1)*hp,KH,
                                      shared,acc,unused);
    exact_tiles::store_tile<C>(acc,out+(size_t)part*M*N,M,N,row,col,shared);
}

template<class Big,class Small>
DEVI void sliced_dispatch(const int8_t* a,const int8_t* w,const __nv_bfloat16* ah,
                         const __nv_bfloat16* wh,const float* sa,const float* sw,const float* bias,
                         __nv_bfloat16* out,int M,int N,int KL,int KH,int group,int hp,int mb,
                         __nv_bfloat16* shared) {
    int block=blockIdx.x,rows=mb/Big::BM;
    if(block<rows*(N/Big::BN))
        sliced_tile<Big>(a,w,ah,wh,sa,sw,bias,out,M,N,KL,KH,group,hp,
                         (block%rows)*Big::BM,(block/rows)*Big::BN,shared);
    else {
        block-=rows*(N/Big::BN);
        const int small=(M-mb+Small::BM-1)/Small::BM;
        sliced_tile<Small>(a,w,ah,wh,sa,sw,bias,out,M,N,KL,KH,group,hp,
                           mb+(block%small)*Small::BM,(block/small)*Small::BN,shared);
    }
}

using L128N128=integer_tiles::Cfg<128,128,32,4,2,3>;
using L64N128=integer_tiles::Cfg<64,128,32,2,4,3>;
using L32N128=integer_tiles::Cfg<32,128,32,2,4,3>;
using L64N128W4=integer_tiles::Cfg<64,128,32,2,2,3>;
using L32N128W4=integer_tiles::Cfg<32,128,32,1,4,3>;
using L128N256=integer_tiles::Cfg<128,256,32,4,2,3>;
using L32N256=integer_tiles::Cfg<32,256,32,2,4,3>;
using L128N128K64=integer_tiles::Cfg<128,128,64,4,2,2>;
using L32N128K64=integer_tiles::Cfg<32,128,64,2,4,2>;

#define SLICE(NAME,BIG,SMALL,T,R) \
extern "C" { __device__ __constant__ unsigned int NAME##_required_smem=BIG::SMEM>SMALL::SMEM?BIG::SMEM:SMALL::SMEM; } \
extern "C" __global__ void __launch_bounds__(T,R) NAME( \
 const int8_t* a,const int8_t* w,const __nv_bfloat16* ah,const __nv_bfloat16* wh, \
 const float* sa,const float* sw,const float* bias,__nv_bfloat16* out, \
 int M,int N,int KL,int KH,int group,int hp,int mb) { \
 extern __shared__ __align__(128) unsigned char shared[]; \
 sliced_dispatch<BIG,SMALL>(a,w,ah,wh,sa,sw,bias,out,M,N,KL,KH,group,hp,mb, \
                           reinterpret_cast<__nv_bfloat16*>(shared)); }
SLICE(slice_m128n128,L128N128,L32N128,256,2)
SLICE(slice_m64n128,L64N128,L32N128,256,2)
SLICE(slice_m64n128w4,L64N128W4,L32N128W4,128,3)
SLICE(slice_m128n256,L128N256,L32N256,256,1)
SLICE(slice_m128n128k64,L128N128K64,L32N128K64,256,1)
