// Canonical K slices provide quantization groups without extra reduction stages.
#include "addressed_sliced_tiles.cuh"

template<class C>
DEVI void sliced_tile(const int8_t* a,const int8_t* w,const __nv_bfloat16* ah,
                     const __nv_bfloat16* wh,const float* sa,const float* sw,const float* bias,
                     __nv_bfloat16* out,int M,int N,int KL,int KH,int group,int hp,
                     int row,int col,__nv_bfloat16* shared) {
    const int part=blockIdx.z;
    float acc[C::MT][C::NT][4],unused[1][1][4];
    addressed_integer::mainloop<C>(reinterpret_cast<const __nv_bfloat16*>(a)+part*group/2,
        reinterpret_cast<const __nv_bfloat16*>(w)+part*group/2,M,group/2,KL/2,row,col,-1,
        shared,acc,nullptr,0,KL/2);
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
        addressed_exact::mainloop<C,0,true>(ah+part*hp,wh+part*hp,M,hp,KH,row,col,-1,
                                          shared,acc,nullptr,0,KH);
    addressed_exact::store_tile<C>(acc,out+(size_t)part*M*N,M,N,row,col,shared);
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


#define SLICED(NAME,BIG,SMALL,T,R) \
extern "C" { __device__ __constant__ unsigned int NAME##_required_smem=BIG::SMEM>SMALL::SMEM?BIG::SMEM:SMALL::SMEM; } \
extern "C" __global__ __launch_bounds__(T,R) void NAME( \
 const int8_t* a,const int8_t* w,const __nv_bfloat16* ah,const __nv_bfloat16* wh, \
 const float* sa,const float* sw,const float* bias,__nv_bfloat16* out, \
 int M,int N,int KL,int KH,int group,int hp,int mb) { \
 extern __shared__ __align__(128) unsigned char shared[]; \
 sliced_dispatch<BIG,SMALL>(a,w,ah,wh,sa,sw,bias,out,M,N,KL,KH,group,hp,mb, \
                           reinterpret_cast<__nv_bfloat16*>(shared)); }
using B0=addressed_integer::Cfg<128,128,32,4,2,3,0>;
using S0=addressed_integer::Cfg<32,128,32,1,8,2,0>;
SLICED(scheduled_0_m128n128k32,B0,S0,256,2)
using B1=addressed_integer::Cfg<128,128,64,4,2,3,0>;
using S1=addressed_integer::Cfg<32,128,32,1,8,2,0>;
SLICED(scheduled_1_m128n128k64,B1,S1,256,1)
using B2=addressed_integer::Cfg<64,128,32,2,4,3,0>;
using S2=addressed_integer::Cfg<32,128,32,1,8,2,0>;
SLICED(scheduled_2_m64n128k32,B2,S2,256,2)
using B3=addressed_integer::Cfg<64,128,32,2,2,3,0>;
using S3=addressed_integer::Cfg<32,128,32,1,4,2,0>;
SLICED(scheduled_3_m64n128k32,B3,S3,128,3)
using B4=addressed_integer::Cfg<128,256,32,2,4,3,0>;
using S4=addressed_integer::Cfg<32,256,32,1,8,2,0>;
SLICED(scheduled_4_m128n256k32,B4,S4,256,1)
using B5=addressed_integer::Cfg<128,256,64,2,4,2,0>;
using S5=addressed_integer::Cfg<32,256,32,1,8,2,0>;
SLICED(scheduled_5_m128n256k64,B5,S5,256,1)
using B6=addressed_integer::Cfg<64,256,32,2,4,3,0>;
using S6=addressed_integer::Cfg<32,256,32,1,8,2,0>;
SLICED(scheduled_6_m64n256k32,B6,S6,256,1)
using B7=addressed_integer::Cfg<64,256,64,2,4,2,0>;
using S7=addressed_integer::Cfg<32,256,32,1,8,2,0>;
SLICED(scheduled_7_m64n256k64,B7,S7,256,1)
