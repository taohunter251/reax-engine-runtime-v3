// Exact protected channels join approximate integer channels before BF16 rounding.
#include "hybrid8_tiles.cuh"

template<class C>
DEVI void hybrid_tile(const int8_t* A,const int8_t* W,const __nv_bfloat16* AH,
 const __nv_bfloat16* WH,const float* sa,const float* sw,const float* bias,
 __nv_bfloat16* out,int M,int N,int K,int KH,int m0,int o0,__nv_bfloat16* smem) {
    float acc[C::MT][C::NT][4],unused[1][1][4];
    integer_tiles::mainloop<C,0>(reinterpret_cast<const __nv_bfloat16*>(A),
      reinterpret_cast<const __nv_bfloat16*>(W),M,K/2,K/2,m0,o0,0,K/2,K/2,
      smem,acc,unused,N/2+o0);
    const int warp=threadIdx.x/32,lane=threadIdx.x%32;
    const int wm=warp/C::WN,wn=warp%C::WN;
    #pragma unroll
    for(int i=0;i<C::MT;++i)
        #pragma unroll
        for(int j=0;j<C::NT;++j)
            #pragma unroll
            for(int q=0;q<4;++q) {
                int row=m0+wm*C::WTM+i*16+lane/4+(q>=2?8:0);
                int local=wn*C::WTN+j*8+(lane%4)*2+(q&1);
                int col=local<C::BN/2?o0+local:N/2+o0+local-C::BN/2;
                float raw=__int2float_rn(__float_as_int(acc[i][j][q]));
                acc[i][j][q]=__fadd_rn(__fmul_rn(__fmul_rn(raw,row<M?sa[row]:0.f),sw[col]),bias[col]);
            }
    exact_tiles::mainloop<C,0,true>(AH,WH,M,KH,KH,m0,o0,0,KH,KH,smem,acc,unused,N/2+o0);
    exact_tiles::store_tile_swiglu<C>(acc,out,M,N/2,m0,o0,smem);
}

template<class Big,class Small>
DEVI void hybrid_hetero(const int8_t* A,const int8_t* W,const __nv_bfloat16* AH,
 const __nv_bfloat16* WH,const float* sa,const float* sw,const float* bias,
 __nv_bfloat16* out,int M,int N,int K,int KH,int mb,__nv_bfloat16* smem) {
    int b=blockIdx.x,rows=mb/Big::BM;
    if(b<rows*(N/Big::BN))
        hybrid_tile<Big>(A,W,AH,WH,sa,sw,bias,out,M,N,K,KH,(b%rows)*Big::BM,(b/rows)*(Big::BN/2),smem);
    else {
        b-=rows*(N/Big::BN);int smallrows=(M-mb+Small::BM-1)/Small::BM;
        hybrid_tile<Small>(A,W,AH,WH,sa,sw,bias,out,M,N,K,KH,mb+(b%smallrows)*Small::BM,(b/smallrows)*(Small::BN/2),smem);
    }
}

using H128256=integer_tiles::Cfg<128,256,64,4,2,2>;
using H64256=integer_tiles::Cfg<64,256,64,2,4,2>;
using H32256=integer_tiles::Cfg<32,256,64,2,4,2>;
using H128128=integer_tiles::Cfg<128,128,64,4,2,3>;
using H32128=integer_tiles::Cfg<32,128,64,2,4,3>;
using H128256K32=integer_tiles::Cfg<128,256,32,4,2,3>;
using H32256K32=integer_tiles::Cfg<32,256,32,2,4,3>;
using H64128K64=integer_tiles::Cfg<64,128,64,4,2,2>;
using H32128K64=integer_tiles::Cfg<32,128,64,2,4,2>;
using H64128K32=integer_tiles::Cfg<64,128,32,4,2,3>;
using H32128K32=integer_tiles::Cfg<32,128,32,2,4,3>;
using H32128W4=integer_tiles::Cfg<32,128,32,2,2,3>;
using H16128W4=integer_tiles::Cfg<16,128,32,1,4,3>;
#define HYBRID8(NAME,BIG,SMALL) \
extern "C" __global__ void __launch_bounds__(256,1) NAME( \
 const int8_t* A,const int8_t* W,const __nv_bfloat16* AH,const __nv_bfloat16* WH, \
 const float* sa,const float* sw,const float* bias,__nv_bfloat16* out, \
 int M,int N,int K,int KH,int mb) { \
 extern __shared__ __align__(128) unsigned char smem[]; \
 hybrid_hetero<BIG,SMALL>(A,W,AH,WH,sa,sw,bias,out,M,N,K,KH,mb,reinterpret_cast<__nv_bfloat16*>(smem)); }

HYBRID8(hybrid8_b128n256_s64n256,H128256,H64256)
HYBRID8(hybrid8_b64n256_s32n256,H64256,H32256)
HYBRID8(hybrid8_b128n128_s32n128,H128128,H32128)
HYBRID8(hybrid8_b128n256k32_s32n256k32,H128256K32,H32256K32)
HYBRID8(hybrid8_b64n128k64_s32n128k64,H64128K64,H32128K64)
HYBRID8(hybrid8_b64n128k32_s32n128k32,H64128K32,H32128K32)

#define HYBRID8_RESIDENT(NAME,BIG,SMALL,THREADS,RESIDENT) \
extern "C" __global__ void __launch_bounds__(THREADS,RESIDENT) NAME( \
 const int8_t* A,const int8_t* W,const __nv_bfloat16* AH,const __nv_bfloat16* WH, \
 const float* sa,const float* sw,const float* bias,__nv_bfloat16* out, \
 int M,int N,int K,int KH,int mb) { \
 extern __shared__ __align__(128) unsigned char smem[]; \
 hybrid_hetero<BIG,SMALL>(A,W,AH,WH,sa,sw,bias,out,M,N,K,KH,mb,reinterpret_cast<__nv_bfloat16*>(smem)); }

HYBRID8_RESIDENT(hybrid8_b64n128k32_min2,H64128K32,H32128K32,256,2)
HYBRID8_RESIDENT(hybrid8_b32n128k32w4_min3,H32128W4,H16128W4,128,3)
