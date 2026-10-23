// Exact protected channels join approximate integer channels before BF16 rounding.
#include "addressed_tiles.cuh"

template<class C>
DEVI void paired_register_store(const float (&value)[C::MT][C::NT][4],
 __nv_bfloat16* output,int M,int stride,int row,int column,__nv_bfloat16*) {
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int wm=warp/C::WN,wn=warp%C::WN;
    constexpr int half=C::NT/2;
    #pragma unroll
    for(int i=0;i<C::MT;++i) {
        #pragma unroll
        for(int h=0;h<2;++h) {
            const int r=row+wm*C::WTM+i*16+lane/4+h*8;
            if(r>=M)continue;
            #pragma unroll
            for(int j=0;j<half;++j) {
                __nv_bfloat162 gate=__floats2bfloat162_rn(value[i][j][2*h],value[i][j][2*h+1]);
                __nv_bfloat162 up=__floats2bfloat162_rn(value[i][j+half][2*h],value[i][j+half][2*h+1]);
                __nv_bfloat162 result;
                result.x=addressed_exact::swiglu1(gate.x,up.x);
                result.y=addressed_exact::swiglu1(gate.y,up.y);
                const int c=column+wn*(C::WTN/2)+j*8+(lane%4)*2;
                *reinterpret_cast<__nv_bfloat162*>(output+(size_t)r*stride+c)=result;
            }
        }
    }
}


template<class C>
DEVI void hybrid_tile(const int8_t* A,const int8_t* W,const __nv_bfloat16* AH,
 const __nv_bfloat16* WH,const float* sa,const float* sw,const float* bias,
 __nv_bfloat16* out,int M,int N,int K,int KH,int m0,int o0,__nv_bfloat16* smem) {
    float acc[C::MT][C::NT][4],unused[1][1][4];
    addressed_integer::mainloop<C>(reinterpret_cast<const __nv_bfloat16*>(A),
      reinterpret_cast<const __nv_bfloat16*>(W),M,K/2,K/2,m0,o0,N/2+o0,smem,acc);
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
                int col=((local%C::WTN)<C::WTN/2?o0:N/2+o0)+(local/C::WTN)*(C::WTN/2)+(local%C::WTN)%(C::WTN/2);
                float raw=__int2float_rn(__float_as_int(acc[i][j][q]));
                acc[i][j][q]=__fadd_rn(__fmul_rn(__fmul_rn(raw,row<M?sa[row]:0.f),sw[col]),bias[col]);
            }
    addressed_exact::mainloop<C,0,true>(AH,WH,M,KH,KH,m0,o0,N/2+o0,smem,acc);
    paired_register_store<C>(acc,out,M,N/2,m0,o0,smem);
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



template<int BM,int BN,int BK,int WM,int WN,int ST>
struct Pipe : addressed_integer::Cfg<BM,BN,BK,WM,WN,ST,1> {
 using Base=addressed_integer::Cfg<BM,BN,BK,WM,WN,ST,1>;
 static constexpr int SMEM=Base::SMEM_PIPE;
};
#define BAND(NAME,BIG,SMALL) \
extern "C" { __device__ __constant__ unsigned int NAME##_required_smem=BIG::SMEM>SMALL::SMEM?BIG::SMEM:SMALL::SMEM; } \
extern "C" __global__ __launch_bounds__(256,1) void NAME( \
 const int8_t* A,const int8_t* W,const __nv_bfloat16* AH,const __nv_bfloat16* WH, \
 const float* sa,const float* sw,const float* bias,__nv_bfloat16* out, \
 int M,int N,int K,int KH,int mb) { \
 extern __shared__ __align__(128) unsigned char memory[]; \
 hybrid_hetero<BIG,SMALL>(A,W,AH,WH,sa,sw,bias,out,M,18432,2432,128,mb,reinterpret_cast<__nv_bfloat16*>(memory)); }
using B_band_k32_t64n256k32=Pipe<128,256,32,2,4,3>;
using S_band_k32_t64n256k32=Pipe<64,256,32,2,4,3>;
BAND(band_k32_t64n256k32,B_band_k32_t64n256k32,S_band_k32_t64n256k32)
using B_band_k32_t64n256k64=Pipe<128,256,32,2,4,3>;
using S_band_k32_t64n256k64=Pipe<64,256,64,2,4,2>;
BAND(band_k32_t64n256k64,B_band_k32_t64n256k64,S_band_k32_t64n256k64)
using B_band_k32_t96n256k32=Pipe<128,256,32,2,4,3>;
using S_band_k32_t96n256k32=Pipe<96,256,32,2,4,3>;
BAND(band_k32_t96n256k32,B_band_k32_t96n256k32,S_band_k32_t96n256k32)
using B_band_k32_t96n256k64=Pipe<128,256,32,2,4,3>;
using S_band_k32_t96n256k64=Pipe<96,256,64,2,4,2>;
BAND(band_k32_t96n256k64,B_band_k32_t96n256k64,S_band_k32_t96n256k64)
using B_band_k32_t160n128k32=Pipe<128,256,32,2,4,3>;
using S_band_k32_t160n128k32=Pipe<160,128,32,2,4,3>;
BAND(band_k32_t160n128k32,B_band_k32_t160n128k32,S_band_k32_t160n128k32)
using B_band_k32_t192n128k32=Pipe<128,256,32,2,4,3>;
using S_band_k32_t192n128k32=Pipe<192,128,32,2,4,3>;
BAND(band_k32_t192n128k32,B_band_k32_t192n128k32,S_band_k32_t192n128k32)
using B_band_k32_t224n128k32=Pipe<128,256,32,2,4,3>;
using S_band_k32_t224n128k32=Pipe<224,128,32,2,4,3>;
BAND(band_k32_t224n128k32,B_band_k32_t224n128k32,S_band_k32_t224n128k32)
using B_band_k32_t160n128k64=Pipe<128,256,32,2,4,3>;
using S_band_k32_t160n128k64=Pipe<160,128,64,2,4,2>;
BAND(band_k32_t160n128k64,B_band_k32_t160n128k64,S_band_k32_t160n128k64)
using B_band_k32_t192n128k64=Pipe<128,256,32,2,4,3>;
using S_band_k32_t192n128k64=Pipe<192,128,64,2,4,2>;
BAND(band_k32_t192n128k64,B_band_k32_t192n128k64,S_band_k32_t192n128k64)
using B_band_k32_t224n128k64=Pipe<128,256,32,2,4,3>;
using S_band_k32_t224n128k64=Pipe<224,128,64,2,4,2>;
BAND(band_k32_t224n128k64,B_band_k32_t224n128k64,S_band_k32_t224n128k64)
using B_band_k64_t64n256k32=Pipe<128,256,64,2,4,2>;
using S_band_k64_t64n256k32=Pipe<64,256,32,2,4,3>;
BAND(band_k64_t64n256k32,B_band_k64_t64n256k32,S_band_k64_t64n256k32)
using B_band_k64_t64n256k64=Pipe<128,256,64,2,4,2>;
using S_band_k64_t64n256k64=Pipe<64,256,64,2,4,2>;
BAND(band_k64_t64n256k64,B_band_k64_t64n256k64,S_band_k64_t64n256k64)
using B_band_k64_t96n256k32=Pipe<128,256,64,2,4,2>;
using S_band_k64_t96n256k32=Pipe<96,256,32,2,4,3>;
BAND(band_k64_t96n256k32,B_band_k64_t96n256k32,S_band_k64_t96n256k32)
using B_band_k64_t96n256k64=Pipe<128,256,64,2,4,2>;
using S_band_k64_t96n256k64=Pipe<96,256,64,2,4,2>;
BAND(band_k64_t96n256k64,B_band_k64_t96n256k64,S_band_k64_t96n256k64)
using B_band_k64_t160n128k32=Pipe<128,256,64,2,4,2>;
using S_band_k64_t160n128k32=Pipe<160,128,32,2,4,3>;
BAND(band_k64_t160n128k32,B_band_k64_t160n128k32,S_band_k64_t160n128k32)
using B_band_k64_t192n128k32=Pipe<128,256,64,2,4,2>;
using S_band_k64_t192n128k32=Pipe<192,128,32,2,4,3>;
BAND(band_k64_t192n128k32,B_band_k64_t192n128k32,S_band_k64_t192n128k32)
using B_band_k64_t224n128k32=Pipe<128,256,64,2,4,2>;
using S_band_k64_t224n128k32=Pipe<224,128,32,2,4,3>;
BAND(band_k64_t224n128k32,B_band_k64_t224n128k32,S_band_k64_t224n128k32)
using B_band_k64_t160n128k64=Pipe<128,256,64,2,4,2>;
using S_band_k64_t160n128k64=Pipe<160,128,64,2,4,2>;
BAND(band_k64_t160n128k64,B_band_k64_t160n128k64,S_band_k64_t160n128k64)
using B_band_k64_t192n128k64=Pipe<128,256,64,2,4,2>;
using S_band_k64_t192n128k64=Pipe<192,128,64,2,4,2>;
BAND(band_k64_t192n128k64,B_band_k64_t192n128k64,S_band_k64_t192n128k64)
using B_band_k64_t224n128k64=Pipe<128,256,64,2,4,2>;
using S_band_k64_t224n128k64=Pipe<224,128,64,2,4,2>;
BAND(band_k64_t224n128k64,B_band_k64_t224n128k64,S_band_k64_t224n128k64)
