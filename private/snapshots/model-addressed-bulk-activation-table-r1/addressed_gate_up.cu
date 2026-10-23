// Exact protected channels join approximate integer channels before BF16 rounding.
#include <cuda_bf16.h>
__device__ __nv_bfloat16 model_silu_values[65536];
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
#define ADDRESSED(NAME,BIG,SMALL,T,R) \
extern "C" { __device__ __constant__ unsigned int NAME##_required_smem=BIG::SMEM>SMALL::SMEM?BIG::SMEM:SMALL::SMEM; } \
extern "C" __global__ __launch_bounds__(T,R) void NAME( \
 const int8_t* A,const int8_t* W,const __nv_bfloat16* AH,const __nv_bfloat16* WH, \
 const float* sa,const float* sw,const float* bias,__nv_bfloat16* out, \
 int M,int N,int K,int KH,int mb) { \
 extern __shared__ __align__(128) unsigned char memory[]; \
 hybrid_hetero<BIG,SMALL>(A,W,AH,WH,sa,sw,bias,out,M,18432,2432,128,mb,reinterpret_cast<__nv_bfloat16*>(memory)); }
using B0=Pipe<128,256,32,2,4,3>;
using S0=Pipe<32,256,32,1,8,2>;
ADDRESSED(addressed_0_m128n256k32,B0,S0,256,1)
using B1=Pipe<128,256,32,4,2,3>;
using S1=Pipe<32,256,32,1,8,2>;
ADDRESSED(addressed_1_m128n256k32,B1,S1,256,1)
using B2=Pipe<128,256,64,2,4,2>;
using S2=Pipe<32,256,32,1,8,2>;
ADDRESSED(addressed_2_m128n256k64,B2,S2,256,1)
using B3=Pipe<128,256,64,4,2,2>;
using S3=Pipe<32,256,32,1,8,2>;
ADDRESSED(addressed_3_m128n256k64,B3,S3,256,1)
using B4=Pipe<64,256,32,2,4,3>;
using S4=Pipe<32,256,32,1,8,2>;
ADDRESSED(addressed_4_m64n256k32,B4,S4,256,1)
using B5=Pipe<64,256,64,2,4,2>;
using S5=Pipe<32,256,32,1,8,2>;
ADDRESSED(addressed_5_m64n256k64,B5,S5,256,1)
using B6=Pipe<128,128,32,4,2,3>;
using S6=Pipe<32,128,32,1,8,2>;
ADDRESSED(addressed_6_m128n128k32,B6,S6,256,2)
using B7=Pipe<128,128,64,4,2,3>;
using S7=Pipe<32,128,32,1,8,2>;
ADDRESSED(addressed_7_m128n128k64,B7,S7,256,1)
using B8=Pipe<128,192,32,4,3,3>;
using S8=Pipe<32,192,32,1,12,2>;
ADDRESSED(addressed_8_m128n192k32,B8,S8,384,1)
using B9=Pipe<64,192,32,2,3,2>;
using S9=Pipe<32,192,32,1,6,2>;
ADDRESSED(addressed_9_m64n192k32,B9,S9,192,2)

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
