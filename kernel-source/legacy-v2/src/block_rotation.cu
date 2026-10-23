#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

template<int G,int T,int B>
__device__ __forceinline__ void prepare_row(const __nv_bfloat16* x,const float* center,
    int8_t* output,float* scales,int M,int K,int chunk,int parts,float* shared) {
    constexpr int V=G/T;
    const int tid=threadIdx.x,row=blockIdx.x,part=blockIdx.y;
    float value[V];
    #pragma unroll
    for(int j=0;j<V;++j) {
        const int offset=tid+j*T,column=part*chunk+offset;
        float v=0.f;
        if(offset<chunk && column<K)v=__bfloat162float(x[(size_t)row*K+column])-center[column];
        const uint32_t bits=uint32_t(column)*1103515245u+12345u;
        value[j]=(bits&65536u)?-v:v;
    }
    #pragma unroll
    for(int bit=0;bit<5;++bit) {
        const int distance=1<<bit;
        #pragma unroll
        for(int j=0;j<V;++j) {
            const float partner=__shfl_xor_sync(0xffffffffu,value[j],distance);
            value[j]=(tid&distance)?partner-value[j]:value[j]+partner;
        }
    }
    #pragma unroll
    for(int distance=32;distance<T && distance<B;distance*=2) {
        #pragma unroll
        for(int j=0;j<V;++j)shared[tid+j*T]=value[j];
        __syncthreads();
        #pragma unroll
        for(int j=0;j<V;++j) {
            const float partner=shared[(tid^distance)+j*T];
            value[j]=(tid&distance)?partner-value[j]:value[j]+partner;
        }
        __syncthreads();
    }
    #pragma unroll
    for(int distance=1;distance<V && distance<B/T;distance*=2) {
        float next[V];
        #pragma unroll
        for(int j=0;j<V;++j)next[j]=(j&distance)?value[j^distance]-value[j]:value[j]+value[j^distance];
        #pragma unroll
        for(int j=0;j<V;++j)value[j]=next[j];
    }
    float maximum=0.f;
    #pragma unroll
    for(int j=0;j<V;++j)maximum=fmaxf(maximum,fabsf(value[j]));
    #pragma unroll
    for(int distance=16;distance;distance/=2)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,distance));
    if(tid%32==0)shared[tid/32]=maximum;
    __syncthreads();
    if(tid==0) {
        float final=1.e-12f;
        #pragma unroll
        for(int warp=0;warp<T/32;++warp)final=fmaxf(final,shared[warp]);
        shared[0]=__fmul_rn(final,1.f/127.f);
        scales[part*M+row]=shared[0];
    }
    __syncthreads();
    const float scale=shared[0];
    #pragma unroll
    for(int j=0;j<V;++j)output[(size_t)row*(parts*G)+part*G+tid+j*T]=int8_t(__float2int_rn(__fdiv_rn(value[j],scale)));
}

extern "C" __global__ void __launch_bounds__(64,1) rotation_g512_b64_t64(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<512,64,64>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(64,1) rotation_g512_b256_t64(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<512,64,256>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(128,1) rotation_g512_b256_t128(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<512,128,256>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(256,1) rotation_g512_b256_t256(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<512,256,256>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(64,1) rotation_g512_b512_t64(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<512,64,512>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(128,1) rotation_g512_b512_t128(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<512,128,512>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(256,1) rotation_g512_b512_t256(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<512,256,512>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(64,1) rotation_g1024_b64_t64(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<1024,64,64>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(64,1) rotation_g1024_b256_t64(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<1024,64,256>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(128,1) rotation_g1024_b256_t128(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<1024,128,256>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(256,1) rotation_g1024_b256_t256(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<1024,256,256>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(64,1) rotation_g1024_b1024_t64(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<1024,64,1024>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(128,1) rotation_g1024_b1024_t128(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<1024,128,1024>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(256,1) rotation_g1024_b1024_t256(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<1024,256,1024>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(64,1) rotation_g1344_b64_t64(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<1344,64,64>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(64,1) rotation_g1408_b64_t64(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<1408,64,64>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(64,1) rotation_g1408_b128_t64(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<1408,64,128>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(128,1) rotation_g1408_b128_t128(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<1408,128,128>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(64,1) rotation_g2304_b64_t64(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<2304,64,64>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(64,1) rotation_g2304_b256_t64(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<2304,64,256>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(128,1) rotation_g2304_b256_t128(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<2304,128,256>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(256,1) rotation_g2304_b256_t256(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<2304,256,256>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(64,1) rotation_g3072_b64_t64(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<3072,64,64>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(64,1) rotation_g3072_b256_t64(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<3072,64,256>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(128,1) rotation_g3072_b256_t128(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<3072,128,256>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(256,1) rotation_g3072_b256_t256(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<3072,256,256>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(64,1) rotation_g3072_b1024_t64(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<3072,64,1024>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(128,1) rotation_g3072_b1024_t128(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<3072,128,1024>(x,center,output,scales,M,K,chunk,parts,shared); }
extern "C" __global__ void __launch_bounds__(256,1) rotation_g3072_b1024_t256(
 const __nv_bfloat16* x,const float* center,int8_t* output,float* scales,int M,int K,int chunk,int parts) {
 extern __shared__ float shared[];prepare_row<3072,256,1024>(x,center,output,scales,M,K,chunk,parts,shared); }
