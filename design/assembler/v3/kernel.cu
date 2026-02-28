// kernel.cu
// GPU kernels for custom GPU processor.
// Compile to PTX with:  nvcc -ptx -arch=sm_80 kernel.cu -o kernel.ptx
//
// Each kernel operates on packed 64-bit registers (4 x int16 or 4 x bfloat16).
// A single call processes one 64-bit "register word" — threadIdx.x selects
// which word in the array each GPU thread handles.

#include <stdint.h>
#include <cuda_bf16.h>

__global__ void vadd_int16(int16_t* a, int16_t* b, int16_t* out) {
    int i = threadIdx.x;
    out[i] = a[i] + b[i];
}

__global__ void vsub_int16(int16_t* a, int16_t* b, int16_t* out) {
    int i = threadIdx.x;
    out[i] = a[i] - b[i];
}

__global__ void relu_int16(int16_t* a, int16_t* out) {
    int i = threadIdx.x;
    out[i] = (a[i] > 0) ? a[i] : (int16_t)0;
}


__global__ void vmul_bf16(__nv_bfloat16* a, __nv_bfloat16* b, __nv_bfloat16* out) {
    int i = threadIdx.x;
    out[i] = __hmul(a[i], b[i]);
}

__global__ void fmac_bf16(__nv_bfloat16* a, __nv_bfloat16* b,
                          __nv_bfloat16* acc, __nv_bfloat16* out) {
    int i = threadIdx.x;
    out[i] = __hfma(a[i], b[i], acc[i]);
}
