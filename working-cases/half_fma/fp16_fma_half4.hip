// half4 probe for the spvcpu fp16 bug family.
//
// Unlike <2 x half> (which the host ABI coerces to i32 and which then breaks
// llvm.fma.v2f16 at link time), <4 x half> is coerced to double and silently
// lowered to llvm.fma.f64.  It links, but it computes a single double FMA on a
// bitcast of the four halfs, not an element-wise 4x half FMA.
#include <hip/hip_runtime.h>
#include <cstdio>

typedef _Float16 half4 __attribute__((ext_vector_type(4)));

extern "C" __device__ half4 llvm_fma_v4f16(half4, half4, half4)
    __asm("llvm.fma.v4f16");

__global__ void fp16_fma_half4_kernel(const half4* a, const half4* b,
                                      const half4* c, half4* d, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    d[i] = llvm_fma_v4f16(a[i], b[i], c[i]);
  }
}

int main() {
  printf("fp16_fma half4 repro\n");
  return 0;
}
