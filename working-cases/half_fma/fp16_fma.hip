// Minimal repro of the spvcpu fp16 bug.
// _Float16 ext_vector_type(2) lowers to i32, not <2 x half>.
// llvm.fma.v2f16 then fails to link.
#include <hip/hip_runtime.h>
#include <cstdio>

typedef _Float16 half2 __attribute__((ext_vector_type(2)));

extern "C" __device__ half2 llvm_fma_v2f16(half2, half2, half2)
    __asm("llvm.fma.v2f16");

__global__ void fp16_fma_kernel(const half2* a,
                                const half2* b,
                                const half2* c,
                                half2* d,
                                int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    d[i] = llvm_fma_v2f16(a[i], b[i], c[i]);
  }
}

int main() {
  printf("fp16_fma repro\n");
  return 0;
}
