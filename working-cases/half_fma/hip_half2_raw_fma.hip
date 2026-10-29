// Raw llvm.fma.v2f16 reached through an __asm-labeled function whose parameter
// type is HIP __half2 (struct wrapping <2 x half>).
// Expected: FAIL at link.  The host ABI coerces the 32-bit __half2 to i32 and
// the emitted declaration becomes `i32 @llvm.fma.v2f16(i32, i32, i32)`, which
// the verifier rejects.  This is the same failure as fp16_fma/fp16_fma.hip.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>

extern "C" __device__ __half2 llvm_fma_v2f16(__half2, __half2, __half2)
    __asm("llvm.fma.v2f16");

__global__ void k(const __half2* a, const __half2* b, const __half2* c,
                  __half2* d, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = llvm_fma_v2f16(a[i], b[i], c[i]);
}

int main() {
  printf("hip_half2 raw_fma\n");
  return 0;
}
