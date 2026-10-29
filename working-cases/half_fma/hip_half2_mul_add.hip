// HIP __half2 through operator* / operator+.
// Expected: PASS.  Goes through the header's C++ operators, not the raw
// llvm.fma.v2f16 intrinsic.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>

__global__ void k(const __half2* a, const __half2* b, const __half2* c,
                  __half2* d, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = a[i] * b[i] + c[i];
}

int main() {
  printf("hip_half2 mul_add\n");
  return 0;
}
