// HIP __half2 (struct wrapping <2 x half>) through __hfma2.
// Expected: PASS.  clang emits llvm.fma.v2f16 internally with its canonical
// <2 x half> signature, so the host ABI never gets a chance to coerce it.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>

__global__ void k(const __half2* a, const __half2* b, const __half2* c,
                  __half2* d, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = __hfma2(a[i], b[i], c[i]);
}

int main() {
  printf("hip_half2 hfma2\n");
  return 0;
}
