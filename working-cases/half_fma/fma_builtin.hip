// __builtin_elementwise_fma as a drop-in replacement for the broken
// `__asm("llvm.fma.v2f16")` declaration.
//
// The __asm form goes through the ordinary function-call path, where the
// spvcpu host ABI coerces <2 x half> to i32 and corrupts the intrinsic
// signature (see fp16_fma/ and hip_half2/ for the failure).  This builtin
// instead goes through clang's intrinsic emission path and emits
// `llvm.fma.v2f16` with its canonical <2 x half> type.
// Expected: PASS.
#include <hip/hip_runtime.h>
#include <cstdio>

typedef _Float16 half2 __attribute__((ext_vector_type(2)));

__device__ half2 llvm_fma_v2f16(half2 a, half2 b, half2 c) {
  return __builtin_elementwise_fma(a, b, c);
}

__global__ void k(const half2* a, const half2* b, const half2* c, half2* d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  d[i] = llvm_fma_v2f16(a[i], b[i], c[i]);
}

int main() {
  // Small integers exactly representable in half -> exact FMA.
  half2 a = {(_Float16)2.0f, (_Float16)3.0f};
  half2 b = {(_Float16)3.0f, (_Float16)4.0f};
  half2 c = {(_Float16)4.0f, (_Float16)5.0f};
  half2 h_a[1] = {a}, h_b[1] = {b}, h_c[1] = {c}, h_d[1];

  half2 *d_a, *d_b, *d_c, *d_d;
  hipMalloc(&d_a, sizeof(half2));
  hipMalloc(&d_b, sizeof(half2));
  hipMalloc(&d_c, sizeof(half2));
  hipMalloc(&d_d, sizeof(half2));
  hipMemcpy(d_a, h_a, sizeof(half2), hipMemcpyHostToDevice);
  hipMemcpy(d_b, h_b, sizeof(half2), hipMemcpyHostToDevice);
  hipMemcpy(d_c, h_c, sizeof(half2), hipMemcpyHostToDevice);
  k<<<1, 1>>>(d_a, d_b, d_c, d_d);
  hipMemcpy(h_d, d_d, sizeof(half2), hipMemcpyDeviceToHost);

  // a*b+c = {2*3+4, 3*4+5} = {10, 17}
  bool ok = ((float)h_d[0].x == 10.0f) && ((float)h_d[0].y == 17.0f);
  printf("%s  (got %.1f, %.1f)\n", ok ? "PASS" : "FAIL", (float)h_d[0].x,
         (float)h_d[0].y);
  return ok ? 0 : 1;
}
