// Same axpy as bfloat16_axpy.hip, but using hip_bf16.h's native __hip_bfloat16
// (which wraps clang's __bf16) instead of the hip_bfloat16 struct.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cmath>

__global__ void bf16_axpy_kernel(const __hip_bfloat16* alpha,
                                 const __hip_bfloat16* x,
                                 __hip_bfloat16* y,
                                 int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = (*alpha) * x[i] + y[i];
  }
}

int main() {
  const int n = 8;
  __hip_bfloat16 hx[n];
  __hip_bfloat16 hy[n];
  for (int i = 0; i < n; ++i) {
    hx[i] = __hip_bfloat16(float(i + 1));
    hy[i] = __hip_bfloat16(10.0f);
  }
  __hip_bfloat16 halpha = __hip_bfloat16(2.0f);

  __hip_bfloat16 *dx, *dy, *dalpha;
  hipMalloc(&dx, n * sizeof(__hip_bfloat16));
  hipMalloc(&dy, n * sizeof(__hip_bfloat16));
  hipMalloc(&dalpha, sizeof(__hip_bfloat16));
  hipMemcpy(dx, hx, n * sizeof(__hip_bfloat16), hipMemcpyHostToDevice);
  hipMemcpy(dy, hy, n * sizeof(__hip_bfloat16), hipMemcpyHostToDevice);
  hipMemcpy(dalpha, &halpha, sizeof(__hip_bfloat16), hipMemcpyHostToDevice);

  bf16_axpy_kernel<<<1, n>>>(dalpha, dx, dy, n);

  __hip_bfloat16 result[n];
  hipMemcpy(result, dy, n * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost);

  bool ok = true;
  for (int i = 0; i < n; ++i) {
    float got = float(result[i]);
    float expected = 10.0f + 2.0f * float(i + 1);
    if (std::fabs(got - expected) > 1e-2f) {
      ok = false;
      printf("  y[%d] = %f, expected %f\n", i, got, expected);
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  hipFree(dx);
  hipFree(dy);
  hipFree(dalpha);
  return ok ? 0 : 1;
}
