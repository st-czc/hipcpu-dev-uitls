#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdio>
#include <cmath>

// y[i] = alpha * x[i] + y[i] in bfloat16 (the rocblas axpy_ex pattern).
__global__ void bf16_axpy_kernel(const hip_bfloat16* alpha,
                                 const hip_bfloat16* x,
                                 hip_bfloat16* y,
                                 int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = (*alpha) * x[i] + y[i];
  }
}

int main() {
  const int n = 8;
  hip_bfloat16 hx[n];
  hip_bfloat16 hy[n];
  for (int i = 0; i < n; ++i) {
    hx[i] = hip_bfloat16(float(i + 1));
    hy[i] = hip_bfloat16(10.0f);
  }
  hip_bfloat16 halpha = hip_bfloat16(2.0f);

  hip_bfloat16 *dx, *dy, *dalpha;
  hipMalloc(&dx, n * sizeof(hip_bfloat16));
  hipMalloc(&dy, n * sizeof(hip_bfloat16));
  hipMalloc(&dalpha, sizeof(hip_bfloat16));
  hipMemcpy(dx, hx, n * sizeof(hip_bfloat16), hipMemcpyHostToDevice);
  hipMemcpy(dy, hy, n * sizeof(hip_bfloat16), hipMemcpyHostToDevice);
  hipMemcpy(dalpha, &halpha, sizeof(hip_bfloat16), hipMemcpyHostToDevice);

  bf16_axpy_kernel<<<1, n>>>(dalpha, dx, dy, n);

  hip_bfloat16 result[n];
  hipMemcpy(result, dy, n * sizeof(hip_bfloat16), hipMemcpyDeviceToHost);

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
