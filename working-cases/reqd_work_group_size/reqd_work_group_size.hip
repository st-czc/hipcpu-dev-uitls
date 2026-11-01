#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void 
#ifdef USE_ATTR
__reqd_work_group_size__(64)
#endif
add_kernel(const float* A, float* B) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  B[gid] = A[gid] + 1.0f;
}

int main() {
  const int n = 65;
  const size_t bytes = size_t(n) * sizeof(float);
  float* A = new float[n];
  float* B = new float[n];
  for (int i = 0; i < n; ++i) A[i] = float(i);
  float* dA;
  float* dB;
  hipMalloc(&dA, bytes);
  hipMalloc(&dB, bytes);
  hipMemcpy(dA, A, bytes, hipMemcpyHostToDevice);
  add_kernel<<<1, n>>>(dA, dB);
  hipMemcpy(B, dB, bytes, hipMemcpyDeviceToHost);
  bool ok = true;
  for (int i = 0; i < n; ++i)
    if (B[i] != A[i] + 1.0f) ok = false;
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
