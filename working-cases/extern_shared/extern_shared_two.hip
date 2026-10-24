#include <hip/hip_runtime.h>
#include <cstdio>

constexpr int THREADS = 256;

__global__ void x_kern_x(const float* A, const float* B, float* C) {
  extern __shared__ float x_smem0_x[];
  extern __shared__ float x_smem1_x[];
  __shared__ float x_SMA_x[THREADS];
  float* SMB = reinterpret_cast<float*>(x_smem0_x);
  float* SMC = reinterpret_cast<float*>(x_smem1_x + THREADS);
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  x_SMA_x[tid] = A[gid];
  SMB[tid] = B[gid];
  SMC[tid] = A[gid] + B[gid];
  __syncthreads();
  C[gid] = x_SMA_x[tid] * SMB[tid] + SMC[tid];
}

int main() {
  const int blocks = 4;
  const int n = THREADS * blocks;
  const size_t bytes = size_t(n) * sizeof(float);
  float* A_h = new float[n];
  float* B_h = new float[n];
  float* C_h = new float[n];
  for (int i = 0; i < n; ++i) {
    A_h[i] = float(i % 7 + 1);
    B_h[i] = float(i % 3 + 2);
  }
  float* A_d;
  float* B_d;
  float* C_d;
  hipMalloc(&A_d, bytes);
  hipMalloc(&B_d, bytes);
  hipMalloc(&C_d, bytes);
  hipMemcpy(A_d, A_h, bytes, hipMemcpyHostToDevice);
  hipMemcpy(B_d, B_h, bytes, hipMemcpyHostToDevice);
  x_kern_x<<<blocks, THREADS, 2 * THREADS * sizeof(float)>>>(A_d, B_d, C_d);
  hipMemcpy(C_h, C_d, bytes, hipMemcpyDeviceToHost);
  bool ok = true;
  for (int i = 0; i < n; ++i) {
    float expected = A_h[i] * B_h[i] + A_h[i] + B_h[i];
    if (C_h[i] != expected) {
      ok = false;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
