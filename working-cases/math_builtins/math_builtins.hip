#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>

constexpr int N = 256;
constexpr float TOL = 1e-4f;

// Uses the AMDGPU builtins that the spvcpu device math library borrows.
__global__ void builtin_kernel(const float* in, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float x = in[i];
  out[0 * N + i] = __builtin_amdgcn_rsqf(x);
  out[1 * N + i] = __builtin_amdgcn_rcpf(x);
  out[2 * N + i] = __builtin_amdgcn_sqrtf(x);
  out[3 * N + i] = __builtin_amdgcn_exp2f(x);
  out[4 * N + i] = __builtin_amdgcn_logf(x);
}

static bool close_enough(float got, float expected) {
  return std::fabs(got - expected) <= TOL * std::fabs(expected);
}

int main() {
  const size_t in_bytes = size_t(N) * sizeof(float);
  const size_t out_bytes = 5 * size_t(N) * sizeof(float);
  float* in_h = new float[N];
  float* out_h = new float[5 * N];
  for (int i = 0; i < N; ++i)
    in_h[i] = 1.0f + 0.01f * float(i);

  float* in_d;
  float* out_d;
  hipMalloc(&in_d, in_bytes);
  hipMalloc(&out_d, out_bytes);
  hipMemcpy(in_d, in_h, in_bytes, hipMemcpyHostToDevice);

  builtin_kernel<<<1, N>>>(in_d, out_d);
  hipMemcpy(out_h, out_d, out_bytes, hipMemcpyDeviceToHost);

  const char* names[5] = {"rsqf", "rcpf", "sqrtf", "exp2f", "logf"};
  bool ok = true;
  for (int f = 0; f < 5; ++f) {
    int bad = 0;
    for (int i = 0; i < N; ++i) {
      float x = in_h[i];
      float expected;
      switch (f) {
      case 0: expected = 1.0f / std::sqrt(x); break;
      case 1: expected = 1.0f / x; break;
      case 2: expected = std::sqrt(x); break;
      case 3: expected = std::exp2(x); break;
      default: expected = std::log2(x); break;
      }
      if (!close_enough(out_h[f * N + i], expected)) ++bad;
    }
    printf("%s: %s (%d/%d out of tolerance)\n", names[f],
           bad == 0 ? "PASS" : "FAIL", bad, N);
    if (bad != 0) ok = false;
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
