#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>

// hypot reaches llvm.frexp.f32.i32/f64.i32 through BUILTIN_FREXP_EXP_F32/F64
// in ocml's hypot implementation. That intrinsic returns a literal struct
// {float, i32} / {double, i32}. The manual kernel spells out the same
// algorithm without calling ocml, and serves as the reference.

constexpr int N = 256;

__global__ void via_hypotf(const float* a, const float* b, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  out[i] = hypotf(a[i], b[i]);
}

__global__ void manual(const float* a, const float* b, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float x = fabsf(a[i]);
  float y = fabsf(b[i]);
  float t = fmaxf(x, y);
  int e;
  frexpf(t, &e);
  float sx = ldexpf(x, -e);
  float sy = ldexpf(y, -e);
  out[i] = ldexpf(sqrtf(fmaf(sx, sx, sy * sy)), e);
}

__global__ void hypot_double(const double* a, const double* b, double* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  out[i] = hypot(a[i], b[i]);
}

int main() {
  static float a[N], b[N], got[N];
  static double ad[N], bd[N], gotd[N];
  for (int i = 0; i < N; ++i) {
    a[i] = 1.0f + 0.25f * float(i % 37);
    b[i] = 0.5f + 0.5f * float(i % 19);
    ad[i] = 1.0 + 0.25 * double(i % 37);
    bd[i] = 0.5 + 0.5 * double(i % 19);
  }

  float *a_d, *b_d, *got_d;
  double *ad_d, *bd_d, *gotd_d;
  hipMalloc(&a_d, sizeof(a));
  hipMalloc(&b_d, sizeof(b));
  hipMalloc(&got_d, sizeof(got));
  hipMalloc(&ad_d, sizeof(ad));
  hipMalloc(&bd_d, sizeof(bd));
  hipMalloc(&gotd_d, sizeof(gotd));
  hipMemcpy(a_d, a, sizeof(a), hipMemcpyHostToDevice);
  hipMemcpy(b_d, b, sizeof(b), hipMemcpyHostToDevice);
  hipMemcpy(ad_d, ad, sizeof(ad), hipMemcpyHostToDevice);
  hipMemcpy(bd_d, bd, sizeof(bd), hipMemcpyHostToDevice);

  int bad = 0;

  via_hypotf<<<1, N>>>(a_d, b_d, got_d);
  hipMemcpy(got, got_d, sizeof(got), hipMemcpyDeviceToHost);
  int bad_hypotf = 0;
  for (int i = 0; i < N; ++i)
    if (got[i] != std::hypot(a[i], b[i])) ++bad_hypotf;
  printf("via_hypotf: %s (%d/%d mismatch), got[0]=%.6g want=%.6g\n",
         bad_hypotf == 0 ? "PASS" : "FAIL", bad_hypotf, N, got[0],
         std::hypot(a[0], b[0]));
  bad += bad_hypotf;

  manual<<<1, N>>>(a_d, b_d, got_d);
  hipMemcpy(got, got_d, sizeof(got), hipMemcpyDeviceToHost);
  int bad_manual = 0;
  for (int i = 0; i < N; ++i)
    if (got[i] != std::hypot(a[i], b[i])) ++bad_manual;
  printf("manual: %s (%d/%d mismatch), got[0]=%.6g want=%.6g\n",
         bad_manual == 0 ? "PASS" : "FAIL", bad_manual, N, got[0],
         std::hypot(a[0], b[0]));
  bad += bad_manual;

  hypot_double<<<1, N>>>(ad_d, bd_d, gotd_d);
  hipMemcpy(gotd, gotd_d, sizeof(gotd), hipMemcpyDeviceToHost);
  int bad_double = 0;
  for (int i = 0; i < N; ++i)
    if (gotd[i] != std::hypot(ad[i], bd[i])) ++bad_double;
  printf("hypot_double: %s (%d/%d mismatch), got[0]=%.6g want=%.6g\n",
         bad_double == 0 ? "PASS" : "FAIL", bad_double, N, gotd[0],
         std::hypot(ad[0], bd[0]));
  bad += bad_double;

  return bad == 0 ? 0 : 1;
}
