#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdio>

// Device side uses __builtin_frexpf(), which lowers to
// llvm.frexp.f32.i32 -> {float, i32}. This is the shape that the HIP
// hypot/norm kernels expose, and it trips the vecz struct-return bug.

constexpr int THREADS = 64;
constexpr int N = 256;

int main() {
  sycl::queue q;
  float* A = sycl::malloc_shared<float>(N, q);
  float* B = sycl::malloc_shared<float>(N, q);
  for (int i = 0; i < N; ++i) {
    A[i] = float(i % 7 + 1);
    B[i] = float(i % 3 + 2);
  }

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(THREADS)),
        [=](sycl::nd_item<1> it) {
          size_t gid = it.get_global_id(0);
          int e;
          float m = __builtin_frexpf(A[gid], &e);
          A[gid] = m * sycl::ldexp(B[gid], e);
        });
  }).wait();

  int bad = 0;
  for (int i = 0; i < N; ++i) {
    int e;
    float m = std::frexp(float(i % 7 + 1), &e);
    float want = m * std::ldexp(float(i % 3 + 2), e);
    if (A[i] != want)
      ++bad;
  }
  printf("sycl_clang_builtin_frexp: %s (%d/%d mismatch)\n",
         bad == 0 ? "PASS" : "FAIL", bad, N);
  return bad == 0 ? 0 : 1;
}
