#include <sycl/sycl.hpp>
#include <cstdio>

constexpr int THREADS = 256;

int main() {
  const int blocks = 4;
  const int n = THREADS * blocks;
  sycl::queue q;
  float* A = sycl::malloc_shared<float>(n, q);
  float* B = sycl::malloc_shared<float>(n, q);
  float* C = sycl::malloc_shared<float>(n, q);
  for (int i = 0; i < n; ++i) {
    A[i] = float(i % 7 + 1);
    B[i] = float(i % 3 + 2);
  }
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> SMA(sycl::range<1>(THREADS), cgh);
    sycl::local_accessor<float, 1> SMB(sycl::range<1>(THREADS), cgh);
    sycl::local_accessor<float, 1> SMC(sycl::range<1>(THREADS), cgh);
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(n), sycl::range<1>(THREADS)),
                     [=](sycl::nd_item<1> it) {
                       size_t tid = it.get_local_id(0);
                       size_t gid = it.get_global_id(0);
                       SMA[tid] = A[gid];
                       SMB[tid] = B[gid];
                       SMC[tid] = A[gid] + B[gid];
                       sycl::group_barrier(it.get_group());
                       C[gid] = SMA[tid] * SMB[tid] + SMC[tid];
                     });
  });
  q.wait();
  bool ok = true;
  for (int i = 0; i < n; ++i) {
    float expected = A[i] * B[i] + A[i] + B[i];
    if (C[i] != expected) {
      ok = false;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  sycl::free(A, q);
  sycl::free(B, q);
  sycl::free(C, q);
  return ok ? 0 : 1;
}
