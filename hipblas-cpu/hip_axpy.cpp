#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include "hipblas/hipblas.h"
#include <cstdio>
#include <cmath>

int main() {
    hipblasHandle_t handle;
    hipblasCreate(&handle);
    int fail = 0;

    { // float
        const int n = 8;
        float x[n] = {1,2,3,4,5,6,7,8};
        float y[n] = {10,10,10,10,10,10,10,10};
        float alpha = 2.0f;
        hipblasStatus_t st = hipblasSaxpy(handle, n, &alpha, x, 1, y, 1);
        bool ok = (st == HIPBLAS_STATUS_SUCCESS);
        for (int i = 0; i < n; i++) {
            float e = 10.0f + 2.0f * x[i];
            if (std::fabs(y[i] - e) > 1e-5f) ok = false;
        }
        printf("hipblasSaxpy status=%d ok=%d\n", (int)st, (int)ok);
        if (!ok) fail++;
    }

    { // double
        const int n = 8;
        double x[n] = {1,2,3,4,5,6,7,8};
        double y[n] = {10,10,10,10,10,10,10,10};
        double alpha = 2.0;
        hipblasStatus_t st = hipblasDaxpy(handle, n, &alpha, x, 1, y, 1);
        bool ok = (st == HIPBLAS_STATUS_SUCCESS);
        for (int i = 0; i < n; i++) {
            double e = 10.0 + 2.0 * x[i];
            if (std::fabs(y[i] - e) > 1e-12) ok = false;
        }
        printf("hipblasDaxpy status=%d ok=%d\n", (int)st, (int)ok);
        if (!ok) fail++;
    }

    hipblasDestroy(handle);
    return fail ? 1 : 0;
}
