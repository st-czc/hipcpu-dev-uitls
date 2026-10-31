#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include "hipblas/hipblas.h"
#include <cstdio>
#include <cmath>

// Column-major reference GEMM (no transpose): C = alpha * A * B + beta * C0.
// Real arithmetic version for sgemm/dgemm.
template <typename Real>
void ref_gemm_real(int         m,
                   int         n,
                   int         k,
                   Real        alpha,
                   const Real* A,
                   int         lda,
                   const Real* B,
                   int         ldb,
                   Real        beta,
                   const Real* C0,
                   Real*       C,
                   int         ldc)
{
    for(int j = 0; j < n; ++j)
        for(int i = 0; i < m; ++i)
        {
            Real sum = 0;
            for(int p = 0; p < k; ++p)
                sum += A[i + p * lda] * B[p + j * ldb];
            C[i + j * ldc] = beta * C0[i + j * ldc] + alpha * sum;
        }
}

// Column-major reference GEMM for complex types (fields .x=real, .y=imag).
template <typename Real, typename Cplx>
void ref_gemm_cplx(int         m,
                   int         n,
                   int         k,
                   const Cplx& alpha,
                   const Cplx* A,
                   int         lda,
                   const Cplx* B,
                   int         ldb,
                   const Cplx& beta,
                   const Cplx* C0,
                   Cplx*       C,
                   int         ldc)
{
    for(int j = 0; j < n; ++j)
        for(int i = 0; i < m; ++i)
        {
            Real sr = 0, si = 0;
            for(int p = 0; p < k; ++p)
            {
                Real ar = A[i + p * lda].x, ai = A[i + p * lda].y;
                Real br = B[p + j * ldb].x, bi = B[p + j * ldb].y;
                sr += ar * br - ai * bi;
                si += ar * bi + ai * br;
            }
            Real c0r = C0[i + j * ldc].x, c0i = C0[i + j * ldc].y;
            Real cr  = beta.x * c0r - beta.y * c0i;
            Real ci  = beta.x * c0i + beta.y * c0r;
            Real pr  = alpha.x * sr - alpha.y * si;
            Real pi  = alpha.x * si + alpha.y * sr;
            C[i + j * ldc].x = cr + pr;
            C[i + j * ldc].y = ci + pi;
        }
}

int main()
{
    hipblasHandle_t handle;
    hipblasCreate(&handle);
    int fail = 0;

    { // float
        const int m = 2, n = 3, k = 4;
        const int lda = m, ldb = k, ldc = m;
        float A[m * k], B[k * n], C[m * n], C0[m * n], Cref[m * n];
        for(int p = 0; p < k; ++p)
            for(int i = 0; i < m; ++i)
                A[i + p * lda] = 0.5f * (i + p + 1);
        for(int j = 0; j < n; ++j)
            for(int p = 0; p < k; ++p)
                B[p + j * ldb] = 0.25f * (p + j + 1);
        for(int j = 0; j < n; ++j)
            for(int i = 0; i < m; ++i)
                C0[i + j * ldc] = C[i + j * ldc] = 1.0f + (i + j);

        float        alpha = 2.0f, beta = 0.5f;
        ref_gemm_real(m, n, k, alpha, A, lda, B, ldb, beta, C0, Cref, ldc);
        hipblasStatus_t st
            = hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
        bool ok = (st == HIPBLAS_STATUS_SUCCESS);
        for(int j = 0; j < n; ++j)
            for(int i = 0; i < m; ++i)
                if(std::fabs(C[i + j * ldc] - Cref[i + j * ldc]) > 1e-5f)
                    ok = false;
        printf("hipblasSgemm status=%d ok=%d\n", (int)st, (int)ok);
        if(!ok)
            fail++;
    }

    { // double
        const int m = 2, n = 3, k = 4;
        const int lda = m, ldb = k, ldc = m;
        double A[m * k], B[k * n], C[m * n], C0[m * n], Cref[m * n];
        for(int p = 0; p < k; ++p)
            for(int i = 0; i < m; ++i)
                A[i + p * lda] = 0.5 * (i + p + 1);
        for(int j = 0; j < n; ++j)
            for(int p = 0; p < k; ++p)
                B[p + j * ldb] = 0.25 * (p + j + 1);
        for(int j = 0; j < n; ++j)
            for(int i = 0; i < m; ++i)
                C0[i + j * ldc] = C[i + j * ldc] = 1.0 + (i + j);

        double         alpha = 2.0, beta = 0.5;
        ref_gemm_real(m, n, k, alpha, A, lda, B, ldb, beta, C0, Cref, ldc);
        hipblasStatus_t st
            = hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
        bool ok = (st == HIPBLAS_STATUS_SUCCESS);
        for(int j = 0; j < n; ++j)
            for(int i = 0; i < m; ++i)
                if(std::fabs(C[i + j * ldc] - Cref[i + j * ldc]) > 1e-12)
                    ok = false;
        printf("hipblasDgemm status=%d ok=%d\n", (int)st, (int)ok);
        if(!ok)
            fail++;
    }

    { // complex float
        const int m = 2, n = 2, k = 2;
        const int lda = m, ldb = k, ldc = m;
        hipComplex A[m * k], B[k * n], C[m * n], C0[m * n], Cref[m * n];
        for(int p = 0; p < k; ++p)
            for(int i = 0; i < m; ++i)
            {
                A[i + p * lda].x = 0.5f * (i + p + 1);
                A[i + p * lda].y = 0.1f * (i + p);
            }
        for(int j = 0; j < n; ++j)
            for(int p = 0; p < k; ++p)
            {
                B[p + j * ldb].x = 0.25f * (p + j + 1);
                B[p + j * ldb].y = 0.05f * (p - j);
            }
        for(int j = 0; j < n; ++j)
            for(int i = 0; i < m; ++i)
            {
                C0[i + j * ldc].x = C[i + j * ldc].x = 1.0f + (i + j);
                C0[i + j * ldc].y = C[i + j * ldc].y = 0.2f * (i - j);
            }

        hipComplex alpha = {1.0f, -0.5f}, beta = {0.5f, 0.25f};
        ref_gemm_cplx<float>(m, n, k, alpha, A, lda, B, ldb, beta, C0, Cref, ldc);
        hipblasStatus_t st
            = hipblasCgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
        bool ok = (st == HIPBLAS_STATUS_SUCCESS);
        for(int j = 0; j < n; ++j)
            for(int i = 0; i < m; ++i)
                if(std::fabs(C[i + j * ldc].x - Cref[i + j * ldc].x) > 1e-5f
                   || std::fabs(C[i + j * ldc].y - Cref[i + j * ldc].y) > 1e-5f)
                    ok = false;
        printf("hipblasCgemm status=%d ok=%d\n", (int)st, (int)ok);
        if(!ok)
            fail++;
    }

    { // complex double
        const int m = 2, n = 2, k = 2;
        const int lda = m, ldb = k, ldc = m;
        hipDoubleComplex A[m * k], B[k * n], C[m * n], C0[m * n], Cref[m * n];
        for(int p = 0; p < k; ++p)
            for(int i = 0; i < m; ++i)
            {
                A[i + p * lda].x = 0.5 * (i + p + 1);
                A[i + p * lda].y = 0.1 * (i + p);
            }
        for(int j = 0; j < n; ++j)
            for(int p = 0; p < k; ++p)
            {
                B[p + j * ldb].x = 0.25 * (p + j + 1);
                B[p + j * ldb].y = 0.05 * (p - j);
            }
        for(int j = 0; j < n; ++j)
            for(int i = 0; i < m; ++i)
            {
                C0[i + j * ldc].x = C[i + j * ldc].x = 1.0 + (i + j);
                C0[i + j * ldc].y = C[i + j * ldc].y = 0.2 * (i - j);
            }

        hipDoubleComplex alpha = {1.0, -0.5}, beta = {0.5, 0.25};
        ref_gemm_cplx<double>(m, n, k, alpha, A, lda, B, ldb, beta, C0, Cref, ldc);
        hipblasStatus_t st
            = hipblasZgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
        bool ok = (st == HIPBLAS_STATUS_SUCCESS);
        for(int j = 0; j < n; ++j)
            for(int i = 0; i < m; ++i)
                if(std::fabs(C[i + j * ldc].x - Cref[i + j * ldc].x) > 1e-12
                   || std::fabs(C[i + j * ldc].y - Cref[i + j * ldc].y) > 1e-12)
                    ok = false;
        printf("hipblasZgemm status=%d ok=%d\n", (int)st, (int)ok);
        if(!ok)
            fail++;
    }

    hipblasDestroy(handle);
    return fail ? 1 : 0;
}
