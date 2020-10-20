#include "gpublas.h"

#include <cstdio>

static gpublas_handle_t handle;

void gpublas_create() {
#ifdef GTENSOR_DEVICE_CUDA
    gtGpuCheck((cudaError_t)cublasCreate(&handle));
#elif defined(GTENSOR_DEVICE_HIP)
    gtGpuCheck((hipError_t)rocblas_create_handle(&handle));
#elif defined(GTENSOR_DEVICE_SYCL)
    handle = &gt::backend::sycl::get_queue();
#endif
}

void gpublas_destroy() {
#ifdef GTENSOR_DEVICE_CUDA
    gtGpuCheck((cudaError_t)cublasDestroy(handle));
#elif defined(GTENSOR_DEVICE_HIP)
    gtGpuCheck((hipError_t)rocblas_destroy_handle(handle));
#endif
}

void gpublas_set_stream(gpublas_stream_t stream_id) {
#ifdef GTENSOR_DEVICE_CUDA
    gtGpuCheck((cudaError_t)cublasSetStream(handle, stream_id));
#elif defined(GTENSOR_DEVICE_HIP)
    gtGpuCheck((hipError_t)rocblas_set_stream(handle, stream_id));
#elif defined(GTENSOR_DEVICE_SYCL)
    handle = stream_id;
#endif
}

void gpublas_get_stream(gpublas_stream_t* stream_id) {
#ifdef GTENSOR_DEVICE_CUDA
    gtGpuCheck((cudaError_t)cublasGetStream(handle, stream_id));
#elif defined(GTENSOR_DEVICE_HIP)
    gtGpuCheck((hipError_t)rocblas_get_stream(handle, stream_id));
#elif defined(GTENSOR_DEVICE_SYCL)
    *stream_id = handle;
#endif
}

/* ---------------- axpy ------------------- */
void gpublas_zaxpy(int n, const gpublas_complex_double_t* a,
                   const gpublas_complex_double_t* x, int incx,
                   gpublas_complex_double_t* y, int incy) {
#ifdef GTENSOR_DEVICE_CUDA
    gtGpuCheck((cudaError_t)cublasZaxpy(handle, n, a, x, incx, y, incy));
#elif defined(GTENSOR_DEVICE_HIP)
    gtGpuCheck((hipError_t)rocblas_zaxpy(handle, n, a, x, incx, y, incy));
#elif defined(GTENSOR_DEVICE_SYCL)
    // TODO: exception handling
    auto e = oneapi::mkl::blas::axpy(*handle, n, a, x, incx, y, incy);
    e.wait();
#endif
}

void gpublas_daxpy(int n, const double* a, const double* x, int incx, double* y,
                   int incy) {
#ifdef GTENSOR_DEVICE_CUDA
    gtGpuCheck((cudaError_t)cublasDaxpy(handle, n, a, x, incx, y, incy));
#elif defined(GTENSOR_DEVICE_HIP)
    gtGpuCheck((hipError_t)rocblas_daxpy(handle, n, a, x, incx, y, incy));
#elif defined(GTENSOR_DEVICE_SYCL)
    // TODO: exception handling
    auto e = oneapi::mkl::blas::axpy(*handle, n, a, x, incx, y, incy);
    e.wait();
#endif
}

/*
void gpublas_caxpy(int n, const gt::complex<float> a,
                   const gt::complex<float>* x, int incx, gt::complex<float>* y,
                   int incy, int* status) {
#ifdef GTENSOR_DEVICE_CUDA
    gtGpuCheck((cudaError_t)cublasCaxpy(handle, n, a, x, incx, y, incy));
    cudaDeviceSynchronize();  // ???
#elif defined(GTENSOR_DEVICE_HIP)
    gtGpuCheck((cudaError_t)rocblas_caxpy(handle, n, a, x, incx, y, incy));
#elif defined(GTENSOR_DEVICE_SYCL)
    // TODO: exception handling
    auto e = oneapi::mkl::blas::axpy(*handle, n, a, x, incx, y, incy);
    e.wait();
#endif
}
*/

void gpublas_zdscal(int n, const double fac, gpublas_complex_double_t* arr,
                    const int incx, int* status) {
#ifdef GTENSOR_DEVICE_CUDA
    gtGpuCheck((cudaError_t)cublasZdscal(handle, n, &fac, arr, incx));
#elif defined(GTENSOR_DEVICE_HIP)
    gtGpuCheck((hipError_t)rocblas_zdscal(handle, n, &fac, arr, incx));
#elif defined(GTENSOR_DEVICE_SYCL)
    // TODO: exception handling
    auto e = oneapi::mkl::blas::scal(*handle, n, fac, arr, incx);
    e.wait();
#endif
}

/* ------------ copy --------------- */
void gpublas_zcopy(int n, const gpublas_complex_double_t* x, int incx,
                   gpublas_complex_double_t* y, int incy, int* status) {
#ifdef GTENSOR_DEVICE_CUDA
    gtGpuCheck((cudaError_t)cublasZcopy(handle, n, x, incx, y, incy));
#elif defined(GTENSOR_DEVICE_HIP)
    gtGpuCheck((hipError_t)rocblas_zcopy(handle, n, x, incx, y, incy));
#elif defined(GTENSOR_DEVICE_SYCL)
    // TODO: exception handling
    auto e = oneapi::mkl::blas::copy(*handle, n, fac, arr, incx);
    e.wait();
#endif
}

void gpublas_zgetrf_batched(int n, gpublas_complex_double_t* d_Aarray[],
                            int lda, int* d_PivotArray, int* d_infoArray,
                            int batchSize, int* status) {
#ifdef GTENSOR_DEVICE_CUDA
    gtGpuCheck((cudaError_t)cublasZgetrfBatched(
        handle, n, d_Aarray, lda, d_PivotArray, d_infoArray, batchSize));
#elif defined(GTENSOR_DEVICE_HIP)
    // Note: extra args are for general n x m size, and strideP for the
    // pivot array stride (we use n).
    gtGpuCheck((hipError_t)rocsolver_zgetrf_batched(
        handle, n, n, d_Aarray, lda, d_PivotArray, n, d_infoArray, batchSize));
#elif defined(GTENSOR_DEVICE_SYCL)
    // TODO: exception handling
    auto scratch_count = oneapi::mkl::lapack::getrf_batch_scratchpad_size(
        *handle, n, n, lda, n * n, n, batchSize);
    auto scratch = sycl::malloc<double>(*handle, scratch_count);

    auto e = oneapi::mkl::lapack::getrf_batch(
        *handle, n, n, d_Aarray, lda, n * n, d_PivotArray, n, d_infoArray,
        batchSize, scratch, scratch_count);
    e.wait();

    sycl::free(*handle, scratch);
#endif
}

void gpublas_zgetrs_batched(int n, int nrhs,
                            const gpublas_complex_double_t* d_Aarray[], int lda,
                            const int* devIpiv,
                            gpublas_complex_double_t* d_Barray[], int ldb,
                            int batchSize, int* status) {
#ifdef GTENSOR_DEVICE_CUDA
    int info;
    gtGpuCheck((cudaError_t)cublasZgetrsBatched(
        handle, CUBLAS_OP_N, n, nrhs, d_Aarray, lda, devIpiv, d_Barray, ldb,
        &info, batchSize));
    if (info != 0) {
        printf("error in ZgetrsBatched: info = %d\n", info);
        std::abort();
    }
#elif defined(GTENSOR_DEVICE_HIP)
    gtGpuCheck((hipError_t)rocsolver_zgetrs_batched(handle, 0, n, nrhs,
                                                    d_Aarray, lda, devIpiv, n,
                                                    d_Barray, ldb, batchSize));
#elif defined(GTENSOR_DEVICE_SYCL)
    // TODO: exception handling
    auto scratch_count = oneapi::mkl::lapack::getrs_batch_scratchpad_size(
        *handle, n, n, lda, n * n, n, batchSize);
    auto scratch = sycl::malloc<double>(*handle, scratch_count);

    auto e = oneapi::mkl::lapack::getrs_batch(
        *handle, oneapi::mkl::transpose::notrans, n, nrhs, d_Aarray, lda, n * n,
        d_Barray, lbd, n * n, batchSize, scratch, scratch_count);
    e.wait();

    sycl::free(*handle, scratch);
#endif
}
