#include "gtensor/gtensor.h"

#ifdef GTENSOR_DEVICE_CUDA

#include "cublas_v2.h"
typedef cublasHandle_t gpublas_handle_t;
typedef cudaStream_t gpublas_stream_t;
typedef cuDoubleComplex gpublas_complex_double_t;
typedef cuComplex gpublas_complex_float_t;

#elif defined(GTENSOR_DEVICE_HIP)

#include "rocblas.h"
#include "rocsolver.h"
using gpublas_handle = rocblas_handle;
using gpublas_stream = hipStream_t;
using gpublas_complex = gt::complex<gpublas_real_t>;

#elif defined(GTENSOR_DEVICE_SYCL)

#include "CL/sycl.hpp"
#include "oneapi/mkl.hpp"
using gpublas_handle = cl::sycl::queue*;
using gpublas_stream = cl::sycl::queue*;
using gpublas_complex = gt::complex<gpublas_real_t>;

#endif

//#ifdef __cplusplus
//extern "C" {
//#endif

void gpublas_create();
void gpublas_destroy();

void gpublas_set_stream(gpublas_stream_t stream_id);
void gpublas_get_stream(gpublas_stream_t* stream_id);

void gpublas_zaxpy(int n, const gpublas_complex_double_t* a,
                   const gpublas_complex_double_t* x, int incx,
                   gpublas_complex_double_t* y, int incy);
void gpublas_daxpy(int n, const double* a, const double* x,
                   int incx, double* y, int incy);

void gpublas_zdscal(int n, const double fac,
                    gpublas_complex_double_t* arr, const int incx,
                    int* status);

void gpublas_zcopy(int n, const gpublas_complex_double_t* x, int incx,
                   gpublas_complex_double_t* y, int incy, int* status);

void gpublas_zgetrf_batched(int n, gpublas_complex_double_t* d_Aarray[],
                            int lda, int* d_PivotArray, int* d_infoArray,
                            int batchSize, int* status);
void gpublas_zgetrs_batched(int n, int nrhs,
                            const gpublas_complex_double_t* d_Aarray[],
                            int lda, const int* devIpiv,
                            gpublas_complex_double_t* d_Barray[], int ldb,
                            int batchSize, int* status);

//#ifdef __cplusplus
//}
//#endif
