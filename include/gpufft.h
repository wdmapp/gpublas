#include "gtensor/gtensor.h"

#ifdef GTENSOR_DEVICE_CUDA

#include <cufft.h>
typedef cufftHandle gpufft_handle_t;
typedef cudaStream_t gpufft_stream_t;

typedef cufftType gpufft_transform_t;
#define GPUFFT_Z2D CUFFT_Z2D
#define GPUFFT_D2Z CUFFT_D2Z
#define GPUFFT_C2R CUFFT_C2R
#define GPUFFT_R2C CUFFT_R2C

typedef cufftDoubleReal gpufft_double_real_t;
typedef cufftReal gpufft_real_t;
typedef cufftDoubleComplex gpufft_double_complex_t;
typedef cufftComplex gpufft_complex_t;

#elif defined(GTENSOR_DEVICE_HIP)

#include "hipfft.h"
typedef hipfftHandle gpufft_handle_t;
typedef hipStream_t gpufft_stream_t;

typedef hipfftType gpufft_transform_t;
#define GPUFFT_Z2D HIPFFT_Z2D
#define GPUFFT_D2Z HIPFFT_D2Z
#define GPUFFT_C2R HIPFFT_C2R
#define GPUFFT_R2C HIPFFT_R2C

typedef hipfftDoubleReal gpufft_double_real_t;
typedef hipfftReal gpufft_real_t;
typedef hipfftDoubleComplex gpufft_double_complex_t;
typedef hipfftComplex gpufft_complex_t;

#elif defined(GTENSOR_DEVICE_SYCL)

#include "CL/sycl.hpp"
#include "oneapi/mkl.hpp"
typedef cl::sycl::queue* gpublas_handle_t;
typedef cl::sycl::queue* gpublas_stream_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

void gpufft_plan_many(gpufft_handle_t* handle, int rank, int* n, int istride,
                      int idist, int ostride, int odist,
                      gpufft_transform_t type, int batchSize);

void gpufft_plan_destroy(gpufft_handle_t handle);

void gpufft_exec_z2d(gpufft_handle_t handle, gpufft_double_complex_t* indata,
                     gpufft_double_real_t* outdata);

void gpufft_exec_d2z(gpufft_handle_t handle, gpufft_double_real_t* indata,
                     gpufft_double_complex_t* outdata);

#ifdef __cplusplus
}
#endif
