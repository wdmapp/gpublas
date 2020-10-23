#include <gtensor/gtensor.h>

#include "gpufft.h"

void gpufft_plan_many(gpufft_handle_t* handle, int rank, int* n, int istride,
                      int idist, int ostride, int odist,
                      gpufft_transform_t type, int batchSize)
{
#ifdef GTENSOR_DEVICE_CUDA
  auto result = cufftPlanMany(handle, rank, n, nullptr, istride, idist, nullptr,
                              ostride, odist, type, batchSize);
  assert(result == CUFFT_SUCCESS);
#elif defined(GTENSOR_DEVICE_HIP)
  auto result = hipfftPlanMany(handle, rank, n, nullptr, istride, idist,
                               nullptr, ostride, odist, type, batchSize);
  assert(result == HIPFFT_SUCCESS);
  /*
  rocfft_plan_description desc = NULL;
  rocfft_plan_description_create(&desc);
  rocfft_plan_description_set_data_layout(
        desc,
        // input data format:
        rocfft_array_type_real,
        // output data format:
        rocfft_array_type_real,
        nullptr, // in offsets
        nullptr, // out offsets
        rank, // input stride length
        istride, // input stride data
        idist, // input batch distance
        rank, // output stride length
        ostride, // output stride data
        odist); // ouptut batch distance
  auto result = rocfft_plan_create(handle, rocfft_placment_inline, type,
                                   rocfft_precision_double, rank, n, batchSize,
  desc); assert(result == rocfft_success);
  */
#endif
}

void gpufft_plan_destroy(gpufft_handle_t handle)
{
#ifdef GTENSOR_DEVICE_CUDA
  auto result = cufftDestroy(handle);
  assert(result == CUFFT_SUCCESS);
#elif defined(GTENSOR_DEVICE_HIP)
  auto result = hipfftDestroy(handle);
  assert(result == rocfft_success);
#endif
}

void gpufft_exec_z2d(gpufft_handle_t handle, gpufft_double_complex_t* indata,
                     gpufft_double_real_t* outdata)
{
#ifdef GTENSOR_DEVICE_CUDA
  auto result = cufftExecZ2D(handle, indata, outdata);
  assert(result == CUFFT_SUCCESS);
#elif defined(GTENSOR_DEVICE_HIP)
  auto result = hipfftExecZ2D(handle, indata, outdata);
  assert(result == rocfft_success);
#endif
}

void gpufft_exec_d2z(gpufft_handle_t handle, gpufft_double_real_t* indata,
                     gpufft_double_complex_t* outdata)
{
#ifdef GTENSOR_DEVICE_CUDA
  auto result = cufftExecD2Z(handle, indata, outdata);
  assert(result == CUFFT_SUCCESS);
#elif defined(GTENSOR_DEVICE_HIP)
  auto result = hipfftExecD2Z(handle, indata, outdata);
  assert(result == rocfft_success);
#endif
}
