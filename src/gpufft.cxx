#include <gtensor/gtensor.h>

#ifdef GTENSOR_DEVICE_SYCL
#include <iostream>
#include <cstdlib>
#endif

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
#elif defined(GTENSOR_DEVICE_SYCL)
  /*
  if (type == GPUFFT_R2C || type == GPUFFT_C2R) {
    auto h = new gpufft_single_handle_t(dims);
  } else if (type == GPUFFT_D2Z || type == GPUFFT_Z2D) {
    auto h = new gpufft_double_descriptor_t(dims);
  }
  */
  gpufft_double_descriptor_t* h;

  try {
    if (rank == 1) {
      h = new gpufft_double_descriptor_t(n[0]);
    } else {
      std::vector<MKL_LONG> dims(rank);
      for (int i = 0; i < rank; i++) {
        dims[i] = n[i];
      }
      assert(dims.size() == rank);

      h = new gpufft_double_descriptor_t(dims);
    }
  } catch(std::exception const& e) {
    std::cerr << "Error creating dft descriptor:" << e.what() << std::endl;
    abort();
  }

  try {
    /*
    h->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, istride);
    h->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, 0);
    h->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, ostride);
    h->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, 0);
    */
    h->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                 batchSize);
    h->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
    h->set_value(oneapi::mkl::dft::config_param::REAL_STORAGE,
                 DFTI_REAL_REAL);
    h->set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
                 DFTI_COMPLEX_COMPLEX);
  } catch(std::exception const& e) {
    std::cerr << "Error setting values on dft descriptor:" << e.what()
              << std::endl;
    abort();
  }

  try {
    h->commit(gt::backend::sycl::get_queue());
  } catch(std::exception const& e) {
    std::cerr << "Error commiting dft descriptor:" << e.what() << std::endl;
    abort();
  }

  *handle = static_cast<void*>(h);
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
#elif defined(GTENSOR_DEVICE_SYCL)
  auto h = static_cast<gpufft_double_descriptor_t*>(handle);
  delete h;
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
#elif defined(GTENSOR_DEVICE_SYCL)
  auto h = static_cast<gpufft_double_descriptor_t*>(handle);
  auto indata_double = reinterpret_cast<double*>(indata);
  auto e = oneapi::mkl::dft::compute_backward(*h, indata_double, outdata);
  e.wait();
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
#elif defined(GTENSOR_DEVICE_SYCL)
  auto h = static_cast<gpufft_double_descriptor_t*>(handle);
  auto outdata_double = reinterpret_cast<double*>(outdata);
  auto e = oneapi::mkl::dft::compute_forward(*h, indata, outdata_double);
  e.wait();
#endif
}
