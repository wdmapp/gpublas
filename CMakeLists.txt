cmake_minimum_required(VERSION 3.13...3.18.2 FATAL_ERROR)
project(gpublas
        VERSION 0.01
        LANGUAGES CXX
        HOMEPAGE_URL https://github.com/wdmapp/gpublas)

find_package(gtensor)
if (${GTENSOR_DEVICE} STREQUAL "cuda")
  # Note: must be called in top level CMakeLists.txt
  enable_language(CUDA)
endif()

add_library(gpublas)
target_gtensor_sources(gpublas PRIVATE src/gpublas.cxx)
target_link_libraries(gpublas gtensor::gtensor)
target_include_directories(gpublas PUBLIC include/)

add_library(gpufft)
target_gtensor_sources(gpufft PRIVATE src/gpufft.cxx)
target_link_libraries(gpufft gtensor::gtensor)
target_include_directories(gpufft PUBLIC include/)

if (${GTENSOR_DEVICE} STREQUAL "cuda")
  target_link_libraries(gpublas "cublas")
  target_link_libraries(gpufft "cufft")
elseif (${GTENSOR_DEVICE} STREQUAL "hip")
  add_library(rocblas INTERFACE IMPORTED)
  target_link_libraries(rocblas INTERFACE "/opt/rocm/lib/librocblas.so")
  target_include_directories(rocblas INTERFACE "/opt/rocm/include")
  add_library(rocsolver INTERFACE IMPORTED)
  target_link_libraries(rocsolver INTERFACE "/opt/rocm/lib/librocsolver.so")
  target_include_directories(rocsolver INTERFACE "/opt/rocm/include")
  add_library(rocfft INTERFACE IMPORTED)
  target_link_libraries(rocfft INTERFACE "/opt/rocm/lib/librocfft.so")
  target_include_directories(rocfft INTERFACE "/opt/rocm/include")

  target_link_libraries(gpublas rocblas rocsolver)
  target_link_libraries(gpufft rocfft)
elseif (${GTENSOR_DEVICE} STREQUAL "sycl")
  add_library(oneapi_mkl_sycl INTERFACE IMPORTED)
  #target_compile_options(oneapi_mkl_sycl INTERFACE -fsycl)
  #target_link_options(oneapi_mkl_sycl INTERFACE -fsycl)
  target_include_directories(oneapi_mkl_sycl
                             INTERFACE "/opt/intel/oneapi/mkl/latest/include")
  target_link_libraries(oneapi_mkl_sycl INTERFACE
              "/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_sycl.so")
  target_link_libraries(oneapi_mkl_sycl INTERFACE
              "/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.so")
  target_link_libraries(oneapi_mkl_sycl INTERFACE
              "/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_sequential.so")
  target_link_libraries(oneapi_mkl_sycl INTERFACE
              "/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so")
  target_link_libraries(oneapi_mkl_sycl INTERFACE
              "/opt/intel/oneapi/compiler/latest/linux/lib/libsycl.so.5")

  target_link_libraries(gpublas oneapi_mkl_sycl)
  target_link_libraries(gpufft oneapi_mkl_sycl)
endif()

include(CTest)
if (BUILD_TESTING)
  message(STATUS "${PROJECT_NAME}: build testing is ON")

  # try to find gtest, otherwise fetch and add to build
  find_package(GTest QUIET)

  if (NOT GTEST_FOUND)
    message(STATUS "${PROJECT_NAME}: googletest not found, fetching source and adding to build")
    include(FetchContent)
    FetchContent_Declare(googletest
      GIT_REPOSITORY    https://github.com/google/googletest.git
      GIT_TAG           release-1.10.0
      )
    FetchContent_GetProperties(googletest)
    if (NOT googletest_POPULATED)
      FetchContent_Populate(googletest)
      add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
    add_library(GTest::GTest INTERFACE IMPORTED)
    target_include_directories(GTest::GTest INTERFACE "${googletest_SOURCE_DIR}/googletest/include")
    target_link_libraries(GTest::GTest INTERFACE gtest)
    add_library(GTest::Main INTERFACE IMPORTED)
    target_link_libraries(GTest::Main INTERFACE gtest_main)
  endif()
  add_subdirectory(tests)
else()
  message(STATUS "${PROJECT_NAME}: build testing is OFF")
endif()


