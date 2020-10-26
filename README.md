# gpublas

C/C++ library wrapping cuBLAS, rocBLAS, and DPC++ oneAPI MKL blas (USM version).
Implementation is C++, the interface provided is C and designed for easy
calling from Fortran code.

This is a prototype, and may be included in gtensor with a refactored API.

Requires gtensor.

## Install gtensor for desired backend

```
git clone https://github.com/wdmapp/gtensor.git
cd gtensor
export GT_DEVICE=cuda # or hip or sycl
# for hip
# export CXX=$(which hipcc)
# for sycl
# export CXX=$(which dpcpp)
cmake -S . -B build-$GT_DEVICE -DGTENSOR_DEVICE=$GT_DEVICE \
  -DCMAKE_INSTALL_PREFIX=$HOME/soft/gtensor/$GT_DEVICE
cmake --build build-$GT_DEVICE -v
cmake --build build-$GT_DEVICE -v -t install
```

## CUDA

```
gtensor_DIR=$HOME/soft/gtensor/cuda \
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGTENSOR_DEVICE=cuda \
  -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build build-cuda -v
cmake --build build-cuda -v -t test
```

## ROCm / HIP

Install libs, for example on ubuntu / debian:
```
apt-get install rocblas rocsparse rocfft
```

Build/run tests:
```
gtensor_DIR=$HOME/soft/gtensor/hip \
CXX=$(which hipcc) \
cmake -S . -B build-hip \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGTENSOR_DEVICE=hip
cmake --build build-hip -v
cmake --build build-hip -v -t test
```

## SYCL

Build/run tests:
```
gtensor_DIR=$HOME/soft/gtensor/sycl \
CXX=$(which dpcpp) \
cmake -S . -B build-sycl \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGTENSOR_DEVICE=sycl
cmake --build build-sycl -v
cmake --build build-sycl -v -t test
```
