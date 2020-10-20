#include <gtest/gtest.h>

#include "gtensor/gtensor.h"

#define DOUBLE_PREC
#include "gpublas.h"

TEST(daxpy, daxpy1)
{
  constexpr int N = 1024;
  using T = double;
  T* h_x = gt::backend::host_allocator<T>::allocate(N);
  T* d_x = gt::backend::device_allocator<T>::allocate(N);
  T* h_y = gt::backend::host_allocator<T>::allocate(N);
  T* d_y = gt::backend::device_allocator<T>::allocate(N);
  T a = 0.5;

  for (int i = 0; i < N; i++) {
    h_x[i] = 2.0 * static_cast<double>(i);
    h_y[i] = static_cast<double>(i);
  }

  gt::backend::device_copy_hd(h_x, d_x, N);
  gt::backend::device_copy_hd(h_y, d_y, N);

  gpublas_create();

  gpublas_daxpy(N, &a, d_x, 1, d_y, 1);

  gpublas_destroy();

  gt::backend::device_copy_dh(d_y, h_y, N);

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h_y[i], static_cast<T>(i * 2.0));
  }

  gt::backend::host_allocator<T>::deallocate(h_x);
  gt::backend::device_allocator<T>::deallocate(d_x);
  gt::backend::host_allocator<T>::deallocate(h_y);
  gt::backend::device_allocator<T>::deallocate(d_y);
}


