#include <gtest/gtest.h>

#include "gtensor/gtensor.h"

#define DOUBLE_PREC
#include "gpublas.h"

TEST(getrfs, dgetrfs_batch1) {
    constexpr int N = 3;
    constexpr int S = N * N;
    constexpr int batch_size = 1;
    using T = double;

    T** h_Aptr = gt::backend::host_allocator<T*>::allocate(batch_size);
    T** d_Aptr = gt::backend::device_allocator<T*>::allocate(batch_size);
    T* h_A = gt::backend::host_allocator<T>::allocate(batch_size * S);
    T* d_A = gt::backend::device_allocator<T>::allocate(batch_size * S);

    int* h_p = gt::backend::host_allocator<int>::allocate(batch_size * N);
    int* d_p = gt::backend::device_allocator<int>::allocate(batch_size * N);
    int* h_info = gt::backend::host_allocator<int>::allocate(batch_size);
    int* d_info = gt::backend::device_allocator<int>::allocate(batch_size);

    // Example from
    // https://courses.physics.illinois.edu/cs357/sp2020/notes/ref-9-linsys.html
    // first column 1, 4, 4
    h_A[0] = 1;
    h_A[1] = 4;
    h_A[2] = 4;
    // second column 2, 4, 6
    h_A[3] = 2;
    h_A[4] = 4;
    h_A[5] = 6;
    // third column 2, 2, 4
    h_A[6] = 2;
    h_A[7] = 2;
    h_A[8] = 4;

    h_Aptr[0] = &d_A[0];

    gt::backend::device_copy_hd(h_A, d_A, batch_size * S);
    gt::backend::device_copy_hd(h_Aptr, d_Aptr, batch_size);

    gpublas_create();

    gpublas_dgetrf_batched(N, d_Aptr, N, d_p, d_info, batch_size);

    gpublas_destroy();

    gt::backend::device_copy_dh(d_A, h_A, batch_size * S);
    gt::backend::device_copy_dh(d_p, h_p, batch_size * N);
    gt::backend::device_copy_dh(d_info, h_info, batch_size);

    // first column factored
    EXPECT_EQ(h_A[0], 4.0);
    EXPECT_EQ(h_A[1], 1.0);
    EXPECT_EQ(h_A[2], 0.25);
    // second column factored
    EXPECT_EQ(h_A[3], 4.0);
    EXPECT_EQ(h_A[4], 2.0);
    EXPECT_EQ(h_A[5], 0.5);
    // third column factored
    EXPECT_EQ(h_A[6], 2.0);
    EXPECT_EQ(h_A[7], 2.0);
    EXPECT_EQ(h_A[8], 0.5);

    // Math notation is 2, 3, 1, but BLAS uses procedural notation, i.e.
    // on the thirst step no swap is done so one-based index of third row
    // is still 3 (no swapping).
    EXPECT_EQ(h_p[0], 2);
    EXPECT_EQ(h_p[1], 3);
    EXPECT_EQ(h_p[2], 3);

    for (int b = 0; b < batch_size; b++) {
        // A_i factored successfully
        EXPECT_EQ(h_info[b], 0);
    }

    gt::backend::host_allocator<T*>::deallocate(h_Aptr);
    gt::backend::device_allocator<T*>::deallocate(d_Aptr);
    gt::backend::host_allocator<T>::deallocate(h_A);
    gt::backend::device_allocator<T>::deallocate(d_A);

    gt::backend::host_allocator<int>::deallocate(h_p);
    gt::backend::device_allocator<int>::deallocate(d_p);
    gt::backend::host_allocator<int>::deallocate(h_info);
    gt::backend::device_allocator<int>::deallocate(d_info);
}

TEST(getrfs, zgetrfs_batch1) {
    constexpr int N = 3;
    constexpr int S = N * N;
    constexpr int batch_size = 1;
    using T = gt::complex<double>;

    T** h_Aptr = gt::backend::host_allocator<T*>::allocate(batch_size);
    T** d_Aptr = gt::backend::device_allocator<T*>::allocate(batch_size);
    T* h_A = gt::backend::host_allocator<T>::allocate(batch_size * S);
    T* d_A = gt::backend::device_allocator<T>::allocate(batch_size * S);

    int* h_p = gt::backend::host_allocator<int>::allocate(batch_size * N);
    int* d_p = gt::backend::device_allocator<int>::allocate(batch_size * N);
    int* h_info = gt::backend::host_allocator<int>::allocate(batch_size);
    int* d_info = gt::backend::device_allocator<int>::allocate(batch_size);

    // Example from
    // https://courses.physics.illinois.edu/cs357/sp2020/notes/ref-9-linsys.html
    // first column 1, 4, 4
    h_A[0] = T(1, 0);
    h_A[1] = T(4, 0);
    h_A[2] = T(4, 0);
    // second column 2, 4, 6
    h_A[3] = T(2, 0);
    h_A[4] = T(4, 0);
    h_A[2] = T(6, 0);
    // third column 2, 2, 4
    h_A[3] = T(2, 0);
    h_A[4] = T(2, 0);
    h_A[2] = T(4, 0);

    h_Aptr[0] = &d_A[0];

    gt::backend::device_copy_hd(h_A, d_A, batch_size * S);
    gt::backend::device_copy_hd(h_Aptr, d_Aptr, batch_size);

    gpublas_create();

    gpublas_zgetrf_batched(N, (gpublas_complex_double_t**)d_Aptr, N, d_p,
                           d_info, batch_size);

    gpublas_destroy();

    gt::backend::device_copy_dh(d_A, h_A, batch_size * S);
    gt::backend::device_copy_dh(d_p, h_p, batch_size * N);
    gt::backend::device_copy_dh(d_info, h_info, batch_size);

    // first column factored
    EXPECT_EQ(h_A[0], 4.0);
    EXPECT_EQ(h_A[1], 1.0);
    EXPECT_EQ(h_A[2], 0.25);
    // second column factored
    EXPECT_EQ(h_A[3], 4.0);
    EXPECT_EQ(h_A[4], 2.0);
    EXPECT_EQ(h_A[5], 0.5);
    // third column factored
    EXPECT_EQ(h_A[6], 2.0);
    EXPECT_EQ(h_A[7], 2.0);
    EXPECT_EQ(h_A[8], 0.5);

    // Math notation is 2, 3, 1, but BLAS uses procedural notation, i.e.
    // on the thirst step no swap is done so one-based index of third row
    // is still 3 (no swapping).
    EXPECT_EQ(h_p[0], 2);
    EXPECT_EQ(h_p[1], 3);
    EXPECT_EQ(h_p[2], 3);

    for (int b = 0; b < batch_size; b++) {
        // A_i factored successfully
        EXPECT_EQ(h_info[b], 0);
    }

    for (int b = 0; b < batch_size; b++) {
        // A_i factored successfully
        EXPECT_EQ(h_info[b], 0);
    }

    gt::backend::host_allocator<T*>::deallocate(h_Aptr);
    gt::backend::device_allocator<T*>::deallocate(d_Aptr);
    gt::backend::host_allocator<T>::deallocate(h_A);
    gt::backend::device_allocator<T>::deallocate(d_A);

    gt::backend::host_allocator<int>::deallocate(h_p);
    gt::backend::device_allocator<int>::deallocate(d_p);
    gt::backend::host_allocator<int>::deallocate(h_info);
    gt::backend::device_allocator<int>::deallocate(d_info);
}