#include "tensor.h"

#include <cstdio>
#include <cstdlib>

namespace dots {

// ---- DeviceBuffer -----------------------------------------------------------

DeviceBuffer::DeviceBuffer(std::size_t bytes, int dev) : cap(0), device(dev) {
    if (bytes == 0) return;
    int prev = 0;
    DOTS_CUDA_CHECK(cudaGetDevice(&prev));
    if (dev != prev) DOTS_CUDA_CHECK(cudaSetDevice(dev));
    DOTS_CUDA_CHECK(cudaMalloc(&ptr, bytes));
    cap = bytes;
    if (dev != prev) DOTS_CUDA_CHECK(cudaSetDevice(prev));
}

DeviceBuffer::~DeviceBuffer() { release(); }

void DeviceBuffer::release() {
    if (ptr) {
        dfree(ptr);
        ptr = nullptr;
    }
    cap = 0;
}

void DeviceBuffer::resize(std::size_t bytes) {
    if (bytes <= cap && ptr) return;
    release();
    if (bytes == 0) return;
    int prev = 0;
    DOTS_CUDA_CHECK(cudaGetDevice(&prev));
    if (device != prev) DOTS_CUDA_CHECK(cudaSetDevice(device));
    DOTS_CUDA_CHECK(cudaMalloc(&ptr, bytes));
    cap = bytes;
    if (device != prev) DOTS_CUDA_CHECK(cudaSetDevice(prev));
}

// ---- DeviceTensor -----------------------------------------------------------

DeviceTensor::DeviceTensor(Dtype dt, int r, int c)
    : dtype(dt), rows(r), cols(c), ld(c) {
    buf.resize(static_cast<std::size_t>(r) * c * dtype_bytes(dt));
}

Tensor DeviceTensor::view() {
    return Tensor(buf.ptr, dtype, rows, cols, ld);
}

int current_device() { int d = 0; DOTS_CUDA_CHECK(cudaGetDevice(&d)); return d; }

size_t device_free_bytes(int dev) {
    size_t free_b = 0, total_b = 0;
    DOTS_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    return free_b;
}

}  // namespace dots
