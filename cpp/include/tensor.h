// Minimal tensor / GPU memory layer for the dots.mocr inference engine.
//
// Deliberately tiny: this is an inference engine, not a general array lib.
// The whole pipeline works in bf16 (the checkpoint dtype) plus float32
// scratch where a reduction needs the range (RMSNorm variance, softmax,
// rotary embedding). There is no autograd, no broadcasting beyond what the
// forward pass uses, and every allocation is explicit — we ship on cards
// with 12–16 GiB and a stray tensor is the difference between running and OOM.
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace dots {

using bf16  = __nv_bfloat16;
using bf16x2 = __nv_bfloat162;

enum class Dtype : int { F32 = 0, BF16 = 1, I32 = 2, I64 = 3, U8 = 4, BOOL = 5 };

inline std::size_t dtype_bytes(Dtype d) {
    switch (d) {
        case Dtype::F32:  return 4;
        case Dtype::BF16: return 2;
        case Dtype::I32:  return 4;
        case Dtype::I64:  return 8;
        case Dtype::U8:   return 1;
        case Dtype::BOOL: return 1;
    }
    return 0;
}

// A non-owning view of device memory with a 2-D logical shape. The engine's
// hottest operands are 2-D [seq, feat], so we specialise for that and keep a
// flat stride. Higher-rank tensors (q/k/v with a head axis) are handled by
// passing raw pointers + explicit shapes into the kernels that need them.
struct Tensor {
    void*   data     = nullptr;   // device pointer
    Dtype   dtype    = Dtype::BF16;
    int     rows     = 0;         // leading dim count
    int     cols     = 0;         // trailing dim count
    int     ld       = 0;         // stride between rows in *elements* (>= cols)
    bool    owned    = false;     // free on destruction?

    Tensor() = default;
    Tensor(void* d, Dtype dt, int r, int c, int leading = 0)
        : data(d), dtype(dt), rows(r), cols(c), ld(leading ? leading : c) {}

    bool   empty()      const { return data == nullptr || rows == 0 || cols == 0; }
    std::size_t numel()  const { return static_cast<std::size_t>(rows) * cols; }
    std::size_t nbytes() const { return numel() * dtype_bytes(dtype); }
};

// RAII device buffer. All engine allocations go through here so the only
// cudaMalloc/cudaFree sites are in this file.
struct DeviceBuffer {
    void*       ptr  = nullptr;
    std::size_t cap  = 0;          // bytes
    int         device = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t bytes, int dev = 0);
    ~DeviceBuffer();
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), cap(o.cap), device(o.device) {
        o.ptr = nullptr; o.cap = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            release();
            ptr = o.ptr; cap = o.cap; device = o.device;
            o.ptr = nullptr; o.cap = 0;
        }
        return *this;
    }
    void release();
    void resize(std::size_t bytes);
};

// Owning 2-D tensor backed by a DeviceBuffer. The shape (rows/cols) is the
// current logical view; the buffer may be larger (reused across calls).
struct DeviceTensor {
    Dtype         dtype = Dtype::BF16;
    int           rows = 0, cols = 0, ld = 0;
    DeviceBuffer  buf;

    DeviceTensor() = default;
    DeviceTensor(Dtype dt, int rows, int cols);

    void*       ptr()       { return buf.ptr; }
    const void* ptr() const { return buf.ptr; }
    Tensor      view();                       // current logical shape
    std::size_t nbytes() const { return static_cast<std::size_t>(rows) * cols * dtype_bytes(dtype); }
};

// ---- CUDA error checking -----------------------------------------------------
#define DOTS_CUDA_CHECK(expr)                                                  \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                    \
                    cudaGetErrorName(_e), __FILE__, __LINE__,                  \
                    cudaGetErrorString(_e));                                    \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

// ---- misc device helpers -----------------------------------------------------
template <typename T>
T* dmalloc(std::size_t n) {
    T* p = nullptr;
    DOTS_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    return p;
}
inline void dfree(void* p) { if (p) DOTS_CUDA_CHECK(cudaFree(p)); }

template <typename T>
void dtoh(const T* src, T* dst, std::size_t n) {
    DOTS_CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDeviceToHost));
}
template <typename T>
void htod(const T* src, T* dst, std::size_t n) {
    DOTS_CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToDevice));
}

int current_device();
size_t device_free_bytes(int dev);

}  // namespace dots
