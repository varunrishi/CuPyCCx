#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace cupyccx {
namespace cuda_utils {

// Throw a std::runtime_error with the CUDA error string
inline void check_cuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error at ") + file + ":" + std::to_string(line)
            + " — " + cudaGetErrorString(err));
    }
}

inline void check_cublas(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("cuBLAS error at ") + file + ":" + std::to_string(line)
            + " — status " + std::to_string(static_cast<int>(status)));
    }
}

#define CUDA_CHECK(x)   cupyccx::cuda_utils::check_cuda((x),   __FILE__, __LINE__)
#define CUBLAS_CHECK(x) cupyccx::cuda_utils::check_cublas((x), __FILE__, __LINE__)

// RAII wrapper around a cuBLAS handle — one per process/thread
struct CublasHandle {
    cublasHandle_t handle;
    CublasHandle()  { CUBLAS_CHECK(cublasCreate(&handle)); }
    ~CublasHandle() { cublasDestroy(handle); }
    // Non-copyable
    CublasHandle(const CublasHandle&)            = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
};

// RAII device buffer
template <typename T>
struct DeviceBuffer {
    T*          ptr  = nullptr;
    std::size_t size = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) : size(n) {
        CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }

    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), size(o.size) { o.ptr = nullptr; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) { if (ptr) cudaFree(ptr); ptr = o.ptr; size = o.size; o.ptr = nullptr; }
        return *this;
    }

    void upload(const T* host, std::size_t n) {
        CUDA_CHECK(cudaMemcpy(ptr, host, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T* host, std::size_t n) const {
        CUDA_CHECK(cudaMemcpy(host, ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    }
};

// Check whether at least one CUDA device is available
inline bool has_cuda_device() {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n > 0;
}

}  // namespace cuda_utils
}  // namespace cupyccx
