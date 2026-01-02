#include "device/CUDAAllocator.h"
#include <stdexcept>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace OwnTensor {

void* CUDAAllocator::allocate(size_t bytes) {
#ifdef WITH_CUDA
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed");
    return ptr;
#else
    throw std::runtime_error("CUDA not supported in this build");
#endif
}

void CUDAAllocator::deallocate(void* ptr) {
#ifdef WITH_CUDA
    if (ptr) cudaFree(ptr);
#endif
}

void CUDAAllocator::memset(void* ptr, int value, size_t bytes) {
#ifdef WITH_CUDA
    if (ptr) cudaMemset(ptr, value, bytes);
#endif
}

void CUDAAllocator::memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
#ifdef WITH_CUDA
    if (dst && src) cudaMemcpy(dst, src, bytes, kind);
#endif
}

void CUDAAllocator::memsetAsync(void* ptr, int value, size_t bytes, cudaStream_t stream) {
#ifdef WITH_CUDA
    if (ptr) cudaMemsetAsync(ptr, value, bytes, stream);
#endif
}

void CUDAAllocator::memcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
#ifdef WITH_CUDA
    if (dst && src) cudaMemcpyAsync(dst, src, bytes, kind, stream);
#endif
}

} // namespace OwnTensor
