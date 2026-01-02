#pragma once
#include <cstddef>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#else
// Define placeholder types if CUDA is not available
typedef void* cudaStream_t;
enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};
#endif

namespace OwnTensor {

class Allocator {
public:
    virtual ~Allocator() = default;
    
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;

    // --- Synchronous API ---
    virtual void memset(void* ptr, int value, size_t bytes) = 0;
    virtual void memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) = 0;

    // --- Asynchronous API ---
    virtual void memsetAsync(void* ptr, int value, size_t bytes, cudaStream_t stream) = 0;
    virtual void memcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) = 0;
};

} // namespace OwnTensor
