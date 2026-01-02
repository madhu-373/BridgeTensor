#include "device/CPUAllocator.h"
#include <cstdlib>
#include <cstring>
#include <new>

namespace OwnTensor {

void* CPUAllocator::allocate(size_t bytes) {
    if (bytes == 0) return nullptr;
    void* ptr = std::malloc(bytes);
    if (!ptr) throw std::bad_alloc();
    return ptr;
}

void CPUAllocator::deallocate(void* ptr) {
    if (ptr) std::free(ptr);
}

void CPUAllocator::memset(void* ptr, int value, size_t bytes) {
    if (ptr) std::memset(ptr, value, bytes);
}

void CPUAllocator::memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
    if (dst && src) std::memcpy(dst, src, bytes);
}

void CPUAllocator::memsetAsync(void* ptr, int value, size_t bytes, cudaStream_t stream) {
    memset(ptr, value, bytes);
}

void CPUAllocator::memcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
    memcpy(dst, src, bytes, kind);
}

} // namespace OwnTensor
