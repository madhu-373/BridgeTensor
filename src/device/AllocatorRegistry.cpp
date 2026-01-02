#include "device/AllocatorRegistry.h"
#include "device/CPUAllocator.h"
#include "device/CUDAAllocator.h"

namespace OwnTensor {

Allocator* AllocatorRegistry::get_cpu_allocator() {
    static CPUAllocator cpu_alloc;
    return &cpu_alloc;
}

Allocator* AllocatorRegistry::get_cuda_allocator() {
    static CUDAAllocator cuda_alloc;
    return &cuda_alloc;
}

Allocator* AllocatorRegistry::get_allocator(Device device) {
    if (device.is_cpu()) return get_cpu_allocator();
    if (device.is_cuda()) return get_cuda_allocator();
    return get_cpu_allocator(); // Default
}

} // namespace OwnTensor
