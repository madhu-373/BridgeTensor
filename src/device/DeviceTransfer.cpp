#include "device/DeviceTransfer.h"
#include "device/AllocatorRegistry.h"
#include "device/Allocator.h"

namespace OwnTensor {
namespace device {

void copy_memory(void* dst, Device dst_device, const void* src, Device src_device, size_t bytes) {
    if (bytes == 0) return;
    if (dst == src && dst_device == src_device) return;

    cudaMemcpyKind kind;
    if (src_device.is_cpu() && dst_device.is_cuda()) {
        kind = cudaMemcpyHostToDevice;
    } else if (src_device.is_cuda() && dst_device.is_cpu()) {
        kind = cudaMemcpyDeviceToHost;
    } else if (src_device.is_cuda() && dst_device.is_cuda()) {
        kind = cudaMemcpyDeviceToDevice;
    } else {
        kind = cudaMemcpyHostToHost;
    }

    // Use the allocator of the device involved (usually the CUDA one for cross-device)
    Allocator* alloc = nullptr;
    if (dst_device.is_cuda()) {
        alloc = AllocatorRegistry::get_cuda_allocator();
    } else if (src_device.is_cuda()) {
        alloc = AllocatorRegistry::get_cuda_allocator();
    } else {
        alloc = AllocatorRegistry::get_cpu_allocator();
    }

    alloc->memcpy(dst, src, bytes, kind);
}

} // namespace device
} // namespace OwnTensor
