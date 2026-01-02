#include <iostream>
#include <cassert>
#include "core/Tensor.h"

using namespace OwnTensor;

void test_allocator_registry() {
    std::cout << "Testing AllocatorRegistry..." << std::endl;
    Allocator* cpu_alloc = AllocatorRegistry::get_cpu_allocator();
    assert(cpu_alloc != nullptr);
    
    Allocator* cuda_alloc = AllocatorRegistry::get_cuda_allocator();
    assert(cuda_alloc != nullptr);
    
    std::cout << "AllocatorRegistry tests passed!" << std::endl;
}

void test_storage_multi_device() {
    std::cout << "Testing Storage with multi-device support..." << std::endl;
    
    // CPU Allocation
    Storage s_cpu(100, DeviceType::CPU);
    assert(s_cpu.device().is_cpu());
    assert(s_cpu.data() != nullptr);
    
    // CUDA Allocation (should fail or throw if CUDA not available/mocked)
    try {
        Storage s_cuda(100, DeviceType::CUDA);
        std::cout << "CUDA Allocation succeeded (likely WITH_CUDA defined or mocked)" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "CUDA Allocation failed as expected: " << e.what() << std::endl;
    }
    
    std::cout << "Storage multi-device tests passed!" << std::endl;
}

void test_tensor_to_device() {
    std::cout << "Testing Tensor::to(Device)..." << std::endl;
    
    Tensor t_cpu = Tensor::ones({2, 2}, Dtype::Float32, DeviceType::CPU);
    assert(t_cpu.device().is_cpu());
    
    // Transfer to same device (should be no-op)
    Tensor t_cpu_new = t_cpu.to(DeviceType::CPU);
    assert(t_cpu_new.device().is_cpu());
    // In our current simple implementation, it returns *this if device matches
    // But if we wanted it to be like PyTorch we might clone if needed.
    // Our implementation does: if (this->device() == device) return *this;
    
    // Transfer to GPU (will throw if not supported/mocked)
    try {
        Tensor t_gpu = t_cpu.to(DeviceType::CUDA);
        std::cout << "Transfer to CUDA succeeded" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Transfer to CUDA failed as expected: " << e.what() << std::endl;
    }
    
    std::cout << "Tensor::to(Device) tests passed!" << std::endl;
}

int main() {
    test_allocator_registry();
    test_storage_multi_device();
    test_tensor_to_device();
    std::cout << "All device/allocator integration tests passed!" << std::endl;
    return 0;
}
