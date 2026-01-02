#include <iostream>
#include <cassert>
#include "core/Tensor.h"

using namespace OwnTensor;

void test_tensor_creation() {
    std::cout << "Testing tensor creation..." << std::endl;
    std::vector<int64_t> sizes = {2, 3};
    Tensor t = Tensor::empty(sizes, Dtype::Float32);
    
    assert(t.defined());
    assert(t.sizes() == sizes);
    assert(t.numel() == 6);
    assert(t.device().is_cpu());
    std::cout << "Tensor creation test passed!" << std::endl;
}

void test_memory_sharing() {
    std::cout << "Testing memory sharing (views)..." << std::endl;
    Tensor t1 = Tensor::empty({4}, Dtype::Float32);
    
    // Fill t1 data manually
    float* data = static_cast<float*>(t1.data_ptr());
    for (int i = 0; i < 4; ++i) data[i] = static_cast<float>(i);
    
    // Create a view
    Tensor t2 = t1.view({2, 2});
    
    assert(t2.defined());
    assert(t2.sizes() == std::vector<int64_t>({2, 2}));
    assert(t2.data_ptr() == t1.data_ptr());
    
    // Modify t2, check t1
    float* data2 = static_cast<float*>(t2.data_ptr());
    data2[0] = 100.0f;
    
    assert(data[0] == 100.0f);
    std::cout << "Memory sharing test passed!" << std::endl;
}

void test_reference_counting() {
    std::cout << "Testing reference counting..." << std::endl;
    {
        Tensor t1 = Tensor::empty({100}, Dtype::Float32);
        void* ptr = t1.data_ptr();
        {
            Tensor t2 = t1;
            // t1 and t2 share the same StorageImpl and TensorImpl
        }
        // t2 is destroyed, but t1 still holds the storage
        assert(t1.data_ptr() == ptr);
    }
    // t1 is destroyed, StorageImpl should be deleted (verified by lack of leak in simple cases)
    std::cout << "Reference counting test passed!" << std::endl;
}

int main() {
    try {
        test_tensor_creation();
        test_memory_sharing();
        test_reference_counting();
        std::cout << "All core tests passed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
