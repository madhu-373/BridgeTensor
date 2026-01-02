#include <iostream>
#include <cassert>
#include "core/Tensor.h"

using namespace OwnTensor;

void demonstrate_storage_efficiency() {
    std::cout << "=== Storage Efficiency Demo ===" << std::endl;
    
    // Create a tensor
    Tensor t1 = Tensor::ones({1000, 1000}, Dtype::Float32);
    std::cout << "Created t1 (1000x1000)" << std::endl;
    std::cout << "  Memory allocated: " << t1.allocated_bytes() / 1024 / 1024 << " MB" << std::endl;
    std::cout << "  Storage use_count: " << t1.storage().use_count() << std::endl;
    std::cout << "  Owns data: " << (t1.owns_data() ? "yes" : "no") << std::endl;
    
    // Copy tensor (shares storage!)
    Tensor t2 = t1;
    std::cout << "\nCopied t1 to t2 (cheap copy!)" << std::endl;
    std::cout << "  t1 storage use_count: " << t1.storage().use_count() << std::endl;
    std::cout << "  t2 storage use_count: " << t2.storage().use_count() << std::endl;
    std::cout << "  Same storage: " << (t1.storage().is_alias_of(t2.storage()) ? "yes" : "no") << std::endl;
    std::cout << "  t1 owns data: " << (t1.owns_data() ? "yes" : "no") << std::endl;
    
    // Create a view
    Tensor t3 = t2.view({1000000});
    std::cout << "\nCreated view t3 from t2" << std::endl;
    std::cout << "  Storage use_count: " << t1.storage().use_count() << std::endl;
    std::cout << "  All share storage: " << (t1.storage().is_alias_of(t3.storage()) ? "yes" : "no") << std::endl;
    
    // Modify through view
    float* data = static_cast<float*>(t3.data_ptr());
    data[0] = 42.0f;
    
    float* t1_data = static_cast<float*>(t1.data_ptr());
    std::cout << "\nModified t3[0] = 42" << std::endl;
    std::cout << "  t1[0] = " << t1_data[0] << " (changed because storage is shared!)" << std::endl;
    
    // Drop references
    {
        Tensor t4 = t1;
        std::cout << "\nCreated t4 temporarily" << std::endl;
        std::cout << "  Storage use_count: " << t1.storage().use_count() << std::endl;
    }
    std::cout << "t4 destroyed" << std::endl;
    std::cout << "  Storage use_count: " << t1.storage().use_count() << std::endl;
}

void demonstrate_external_memory() {
    std::cout << "\n=== External Memory Sharing Demo ===" << std::endl;
    
    // Allocate external memory
    const size_t size = 100;
    float* external_data = new float[size];
    for (size_t i = 0; i < size; ++i) {
        external_data[i] = static_cast<float>(i);
    }
    std::cout << "Allocated external memory: " << size * sizeof(float) << " bytes" << std::endl;
    
    // Create a tensor with unique storage
    Tensor t = Tensor::empty({100}, Dtype::Float32);
    std::cout << "Created empty tensor" << std::endl;
    std::cout << "  Owns data: " << (t.owns_data() ? "yes" : "no") << std::endl;
    
    // Share external pointer (with custom deleter)
    auto deleter = [](void* ptr) {
        std::cout << "  Custom deleter called!" << std::endl;
        delete[] static_cast<float*>(ptr);
    };
    
    t.storage().UniqueStorageShareExternalPointer(
        external_data, 
        size * sizeof(float),
        deleter
    );
    
    std::cout << "Shared external memory with tensor" << std::endl;
    
    // Verify data
    float* t_data = static_cast<float*>(t.data_ptr());
    std::cout << "  t[0] = " << t_data[0] << std::endl;
    std::cout << "  t[99] = " << t_data[99] << std::endl;
    
    std::cout << "Tensor will clean up external memory on destruction..." << std::endl;
}

int main() {
    try {
        demonstrate_storage_efficiency();
        demonstrate_external_memory();
        
        std::cout << "\n✓ All demonstrations completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
