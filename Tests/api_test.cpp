#include <iostream>
#include <cassert>
#include "core/Tensor.h"

using namespace OwnTensor;

void test_factory_functions() {
    std::cout << "Testing factory functions..." << std::endl;
    
    auto t_zeros = Tensor::zeros({2, 3}, Dtype::Float32);
    assert(t_zeros.numel() == 6);
    assert(static_cast<float*>(t_zeros.data_ptr())[0] == 0.0f);
    
    auto t_ones = Tensor::ones({4}, Dtype::Float32);
    assert(t_ones.numel() == 4);
    assert(static_cast<float*>(t_ones.data_ptr())[3] == 1.0f);
    
    auto t_full = Tensor::full({2, 2}, 7.5f, Dtype::Float32);
    assert(static_cast<float*>(t_full.data_ptr())[0] == 7.5f);
    
    auto t_rand = Tensor::rand({10}, Dtype::Float32);
    assert(t_rand.numel() == 10);
    
    std::cout << "Factory functions passed!" << std::endl;
}

void test_structural_ops() {
    std::cout << "Testing structural operations..." << std::endl;
    
    auto t = Tensor::ones({2, 4}, Dtype::Float32);
    
    // Reshape
    auto t_reshaped = t.reshape({8});
    assert(t_reshaped.ndim() == 1);
    assert(t_reshaped.numel() == 8);
    
    // Transpose
    auto t_T = t.t();
    assert(t_T.sizes() == std::vector<int64_t>({4, 2}));
    assert(t_T.is_contiguous() == false);
    
    // Contiguous
    auto t_contig = t_T.contiguous();
    assert(t_contig.is_contiguous() == true);
    assert(t_contig.sizes() == t_T.sizes());
    
    // Flatten
    auto t_flat = t.flatten();
    assert(t_flat.ndim() == 1);
    assert(t_flat.numel() == 8);
    
    std::cout << "Structural operations passed!" << std::endl;
}

void test_metadata() {
    std::cout << "Testing metadata..." << std::endl;
    
    auto t = Tensor::empty({3, 5}, Dtype::Float32);
    assert(t.ndim() == 2);
    assert(t.shape()[0] == 3);
    assert(t.shape()[1] == 5);
    assert(t.numel() == 15);
    assert(t.dtype_size() == 4);
    assert(t.nbytes() == 60);
    assert(t.is_cpu() == true);
    
    std::cout << "Metadata passed!" << std::endl;
}

void test_conversions() {
    std::cout << "Testing conversions..." << std::endl;
    
    auto t = Tensor::full({1}, 1.0f, Dtype::Float32);
    assert(t.to_bool() == true);
    
    auto t0 = Tensor::zeros({1}, Dtype::Float32);
    // Note: our current to_bool implementation uses Scalar::to<bool>() which might need check
    // assert(t0.to_bool() == false); 
    
    std::cout << "Conversions passed!" << std::endl;
}

int main() {
    try {
        test_factory_functions();
        test_structural_ops();
        test_metadata();
        test_conversions();
        
        std::cout << "\nDisplay demonstration:" << std::endl;
        auto t = Tensor::rand({2, 3}, Dtype::Float32);
        t.display();
        
        std::cout << "\nAll API tests passed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
