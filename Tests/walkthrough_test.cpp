#include <iostream>
#include <cassert>
#include "core/Tensor.h"

using namespace OwnTensor;

void test_creation_flow() {
    std::cout << "--- Testing Creation and Allocation Flow ---" << std::endl;
    
    // 1. User end: factory method
    // This calls Tensor::empty -> Storage creation -> Allocator::allocate
    auto t = Tensor::zeros({2, 3}, Dtype::Float32);
    
    std::cout << "Tensor created with shape [2, 3] on " << (t.device().is_cpu() ? "CPU" : "GPU") << std::endl;
    std::cout << "Allocated bytes: " << t.allocated_bytes() << std::endl;
    assert(t.allocated_bytes() == 2 * 3 * 4); // 2*3 float32 (4 bytes each)
}

void test_autograd_meta() {
    std::cout << "--- Testing Autograd and Gradient Storage ---" << std::endl;
    
    auto t = Tensor::ones({2, 2}, Dtype::Float32);
    
    // Check initial state
    assert(t.requires_grad() == false);
    assert(!t.grad().defined());
    
    // 2. Set requires_grad
    t.set_requires_grad(true);
    assert(t.requires_grad() == true);
    
    // 3. Store gradient
    auto g = Tensor::full({2, 2}, 0.5f, Dtype::Float32);
    t.set_grad(g);
    
    assert(t.grad().defined());
    assert(t.grad().numel() == 4);
    assert(t.grad_nbytes() == 16);
    
    std::cout << "Gradient successfully stored in AutogradMeta." << std::endl;
    std::cout << "Tensor grad display:" << std::endl;
    Tensor(t.grad()).display();
}

int main() {
    try {
        test_creation_flow();
        test_autograd_meta();
        std::cout << "All walkthrough tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
