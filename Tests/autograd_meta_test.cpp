#include <iostream>
#include <cassert>
#include "core/Tensor.h"
#include "autograd/Hooks.h"
#include <thread>

using namespace OwnTensor;

void test_basic_autograd_meta() {
    std::cout << "=== Test: Basic AutogradMeta ===" << std::endl;
    
    Tensor t = Tensor::ones({2, 2}, Dtype::Float32);
    assert(!t.requires_grad());
    
    t.set_requires_grad(true);
    assert(t.requires_grad());
    
    t.set_grad(Tensor::zeros({2, 2}, Dtype::Float32));
    assert(t.grad().defined());
    
    std::cout << "✓ Basic autograd meta works!" << std::endl;
}

void test_view_tracking() {
    std::cout << "\n=== Test: View Tracking ===" << std::endl;
    
    Tensor base = Tensor::ones({4}, Dtype::Float32);
    base.set_requires_grad(true);
    
    Tensor view = base.view({2, 2});
    assert(!view.is_view());  // Not set yet
    
    view.set_is_view(true);
    assert(view.is_view());
    
    std::cout << "✓ View tracking works!" << std::endl;
}

void test_gradient_function() {
    std::cout << "\n=== Test: Gradient Function ===" << std::endl;
    
    Tensor t = Tensor::ones({3}, Dtype::Float32);
    assert(t.grad_fn() == nullptr);  // Leaf tensor
    
    // In a real implementation, operations would set this
    // For now, just create a dummy node
    class DummyNode : public Node {
    public:
        std::vector<TensorBase> apply(std::vector<TensorBase>&& grads) override {
            return std::vector<TensorBase>();
        }
    };
    
    auto dummy_fn = std::make_shared<DummyNode>();
    t.set_grad_fn(dummy_fn);
    
    assert(t.grad_fn() != nullptr);
    assert(t.requires_grad());  // Should be true because has grad_fn
    
    std::cout << "✓ Gradient function tracking works!" << std::endl;
}

void test_output_number() {
    std::cout << "\n=== Test: Output Number ===" << std::endl;
    
    Tensor t = Tensor::ones({2}, Dtype::Float32);
    assert(t.output_nr() == 0);  // Default
    
    t.set_output_nr(2);
    assert(t.output_nr() == 2);
    
    std::cout << "✓ Output number tracking works!" << std::endl;
}

void test_retains_grad() {
    std::cout << "\n=== Test: Retains Grad ===" << std::endl;
    
    Tensor t = Tensor::ones({2}, Dtype::Float32);
    assert(!t.retains_grad());
    
    t.set_retains_grad(true);
    assert(t.retains_grad());
    
    std::cout << "✓ Retains grad flag works!" << std::endl;
}

void test_hooks() {
    std::cout << "\n=== Test: Hooks ===" << std::endl;
    
    Tensor t = Tensor::ones({3}, Dtype::Float32);
    t.set_requires_grad(true);
    
    // Test pre-hook
    int pre_hook_called = 0;
    t.register_hook(make_pre_hook([&](const TensorBase& grad) -> TensorBase {
        pre_hook_called++;
        std::cout << "  Pre-hook called!" << std::endl;
        return grad;  // Can modify gradient here
    }));
    
    // Test post-accumulation hook
    int post_hook_called = 0;
    t.register_post_acc_hook(make_post_acc_hook([&](const TensorBase& grad) {
        post_hook_called++;
        std::cout << "  Post-acc hook called!" << std::endl;
    }));
    
    // In a real backward pass, these would be called automatically
    // For now, we can't test them without a full backward implementation
    
    std::cout << "✓ Hooks registered successfully!" << std::endl;
    
    // Clear hooks
    t.clear_hooks();
    std::cout << "✓ Hooks cleared!" << std::endl;
}

void test_thread_safety() {
    std::cout << "\n=== Test: Thread Safety ===" << std::endl;
    
    Tensor t = Tensor::ones({100}, Dtype::Float32);
    t.set_requires_grad(true);
    
    // Test concurrent access to autograd meta
    std::thread t1([&]() {
        for (int i = 0; i < 100; ++i) {
            t.set_is_view(i % 2 == 0);
        }
    });
    
    std::thread t2([&]() {
        for (int i = 0; i < 100; ++i) {
            bool is_view = t.is_view();
            (void)is_view;  // Suppress unused warning
        }
    });
    
    t1.join();
    t2.join();
    
    std::cout << "✓ Thread safety check passed (no crash)!" << std::endl;
}

int main() {
    try {
        test_basic_autograd_meta();
        test_view_tracking();
        test_gradient_function();
        test_output_number();
        test_retains_grad();
        test_hooks();
        test_thread_safety();
        
        std::cout << "\n✅ All AutogradMeta tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
