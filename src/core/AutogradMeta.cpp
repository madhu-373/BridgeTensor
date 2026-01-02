#include "core/AutogradMeta.h"
#include "core/TensorBase.h"
#include "core/TensorImpl.h"
#include "dtype/DtypeTraits.h"
#include <stdexcept>

namespace OwnTensor {

// Helper function to check if dtype supports gradients
static bool isDifferentiableType(Dtype dtype) {
    return dtype == Dtype::Float32 || dtype == Dtype::Float64;
}

AutogradMeta::AutogradMeta(
    TensorImpl* self_impl,
    bool requires_grad,
    Edge gradient_edge)
    : grad_fn_(std::move(gradient_edge.function)),
      output_nr_(gradient_edge.input_nr) {
    
    // Validate: if we have a grad_fn, requires_grad should be false
    // (non-leaf tensors don't have the requires_grad flag set)
    if (grad_fn_ && requires_grad) {
        throw std::invalid_argument(
            "AutogradMeta: requires_grad should be false if grad_fn is set");
    }

    // Set requires_grad (this also validates dtype)
    if (requires_grad) {
        if (!self_impl) {
            throw std::invalid_argument(
                "AutogradMeta: self_impl required when requires_grad=true");
        }
        set_requires_grad(requires_grad, self_impl);
    }
}

void AutogradMeta::set_requires_grad(bool requires_grad, TensorImpl* self_impl) {
    if (!self_impl) {
        throw std::invalid_argument(
            "AutogradMeta::set_requires_grad: self_impl cannot be null");
    }

    // Validate that only floating point tensors can require gradients
    if (requires_grad) {
        Dtype dtype = self_impl->dtype();
        if (!isDifferentiableType(dtype)) {
            throw std::invalid_argument(
                "Only Tensors of floating point dtype can require gradients");
        }
    }

    requires_grad_ = requires_grad;
}

TensorBase& AutogradMeta::mutable_grad() {
    // Lazy initialization: create undefined grad if not set
    // This matches PyTorch's behavior
    if (!grad_) {
        grad_ = std::make_shared<TensorBase>();  // Creates undefined tensor
    }
    return *grad_;
}

const TensorBase& AutogradMeta::grad() const {
    // For const access, we don't lazily initialize
    // Just return undefined if not set
    if (!grad_) {
        static thread_local TensorBase undefined_grad;
        return undefined_grad;
    }
    return *grad_;
}

} // namespace OwnTensor
