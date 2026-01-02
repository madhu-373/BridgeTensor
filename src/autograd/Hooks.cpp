#include "autograd/Hooks.h"
#include "core/TensorBase.h"

namespace OwnTensor {

// LambdaPreHook implementation
LambdaPreHook::LambdaPreHook(hook_fn fn) : fn_(std::move(fn)) {}

TensorBase LambdaPreHook::operator()(const TensorBase& grad) {
    return fn_(grad);
}

// LambdaPostAccHook implementation
LambdaPostAccHook::LambdaPostAccHook(hook_fn fn) : fn_(std::move(fn)) {}

void LambdaPostAccHook::operator()(const TensorBase& grad) {
    fn_(grad);
}

} // namespace OwnTensor
