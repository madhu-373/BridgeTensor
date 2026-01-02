#include "core/TensorBase.h"
#include "core/Tensor.h" 
#include "core/AutogradMeta.h"
#include "dtype/DtypeTraits.h"

namespace OwnTensor {

size_t TensorBase::dtype_size() const {
    return elementSize(dtype());
}

size_t TensorBase::nbytes() const {
    return numel() * dtype_size();
}

bool TensorBase::is_contiguous() const {
    if (!defined()) return true;
    const auto& sz = sizes();
    const auto& st = strides();
    int64_t expected_stride = 1;
    for (int i = (int)sz.size() - 1; i >= 0; --i) {
        if (sz[i] > 1 && st[i] != expected_stride) return false;
        expected_stride *= sz[i];
    }
    return true;
}

void* TensorBase::data_ptr() const {
    if (!defined()) return nullptr;
    return static_cast<char*>(impl_->storage().mutable_data()) + impl_->storage_offset() * dtype_size();
}

bool TensorBase::requires_grad() const {
    if (!defined() || !impl_->autograd_meta()) return false;
    return static_cast<AutogradMeta*>(impl_->autograd_meta())->requires_grad;
}

TensorBase TensorBase::grad() const {
    if (!defined() || !impl_->autograd_meta()) return TensorBase();
    auto meta = static_cast<AutogradMeta*>(impl_->autograd_meta());
    if (!meta->grad) return TensorBase();
    return *(meta->grad);
}

} // namespace OwnTensor
