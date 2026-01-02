#pragma once

#include "TensorImpl.h"

namespace OwnTensor {

/**
 * @brief A lightweight, reference-counted handle to TensorImpl.
 */
class TensorBase {
public:
    TensorBase() = default;
    TensorBase(IntrusivePtr<TensorImpl> impl) : impl_(std::move(impl)) {}

    bool defined() const { return static_cast<bool>(impl_); }
    bool is_valid() const { return defined(); }

    int64_t ndim() const { return impl_->sizes().size(); }
    const std::vector<int64_t>& shape() const { return impl_->sizes(); }
    const std::vector<int64_t>& sizes() const { return impl_->sizes(); }
    const std::vector<int64_t>& strides() const { return impl_->strides(); }
    int64_t numel() const { return impl_->numel(); }
    Dtype dtype() const { return impl_->dtype(); }
    Device device() const { return impl_->device(); }
    int64_t storage_offset() const { return impl_->storage_offset(); }
    const Storage& storage() const { return impl_->storage(); }

    bool is_cpu() const { return device().is_cpu(); }
    bool is_cuda() const { return device().is_cuda(); }

    size_t dtype_size() const;
    size_t nbytes() const;

    bool is_contiguous() const;

    void* data_ptr() const;

    // Support for autograd metadata (placeholders)
    bool requires_grad() const;
    TensorBase grad() const;

    void reset() { impl_ = nullptr; }
    void release() { reset(); }

protected:
    IntrusivePtr<TensorImpl> impl_;
};

} // namespace OwnTensor
