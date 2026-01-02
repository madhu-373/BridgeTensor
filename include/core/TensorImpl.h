#pragma once

#include "Storage.h"
#include "AutogradMeta.h"
#include "dtype/Dtype.h"
#include <vector>
#include <numeric>

namespace OwnTensor {

/**
 * @brief The core tensor implementation class.
 */
class TensorImpl : public Retainable {
public:
    TensorImpl(Storage storage, Dtype dtype, const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides = {});

    const Storage& storage() const { return storage_; }
    Storage& storage() { return storage_; }
    Dtype dtype() const { return dtype_; }
    Device device() const { return device_; }
    const std::vector<int64_t>& sizes() const { return sizes_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    int64_t storage_offset() const { return storage_offset_; }
    int64_t numel() const { return numel_; }

    void set_storage_offset(int64_t offset) { storage_offset_ = offset; }
    
    AutogradMetaInterface* autograd_meta() const { return autograd_meta_.get(); }
    void set_autograd_meta(std::unique_ptr<AutogradMetaInterface> meta) {
        autograd_meta_ = std::move(meta);
    }

private:
    Storage storage_;       //for storing data values
    Dtype dtype_;
    Device device_;
    std::vector<int64_t> sizes_;
    std::vector<int64_t> strides_;
    int64_t storage_offset_;
    int64_t numel_;
    std::unique_ptr<AutogradMetaInterface> autograd_meta_;
};

} // namespace OwnTensor
