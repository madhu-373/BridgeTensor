#include "core/Storage.h"
#include "device/AllocatorRegistry.h"
#include <stdexcept>

namespace OwnTensor {

StorageImpl::StorageImpl(size_t size_bytes, Device device)
    : size_bytes_(size_bytes), device_(device) {
    if (size_bytes_ > 0) {
        allocator_ = AllocatorRegistry::get_allocator(device_);
        data_ = allocator_->allocate(size_bytes_);
    } else {
        data_ = nullptr;
        allocator_ = nullptr;
    }
}

StorageImpl::~StorageImpl() {
    if (data_ && allocator_) {
        allocator_->deallocate(data_);
    }
}

} // namespace OwnTensor
