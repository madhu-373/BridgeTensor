#include "core/StorageImpl.h"
#include "device/AllocatorRegistry.h"
#include <stdexcept>
#include <cassert>

namespace OwnTensor {

// Helper function to create a deleter for allocator-managed memory
static void allocator_deleter(void* ctx) {
    // ctx contains both the pointer and allocator
    // For now, we'll use a simpler approach where the allocator is stored
    // in the StorageImpl and cleanup happens in reset()
}

StorageImpl::StorageImpl(
    use_byte_size_t /*use_byte_size*/,
    size_t size_bytes,
    DataPtr data_ptr,
    Allocator* allocator,
    bool resizable)
    : data_ptr_(std::move(data_ptr)),
      size_bytes_(size_bytes),
      resizable_(resizable),
      allocator_(allocator) {
    
    // Validation: resizable storage must have an allocator
    if (resizable_ && !allocator_) {
        throw std::invalid_argument(
            "StorageImpl: resizable storage must have an allocator");
    }
}

StorageImpl::StorageImpl(
    use_byte_size_t /*use_byte_size*/,
    size_t size_bytes,
    Allocator* allocator,
    bool resizable)
    : size_bytes_(size_bytes),
      resizable_(resizable),
      allocator_(allocator) {
    
    // Validation: must have allocator to allocate memory
    if (!allocator_) {
        throw std::invalid_argument(
            "StorageImpl: allocator required when allocating memory");
    }

    // Validation: resizable storage must have an allocator
    if (resizable_ && !allocator_) {
        throw std::invalid_argument(
            "StorageImpl: resizable storage must have an allocator");
    }

    // Allocate memory
    void* data = nullptr;
    if (size_bytes_ > 0) {
        data = allocator_->allocate(size_bytes_);
    }

    // Determine device from allocator
    // For now, we'll infer from the allocator registry
    Device device = DeviceType::CPU; // Default
    
    // Create DataPtr with custom deleter that uses our allocator
    // We'll store the allocator in the StorageImpl and handle deletion in reset()
    data_ptr_ = DataPtr(data, device);
}

StorageImpl::~StorageImpl() {
    // Clean up allocator-managed memory
    if (data_ptr_ && allocator_) {
        void* ptr = data_ptr_.mutable_get();
        if (ptr) {
            allocator_->deallocate(ptr);
        }
    }
    // DataPtr destructor will handle custom deleters automatically
}

DataPtr StorageImpl::set_data_ptr(DataPtr&& data_ptr) {
    DataPtr old_data_ptr = std::move(data_ptr_);
    data_ptr_ = std::move(data_ptr);
    return old_data_ptr;
}

void StorageImpl::set_data_ptr_noswap(DataPtr&& data_ptr) {
    data_ptr_ = std::move(data_ptr);
}

void StorageImpl::set_resizable(bool resizable) {
    if (resizable && !allocator_) {
        throw std::invalid_argument(
            "StorageImpl: cannot make storage resizable without an allocator");
    }
    resizable_ = resizable;
}

void StorageImpl::UniqueStorageShareExternalPointer(
    void* src,
    size_t size_bytes,
    DeleterFnPtr d) {
    
    // Check that we have unique ownership
    if (use_count() != 1) {
        throw std::runtime_error(
            "UniqueStorageShareExternalPointer can only be called when use_count == 1");
    }

    // Clean up old allocator-managed memory FIRST
    if (data_ptr_ && allocator_) {
        void* ptr = data_ptr_.mutable_get();
        if (ptr) {
            allocator_->deallocate(ptr);
        }
    }

    // Get device from current data_ptr
    Device device = data_ptr_.device();

    // Create new DataPtr with custom deleter
    DataPtr new_data_ptr(src, src, d, device);

    // Replace data
    data_ptr_ = std::move(new_data_ptr);
    size_bytes_ = size_bytes;
    allocator_ = nullptr;  // External memory, no allocator
    resizable_ = false;    // Can't resize external memory
}

void StorageImpl::UniqueStorageShareExternalPointer(
    DataPtr&& data_ptr,
    size_t size_bytes) {
    
    // Check that we have unique ownership
    if (use_count() != 1) {
        throw std::runtime_error(
            "UniqueStorageShareExternalPointer can only be called when use_count == 1");
    }

    // Clean up old allocator-managed memory FIRST
    if (data_ptr_ && allocator_) {
        void* ptr = data_ptr_.mutable_get();
        if (ptr) {
            allocator_->deallocate(ptr);
        }
    }

    // Replace data
    data_ptr_ = std::move(data_ptr);
    size_bytes_ = size_bytes;
    allocator_ = nullptr;  // External memory, no allocator
    resizable_ = false;    // Can't resize external memory
}

void StorageImpl::reset() {
    // If we have allocator-managed memory, deallocate it
    if (data_ptr_ && allocator_) {
        void* ptr = data_ptr_.mutable_get();
        if (ptr) {
            allocator_->deallocate(ptr);
        }
    }
    
    // Clear DataPtr (this also calls custom deleter if any)
    data_ptr_.clear();
    size_bytes_ = 0;
}

} // namespace OwnTensor
