#pragma once

#include "device/Device.h"
#include "device/Allocator.h"
#include <utility>

namespace OwnTensor {

// Function pointer type for custom deleters
using DeleterFnPtr = void (*)(void*);

/**
 * @brief Smart pointer for managing raw data with custom deleters.
 * 
 * DataPtr encapsulates a raw pointer along with device information and
 * a custom deleter. This allows for flexible memory management across
 * different device types (CPU, CUDA) and memory sources (allocated,
 * borrowed, external).
 */
class DataPtr {
public:
    // Default constructor - null pointer
    DataPtr() 
        : data_(nullptr), 
          ctx_(nullptr), 
          deleter_(nullptr), 
          device_(DeviceType::CPU) {}

    // Constructor with data, context, deleter, and device
    DataPtr(void* data, void* ctx, DeleterFnPtr deleter, Device device)
        : data_(data), 
          ctx_(ctx), 
          deleter_(deleter), 
          device_(device) {}

    // Constructor with just data and device (no custom deleter)
    DataPtr(void* data, Device device)
        : data_(data), 
          ctx_(data), 
          deleter_(nullptr), 
          device_(device) {}

    // Move constructor
    DataPtr(DataPtr&& other) noexcept
        : data_(other.data_),
          ctx_(other.ctx_),
          deleter_(other.deleter_),
          device_(other.device_) {
        other.data_ = nullptr;
        other.ctx_ = nullptr;
        other.deleter_ = nullptr;
    }

    // Move assignment
    DataPtr& operator=(DataPtr&& other) noexcept {
        if (this != &other) {
            // Clean up current data
            clear();
            
            // Transfer ownership
            data_ = other.data_;
            ctx_ = other.ctx_;
            deleter_ = other.deleter_;
            device_ = other.device_;
            
            // Clear other
            other.data_ = nullptr;
            other.ctx_ = nullptr;
            other.deleter_ = nullptr;
        }
        return *this;
    }

    // Disable copy operations (unique ownership)
    DataPtr(const DataPtr&) = delete;
    DataPtr& operator=(const DataPtr&) = delete;

    // Destructor
    ~DataPtr() {
        clear();
    }

    // Get immutable pointer to data
    const void* get() const {
        return data_;
    }

    // Get mutable pointer to data
    void* mutable_get() {
        return data_;
    }

    // Get device
    Device device() const {
        return device_;
    }

    // Check if valid
    explicit operator bool() const {
        return data_ != nullptr;
    }

    // Release ownership and clean up
    void clear() {
        if (data_ && deleter_) {
            deleter_(ctx_);
        }
        data_ = nullptr;
        ctx_ = nullptr;
        deleter_ = nullptr;
    }

    // Compare data pointers
    bool operator==(const DataPtr& other) const {
        return data_ == other.data_;
    }

    bool operator!=(const DataPtr& other) const {
        return data_ != other.data_;
    }

private:
    void* data_;              // Actual data pointer
    void* ctx_;               // Context for deleter (often same as data_)
    DeleterFnPtr deleter_;    // Custom deleter function
    Device device_;           // Device where data resides
};

} // namespace OwnTensor
