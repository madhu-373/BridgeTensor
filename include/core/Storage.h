#pragma once

#include "IntrusivePtr.h"
#include "device/Device.h"
#include "device/AllocatorRegistry.h"
#include <cstdlib>
#include <stdexcept>

namespace OwnTensor {

/**
 * @brief Raw data storage implementation.
 */
class StorageImpl : public Retainable {
public:
    StorageImpl(size_t size_bytes, Device device = DeviceType::CPU);
    ~StorageImpl();

    void* data() const { return data_; }
    size_t size_bytes() const { return size_bytes_; }
    Device device() const { return device_; }
    Allocator* allocator() const { return allocator_; }

private:
    void* data_;
    size_t size_bytes_;
    Device device_;
    Allocator* allocator_;
};

/**
 * @brief Handle for StorageImpl.
 */
struct Storage {
    IntrusivePtr<StorageImpl> impl;

    Storage() = default;
    Storage(IntrusivePtr<StorageImpl> impl) : impl(std::move(impl)) {}
    Storage(size_t size_bytes, Device device = DeviceType::CPU)
        : impl(new StorageImpl(size_bytes, device)) {}

    void* data() const { return impl ? impl->data() : nullptr; }
    size_t size_bytes() const { return impl ? impl->size_bytes() : 0; }
    Device device() const { return impl ? impl->device() : Device(); }
};

} // namespace OwnTensor
