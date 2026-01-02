#include "core/Storage.h"
#include "device/AllocatorRegistry.h"
#include <stdexcept>

namespace OwnTensor {

// Constructor that allocates memory
Storage::Storage(
    use_byte_size_t /*use_byte_size*/,
    size_t size_bytes,
    Allocator* allocator,
    bool resizable)
    : storage_impl_(new StorageImpl(
        StorageImpl::use_byte_size_t(),
        size_bytes,
        allocator,
        resizable)) {}

// Constructor with pre-allocated memory
Storage::Storage(
    use_byte_size_t /*use_byte_size*/,
    size_t size_bytes,
    DataPtr data_ptr,
    Allocator* allocator,
    bool resizable)
    : storage_impl_(new StorageImpl(
        StorageImpl::use_byte_size_t(),
        size_bytes,
        std::move(data_ptr),
        allocator,
        resizable)) {}

// Legacy constructor for backward compatibility
Storage::Storage(size_t size_bytes, Device device)
    : storage_impl_(new StorageImpl(
        StorageImpl::use_byte_size_t(),
        size_bytes,
        AllocatorRegistry::get_allocator(device),
        false)) {}

} // namespace OwnTensor
