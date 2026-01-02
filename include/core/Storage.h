#pragma once

#include "IntrusivePtr.h"
#include "StorageImpl.h"
#include "DataPtr.h"
#include "device/Device.h"
#include "device/Allocator.h"
#include <cstdlib>

namespace OwnTensor {

/**
 * @brief Lightweight handle for StorageImpl.
 * 
 * Storage is a thin wrapper around IntrusivePtr<StorageImpl> that provides
 * a convenient interface for memory management. Multiple Storage objects
 * can share the same underlying StorageImpl (reference counted).
 * 
 * This design enables:
 * - Cheap copies (just pointer + refcount increment)
 * - Easy aliasing detection
 * - Shared memory between tensors (views, slices)
 * - Const-correct API (const Storage can modify data, but not the handle)
 */
class Storage {
public:
    // Tag for byte size constructor
    struct use_byte_size_t {};

    // --- Constructors ---

    /**
     * @brief Default constructor - empty storage.
     */
    Storage() = default;

    /**
     * @brief Construct from existing StorageImpl.
     */
    Storage(IntrusivePtr<StorageImpl> ptr)
        : storage_impl_(std::move(ptr)) {}

    /**
     * @brief Allocate memory using allocator.
     * 
     * @param use_byte_size Tag for disambiguation
     * @param size_bytes Number of bytes to allocate
     * @param allocator Allocator to use (nullptr = default for CPU)
     * @param resizable Whether storage can be resized
     */
    Storage(
        use_byte_size_t /*use_byte_size*/,
        size_t size_bytes,
        Allocator* allocator = nullptr,
        bool resizable = false);

    /**
     * @brief Create storage with pre-allocated memory.
     * 
     * @param use_byte_size Tag for disambiguation
     * @param size_bytes Size of the allocation
     * @param data_ptr Pre-allocated memory (takes ownership)
     * @param allocator Allocator for potential resizing
     * @param resizable Whether storage can be resized
     */
    Storage(
        use_byte_size_t /*use_byte_size*/,
        size_t size_bytes,
        DataPtr data_ptr,
        Allocator* allocator = nullptr,
        bool resizable = false);

    // Legacy constructor for backward compatibility
    Storage(size_t size_bytes, Device device = DeviceType::CPU);

    // --- Data Access ---
    // Note: These are const methods that return mutable data!
    // This is intentional - the Storage handle is const, but data isn't.

    /**
     * @brief Get immutable pointer to data.
     */
    const void* data() const {
        return storage_impl_ ? storage_impl_->data() : nullptr;
    }

    /**
     * @brief Get mutable pointer to data (const method!).
     */
    void* mutable_data() const {
        return storage_impl_ ? storage_impl_->mutable_data() : nullptr;
    }

    /**
     * @brief Get immutable DataPtr reference.
     */
    const DataPtr& data_ptr() const {
        return storage_impl_->data_ptr();
    }

    /**
     * @brief Get mutable DataPtr reference (const method!).
     */
    DataPtr& mutable_data_ptr() const {
        return storage_impl_->mutable_data_ptr();
    }

    /**
     * @brief Replace data pointer and return the old one.
     */
    DataPtr set_data_ptr(DataPtr&& data_ptr) const {
        return storage_impl_->set_data_ptr(std::move(data_ptr));
    }

    /**
     * @brief Set data pointer without swapping.
     */
    void set_data_ptr_noswap(DataPtr&& data_ptr) const {
        storage_impl_->set_data_ptr_noswap(std::move(data_ptr));
    }

    // --- Size Management ---

    /**
     * @brief Get number of bytes in storage.
     */
    size_t nbytes() const {
        return storage_impl_ ? storage_impl_->nbytes() : 0;
    }

    /**
     * @brief Set number of bytes (doesn't reallocate).
     */
    void set_nbytes(size_t size_bytes) const {
        storage_impl_->set_nbytes(size_bytes);
    }

    // --- Device & Allocator ---

    /**
     * @brief Get device type.
     */
    DeviceType device_type() const {
        return storage_impl_ ? storage_impl_->device_type() : DeviceType::CPU;
    }

    /**
     * @brief Get device.
     */
    Device device() const {
        return storage_impl_ ? storage_impl_->device() : Device();
    }

    /**
     * @brief Get allocator.
     */
    Allocator* allocator() const {
        return storage_impl_ ? storage_impl_->allocator() : nullptr;
    }

    // --- Resizability ---

    /**
     * @brief Check if storage is resizable.
     */
    bool resizable() const {
        return storage_impl_ ? storage_impl_->resizable() : false;
    }

    // --- Reference Counting ---

    /**
     * @brief Get reference count.
     */
    size_t use_count() const {
        return storage_impl_ ? storage_impl_->use_count() : 0;
    }

    /**
     * @brief Check if this is the only reference.
     */
    bool unique() const {
        return storage_impl_ && storage_impl_->use_count() == 1;
    }

    /**
     * @brief Check if two Storage objects share the same underlying data.
     */
    bool is_alias_of(const Storage& other) const {
        return storage_impl_ == other.storage_impl_;
    }

    // --- Validity ---

    /**
     * @brief Check if storage is valid (not null).
     */
    explicit operator bool() const {
        return storage_impl_ != nullptr;
    }

    // --- External Memory Sharing ---

    /**
     * @brief Share external pointer (only when use_count == 1).
     */
    void UniqueStorageShareExternalPointer(
        void* src,
        size_t capacity,
        DeleterFnPtr d = nullptr) const {
        if (!unique()) {
            throw std::runtime_error(
                "UniqueStorageShareExternalPointer can only be called when use_count == 1");
        }
        storage_impl_->UniqueStorageShareExternalPointer(src, capacity, d);
    }

    /**
     * @brief Share external pointer via DataPtr (only when use_count == 1).
     */
    void UniqueStorageShareExternalPointer(
        DataPtr&& data_ptr,
        size_t capacity) const {
        if (!unique()) {
            throw std::runtime_error(
                "UniqueStorageShareExternalPointer can only be called when use_count == 1");
        }
        storage_impl_->UniqueStorageShareExternalPointer(std::move(data_ptr), capacity);
    }

    // --- Internal Access ---

    /**
     * @brief Get raw pointer to StorageImpl (use with caution!).
     */
    StorageImpl* unsafeGetStorageImpl() const {
        return storage_impl_.get();
    }

    // Backward compatibility
    size_t size_bytes() const { return nbytes(); }

private:
    IntrusivePtr<StorageImpl> storage_impl_;
};

} // namespace OwnTensor
