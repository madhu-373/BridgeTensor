#pragma once

#include "IntrusivePtr.h"
#include "DataPtr.h"
#include "device/Device.h"
#include "device/Allocator.h"
#include <cstdlib>
#include <stdexcept>

namespace OwnTensor {

/**
 * @brief Core storage implementation managing raw memory.
 * 
 * StorageImpl is the heavy implementation class that manages:
 * - Raw memory buffer via DataPtr
 * - Size tracking
 * - Allocator for potential resizing
 * - Resizability flag
 * - Device information
 * 
 * This class is reference-counted via Retainable base class.
 * Multiple Storage handles can point to the same StorageImpl.
 */
class StorageImpl : public Retainable {
public:
    // Tag for specifying byte size explicitly
    struct use_byte_size_t {};

    /**
     * @brief Construct storage with pre-allocated memory.
     * 
     * @param use_byte_size Tag for disambiguation
     * @param size_bytes Number of bytes in the allocation
     * @param data_ptr Pre-allocated memory (takes ownership)
     * @param allocator Allocator for potential resizing (can be nullptr if not resizable)
     * @param resizable Whether this storage can be resized
     */
    StorageImpl(
        use_byte_size_t /*use_byte_size*/,
        size_t size_bytes,
        DataPtr data_ptr,
        Allocator* allocator,
        bool resizable);

    /**
     * @brief Construct storage and allocate memory.
     * 
     * @param use_byte_size Tag for disambiguation
     * @param size_bytes Number of bytes to allocate
     * @param allocator Allocator to use (required)
     * @param resizable Whether this storage can be resized
     */
    StorageImpl(
        use_byte_size_t /*use_byte_size*/,
        size_t size_bytes,
        Allocator* allocator,
        bool resizable);

    ~StorageImpl() override;

    // Disable copy and move
    StorageImpl(const StorageImpl&) = delete;
    StorageImpl& operator=(const StorageImpl&) = delete;
    StorageImpl(StorageImpl&&) = delete;
    StorageImpl& operator=(StorageImpl&&) = delete;

    // --- Data Access ---

    /**
     * @brief Get immutable pointer to data.
     */
    const void* data() const {
        return data_ptr_.get();
    }

    /**
     * @brief Get mutable pointer to data.
     */
    void* mutable_data() {
        return data_ptr_.mutable_get();
    }

    /**
     * @brief Get immutable reference to DataPtr.
     */
    const DataPtr& data_ptr() const {
        return data_ptr_;
    }

    /**
     * @brief Get mutable reference to DataPtr.
     */
    DataPtr& mutable_data_ptr() {
        return data_ptr_;
    }

    /**
     * @brief Replace the data pointer and return the old one.
     * 
     * This is useful for sharing external memory or swapping buffers.
     */
    DataPtr set_data_ptr(DataPtr&& data_ptr);

    /**
     * @brief Set data pointer without swapping (just replaces).
     */
    void set_data_ptr_noswap(DataPtr&& data_ptr);

    // --- Size Management ---

    /**
     * @brief Get number of bytes in storage.
     */
    size_t nbytes() const {
        return size_bytes_;
    }

    /**
     * @brief Set number of bytes (doesn't reallocate).
     */
    void set_nbytes(size_t size_bytes) {
        size_bytes_ = size_bytes;
    }

    // --- Device & Allocator ---

    /**
     * @brief Get device type (CPU, CUDA, etc.).
     */
    DeviceType device_type() const {
        return data_ptr_.device().type;
    }

    /**
     * @brief Get device.
     */
    Device device() const {
        return data_ptr_.device();
    }

    /**
     * @brief Get allocator (may be nullptr for non-resizable storage).
     */
    Allocator* allocator() const {
        return allocator_;
    }

    /**
     * @brief Set allocator (be careful with this!).
     */
    void set_allocator(Allocator* allocator) {
        allocator_ = allocator;
    }

    // --- Resizability ---

    /**
     * @brief Check if storage is resizable.
     */
    bool resizable() const {
        return resizable_;
    }

    /**
     * @brief Set whether storage is resizable.
     * If setting to true, must have an allocator.
     */
    void set_resizable(bool resizable);

    // --- External Memory Sharing ---

    /**
     * @brief Share external pointer (can only call when use_count == 1).
     * 
     * This allows storage to take ownership of externally allocated memory
     * (e.g., from NumPy, PyTorch, or shared memory).
     * 
     * @param src Source pointer
     * @param size_bytes Size in bytes
     * @param d Custom deleter (nullptr for no cleanup)
     */
    void UniqueStorageShareExternalPointer(
        void* src,
        size_t size_bytes,
        DeleterFnPtr d = nullptr);

    /**
     * @brief Share external pointer via DataPtr (can only call when use_count == 1).
     */
    void UniqueStorageShareExternalPointer(
        DataPtr&& data_ptr,
        size_t size_bytes);

    /**
     * @brief Reset storage to empty state.
     */
    void reset();

private:
    DataPtr data_ptr_;        // Smart pointer managing memory
    size_t size_bytes_;       // Number of bytes allocated
    bool resizable_;          // Whether storage can be resized
    Allocator* allocator_;    // Allocator for resizing (can be nullptr)
};

} // namespace OwnTensor
