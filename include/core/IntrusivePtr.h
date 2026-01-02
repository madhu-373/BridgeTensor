#pragma once

#include <atomic>
#include <stdexcept>

namespace OwnTensor {

/**
 * @brief Base class for intrusive reference counting.
 * Objects wanting reference counting must inherit from this.
 */
class Retainable {
public:
    Retainable() : ref_count_(0) {}
    virtual ~Retainable() = default;

    void retain() const {
        ref_count_.fetch_add(1, std::memory_order_relaxed);
    }

    void release() const {
        if (ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete this;
        }
    }

    int32_t use_count() const {
        return ref_count_.load(std::memory_order_relaxed);
    }

private:
    mutable std::atomic<int32_t> ref_count_;
};

/**
 * @brief A smart pointer for objects inheriting from Retainable.
 */
template <typename T>
class IntrusivePtr {
public:
    IntrusivePtr() : ptr_(nullptr) {}
    IntrusivePtr(T* ptr) : ptr_(ptr) {
        if (ptr_) ptr_->retain();
    }
    IntrusivePtr(const IntrusivePtr& other) : ptr_(other.ptr_) {
        if (ptr_) ptr_->retain();
    }
    IntrusivePtr(IntrusivePtr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    ~IntrusivePtr() {
        if (ptr_) ptr_->release();
    }

    IntrusivePtr& operator=(const IntrusivePtr& other) {
        if (this != &other) {
            if (ptr_) ptr_->release();
            ptr_ = other.ptr_;
            if (ptr_) ptr_->retain();
        }
        return *this;
    }

    IntrusivePtr& operator=(IntrusivePtr&& other) noexcept {
        if (this != &other) {
            if (ptr_) ptr_->release();
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    T* get() const { return ptr_; }
    T* operator->() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }

    bool operator==(const IntrusivePtr& other) const { return ptr_ == other.ptr_; }
    bool operator!=(const IntrusivePtr& other) const { return ptr_ != other.ptr_; }

private:
    T* ptr_;
};

} // namespace OwnTensor
