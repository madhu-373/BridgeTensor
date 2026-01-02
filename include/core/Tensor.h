#pragma once

#include "TensorBase.h"
#include "Scalar.h"
#include <memory>

namespace OwnTensor {

// Forward declarations
class Node;
class FunctionPreHook;
class PostAccumulateGradHook;

/**
 * @brief The main user-facing Tensor class.
 * Inherits from TensorBase and provides the full API.
 */
class Tensor : public TensorBase {
public:
    using TensorBase::TensorBase;
    Tensor(const TensorBase& base) : TensorBase(base) {}

    // Factory methods
    static Tensor empty(const std::vector<int64_t>& sizes, Dtype dtype, Device device = DeviceType::CPU);
    static Tensor zeros(const std::vector<int64_t>& sizes, Dtype dtype, Device device = DeviceType::CPU);
    static Tensor ones(const std::vector<int64_t>& sizes, Dtype dtype, Device device = DeviceType::CPU);
    static Tensor full(const std::vector<int64_t>& sizes, Scalar value, Dtype dtype, Device device = DeviceType::CPU);
    static Tensor rand(const std::vector<int64_t>& sizes, Dtype dtype, Device device = DeviceType::CPU);
    static Tensor randn(const std::vector<int64_t>& sizes, Dtype dtype, Device device = DeviceType::CPU);

    // Structural operations
    Tensor view(const std::vector<int64_t>& new_sizes) const;
    Tensor reshape(const std::vector<int64_t>& new_sizes) const;
    Tensor transpose(int64_t dim0, int64_t dim1) const;
    Tensor t() const; // Transpose 2D
    Tensor flatten(int64_t start_dim = 0, int64_t end_dim = -1) const;
    Tensor unflatten(int64_t dim, const std::vector<int64_t>& sizes) const;
    Tensor contiguous() const;

    // Data operations
    Tensor clone() const;
    Tensor& copy_(const Tensor& src);
    Tensor& fill_(Scalar value);
    
    template <typename T>
    void fill(T value) { fill_(Scalar(value)); }

    // Conversions
    Tensor to(Device device) const;
    Tensor to(Dtype dtype) const;
    Tensor to_cpu() const { return to(DeviceType::CPU); }
    Tensor to_cuda() const { return to(DeviceType::CUDA); }
    bool to_bool() const;
    Tensor as_type(Dtype dtype) const { return to(dtype); }

    // Autograd
    void set_requires_grad(bool requires_grad);
    void set_grad(const Tensor& grad);
    
    // View tracking
    bool is_view() const;
    void set_is_view(bool is_view);
    
    // Gradient function
    std::shared_ptr<Node> grad_fn() const;
    void set_grad_fn(std::shared_ptr<Node> fn);
    
    // Output number (for multi-output operations)
    uint32_t output_nr() const;
    void set_output_nr(uint32_t nr);
    
    // Gradient retention (for non-leaves)
    bool retains_grad() const;
    void set_retains_grad(bool retains);
    
    // Hooks
    void register_hook(std::unique_ptr<FunctionPreHook> hook);
    void register_post_acc_hook(std::unique_ptr<PostAccumulateGradHook> hook);
    void clear_hooks();

    // Information
    size_t allocated_bytes() const;
    size_t grad_allocated_bytes() const;
    bool owns_data() const;
    bool owns_grad() const;

    size_t grad_nbytes() const;

    void display() const;

    void set_data(const Tensor& new_data);
    template <typename T>
    void set_data_ptr(void* ptr);
};

} // namespace OwnTensor
