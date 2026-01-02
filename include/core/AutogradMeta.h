#pragma once

#include <memory>
#include <vector>
#include <mutex>
#include <optional>
#include "autograd/Node.h"
#include "autograd/Hooks.h"

namespace OwnTensor {

// Forward declarations
class TensorBase;
class TensorImpl;
enum class Dtype;  // Forward declare Dtype enum

/**
 * @brief Interface for autograd metadata.
 * 
 * This interface allows the core tensor implementation to remain independent
 * of the autograd system. Different autograd implementations can be plugged in.
 */
struct AutogradMetaInterface {
    virtual ~AutogradMetaInterface() = default;

    /// Set whether this tensor requires gradient
    virtual void set_requires_grad(bool requires_grad, TensorImpl* self_impl) = 0;
    
    /// Check if this tensor requires gradient
    virtual bool requires_grad() const = 0;
    
    /// Get mutable reference to gradient tensor
    virtual TensorBase& mutable_grad() = 0;
    
    /// Get const reference to gradient tensor
    virtual const TensorBase& grad() const = 0;
};

/**
 * @brief Full autograd metadata implementation (PyTorch-style).
 * 
 * AutogradMeta stores all information needed for automatic differentiation:
 * - Gradient storage (for leaf tensors)
 * - Computational graph connections (grad_fn, edges)
 * - Hooks for custom backward logic
 * - View tracking information
 * - Thread-safety primitives
 * 
 * Design principles:
 * - Leaf tensors: requires_grad_ = true, grad_fn_ = nullptr
 * - Non-leaf tensors: requires_grad_ = false, grad_fn_ != nullptr
 * - weak_ptr for grad_accumulator to prevent reference cycles
 * - Lazy initialization of expensive fields (protected by mutex)
 */
struct AutogradMeta : public AutogradMetaInterface {
    // =================================================================
    // GRADIENT STORAGE
    // =================================================================

    /// Accumulated gradient (for leaf tensors mainly)
    std::shared_ptr<TensorBase> grad_;

    /// Function that created this tensor (nullptr for leaves)
    std::shared_ptr<Node> grad_fn_;

    /// Gradient accumulator for leaf tensors (weak to prevent cycles!)
    std::weak_ptr<Node> grad_accumulator_;

    // =================================================================
    // HOOKS
    // =================================================================

    /// Pre-backward hooks (run before gradient computation)
    std::vector<std::unique_ptr<FunctionPreHook>> hooks_;

    /// Post-accumulation hook (run after .grad update on leaves)
    std::unique_ptr<PostAccumulateGradHook> post_acc_grad_hook_;

    // =================================================================
    // FLAGS & METADATA
    // =================================================================

    /// Only meaningful on leaf variables (must be false otherwise)
    bool requires_grad_{false};

    /// Whether non-leaf should retain gradient (normally cleared)
    bool retains_grad_{false};

    /// Is this tensor a view of another tensor?
    bool is_view_{false};

    /// Which output of grad_fn_ is this (for multi-output operations)
    uint32_t output_nr_{0};

    /// The dtype of the grad field; when nullopt, defaults to tensor's dtype
    std::optional<Dtype> grad_dtype_;

    /// When true, allows gradient dtype to be different from tensor dtype
    bool allow_grad_dtype_mismatch_{false};

    // =================================================================
    // THREAD SAFETY
    // =================================================================

    /// Mutex for thread-safe access to lazy fields
    /// Mutable because we need to lock in const methods
    mutable std::mutex mutex_;

    // =================================================================
    // CONSTRUCTORS
    // =================================================================

    /**
     * @brief Default constructor.
     */
    AutogradMeta() = default;

    /**
     * @brief Construct with requires_grad flag.
     */
    explicit AutogradMeta(bool requires_grad)
        : requires_grad_(requires_grad) {}

    /**
     * @brief Construct with gradient function (for non-leaf tensors).
     * 
     * @param self_impl Pointer to TensorImpl (for validation)
     * @param requires_grad Whether to require gradient
     * @param gradient_edge Edge to the creating function
     */
    AutogradMeta(
        TensorImpl* self_impl,
        bool requires_grad,
        Edge gradient_edge);

    /**
     * @brief Destructor.
     */
    ~AutogradMeta() override = default;

    // =================================================================
    // INTERFACE IMPLEMENTATION
    // =================================================================

    /**
     * @brief Set requires_grad property.
     * 
     * Only leaves can have requires_grad = true.
     * Non-leaves implicitly require grad if they have a grad_fn.
     */
    void set_requires_grad(bool requires_grad, TensorImpl* self_impl) override;

    /**
     * @brief Check if this tensor requires gradient.
     * 
     * Returns true if:
     * - This is a leaf with requires_grad_ = true, OR
     * - This is a non-leaf with grad_fn_ != nullptr
     */
    bool requires_grad() const override {
        return requires_grad_ || grad_fn_ != nullptr;
    }

    /**
     * @brief Get mutable gradient.
     */
    TensorBase& mutable_grad() override;

    /**
     * @brief Get const gradient.
     */
    const TensorBase& grad() const override;

    // =================================================================
    // ADDITIONAL METHODS
    // =================================================================

    /**
     * @brief Get the gradient function.
     */
    const std::shared_ptr<Node>& grad_fn() const {
        return grad_fn_;
    }

    /**
     * @brief Set the gradient function.
     */
    void set_grad_fn(std::shared_ptr<Node> fn) {
        grad_fn_ = std::move(fn);
    }

    /**
     * @brief Check if this is a leaf tensor.
     * 
     * A tensor is a leaf if it has no grad_fn (wasn't created by an operation).
     */
    bool is_leaf() const {
        return grad_fn_ == nullptr;
    }

    /**
     * @brief Get output number.
     */
    uint32_t output_nr() const {
        return output_nr_;
    }

    /**
     * @brief Set output number.
     */
    void set_output_nr(uint32_t nr) {
        output_nr_ = nr;
   }

    /**
     * @brief Check if this is a view.
     */
    bool is_view() const {
        return is_view_;
    }

    /**
     * @brief Set view flag.
     */
    void set_is_view(bool is_view) {
        is_view_ = is_view;
    }

    /**
     * @brief Check if gradient should be retained (for non-leaves).
     */
    bool retains_grad() const {
        return retains_grad_;
    }

    /**
     * @brief Set whether to retain gradient.
     */
    void set_retains_grad(bool retains) {
        retains_grad_ = retains;
    }

    /**
     * @brief Add a pre-backward hook.
     */
    void add_hook(std::unique_ptr<FunctionPreHook> hook) {
        std::lock_guard<std::mutex> lock(mutex_);
        hooks_.push_back(std::move(hook));
    }

    /**
     * @brief Set post-accumulation hook.
     */
    void set_post_acc_hook(std::unique_ptr<PostAccumulateGradHook> hook) {
        std::lock_guard<std::mutex> lock(mutex_);
        post_acc_grad_hook_ = std::move(hook);
    }

    /**
     * @brief Get gradient dtype (nullopt means use tensor dtype).
     */
    std::optional<Dtype> grad_dtype() const {
        return grad_dtype_;
    }

    /**
     * @brief Set gradient dtype.
     */
    void set_grad_dtype(std::optional<Dtype> dtype) {
        grad_dtype_ = dtype;
    }

    /**
     * @brief Check if gradient dtype mismatch is allowed.
     */
    bool allows_grad_dtype_mismatch() const {
        return allow_grad_dtype_mismatch_;
    }

    /**
     * @brief Set whether to allow gradient dtype mismatch.
     */
    void set_allow_grad_dtype_mismatch(bool allow) {
        allow_grad_dtype_mismatch_ = allow;
    }

    /**
     * @brief Clear all hooks.
     */
    void clear_hooks() {
        std::lock_guard<std::mutex> lock(mutex_);
        hooks_.clear();
        post_acc_grad_hook_.reset();
    }
};

} // namespace OwnTensor
