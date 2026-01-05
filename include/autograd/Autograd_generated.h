#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {

// Forward declaration
class TensorBase;

// ============================================================================
// Generated Backward Functions
// ============================================================================

/**
 * Backward function for add
 * Signature: add(Tensor self, Tensor other) -> Tensor
 */
class AddBackward : public Node {
public:
    AddBackward() = default;


    std::vector<TensorBase> apply(std::vector<TensorBase>&& grads) override {
        auto grad = grads[0];
        return {
            grad,
            grad
        };
    }

};

/**
 * Backward function for matmul
 * Signature: matmul(Tensor self, Tensor other) -> Tensor
 */
class MatmulBackward : public Node {
public:
    MatmulBackward(TensorBase other, TensorBase self) : other_(other), self_(self) {}


    std::vector<TensorBase> apply(std::vector<TensorBase>&& grads) override {
        auto grad = grads[0];
        return {
            grad.matmul(other_.t()),
            self_.t().matmul(grad)
        };
    }

private:
    TensorBase other_;
    TensorBase self_;
};

/**
 * Backward function for relu
 * Signature: relu(Tensor self) -> Tensor
 */
class ReluBackward : public Node {
public:
    ReluBackward(TensorBase self) : self_(self) {}


    std::vector<TensorBase> apply(std::vector<TensorBase>&& grads) override {
        auto grad = grads[0];
        return {
            grad * (self_ > 0)
        };
    }

private:
    TensorBase self_;
};

} // namespace OwnTensor