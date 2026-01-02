#pragma once

#include <memory>

namespace OwnTensor {

class TensorBase; // Forward declaration

/**
 * @brief Interface for autograd metadata.
 */
struct AutogradMetaInterface {
    virtual ~AutogradMetaInterface() = default;
};

/**
 * @brief Basic implementation of autograd metadata.
 */
struct AutogradMeta : public AutogradMetaInterface {
    // Basic placeholders for now
    bool requires_grad = false;
    std::shared_ptr<TensorBase> grad; 
    
    AutogradMeta(bool req_grad = false) : requires_grad(req_grad) {}
};

} // namespace OwnTensor
