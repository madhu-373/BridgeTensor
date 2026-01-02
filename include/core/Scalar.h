#pragma once

#include "dtype/Types.h"
#include "dtype/Dtype.h"
#include <variant>

namespace OwnTensor {

/**
 * @brief A variant-like class to represent a single scalar value.
 * Used for filling tensors or as operation arguments.
 */
class Scalar {
public:
    enum class Tag { Int, Float, Complex, Bool };

    Scalar(int v) : val_(static_cast<int64_t>(v)), tag_(Tag::Int) {}
    Scalar(int64_t v) : val_(v), tag_(Tag::Int) {}
    Scalar(float v) : val_(static_cast<double>(v)), tag_(Tag::Float) {}
    Scalar(double v) : val_(v), tag_(Tag::Float) {}
    Scalar(complex128_t v) : val_(v), tag_(Tag::Complex) {}
    Scalar(bool v) : val_(v), tag_(Tag::Bool) {}

    template <typename T>
    T to() const {
        if (std::holds_alternative<int64_t>(val_)) return static_cast<T>(std::get<int64_t>(val_));
        if (std::holds_alternative<double>(val_)) return static_cast<T>(std::get<double>(val_));
        if (std::holds_alternative<bool>(val_)) return static_cast<T>(std::get<bool>(val_));
        // Complex handling needs care
        return T{}; 
    }

    bool is_floating_point() const { return tag_ == Tag::Float; }
    bool is_integral() const { return tag_ == Tag::Int; }
    bool is_complex() const { return tag_ == Tag::Complex; }
    bool is_bool() const { return tag_ == Tag::Bool; }

private:
    std::variant<int64_t, double, complex128_t, bool> val_;
    Tag tag_;
};

} // namespace OwnTensor
