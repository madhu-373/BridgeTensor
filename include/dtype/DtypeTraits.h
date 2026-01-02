#pragma once

#include "Dtype.h"
#include "Types.h"
#include <cstddef>

namespace OwnTensor {

template <Dtype T>
struct DtypeTraits;

#define DEFINE_DTYPE_TRAITS(dtype, type_name) \
template <> \
struct DtypeTraits<Dtype::dtype> { \
    using type = type_name; \
    static constexpr size_t size = sizeof(type_name); \
    static constexpr const char* name = #dtype; \
};

DEFINE_DTYPE_TRAITS(Int8, int8_t)
DEFINE_DTYPE_TRAITS(Int16, int16_t)
DEFINE_DTYPE_TRAITS(Int32, int32_t)
DEFINE_DTYPE_TRAITS(Int64, int64_t)
DEFINE_DTYPE_TRAITS(UInt8, uint8_t)
DEFINE_DTYPE_TRAITS(UInt16, uint16_t)
DEFINE_DTYPE_TRAITS(UInt32, uint32_t)
DEFINE_DTYPE_TRAITS(UInt64, uint64_t)
DEFINE_DTYPE_TRAITS(Float16, float16_t)
DEFINE_DTYPE_TRAITS(Bfloat16, bfloat16_t)
DEFINE_DTYPE_TRAITS(Float32, float)
DEFINE_DTYPE_TRAITS(Float64, double)
DEFINE_DTYPE_TRAITS(Bool, bool)
DEFINE_DTYPE_TRAITS(Complex32, complex32_t)
DEFINE_DTYPE_TRAITS(Complex64, complex64_t)
DEFINE_DTYPE_TRAITS(Complex128, complex128_t)

inline size_t elementSize(Dtype dtype) {
    switch (dtype) {
        case Dtype::Int8: return sizeof(int8_t);
        case Dtype::Int16: return sizeof(int16_t);
        case Dtype::Int32: return sizeof(int32_t);
        case Dtype::Int64: return sizeof(int64_t);
        case Dtype::UInt8: return sizeof(uint8_t);
        case Dtype::UInt16: return sizeof(uint16_t);
        case Dtype::UInt32: return sizeof(uint32_t);
        case Dtype::UInt64: return sizeof(uint64_t);
        case Dtype::Float16: return sizeof(float16_t);
        case Dtype::Bfloat16: return sizeof(bfloat16_t);
        case Dtype::Float32: return sizeof(float);
        case Dtype::Float64: return sizeof(double);
        case Dtype::Bool: return sizeof(bool);
        case Dtype::Complex32: return sizeof(complex32_t);
        case Dtype::Complex64: return sizeof(complex64_t);
        case Dtype::Complex128: return sizeof(complex128_t);
        default: return 0;
    }
}

} // namespace OwnTensor
