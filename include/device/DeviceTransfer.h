#pragma once
#include "device/Device.h"
#include <cstddef>

namespace OwnTensor {
namespace device {

void copy_memory(void* dst, Device dst_device, const void* src, Device src_device, size_t bytes);

} // namespace device
} // namespace OwnTensor
