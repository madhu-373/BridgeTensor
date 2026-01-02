#pragma once
#include "device/Device.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#else
typedef void* cudaStream_t;
#endif

namespace OwnTensor {
namespace device {

bool cuda_available();
int cuda_device_count();
int get_current_cuda_device();

} // namespace device

namespace cuda {

void setCurrentStream(cudaStream_t stream);
cudaStream_t getCurrentStream();

} // namespace cuda
} // namespace OwnTensor
