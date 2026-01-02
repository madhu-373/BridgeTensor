#include "device/DeviceCore.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace OwnTensor {
namespace device {

bool cuda_available() {
#ifdef WITH_CUDA
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
#else
    return false;
#endif
}

int cuda_device_count() {
#ifdef WITH_CUDA
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
#else
    return 0;
#endif
}

int get_current_cuda_device() {
#ifdef WITH_CUDA
    int device = -1;
    cudaGetDevice(&device);
    return device;
#else
    return -1;
#endif
}

} // namespace device

namespace cuda {

#ifdef WITH_CUDA
static thread_local cudaStream_t current_stream = nullptr;

void setCurrentStream(cudaStream_t stream) {
    current_stream = stream;
}

cudaStream_t getCurrentStream() {
    return current_stream;
}
#else
void setCurrentStream(cudaStream_t stream) {}
cudaStream_t getCurrentStream() { return nullptr; }
#endif

} // namespace cuda
} // namespace OwnTensor
