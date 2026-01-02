#include "core/TensorImpl.h"
#include <numeric>

namespace OwnTensor {

TensorImpl::TensorImpl(Storage storage, Dtype dtype, const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides)
    : storage_(std::move(storage)),
      dtype_(dtype),
      device_(storage_.device()),
      sizes_(sizes),
      storage_offset_(0) {
    
    if (strides.empty()) {
        strides_.resize(sizes_.size());
        int64_t s = 1;
        for (int i = (int)sizes_.size() - 1; i >= 0; --i) {
            strides_[i] = s;
            s *= sizes_[i];
        }
    } else {
        strides_ = strides;
    }

    numel_ = 1;
    for (auto s : sizes_) numel_ *= s;
}

} // namespace OwnTensor
