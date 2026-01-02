#include "core/Tensor.h"
#include "dtype/DtypeTraits.h"
#include "device/DeviceTransfer.h"
#include "device/AllocatorRegistry.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <cstring>

namespace OwnTensor {

// --- Tensor Implementation ---

Tensor Tensor::empty(const std::vector<int64_t>& sizes, Dtype dtype, Device device) {
    int64_t numel = 1;
    for (auto s : sizes) numel *= s;
    Storage storage(numel * elementSize(dtype), device);
    return Tensor(IntrusivePtr<TensorImpl>(new TensorImpl(storage, dtype, sizes)));
}

Tensor Tensor::zeros(const std::vector<int64_t>& sizes, Dtype dtype, Device device) {
    auto t = empty(sizes, dtype, device);
    t.fill_(0);
    return t;
}

Tensor Tensor::ones(const std::vector<int64_t>& sizes, Dtype dtype, Device device) {
    auto t = empty(sizes, dtype, device);
    t.fill_(1);
    return t;
}

Tensor Tensor::full(const std::vector<int64_t>& sizes, Scalar value, Dtype dtype, Device device) {
    auto t = empty(sizes, dtype, device);
    t.fill_(value);
    return t;
}

Tensor Tensor::rand(const std::vector<int64_t>& sizes, Dtype dtype, Device device) {
    auto t = empty(sizes, dtype, device);
    // Simple CPU filling for now
    if (device.is_cpu()) {
        float* data = static_cast<float*>(t.data_ptr());
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0, 1.0);
        for (int64_t i = 0; i < t.numel(); ++i) data[i] = dis(gen);
    } else {
        // GPU random would need a kernel. Fallback:
        auto cpu_t = Tensor::rand(sizes, dtype, DeviceType::CPU);
        t.copy_(cpu_t);
    }
    return t;
}

Tensor Tensor::randn(const std::vector<int64_t>& sizes, Dtype dtype, Device device) {
    auto t = empty(sizes, dtype, device);
    if (device.is_cpu()) {
        float* data = static_cast<float*>(t.data_ptr());
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dis(0.0, 1.0);
        for (int64_t i = 0; i < t.numel(); ++i) data[i] = dis(gen);
    } else {
        auto cpu_t = Tensor::randn(sizes, dtype, DeviceType::CPU);
        t.copy_(cpu_t);
    }
    return t;
}

Tensor Tensor::view(const std::vector<int64_t>& new_sizes) const {
    auto new_impl = new TensorImpl(impl_->storage(), impl_->dtype(), new_sizes);
    new_impl->set_storage_offset(impl_->storage_offset());
    return Tensor(IntrusivePtr<TensorImpl>(new_impl));
}

Tensor Tensor::reshape(const std::vector<int64_t>& new_sizes) const {
    if (is_contiguous()) return view(new_sizes);
    return contiguous().view(new_sizes);
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    auto new_sizes = sizes();
    auto new_strides = strides();
    std::swap(new_sizes[dim0], new_sizes[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);
    auto new_impl = new TensorImpl(impl_->storage(), impl_->dtype(), new_sizes, new_strides);
    new_impl->set_storage_offset(impl_->storage_offset());
    return Tensor(IntrusivePtr<TensorImpl>(new_impl));
}

Tensor Tensor::t() const {
    if (ndim() != 2) throw std::runtime_error("t() expects a 2D tensor");
    return transpose(0, 1);
}

Tensor Tensor::unflatten(int64_t dim, const std::vector<int64_t>& sizes) const {
    std::vector<int64_t> new_sizes;
    for (int64_t i = 0; i < dim; ++i) new_sizes.push_back(this->sizes()[i]);
    for (auto s : sizes) new_sizes.push_back(s);
    for (int64_t i = dim + 1; i < (int64_t)ndim(); ++i) new_sizes.push_back(this->sizes()[i]);
    return reshape(new_sizes);
}

Tensor Tensor::to(Device device) const {
    if (this->device() == device) return *this;
    auto t = empty(sizes(), dtype(), device);
    device::copy_memory(t.data_ptr(), device, this->data_ptr(), this->device(), nbytes());
    return t;
}

Tensor Tensor::to(Dtype dtype) const {
    if (this->dtype() == dtype) return *this;
    auto t = empty(sizes(), dtype, device());
    t.copy_(*this); 
    return t;
}

bool Tensor::to_bool() const {
    if (numel() != 1) throw std::runtime_error("to_bool() expects a scalar tensor");
    if (device().is_cpu()) {
        return Scalar(*static_cast<float*>(data_ptr())).to<bool>();
    } else {
        return this->to(DeviceType::CPU).to_bool();
    }
}

void Tensor::set_requires_grad(bool requires_grad) {
    if (!impl_->autograd_meta()) {
        if (!requires_grad) return;
        impl_->set_autograd_meta(std::unique_ptr<AutogradMetaInterface>(new AutogradMeta(requires_grad)));
    } else {
        static_cast<AutogradMeta*>(impl_->autograd_meta())->requires_grad = requires_grad;
    }
}

void Tensor::set_grad(const Tensor& grad) {
    if (!impl_->autograd_meta()) {
        impl_->set_autograd_meta(std::unique_ptr<AutogradMetaInterface>(new AutogradMeta(true)));
    }
    auto meta = static_cast<AutogradMeta*>(impl_->autograd_meta());
    meta->grad = std::make_shared<TensorBase>(grad);
}

size_t Tensor::grad_nbytes() const { 
    auto g = grad();
    return g.defined() ? g.nbytes() : 0;
}

size_t Tensor::grad_allocated_bytes() const { 
    auto g = grad();
    return g.defined() ? static_cast<const Tensor&>(g).allocated_bytes() : 0; 
}

bool Tensor::owns_grad() const { 
    auto g = grad();
    return g.defined() ? static_cast<const Tensor&>(g).owns_data() : false;
}

void Tensor::set_data(const Tensor& new_data) {
    if (new_data.sizes() != sizes()) throw std::runtime_error("set_data dimension mismatch");
    impl_->storage() = new_data.storage();
}

Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim) const {
    int64_t flat_size = numel();
    return reshape({flat_size});
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;
    auto t = empty(sizes(), dtype(), device());
    t.copy_(*this);
    return t;
}

Tensor Tensor::clone() const {
    auto t = empty(sizes(), dtype(), device());
    t.copy_(*this);
    return t;
}

Tensor& Tensor::copy_(const Tensor& src) {
    if (numel() == 0) return *this;
    device::copy_memory(data_ptr(), device(), src.data_ptr(), src.device(), nbytes());
    return *this;
}

Tensor& Tensor::fill_(Scalar value) {
    if (numel() == 0) return *this;

    if (device().is_cpu()) {
        float v = value.to<float>();
        float* data = static_cast<float*>(data_ptr());
        std::fill(data, data + numel(), v);
    } else {
        // GPU support
        // If value is 0, we can use memsetAsync
        bool is_zero = false;
        try { is_zero = (value.to<double>() == 0.0); } catch (...) {}

        if (is_zero) {
            auto alloc = AllocatorRegistry::get_allocator(device());
            alloc->memset(data_ptr(), 0, nbytes());
        } else {
            // For non-zero fill on GPU, we currently need a kernel or a CPU-GPU transfer fallback
            // For now, let's do the transfer fallback to ensure "working condition"
            auto cpu_t = this->to(DeviceType::CPU);
            cpu_t.fill_(value);
            this->copy_(cpu_t);
        }
    }
    return *this;
}

void Tensor::display() const {
    std::cout << "Tensor(" << (device().is_cpu() ? "cpu" : "cuda") << ", " << numel() << " elements, dtype=" << (int)dtype() << ")" << std::endl;
    std::cout << "Shape: [";
    for (size_t i = 0; i < sizes().size(); ++i) std::cout << sizes()[i] << (i == sizes().size() - 1 ? "" : ", ");
    std::cout << "]" << std::endl;
    
    if (numel() > 0) {
        Tensor cpu_t = (device().is_cpu()) ? *this : this->to(DeviceType::CPU);
        float* data = static_cast<float*>(cpu_t.data_ptr());
        std::cout << "Data: [";
        for (int64_t i = 0; i < std::min(numel(), (int64_t)10); ++i) std::cout << data[i] << (i == std::min(numel(), (int64_t)10) - 1 ? "" : ", ");
        if (numel() > 10) std::cout << "...";
        std::cout << "]" << std::endl;
    }
}

size_t Tensor::allocated_bytes() const {
    return impl_->storage().size_bytes();
}

bool Tensor::owns_data() const {
    return impl_->storage().impl.get()->use_count() == 1;
}

} // namespace OwnTensor
