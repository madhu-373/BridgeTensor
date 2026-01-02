#pragma once

#include <cstdint>
#include <string>

namespace OwnTensor {

enum class DeviceType : int8_t {
    CPU = 0,
    CUDA = 1,
};

struct Device {
    DeviceType type;
    int8_t index;

    Device(DeviceType type = DeviceType::CPU, int8_t index = -1)
        : type(type), index(index) {}

    bool operator==(const Device& other) const {
        return type == other.type && index == other.index;
    }

    bool operator!=(const Device& other) const {
        return !(*this == other);
    }

    bool is_cpu() const { return type == DeviceType::CPU; }
    bool is_cuda() const { return type == DeviceType::CUDA; }

    std::string to_string() const {
        std::string s = (type == DeviceType::CPU) ? "cpu" : "cuda";
        if (index != -1) {
            s += ":" + std::to_string(index);
        }
        return s;
    }
};

} // namespace OwnTensor
