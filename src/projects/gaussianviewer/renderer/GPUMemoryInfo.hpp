#include <nvml.h>
#include <iostream>
#include <vector>

class GPUInfoManager {
public:
    GPUInfoManager() : device_(nullptr), isInitialized_(false) {
        nvmlReturn_t result = nvmlInit();
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
            throw std::runtime_error("NVML initialization failed");
        }
        isInitialized_ = true;

        result = nvmlDeviceGetHandleByIndex(0, &device_);
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to get handle for device 0: " << nvmlErrorString(result) << std::endl;
            throw std::runtime_error("Failed to get device handle");
        }
    }

    ~GPUInfoManager() {
        if (isInitialized_) {
            nvmlShutdown();
        }
    }

    std::pair<float, float> getMemoryUsage() {
        if (!device_) {
            throw std::runtime_error("Device handle is not initialized");
        }

        nvmlMemory_t memInfo;
        nvmlReturn_t result = nvmlDeviceGetMemoryInfo(device_, &memInfo);
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to get memory info: " << nvmlErrorString(result) << std::endl;
            throw std::runtime_error("Failed to get memory info");
        }

        float usage = 100.f * static_cast<float>(memInfo.used) / static_cast<float>(memInfo.total);
        if (usage_buffer.size() >= buffer_size) {
            usage_buffer.erase(usage_buffer.begin());
        }
        usage_buffer.push_back(usage);

        // return static_cast<float>(memInfo.used) / memInfo.total;
        return std::make_pair(static_cast<float>(memInfo.used), static_cast<float>(memInfo.total));
    }

    std::vector<float> getMemoryUsageBuffer() {
        return usage_buffer;
    }

private:
    nvmlDevice_t device_;
    bool isInitialized_;
    std::vector<float> usage_buffer;
    const size_t buffer_size = 100;
};
