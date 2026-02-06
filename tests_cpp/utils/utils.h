#ifndef TESTS_UTILS_H
#define TESTS_UTILS_H
#include <iostream>
#include <filesystem>
#include <vector>
#include <NvInferRuntime.h>
#include "common/safetensorsUtils.h"
#include "utils/path_config.h"
#include "utils/cuda_check.h"

namespace trt_edgellm {
namespace rt {

inline std::filesystem::path getResourcesPath() {
    return trt_edgellm::tests::PathConfig::getInstance().getResourcesPath();
}

inline std::filesystem::path getTestDataPath() {
    return trt_edgellm::tests::PathConfig::getInstance().getTestDataPath();
}

inline std::filesystem::path getSafetensorPath(const std::string& filename) {
    return trt_edgellm::tests::PathConfig::getInstance().getSafetensorPath(filename);
}

inline void setCustomPath(const std::string& key, const std::filesystem::path& path) {
    trt_edgellm::tests::PathConfig::getInstance().setCustomPath(key, path);
}

constexpr char const* getDataTypeString(nvinfer1::DataType const dataType) {
    switch (dataType) {
        case nvinfer1::DataType::kINT64: return "INT64";
        case nvinfer1::DataType::kINT32: return "INT32";
        case nvinfer1::DataType::kFLOAT: return "FLOAT32";
        case nvinfer1::DataType::kHALF: return "FLOAT16";
        case nvinfer1::DataType::kBF16: return "BFLOAT16";
        case nvinfer1::DataType::kFP8: return "FLOAT8_E4M3";
        case nvinfer1::DataType::kINT8: return "INT8";
        case nvinfer1::DataType::kUINT8: return "UINT8";
        default: return "UNKNOWN";
    }

    return "UNKNOWN";
}

void printTensorInfo(const std::vector<Tensor>& tensors);

bool compareTensors(const Tensor& actual,
                    const Tensor& reference,
                    cudaStream_t stream,
                    float tolerance = 1e-4f);
bool compareHalfArrays(const half* actual,
                       const half* expected,
                       size_t numElements,
                       float tolerance = 1e-4f);
bool loadSafetensors(const std::filesystem::path& filepath,
                     std::vector<Tensor>& tensors,
                     cudaStream_t stream);
bool saveSafetensors(const std::filesystem::path& filepath,
                     const std::vector<Tensor>& tensors,
                     cudaStream_t stream);
bool loadMoeData(std::vector<Tensor>& moeInputTensors,
                 std::vector<Tensor>& moeWeightsTensors,
                 cudaStream_t stream);
bool loadMoeIntermediateRes(std::vector<Tensor>& intermediateResTensors,
                            cudaStream_t stream);
bool loadReferenceOutput(std::vector<trt_edgellm::rt::Tensor>& referenceTensors,
                         cudaStream_t stream);
bool saveOutputTensor(std::vector<Tensor>& outputTensors,
                      cudaStream_t stream);
}
}


namespace trt_edgellm {
namespace rt {

template<typename T>
void printDeviceData(const T* device_ptr, size_t count,
                     cudaStream_t stream, const std::string& label,
                     int print_count = 5, bool show_indices = false) {
    std::vector<T> host_data(count);
    cudaMemcpyAsync(host_data.data(), device_ptr,
                   count * sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    LOG_INFO("%s:", label.c_str());
    int actual_print_count = std::min(print_count, static_cast<int>(count));

    if constexpr (std::is_same<T, half>::value) {
        LOG_INFO("  Values (first %d):", actual_print_count);
        for (int i = 0; i < actual_print_count; ++i) {
            LOG_INFO("    [%d] = %f", i, __half2float(host_data[i]));
        }
    } else if constexpr (std::is_same<T, int32_t>::value) {
        if (show_indices) {
            LOG_INFO("  Indices (first %d):", actual_print_count);
            for (int i = 0; i < actual_print_count; ++i) {
                LOG_INFO("    Token[%d] -> Expert[%d]", i, host_data[i]);
            }
        } else {
            LOG_INFO("  Values (first %d):", actual_print_count);
            for (int i = 0; i < actual_print_count; ++i) {
                LOG_INFO("    [%d] = %d", i, host_data[i]);
            }
        }
    } else {
        LOG_INFO("  Values (first %d):", actual_print_count);
        for (int i = 0; i < actual_print_count; ++i) {
            LOG_INFO("    [%d] = %d", i, static_cast<int>(host_data[i]));
        }
    }
}

}
}

#endif // TESTS_UTILS_H