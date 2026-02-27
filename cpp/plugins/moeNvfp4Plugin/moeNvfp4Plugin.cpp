#include "moeNvfp4Plugin.h"
#include "kernels/moeNvfp4Kernels/moeNvfp4Kernels.h"
#include "plugins/utils/pluginUtils.h"

#include <cassert>
#include <cuda_fp16.h>
#include <mutex>
#include <optional>
#include <iostream>
#include <numeric>

using namespace trt_edgellm::kernel::moe_nvfp4;

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

using namespace nvinfer1;

namespace trt_edgellm {
namespace plugins {

namespace {
constexpr char const* kMOE_NVFP4_PLUGIN_VERSION{"1"};
constexpr char const* kMOE_NVFP4_PLUGIN_NAME{"MoENvFp4Plugin"};
} // namespace

// Static class fields initialization
PluginFieldCollection MoeNvfp4PluginCreator::mFieldCollection{};
std::vector<PluginField> MoeNvfp4PluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(MoeNvfp4PluginCreator);

MoeNvfp4Plugin::MoeNvfp4Plugin(std::string const& name, 
                               int32_t expertsNum, 
                               int32_t expertsTopK,
                               std::vector<float> gateAlpha,
                               std::vector<float> upAlpha,
                               std::vector<float> downAlpha)
    : mLayerName(name)
    , mExpertsNum(expertsNum)
    , mExpertsTopK(expertsTopK)
    , mGateAlpha(gateAlpha)
    , mUpAlpha(upAlpha)
    , mDownAlpha(downAlpha) {
    
    mHiddenSize = 0;
    mIntermediateSize = 0;

    if (expertsTopK != 1) {
        printf("Warning: experts_topk > 1 not yet implemented, will use top-1\n");
    }
    
    // Validate alpha vectors size
    if (gateAlpha.size() != static_cast<size_t>(mExpertsNum) ||
        upAlpha.size() != static_cast<size_t>(mExpertsNum) ||
        downAlpha.size() != static_cast<size_t>(mExpertsNum)) {
        printf("Error: Alpha vectors must have size equal to experts_num\n");
        throw std::runtime_error("Invalid alpha vectors size");
    }

#if 0
    std::cout << "Gate alpha: ";
    for (auto alpha : mGateAlpha) {
        std::cout << alpha << " ";
    }
    std::cout << std::endl;
    std::cout << "Up alpha: ";
    for (auto alpha : mUpAlpha) {
        std::cout << alpha << " ";
    }
    std::cout << std::endl;
    std::cout << "Down alpha: ";
    for (auto alpha : mDownAlpha) {
        std::cout << alpha << " ";
    }
    std::cout << std::endl;
#endif
}

MoeNvfp4Plugin::MoeNvfp4Plugin(std::string const& name, void const* data, size_t length)
    : mLayerName(name) {
#if 0
    std::cout << "Deserialized data.........." << std::endl;
    std::cout << "len: " << length << std::endl;
    for (size_t i = 0; i < length; ++i) {
        std::cout << static_cast<const float*>(data)[i] << " ";
    }
    std::cout << std::endl;
        for (size_t i = 0; i < length; ++i) {
        std::cout << static_cast<const int32_t*>(data)[i] << " ";
    }
    std::cout << std::endl;
#endif

    deserializeValue(&data, &length, &mHiddenSize);
    deserializeValue(&data, &length, &mIntermediateSize);
    deserializeValue(&data, &length, &mExpertsNum);
    deserializeValue(&data, &length, &mExpertsTopK);

    int32_t gateAlphaSize, upAlphaSize, downAlphaSize;
    deserializeValue(&data, &length, &gateAlphaSize);
    deserializeValue(&data, &length, &upAlphaSize);
    deserializeValue(&data, &length, &downAlphaSize);

    mGateAlpha.resize(gateAlphaSize);
    mUpAlpha.resize(upAlphaSize);
    mDownAlpha.resize(downAlphaSize);

    deserializeArray(&data, &length, mGateAlpha.data(), gateAlphaSize);
    deserializeArray(&data, &length, mUpAlpha.data(), upAlphaSize);
    deserializeArray(&data, &length, mDownAlpha.data(), downAlphaSize);

#if 0
    std::cout << "Gate alpha: ";
    for (auto alpha : mGateAlpha) {
        std::cout << alpha << " ";
    }
    std::cout << std::endl;
    std::cout << "Up alpha: ";
    for (auto alpha : mUpAlpha) {
        std::cout << alpha << " ";
    }
    std::cout << std::endl;
    std::cout << "Down alpha: ";
    for (auto alpha : mDownAlpha) {
        std::cout << alpha << " ";
    }
    std::cout << std::endl;
#endif

    if (mExpertsTopK != 1) {
        printf("Warning: experts_topk > 1 not yet implemented, will use top-1\n");
    }
}

MoeNvfp4Plugin::~MoeNvfp4Plugin() {
    freeWorkspace();
}

int MoeNvfp4Plugin::getScalePadSize(int M, int N) const {
    int scale_n = N / 16;
    int rounded_m = ((M + 128 - 1) / 128) * 128;
    int rounded_n = ((scale_n + 4 - 1) / 4) * 4;
    return rounded_m * rounded_n;
}

void MoeNvfp4Plugin::allocateWorkspace() {
    if (mIsWorkspaceAllocated) {
        return;
    }
    
    // Allocate padded router weights and bias
    auto padSize = (mExpertsNum + 7) / 8 * 8;
    
    mWorkspace["router_weight"] = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mWorkspace["router_weight"]), 
                         mHiddenSize * padSize * sizeof(half)));
    
    mWorkspace["router_bias"] = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mWorkspace["router_bias"]), 
                         padSize * sizeof(half)));
    
    mIsWorkspaceAllocated = true;
}

void MoeNvfp4Plugin::freeWorkspace() {
    for (auto& [key, ptr] : mWorkspace) {
        cudaFree(ptr);
        ptr = nullptr;
    }
    mIsWorkspaceAllocated = false;
}

IPluginV2DynamicExt* MoeNvfp4Plugin::clone() const noexcept {
    try {
        MoeNvfp4Plugin* plugin = new MoeNvfp4Plugin(mLayerName, 
                                                   mExpertsNum, 
                                                   mExpertsTopK,
                                                   mGateAlpha,
                                                   mUpAlpha,
                                                   mDownAlpha);
        plugin->mHiddenSize = mHiddenSize;
        plugin->mIntermediateSize = mIntermediateSize;
        plugin->mIsWorkspaceAllocated = false;
        plugin->mIsDataInitialized = false;
        return plugin;
    } catch (std::exception const& e) {
        printf("Error cloning MoE plugin: %s\n", e.what());
        return nullptr;
    }
}

char const* MoeNvfp4Plugin::getPluginType() const noexcept {
    return kMOE_NVFP4_PLUGIN_NAME;
}

char const* MoeNvfp4Plugin::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

void MoeNvfp4Plugin::setPluginNamespace(char const* pluginNamespace) noexcept {
    mNamespace = std::string(pluginNamespace);
}

char const* MoeNvfp4Plugin::getPluginVersion() const noexcept {
    return kMOE_NVFP4_PLUGIN_VERSION;
}

int32_t MoeNvfp4Plugin::getNbOutputs() const noexcept {
    return 1;
}

bool MoeNvfp4Plugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    try {
        // 15 inputs + 1 output
        assert(nbInputs == 15 && nbOutputs == 1);
        assert(pos < (nbInputs + nbOutputs));
        
        auto const& tensorDesc = inOut[pos];
        bool status{true};

        switch (pos) {
            case 0: { // hidden_state [batch_size, seq_len, hidden_size]
                status &= tensorDesc.type == DataType::kHALF;
                status &= tensorDesc.format == TensorFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 3;
                break;
            }
            case 1: { // router_weight [hidden_size, experts_num]
                status &= tensorDesc.type == DataType::kHALF;
                status &= tensorDesc.format == TensorFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 2;
                break;
            }
            case 2: { // router_bias [experts_num]
                status &= tensorDesc.type == DataType::kHALF;
                status &= tensorDesc.format == TensorFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 1;
                break;
            }
            // Gate projection quantized weights
            case 3: { // gate_qweight [experts_num, intermediate_size, hidden_size // 2]
                status &= tensorDesc.type == DataType::kINT8;
                status &= tensorDesc.format == TensorFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 3;
                break;
            }
            case 4: { // gate_qscales [experts_num, padd, padd]
                status &= tensorDesc.type == DataType::kINT8;
                status &= tensorDesc.format == TensorFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 3;
                break;
            }
            case 5: { // gate_input_global_scale [experts_num]
                status &= tensorDesc.type == DataType::kFLOAT;
                status &= tensorDesc.format == TensorFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 1;
                break;
            }
            case 6: { // gate_weight_global_scale [experts_num]
                status &= tensorDesc.type == DataType::kFLOAT;
                status &= tensorDesc.format == TensorFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 1;
                break;
            }
            // Up projection quantized weights
            case 7: { // up_qweight [experts_num, intermediate_size, hidden_size // 2]
                status &= tensorDesc.type == DataType::kINT8;
                status &= tensorDesc.format == TensorFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 3;
                break;
            }
            case 8: { // up_qscales [experts_num, padd, padd]
                status &= tensorDesc.type == DataType::kINT8;
                status &= tensorDesc.format == TensorFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 3;
                break;
            }
            case 9: { // up_input_global_scale [experts_num]
                status &= tensorDesc.type == DataType::kFLOAT;
                status &= tensorDesc.format == TensorFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 1;
                break;
            }
            case 10: { // up_weight_global_scale [experts_num]
                status &= tensorDesc.type == DataType::kFLOAT;
                status &= tensorDesc.format == TensorFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 1;
                break;
            }
            // Down projection quantized weights
            case 11: { // down_qweight [experts_num, hidden_size, intermediate_size // 2]
                status &= tensorDesc.type == DataType::kINT8;
                status &= tensorDesc.format == TensorFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 3;
                break;
            }
            case 12: { // down_qscales [experts_num, padd, padd]
                status &= tensorDesc.type == DataType::kINT8;
                status &= tensorDesc.format == TensorFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 3;
                break;
            }
            case 13: { // down_input_global_scale [experts_num]
                status &= tensorDesc.type == DataType::kFLOAT;
                status &= tensorDesc.format == TensorFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 1;
                break;
            }
            case 14: { // down_weight_global_scale [experts_num]
                status &= tensorDesc.type == DataType::kFLOAT;
                status &= tensorDesc.format == TensorFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 1;
                break;
            }
            // Output
            case 15: { // output [batch_size, seq_len, hidden_size]
                status &= tensorDesc.type == DataType::kHALF;
                status &= tensorDesc.format == TensorFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 3;
                break;
            }
            default: 
                status = false;
                break;
        }
        // printf("[MoeNvfp4Plugin] %s: %s, %d\n", __FUNCTION__, status ? "SUCCESS" : "FAIL", pos);
        return status;
    } catch (std::exception const& e) {
        return false;
    }
}

DataType MoeNvfp4Plugin::getOutputDataType([[maybe_unused]] int32_t index,
    [[maybe_unused]] nvinfer1::DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept {
    return DataType::kHALF;
}

DimsExprs MoeNvfp4Plugin::getOutputDimensions([[maybe_unused]] int32_t outputIndex,
    nvinfer1::DimsExprs const* inputs, [[maybe_unused]] int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    DimsExprs output;
    output.nbDims = 3;
    output.d[0] = inputs[0].d[0];
    output.d[1] = inputs[0].d[1];
    output.d[2] = inputs[0].d[2];
    return output;
}

void MoeNvfp4Plugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] nvinfer1::DynamicPluginTensorDesc const* out,
    [[maybe_unused]] int32_t nbOutputs) noexcept {
    try {
        if (in[0].desc.dims.nbDims == 3) {
            mHiddenSize = in[0].desc.dims.d[2];
        } else {
            throw std::runtime_error("Invalid hidden_state dimensions");
        }

        // Verify router weight dimensions
        if (in[1].desc.dims.nbDims == 2) {
            int inferred_hidden_size = in[1].desc.dims.d[0];
            int inferred_experts_num = in[1].desc.dims.d[1];
            
            if (mHiddenSize == 0) {
                mHiddenSize = inferred_hidden_size;
            } else if (mHiddenSize != inferred_hidden_size) {
                throw std::runtime_error("Hidden size mismatch between hidden_state and router_weight");
            }

            if (inferred_experts_num != mExpertsNum) {
                throw std::runtime_error("Experts number mismatch in router_weight");
            }
        }

        // Verify quantized weight dimensions
        // Gate projection
        if (in[3].desc.dims.nbDims == 3) {
            int inferred_experts_num = in[3].desc.dims.d[0];
            int inferred_intermediate_size = in[3].desc.dims.d[1];
            int inferred_hidden_half = in[3].desc.dims.d[2];
            
            if (inferred_experts_num != mExpertsNum) {
                throw std::runtime_error("Experts number mismatch in gate_qweight");
            }
            
            if (mIntermediateSize == 0) {
                mIntermediateSize = inferred_intermediate_size;
            } else if (mIntermediateSize != inferred_intermediate_size) {
                throw std::runtime_error("Intermediate size mismatch in gate_qweight");
            }
            
            if (inferred_hidden_half != mHiddenSize / 2) {
                throw std::runtime_error("Hidden size mismatch in gate_qweight");
            }
        }

        // Up projection
        if (in[7].desc.dims.nbDims == 3) {
            int inferred_experts_num = in[7].desc.dims.d[0];
            int inferred_intermediate_size = in[7].desc.dims.d[1];
            int inferred_hidden_half = in[7].desc.dims.d[2];
            
            if (inferred_experts_num != mExpertsNum) {
                throw std::runtime_error("Experts number mismatch in up_qweight");
            }
            
            if (inferred_intermediate_size != mIntermediateSize) {
                throw std::runtime_error("Intermediate size mismatch in up_qweight");
            }
            
            if (inferred_hidden_half != mHiddenSize / 2) {
                throw std::runtime_error("Hidden size mismatch in up_qweight");
            }
        }

        // Down projection
        if (in[11].desc.dims.nbDims == 3) {
            int inferred_experts_num = in[11].desc.dims.d[0];
            int inferred_hidden_size = in[11].desc.dims.d[1];
            int inferred_intermediate_half = in[11].desc.dims.d[2];
            
            if (inferred_experts_num != mExpertsNum) {
                throw std::runtime_error("Experts number mismatch in down_qweight");
            }
            
            if (inferred_hidden_size != mHiddenSize) {
                throw std::runtime_error("Hidden size mismatch in down_qweight");
            }
            
            if (inferred_intermediate_half != mIntermediateSize / 2) {
                throw std::runtime_error("Intermediate size mismatch in down_qweight");
            }
        }

        // Verify scale vectors have correct size
        if (in[5].desc.dims.nbDims == 1 && in[5].desc.dims.d[0] != mExpertsNum) {
            throw std::runtime_error("gate_input_global_scale size mismatch");
        }
        
        if (in[6].desc.dims.nbDims == 1 && in[6].desc.dims.d[0] != mExpertsNum) {
            throw std::runtime_error("gate_weight_global_scale size mismatch");
        }
        
        if (in[9].desc.dims.nbDims == 1 && in[9].desc.dims.d[0] != mExpertsNum) {
            throw std::runtime_error("up_input_global_scale size mismatch");
        }
        
        if (in[10].desc.dims.nbDims == 1 && in[10].desc.dims.d[0] != mExpertsNum) {
            throw std::runtime_error("up_weight_global_scale size mismatch");
        }
        
        if (in[13].desc.dims.nbDims == 1 && in[13].desc.dims.d[0] != mExpertsNum) {
            throw std::runtime_error("down_input_global_scale size mismatch");
        }
        
        if (in[14].desc.dims.nbDims == 1 && in[14].desc.dims.d[0] != mExpertsNum) {
            throw std::runtime_error("down_weight_global_scale size mismatch");
        }

        if (mExpertsTopK < 1 || mExpertsTopK > mExpertsNum) {
            throw std::runtime_error("experts_topk must be between 1 and experts_num");
        }

        if (mExpertsTopK != 1) {
            printf("Warning: experts_topk > 1 not yet implemented, using top-1\n");
        }
        
        // Verify alpha vectors
        if (mGateAlpha.size() != static_cast<size_t>(mExpertsNum) ||
            mUpAlpha.size() != static_cast<size_t>(mExpertsNum) ||
            mDownAlpha.size() != static_cast<size_t>(mExpertsNum)) {
            throw std::runtime_error("Alpha vectors size mismatch with experts_num");
        }

    } catch (std::exception const& e) {
        printf("Error configuring MoE Nvfp4 plugin: %s\n", e.what());
        throw std::runtime_error("Error configuring MoE Nvfp4 plugin");
    }
}

size_t MoeNvfp4Plugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] nvinfer1::PluginTensorDesc const* outputs,
    [[maybe_unused]] int32_t nbOutputs) const noexcept {
    
    if (mHiddenSize == 0 || mIntermediateSize == 0) {
        printf("Warning: getWorkspaceSize called before configurePlugin\n");
        return 0;
    }

    int32_t batchSize = inputs[0].dims.d[0];
    int32_t seqLen = inputs[0].dims.d[1];
    size_t workspaceSize = 0;

    // 计算各个缓冲区的大小
    int total_tokens = batchSize * seqLen;
    auto padSize = (mExpertsNum + 7) / 8 * 8;

    // expert_indices: [total_tokens] (int32)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens}, DataType::kINT32);

    // router_logits: [total_tokens, padded_experts_num] (half)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens, padSize}, DataType::kHALF);

    // expert_counts: [experts_num] (int32)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{mExpertsNum}, DataType::kINT32);

    // expert_offsets: [experts_num + 1] (int32)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{mExpertsNum + 1}, DataType::kINT32);

    // current_offsets: [experts_num] (int32)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{mExpertsNum}, DataType::kINT32);

    // token_to_buffer_map: [total_tokens] (int32)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens}, DataType::kINT32);

    // expert_input_buffer: [total_tokens, hidden_size] (half)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens, mHiddenSize}, DataType::kHALF);

    // expert_output_buffer: [total_tokens, hidden_size] (half)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens, mHiddenSize}, DataType::kHALF);

    // silu_output: [total_tokens, intermediate_size] (half)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens, mIntermediateSize}, DataType::kHALF);

    // up_output: [total_tokens, intermediate_size] (half)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens, mIntermediateSize}, DataType::kHALF);

    // hadamard_output: [total_tokens, intermediate_size] (half)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens, mIntermediateSize}, DataType::kHALF);

    // Quantization buffers for 4-bit weights
    // gate_input_quant: [total_tokens, hidden_size // 2] (uint8)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens, mHiddenSize / 2}, DataType::kINT8);

    // gate_input_qscales: [total_tokens, padd] (uint8)
    int gate_scale_size = getScalePadSize(total_tokens, mHiddenSize);
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{gate_scale_size}, DataType::kINT8);

    // up_input_quant: [total_tokens, hidden_size // 2] (uint8)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens, mHiddenSize / 2}, DataType::kINT8);

    // up_input_qscales: [total_tokens, padd] (uint8)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{gate_scale_size}, DataType::kINT8);

    // down_input_quant: [total_tokens, intermediate_size // 2] (uint8)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens, mIntermediateSize / 2}, DataType::kINT8);

    // down_input_qscales: [total_tokens, padd] (uint8)
    int down_scale_size = getScalePadSize(total_tokens, mIntermediateSize);
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{down_scale_size}, DataType::kINT8);

    // 添加额外的对齐空间
    workspaceSize += kDEVICE_ALIGNMENT;
    printf("MoE NVFP4 Plugin workspace size: %zu bytes\n", workspaceSize);

    return workspaceSize;
}

template<typename T>
void printDeviceData(const T* device_ptr, size_t count,
                     cudaStream_t stream, const std::string& label,
                     int print_count = 5, bool show_indices = false) {
    std::vector<T> host_data(count);
    cudaMemcpyAsync(host_data.data(), device_ptr,
                   count * sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    printf("%s:\n", label.c_str());
    int actual_print_count = std::min(print_count, static_cast<int>(count));

    if constexpr (std::is_same<T, half>::value) {
        printf("  Values (first %d):\n", actual_print_count);
        for (int i = 0; i < actual_print_count; ++i) {
            printf("    [%d] = %f\n", i, __half2float(host_data[i]));
        }
    } else if constexpr (std::is_same<T, int32_t>::value) {
        printf("  Indices (first %d):\n", actual_print_count);
        for (int i = 0; i < actual_print_count; ++i) {
            if (show_indices) {
                printf("    Token[%d] -> Expert[%d]\n", i, host_data[i]);
            } else {
                printf("    [%d] = %d\n", i, host_data[i]);
            }
        }
    } else if constexpr (std::is_same<T, uint8_t>::value) {
        printf("  Quantized Values (first %d):\n", actual_print_count);
        for (int i = 0; i < actual_print_count; ++i) {
            printf("    [%d] = %u\n", i, host_data[i]);
        }
    } else if constexpr (std::is_same<T, float>::value) {
        printf("  Global Scales (first %d):\n", actual_print_count);
        for (int i = 0; i < actual_print_count; ++i) {
            std::cout << "    [" << i << "] = " << host_data[i] << std::endl;
        }
    }
}

template<typename T>
void printHostData(const T* host_ptr, size_t count, const std::string& label,
                     int print_count = 5) {
    std::vector<T> host_data(count);
    memcpy(host_data.data(), host_ptr, count * sizeof(T));

    printf("%s:\n", label.c_str());
    int actual_print_count = std::min(print_count, static_cast<int>(count));

    if constexpr (std::is_same<T, float>::value) {
        printf("  Global Alpha (first %d):\n", actual_print_count);
        for (int i = 0; i < actual_print_count; ++i) {
            std::cout << "    [" << i << "] = " << host_data[i] << std::endl;
        }
    }
}

int getScalePadSize(int M, int N) {
    int scale_n = N / 16;
    int rounded_m = ((M + 128 - 1) / 128) * 128;
    int rounded_n = ((scale_n + 4 - 1) / 4) * 4;
    return rounded_m * rounded_n;
}

void printDataInfo(MoeNvfp4InputsParams &params, cudaStream_t stream) {
    printf("MoE Nvfp4 Input Parameters:\n");
    auto batch_size = params.batch_size;
    auto seq_len = params.seq_len;
    auto hidden_size = params.hidden_size;
    auto intermediate_size = params.intermediate_size;
    auto experts_num = params.experts_num;
    auto experts_topk = params.experts_topk;
    auto total_tokens = params.total_tokens();
    printf("  Batch Size: %d\n", batch_size);
    printf("  Sequence Length: %d\n", seq_len);
    printf("  Hidden Size: %d\n", hidden_size);
    printf("  Intermediate Size: %d\n", intermediate_size);
    printf("  Experts Number: %d\n", experts_num);
    printf("  Experts Top-K: %d\n", experts_topk);
    printf("  Total Tokens: %d\n", total_tokens);

    printf("Input Tensor Device Pointer: %p\n", (void*)params.hidden_state);
    printf("Output Tensor Device Pointer: %p\n", (void*)params.output);
    printf("Expert Indices Device Pointer: %p\n", (void*)params.expert_indices);
    printf("Expert Weights Device Pointer: %p\n", (void*)params.token_to_buffer_map);
    printf("Expert Counts Device Pointer: %p\n", (void*)params.expert_counts);
    printf("Expert Offsets Device Pointer: %p\n", (void*)params.expert_offsets);
    printf("Current Offsets Device Pointer: %p\n", (void*)params.current_offsets);
    printf("Expert Input Buffer Device Pointer: %p\n", (void*)params.expert_input_buffer);
    printf("Up Output Device Pointer: %p\n", (void*)params.up_output);
    printf("SiLU Output Device Pointer: %p\n", (void*)params.silu_output);
    printf("Hadamard Output Device Pointer: %p\n", (void*)params.hadamard_output);
    printf("Expert Output Buffer Device Pointer: %p\n", (void*)params.expert_output_buffer);

    printf("Weight Data:\n");
    printDeviceData<half>(params.router_weight,
                          hidden_size * experts_num,
                          stream, "Router weight data");
    for(int i = 0; i < experts_num; ++i) {
        printf("Expert %d:\n", i);
        // Gate
        printDeviceData<uint8_t>(params.gate_qweight + i * hidden_size * intermediate_size / 2,
                         hidden_size * intermediate_size / 2,
                         stream, "Gate projection qweight data");
        auto scale_size = getScalePadSize(intermediate_size, hidden_size);
        printDeviceData<uint8_t>(params.gate_qscales + i * scale_size,
                         scale_size,
                         stream, "Gate projection qscale data");
        printDeviceData<float>(params.gate_input_global_scale + i,
                            1,
                            stream, "Gate input global scale data");
        printDeviceData<float>(params.gate_weight_global_scale + i,
                            1,
                            stream, "Gate weight global scale data");
        // Up
        printDeviceData<uint8_t>(params.up_qweight + i * hidden_size * intermediate_size / 2,
                            hidden_size * intermediate_size / 2,
                            stream, "Up projection qweight data");
        printDeviceData<uint8_t>(params.up_qscales + i * scale_size,
                         scale_size,
                         stream, "Up projection qscale data");
        
        printDeviceData<float>(params.up_input_global_scale + i,
                            1,
                            stream, "Up input global scale data");
        printDeviceData<float>(params.up_weight_global_scale + i,
                            1,
                            stream, "Up weight global scale data");
        // Down
        printDeviceData<uint8_t>(params.down_qweight + i * hidden_size * intermediate_size / 2,
                            hidden_size * intermediate_size / 2,
                            stream, "Down projection qweight data");
        scale_size = getScalePadSize(hidden_size, intermediate_size);
        printDeviceData<uint8_t>(params.down_qscales + i * scale_size,
                         scale_size,
                         stream, "Down projection qscale data");
        printDeviceData<float>(params.down_input_global_scale + i,
                            1,
                            stream, "Down input global scale data");
        printDeviceData<float>(params.down_weight_global_scale + i,
                            1,
                            stream, "Down weight global scale data");

        printHostData<float>(params.gate_alpha + i,
                            1, "Gate alpha data");
        printHostData<float>(params.up_alpha + i,
                            1, "Up alpha data");
        printHostData<float>(params.down_alpha + i,
                            1, "Down alpha data");
    }

    printf("Processing Data:\n");
    printDeviceData<int32_t>(params.expert_indices,
                             total_tokens,
                             stream, "Router indices", 5, true);
    for (int i = 0; i < total_tokens; ++i) {
        printf("[Token %d]:\n", i);
        printDeviceData<half>(params.hidden_state + i * hidden_size,
                              hidden_size,
                              stream, "Input data");
        printDeviceData<half>(params.up_output + i * intermediate_size,
                              intermediate_size,
                              stream, "Up projection output data");
        printDeviceData<half>(params.silu_output + i * intermediate_size,
                              intermediate_size,
                              stream, "Silu output data");
        printDeviceData<half>(params.hadamard_output + i * intermediate_size,
                              intermediate_size,
                              stream, "Hadamard output data");
        printDeviceData<half>(params.output + i * hidden_size,
                              hidden_size,
                              stream, "Output data");
        printDeviceData<uint8_t>(params.gate_input_quant + i * hidden_size / 2,
                              hidden_size / 2,
                              stream, "Gate input quantized data");
        printDeviceData<uint8_t>(params.up_input_quant + i * hidden_size / 2,
                              hidden_size / 2,
                              stream, "Up input quantized data");
        printDeviceData<uint8_t>(params.down_input_quant + i * intermediate_size / 2,
                              intermediate_size / 2,
                              stream, "Down input quantized data");
    }
}

// static int g_count = 0;
int32_t MoeNvfp4Plugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    [[maybe_unused]] nvinfer1::PluginTensorDesc const* outputDesc, 
    void const* const* inputs, 
    void* const* outputs,
    void* workspace, 
    cudaStream_t stream) noexcept {

    // cudaGetLastError();
    CUDA_CHECK(cudaGetLastError());
    // if (g_count++ > 0) {
    //     return 0;
    // }
    
    try {
        // Extract input pointers
        half* hiddenState = reinterpret_cast<half*>(const_cast<void*>(inputs[0]));
        half* routerWeight = reinterpret_cast<half*>(const_cast<void*>(inputs[1]));
        half* routerBias = reinterpret_cast<half*>(const_cast<void*>(inputs[2]));
        
        // Gate projection quantized weights
        uint8_t* gateQWeight = reinterpret_cast<uint8_t*>(const_cast<void*>(inputs[3]));
        uint8_t* gateQScales = reinterpret_cast<uint8_t*>(const_cast<void*>(inputs[4]));
        float* gateInputGlobalScale = reinterpret_cast<float*>(const_cast<void*>(inputs[5]));
        float* gateWeightGlobalScale = reinterpret_cast<float*>(const_cast<void*>(inputs[6]));
        
        // Up projection quantized weights
        uint8_t* upQWeight = reinterpret_cast<uint8_t*>(const_cast<void*>(inputs[7]));
        uint8_t* upQScales = reinterpret_cast<uint8_t*>(const_cast<void*>(inputs[8]));
        float* upInputGlobalScale = reinterpret_cast<float*>(const_cast<void*>(inputs[9]));
        float* upWeightGlobalScale = reinterpret_cast<float*>(const_cast<void*>(inputs[10]));
        
        // Down projection quantized weights
        uint8_t* downQWeight = reinterpret_cast<uint8_t*>(const_cast<void*>(inputs[11]));
        uint8_t* downQScales = reinterpret_cast<uint8_t*>(const_cast<void*>(inputs[12]));
        float* downInputGlobalScale = reinterpret_cast<float*>(const_cast<void*>(inputs[13]));
        float* downWeightGlobalScale = reinterpret_cast<float*>(const_cast<void*>(inputs[14]));
        
        half* moeOut = reinterpret_cast<half*>(outputs[0]);

        int32_t batchSize = inputDesc[0].dims.d[0];
        int32_t seqLen = inputDesc[0].dims.d[1];

        if (mHiddenSize == 0 || mIntermediateSize == 0) {
            printf("Error: Plugin not properly configured before enqueue\n");
            return -1;
        }

        if ((int)inputDesc[0].dims.d[2] != mHiddenSize) {
            printf("Error: Input hidden_size mismatch: expected %d, got %ld\n",
                   mHiddenSize, inputDesc[0].dims.d[2]);
            return -1;
        }

        // 对齐workspace指针
        void* alignedWorkspacePtr = alignDevicePtr(workspace);

        // 计算需要的缓冲区大小
        int total_tokens = batchSize * seqLen;
        auto padSize = (mExpertsNum + 7) / 8 * 8;

        // 分配workspace缓冲区
        rt::Tensor expert_indices_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{total_tokens}, DataType::kINT32);

        rt::Tensor router_logits_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{total_tokens, padSize}, DataType::kHALF);

        rt::Tensor expert_counts_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{mExpertsNum}, DataType::kINT32);

        rt::Tensor expert_offsets_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{mExpertsNum + 1}, DataType::kINT32);

        rt::Tensor current_offsets_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{mExpertsNum}, DataType::kINT32);

        rt::Tensor token_to_buffer_map_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{total_tokens}, DataType::kINT32);

        rt::Tensor expert_input_buffer_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{total_tokens, mHiddenSize}, DataType::kHALF);

        rt::Tensor expert_output_buffer_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{total_tokens, mHiddenSize}, DataType::kHALF);

        rt::Tensor silu_output_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{total_tokens, mIntermediateSize}, DataType::kHALF);

        rt::Tensor up_output_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{total_tokens, mIntermediateSize}, DataType::kHALF);

        rt::Tensor hadamard_output_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{total_tokens, mIntermediateSize}, DataType::kHALF);

        // Quantization buffers
        int gate_scale_size = getScalePadSize(total_tokens, mHiddenSize);
        int down_scale_size = getScalePadSize(total_tokens, mIntermediateSize);

        rt::Tensor gate_input_quant_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{total_tokens, mHiddenSize / 2}, DataType::kINT8);

        rt::Tensor gate_input_qscales_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{gate_scale_size}, DataType::kINT8);

        rt::Tensor up_input_quant_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{total_tokens, mHiddenSize / 2}, DataType::kINT8);

        rt::Tensor up_input_qscales_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{gate_scale_size}, DataType::kINT8);

        rt::Tensor down_input_quant_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{total_tokens, mIntermediateSize / 2}, DataType::kINT8);

        rt::Tensor down_input_qscales_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{down_scale_size}, DataType::kINT8);

        MoeNvfp4InputsParams params;
        
        // Input and output
        params.hidden_state = hiddenState;
        params.output = moeOut;
        
        // Router parameters
        params.router_weight = routerWeight;
        params.router_bias = routerBias;
        
        // Gate projection parameters
        params.gate_qweight = gateQWeight;
        params.gate_qscales = gateQScales;
        params.gate_input_global_scale = gateInputGlobalScale;
        params.gate_weight_global_scale = gateWeightGlobalScale;
        
        // Up projection parameters
        params.up_qweight = upQWeight;
        params.up_qscales = upQScales;
        params.up_input_global_scale = upInputGlobalScale;
        params.up_weight_global_scale = upWeightGlobalScale;
        
        // Down projection parameters
        params.down_qweight = downQWeight;
        params.down_qscales = downQScales;
        params.down_input_global_scale = downInputGlobalScale;
        params.down_weight_global_scale = downWeightGlobalScale;
        
        // Workspace buffers
        params.expert_indices = expert_indices_tensor.dataPointer<int32_t>();
        params.expert_counts = expert_counts_tensor.dataPointer<int32_t>();
        params.expert_offsets = expert_offsets_tensor.dataPointer<int32_t>();
        params.current_offsets = current_offsets_tensor.dataPointer<int32_t>();
        params.token_to_buffer_map = token_to_buffer_map_tensor.dataPointer<int32_t>();
        params.router_logits = router_logits_tensor.dataPointer<half>();
        
        params.expert_input_buffer = expert_input_buffer_tensor.dataPointer<half>();
        params.expert_output_buffer = expert_output_buffer_tensor.dataPointer<half>();
        params.silu_output = silu_output_tensor.dataPointer<half>();
        params.up_output = up_output_tensor.dataPointer<half>();
        params.hadamard_output = hadamard_output_tensor.dataPointer<half>();
        
        params.gate_input_quant = gate_input_quant_tensor.dataPointer<uint8_t>();
        params.gate_input_qscales = gate_input_qscales_tensor.dataPointer<uint8_t>();
        params.up_input_quant = up_input_quant_tensor.dataPointer<uint8_t>();
        params.up_input_qscales = up_input_qscales_tensor.dataPointer<uint8_t>();
        params.down_input_quant = down_input_quant_tensor.dataPointer<uint8_t>();
        params.down_input_qscales = down_input_qscales_tensor.dataPointer<uint8_t>();
        
        // Shape parameters
        params.batch_size = batchSize;
        params.seq_len = seqLen;
        params.hidden_size = mHiddenSize;
        params.intermediate_size = mIntermediateSize;
        params.experts_num = mExpertsNum;
        params.experts_topk = mExpertsTopK;
        params.stream = stream;

        // Allocate workspace and copy alpha vectors
        allocateWorkspace();
        
        if (!mIsDataInitialized) {
            // Copy and pad router weights and bias
            CUDA_CHECK(cudaMemcpyAsync(mWorkspace["router_bias"],
                                       routerBias,
                                       mExpertsNum * sizeof(half),
                                       cudaMemcpyDeviceToDevice,
                                       stream));
            
            CUDA_CHECK(cudaMemcpy2DAsync(
                mWorkspace["router_weight"],
                padSize * sizeof(half),
                routerWeight,
                mExpertsNum * sizeof(half),
                mExpertsNum * sizeof(half),
                mHiddenSize,
                cudaMemcpyDeviceToDevice,
                stream));
            
            mIsDataInitialized = true;
        }
        
        // Set padded router parameters
        params.router_weights_padded = reinterpret_cast<half*>(mWorkspace["router_weight"]);
        params.router_bias_padded = reinterpret_cast<half*>(mWorkspace["router_bias"]);
        
        // Set alpha parameters
        params.gate_alpha = mGateAlpha.data();
        params.up_alpha = mUpAlpha.data();
        params.down_alpha = mDownAlpha.data();

        // Call the kernel
        moe_nvfp4_forward_cuda(params);
#if ENABLE_MOE_DEBUG
        printDataInfo(params, stream);
#endif
        return 0;
        
    } catch (std::exception const& e) {
        printf("Error in MoE NVFP4 enqueue: %s\n", e.what());
        return -1;
    }
}

size_t MoeNvfp4Plugin::getSerializationSize() const noexcept {
    return sizeof(mHiddenSize) + sizeof(mIntermediateSize) +
           sizeof(mExpertsNum) + sizeof(mExpertsTopK) +
           sizeof(int32_t) * 3 + // for gate_alpha, up_alpha, down_alpha sizes
           mGateAlpha.size() * sizeof(float) +
           mUpAlpha.size() * sizeof(float) +
           mDownAlpha.size() * sizeof(float);
}

void MoeNvfp4Plugin::serialize(void* buffer) const noexcept {
    serializeValue(&buffer, mHiddenSize);
    serializeValue(&buffer, mIntermediateSize);
    serializeValue(&buffer, mExpertsNum);
    serializeValue(&buffer, mExpertsTopK);

    int32_t gate_size = mGateAlpha.size();
    int32_t up_size = mUpAlpha.size();
    int32_t down_size = mDownAlpha.size();
    serializeValue(&buffer, gate_size);
    serializeValue(&buffer, up_size);
    serializeValue(&buffer, down_size);

    serializeArray(&buffer, mGateAlpha.data(), gate_size);
    serializeArray(&buffer, mUpAlpha.data(), up_size);
    serializeArray(&buffer, mDownAlpha.data(), down_size);
}

int32_t MoeNvfp4Plugin::initialize() noexcept {
    return 0;
}

void MoeNvfp4Plugin::terminate() noexcept {
    freeWorkspace();
}

void MoeNvfp4Plugin::destroy() noexcept {
    delete this;
}

MoeNvfp4PluginCreator::MoeNvfp4PluginCreator() {
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("experts_num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("experts_topk", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("experts_gate_alpha", nullptr, PluginFieldType::kFLOAT32, -1));
    mPluginAttributes.emplace_back(PluginField("experts_up_alpha", nullptr, PluginFieldType::kFLOAT32, -1));
    mPluginAttributes.emplace_back(PluginField("experts_down_alpha", nullptr, PluginFieldType::kFLOAT32, -1));

    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* MoeNvfp4PluginCreator::getPluginName() const noexcept {
    return kMOE_NVFP4_PLUGIN_NAME;
}

nvinfer1::PluginFieldCollection const* MoeNvfp4PluginCreator::getFieldNames() noexcept {
    return &mFieldCollection;
}

void MoeNvfp4PluginCreator::setPluginNamespace(char const* libNamespace) noexcept {
    mNamespace = libNamespace;
}

char const* MoeNvfp4PluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

char const* MoeNvfp4PluginCreator::getPluginVersion() const noexcept {
    return kMOE_NVFP4_PLUGIN_VERSION;
}

nvinfer1::IPluginV2* MoeNvfp4PluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept {
    try {
        std::optional<int32_t> expertsNum = parsePluginScalarField<int32_t>("experts_num", fc);
        std::optional<int32_t> expertsTopK = parsePluginScalarField<int32_t>("experts_topk", fc);
        // std::cout << "Parsed experts_num: " << (expertsNum.has_value() ? std::to_string(expertsNum.value()) : "not found") << std::endl;
        // std::cout << "Parsed experts_topk: " << (expertsTopK.has_value() ? std::to_string(expertsTopK.value()) : "not found") << std::endl;
        
        // Parse alpha vectors
        std::vector<float> gateAlpha, upAlpha, downAlpha;
        
        for (int i = 0; i < fc->nbFields; ++i) {
            // std::cout << "Parsing field: " << fc->fields[i].name << ", length: " << fc->fields[i].length << std::endl;
            if (!strcmp(fc->fields[i].name, "experts_gate_alpha")) {
                gateAlpha.resize(fc->fields[i].length);
                memcpy(gateAlpha.data(), fc->fields[i].data, fc->fields[i].length * sizeof(float));
            } else if (!strcmp(fc->fields[i].name, "experts_up_alpha")) {
                upAlpha.resize(fc->fields[i].length);
                memcpy(upAlpha.data(), fc->fields[i].data, fc->fields[i].length * sizeof(float));
            } else if (!strcmp(fc->fields[i].name, "experts_down_alpha")) {
                downAlpha.resize(fc->fields[i].length);
                memcpy(downAlpha.data(), fc->fields[i].data, fc->fields[i].length * sizeof(float));
            }
        }

        if (!expertsNum.has_value() || !expertsTopK.has_value()) {
            printf("Error: Missing required attributes for MoeNvfp4Plugin\n");
            return nullptr;
        }
        
        // Validate alpha vectors
        if (gateAlpha.empty() || upAlpha.empty() || downAlpha.empty()) {
            printf("Error: Alpha vectors are required for MoeNvfp4Plugin\n");
            return nullptr;
        }
        
        if (gateAlpha.size() != static_cast<size_t>(expertsNum.value()) ||
            upAlpha.size() != static_cast<size_t>(expertsNum.value()) ||
            downAlpha.size() != static_cast<size_t>(expertsNum.value())) {
            printf("Error: Alpha vectors must have size equal to experts_num\n");
            return nullptr;
        }

        MoeNvfp4Plugin* plugin = new MoeNvfp4Plugin(
            std::string(name),
            expertsNum.value(),
            expertsTopK.value(),
            std::move(gateAlpha),
            std::move(upAlpha),
            std::move(downAlpha)
        );

        return plugin;
        
    } catch (std::exception const& e) {
        printf("Error creating MoE NVFP4 plugin: %s\n", e.what());
    }
    return nullptr;
}

nvinfer1::IPluginV2* MoeNvfp4PluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept {
    
    try {
        return new MoeNvfp4Plugin(name, serialData, serialLength);
    } catch (std::exception const& e) {
        printf("Error deserializing MoE NVFP4 plugin: %s\n", e.what());
    }
    return nullptr;
}

} // namespace plugins
} // namespace trt_edgellm
