#include "moeFp16Plugin.h"
#include "kernels/moeKernels/moeFp16Kernels.h"
#include "kernels/moeCutlassKernels/moeCutlassFp16Kernels.h"
#include "plugins/utils/pluginUtils.h"

#include <cassert>
#include <cuda_fp16.h>
#include <mutex>
#include <optional>

#ifdef USE_MOE_CUTLASS
using namespace trt_edgellm::kernel::cutlass_moe;
#else
using namespace trt_edgellm::kernel::cuda;
#endif

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
constexpr char const* kMOE_PLUGIN_VERSION{"1"};
constexpr char const* kMOE_PLUGIN_NAME{"MoEFp16Plugin"};
} // namespace

// Static class fields initialization
PluginFieldCollection MoeFp16PluginCreator::mFieldCollection{};
std::vector<PluginField> MoeFp16PluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(MoeFp16PluginCreator);

MoeFp16Plugin::MoeFp16Plugin(std::string const& name, int32_t expertsNum, int32_t expertsTopK)
    : mLayerName(name)
    , mExpertsNum(expertsNum)
    , mExpertsTopK(expertsTopK) {
    mHiddenSize = 0;
    mIntermediateSize = 0;

    if (expertsTopK != 1) {
        printf("Warning: experts_topk > 1 not yet implemented, will use top-1\n");
    }
}

MoeFp16Plugin::MoeFp16Plugin(std::string const& name, void const* data, size_t length)
    : mLayerName(name) {
    deserializeValue(&data, &length, &mHiddenSize);
    deserializeValue(&data, &length, &mIntermediateSize);
    deserializeValue(&data, &length, &mExpertsNum);
    deserializeValue(&data, &length, &mExpertsTopK);

    if (mExpertsTopK != 1) {
        printf("Warning: experts_topk > 1 not yet implemented, will use top-1\n");
    }
}

MoeFp16Plugin::~MoeFp16Plugin() {
    freeWorkspace();
}

void MoeFp16Plugin::allocateWorkspace() {
    if (mIsWorkspaceAllocated) {
        return;
    }
    mWorkspace["router_weight"] = nullptr;
    auto padSize = (mExpertsNum + 7) / 8 * 8;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mWorkspace["router_weight"]), mHiddenSize * padSize * sizeof(__half)));
    mWorkspace["router_bias"] = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mWorkspace["router_bias"]), padSize * sizeof(__half)));
    mIsWorkspaceAllocated = true;
}

void MoeFp16Plugin::freeWorkspace() {
    for (auto& [key, ptr] : mWorkspace) {
        cudaFree(ptr);
        ptr = nullptr;
    }
    mIsWorkspaceAllocated = false;
}

IPluginV2DynamicExt* MoeFp16Plugin::clone() const noexcept {
    MoeFp16Plugin* plugin = new MoeFp16Plugin(mLayerName, mExpertsNum, mExpertsTopK);
    plugin->mHiddenSize = mHiddenSize;
    plugin->mIntermediateSize = mIntermediateSize;
    return plugin;
}

char const* MoeFp16Plugin::getPluginType() const noexcept {
    return kMOE_PLUGIN_NAME;
}

char const* MoeFp16Plugin::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

void MoeFp16Plugin::setPluginNamespace(char const* pluginNamespace) noexcept {
    mNamespace = std::string(pluginNamespace);
}

char const* MoeFp16Plugin::getPluginVersion() const noexcept {
    return kMOE_PLUGIN_VERSION;
}

int32_t MoeFp16Plugin::getNbOutputs() const noexcept {
    return 1;
}

bool MoeFp16Plugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    try {
        assert(nbInputs == 6 && nbOutputs == 1);
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
                status &= tensorDesc.dims.d[1] == mExpertsNum;
                break;
            }
            case 2: { // router_bias [experts_num]
                status &= tensorDesc.type == DataType::kHALF;
                status &= tensorDesc.format == TensorFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 1;
                status &= tensorDesc.dims.d[0] == mExpertsNum;
                break;
            }
            case 3: { // experts_gate_proj_weight [experts_num, hidden_size, intermediate_size]
                status &= tensorDesc.type == DataType::kHALF;
                status &= tensorDesc.format == TensorFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 3;
                status &= tensorDesc.dims.d[0] == mExpertsNum;
                break;
            }
            case 4: { // experts_up_proj_weight [experts_num, hidden_size, intermediate_size]
                status &= tensorDesc.type == DataType::kHALF;
                status &= tensorDesc.format == TensorFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 3;
                status &= tensorDesc.dims.d[0] == mExpertsNum;
                break;
            }
            case 5: { // experts_down_proj_weight [experts_num, intermediate_size, hidden_size]
                status &= tensorDesc.type == DataType::kHALF;
                status &= tensorDesc.format == TensorFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 3;
                status &= tensorDesc.dims.d[0] == mExpertsNum;
                break;
            }
            case 6: { // moe_out [batch_size, seq_len, hidden_size]
                status &= tensorDesc.type == DataType::kHALF;
                status &= tensorDesc.format == TensorFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 3;
                break;
            }
            default: break;
        }
        return status;
    } catch (std::exception const& e) {
        return false;
    }
}

DataType MoeFp16Plugin::getOutputDataType([[maybe_unused]] int32_t index,
    [[maybe_unused]] nvinfer1::DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept {
    return DataType::kHALF;
}

DimsExprs MoeFp16Plugin::getOutputDimensions([[maybe_unused]] int32_t outputIndex,
    nvinfer1::DimsExprs const* inputs, [[maybe_unused]] int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    DimsExprs output;
    output.nbDims = 3;
    output.d[0] = inputs[0].d[0];
    output.d[1] = inputs[0].d[1];
    output.d[2] = inputs[0].d[2];
    return output;
}

void MoeFp16Plugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] nvinfer1::DynamicPluginTensorDesc const* out,
    [[maybe_unused]] int32_t nbOutputs) noexcept {
    try {
        if (in[0].desc.dims.nbDims == 3) {
            mHiddenSize = in[0].desc.dims.d[2];
        } else {
            throw std::runtime_error("Invalid hidden_state dimensions");
        }

        if (in[1].desc.dims.nbDims == 2) { // router_weight [hidden_size, experts_num]
            int inferred_hidden_size = in[1].desc.dims.d[0];
            if (mHiddenSize == 0) {
                mHiddenSize = inferred_hidden_size;
            } else if (mHiddenSize != inferred_hidden_size) {
                throw std::runtime_error("Hidden size mismatch between hidden_state and router_weight");
            }

            int inferred_experts_num = in[1].desc.dims.d[1];
            if (inferred_experts_num != mExpertsNum) {
                throw std::runtime_error("Experts number mismatch");
            }
        }

        if (in[3].desc.dims.nbDims == 3) { // gate_proj_weight [experts_num, hidden_size, intermediate_size]
            int inferred_experts_num = in[3].desc.dims.d[0];
            int inferred_hidden_size = in[3].desc.dims.d[1];
            mIntermediateSize = in[3].desc.dims.d[2];

            if (inferred_experts_num != mExpertsNum) {
                throw std::runtime_error("Experts number mismatch in gate_proj_weight");
            }

            if (mHiddenSize != 0 && inferred_hidden_size != mHiddenSize) {
                throw std::runtime_error("Hidden size mismatch in gate_proj_weight");
            }

            if (mHiddenSize == 0) {
                mHiddenSize = inferred_hidden_size;
            }
        }

        if (in[4].desc.dims.nbDims == 3) { // up_proj_weight
            if (in[4].desc.dims.d[0] != mExpertsNum ||
                in[4].desc.dims.d[1] != mHiddenSize ||
                in[4].desc.dims.d[2] != mIntermediateSize) {
                throw std::runtime_error("up_proj_weight dimensions mismatch");
            }
        }

        if (in[5].desc.dims.nbDims == 3) { // down_proj_weight
            if (in[5].desc.dims.d[0] != mExpertsNum ||
                in[5].desc.dims.d[1] != mIntermediateSize ||
                in[5].desc.dims.d[2] != mHiddenSize) {
                throw std::runtime_error("down_proj_weight dimensions mismatch");
            }
        }

        if (mExpertsTopK < 1 || mExpertsTopK > mExpertsNum) {
            throw std::runtime_error("experts_topk must be between 1 and experts_num");
        }

        if (mExpertsTopK != 1) {
            printf("Warning: experts_topk > 1 not yet implemented, using top-1\n");
        }
    } catch (std::exception const& e) {
        printf("Error configuring MoE plugin: %s\n", e.what());
        throw std::runtime_error("Error configuring MoE plugin");
    }
}

size_t MoeFp16Plugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] nvinfer1::PluginTensorDesc const* outputs,
    [[maybe_unused]] int32_t nbOutputs) const noexcept
{
    if (mHiddenSize == 0 || mIntermediateSize == 0) {
        printf("Warning: getWorkspaceSize called before configurePlugin\n");
        return 0;
    }

    int32_t batchSize = inputs[0].dims.d[0];
    int32_t seqLen = inputs[0].dims.d[1];

    size_t workspaceSize = 0;

    // 计算各个缓冲区的大小
    int total_tokens = batchSize * seqLen;

    // expert_indices: [total_tokens] (int32)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens}, DataType::kINT32);

    workspaceSize = plugins::accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens}, nvinfer1::DataType::kINT32);

    // router_logits: [total_tokens, experts_num] (half)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens, (mExpertsNum + 7) / 8 * 8}, DataType::kHALF);

    // expert_counts: [experts_num] (int32)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{mExpertsNum}, DataType::kINT32);

    // expert_offsets: [experts_num + 1] (int32)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{mExpertsNum + 1}, DataType::kINT32);

    // current_offsets: [experts_num] (int32)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{mExpertsNum}, DataType::kINT32);

    // expert_input_buffer: [total_tokens, hidden_size] (half)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens, mHiddenSize}, DataType::kHALF);

    // gate_output: [total_tokens, intermediate_size] (half)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens, mIntermediateSize}, DataType::kHALF);

    // up_output: [total_tokens, intermediate_size] (half)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens, mIntermediateSize}, DataType::kHALF);

    // silu_output: [total_tokens, intermediate_size] (half)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens, mIntermediateSize}, DataType::kHALF);

    // hadamard_output: [total_tokens, intermediate_size] (half)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens, mIntermediateSize}, DataType::kHALF);

    // expert_output_buffer: [total_tokens, hidden_size] (half)
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens, mHiddenSize}, DataType::kHALF);

    // 添加额外的对齐空间
    workspaceSize += kDEVICE_ALIGNMENT;
    printf("MoE Plugin workspace size: %zu bytes\n", workspaceSize);

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
    }
}

void printDataInfo(MoeInputsParams &params, cudaStream_t stream) {
    printf("MoE Input Parameters:\n");
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
    printf("Gate Output Device Pointer: %p\n", (void*)params.gate_output);
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
        printDeviceData<half>(params.gate_proj_weight + i * hidden_size * intermediate_size,
                         hidden_size * intermediate_size,
                         stream, "Gate projection weight data");
        printDeviceData<half>(params.up_proj_weight + i * hidden_size * intermediate_size,
                            hidden_size * intermediate_size,
                            stream, "Up projection weight data");
        printDeviceData<half>(params.down_proj_weight + i * hidden_size * intermediate_size,
                            hidden_size * intermediate_size,
                            stream, "Down projection weight data");
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
        printDeviceData<half>(params.gate_output + i * intermediate_size,
                              intermediate_size,
                              stream, "Gate output data");
        printDeviceData<half>(params.up_output + i * intermediate_size,
                              intermediate_size,
                              stream, "Up projection output data");
        printDeviceData<half>(params.silu_output + i * intermediate_size,
                              intermediate_size,
                              stream, "SiLU output data");
        printDeviceData<half>(params.hadamard_output + i * intermediate_size,
                              intermediate_size,
                              stream, "Hadamard output data");
        printDeviceData<half>(params.output + i * hidden_size,
                              hidden_size,
                              stream, "Output data");
    }
}

int32_t MoeFp16Plugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    [[maybe_unused]] nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept {
    try {
        half* hiddenState = reinterpret_cast<half*>(const_cast<void*>(inputs[0]));
        half* routerWeight = reinterpret_cast<half*>(const_cast<void*>(inputs[1]));
        half* routerBias = reinterpret_cast<half*>(const_cast<void*>(inputs[2]));
        half* gateProjWeight = reinterpret_cast<half*>(const_cast<void*>(inputs[3]));
        half* upProjWeight = reinterpret_cast<half*>(const_cast<void*>(inputs[4]));
        half* downProjWeight = reinterpret_cast<half*>(const_cast<void*>(inputs[5]));
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

        // 分配workspace缓冲区
        rt::Tensor expert_indices_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{total_tokens}, DataType::kINT32);

        rt::Tensor router_logits_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{total_tokens, (mExpertsNum + 7) / 8 * 8}, DataType::kHALF);

        rt::Tensor token_to_buffer_map_tensor = plugins::assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{total_tokens}, nvinfer1::DataType::kINT32);

        rt::Tensor expert_counts_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{mExpertsNum}, DataType::kINT32);

        rt::Tensor expert_offsets_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{mExpertsNum + 1}, DataType::kINT32);

        rt::Tensor current_offsets_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{mExpertsNum}, DataType::kINT32);

        rt::Tensor expert_input_buffer_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{total_tokens, mHiddenSize}, DataType::kHALF);

        rt::Tensor gate_output_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{total_tokens, mIntermediateSize}, DataType::kHALF);

        rt::Tensor up_output_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{total_tokens, mIntermediateSize}, DataType::kHALF);

        rt::Tensor silu_output_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{total_tokens, mIntermediateSize}, DataType::kHALF);

        rt::Tensor hadamard_output_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{total_tokens, mIntermediateSize}, DataType::kHALF);

        rt::Tensor expert_output_buffer_tensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, rt::Coords{total_tokens, mHiddenSize}, DataType::kHALF);

        // 初始化workspace内存，todo @hy
        // cudaMemsetAsync(expert_indices_tensor.rawPointer(), 0,
        //                expert_indices_tensor.getMemoryCapacity(), stream);
        // cudaMemsetAsync(expert_counts_tensor.rawPointer(), 0,
        //                expert_counts_tensor.getMemoryCapacity(), stream);
        // cudaMemsetAsync(current_offsets_tensor.rawPointer(), 0,
        //                current_offsets_tensor.getMemoryCapacity(), stream);
        // cudaMemsetAsync(expert_offsets_tensor.rawPointer(), 0,
        //                expert_offsets_tensor.getMemoryCapacity(), stream);
        // cudaMemsetAsync(token_to_buffer_map_tensor.rawPointer(), 0,
        //                token_to_buffer_map_tensor.getMemoryCapacity(), stream);

        // 设置MoeInputsParams
        MoeInputsParams params;
        params.hidden_state = hiddenState;
        params.router_weight = routerWeight;
        params.router_bias = routerBias;
        params.gate_proj_weight = gateProjWeight;
        params.up_proj_weight = upProjWeight;
        params.down_proj_weight = downProjWeight;
        params.output = moeOut;

        params.expert_indices = expert_indices_tensor.dataPointer<int32_t>();
        params.router_logits = router_logits_tensor.dataPointer<half>();
        params.token_to_buffer_map = token_to_buffer_map_tensor.dataPointer<int32_t>();
        params.expert_counts = expert_counts_tensor.dataPointer<int32_t>();
        params.expert_offsets = expert_offsets_tensor.dataPointer<int32_t>();
        params.current_offsets = current_offsets_tensor.dataPointer<int32_t>();
        params.expert_input_buffer = expert_input_buffer_tensor.dataPointer<half>();
        params.gate_output = gate_output_tensor.dataPointer<half>();
        params.up_output = up_output_tensor.dataPointer<half>();
        params.silu_output = silu_output_tensor.dataPointer<half>();
        params.hadamard_output = hadamard_output_tensor.dataPointer<half>();
        params.expert_output_buffer = expert_output_buffer_tensor.dataPointer<half>();

        params.batch_size = batchSize;
        params.seq_len = seqLen;
        params.hidden_size = mHiddenSize;
        params.intermediate_size = mIntermediateSize;
        params.experts_num = mExpertsNum;
        params.experts_topk = mExpertsTopK;
        params.stream = stream;

        allocateWorkspace();

        if (!mIsDataInitialized) {
            auto padSize = (mExpertsNum + 7) / 8 * 8;

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
        params.router_weights_padded = reinterpret_cast<half*>(mWorkspace["router_weight"]);
        params.router_bias_padded = reinterpret_cast<half*>(mWorkspace["router_bias"]);

        if (seqLen > 1) {
            // Context mode (prefill)
            context_moe_fp16_forward_cuda(params);
        } else {
            // Decode mode (generation)
            decode_moe_fp16_forward_cuda(params);
        }
#if ENABLE_MOE_DEBUG
        for(int i = 0; i < 7; ++i) {
            printf("Input[%d] ptr: %p\n", i, inputs[i]);
        }
        printf("Output Tensor Device Pointer: %p\n", (void*)outputs[0]);
        printDataInfo(params, stream);
#endif

        return 0;
    } catch (std::exception const& e) {
        printf("Error in MoE enqueue: %s\n", e.what());
        return -1;
    }
}


size_t MoeFp16Plugin::getSerializationSize() const noexcept {
    return sizeof(mHiddenSize) + sizeof(mIntermediateSize) +
           sizeof(mExpertsNum) + sizeof(mExpertsTopK);
}

void MoeFp16Plugin::serialize(void* buffer) const noexcept {
    serializeValue(&buffer, mHiddenSize);
    serializeValue(&buffer, mIntermediateSize);
    serializeValue(&buffer, mExpertsNum);
    serializeValue(&buffer, mExpertsTopK);
}

int32_t MoeFp16Plugin::initialize() noexcept {
    return 0;
}

void MoeFp16Plugin::terminate() noexcept {}

void MoeFp16Plugin::destroy() noexcept {
    delete this;
}

MoeFp16PluginCreator::MoeFp16PluginCreator() {
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("experts_num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("experts_topk", nullptr, PluginFieldType::kINT32, 1));

    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* MoeFp16PluginCreator::getPluginName() const noexcept {
    return kMOE_PLUGIN_NAME;
}

nvinfer1::PluginFieldCollection const* MoeFp16PluginCreator::getFieldNames() noexcept {
    return &mFieldCollection;
}

void MoeFp16PluginCreator::setPluginNamespace(char const* libNamespace) noexcept {
    mNamespace = libNamespace;
}

char const* MoeFp16PluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

char const* MoeFp16PluginCreator::getPluginVersion() const noexcept {
    return kMOE_PLUGIN_VERSION;
}

nvinfer1::IPluginV2* MoeFp16PluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept {
    try {
        std::optional<int32_t> expertsNum = parsePluginScalarField<int32_t>("experts_num", fc);
        std::optional<int32_t> expertsTopK = parsePluginScalarField<int32_t>("experts_topk", fc);

        if (!expertsNum.has_value() || !expertsTopK.has_value()) {
            printf("Error: Missing required attributes for MoeFp16Plugin\n");
            return nullptr;
        }

        MoeFp16Plugin* plugin = new MoeFp16Plugin(
            std::string(name),
            expertsNum.value(),
            expertsTopK.value()
        );

        return plugin;
    } catch (std::exception const& e) {
        printf("Error creating MoE plugin: %s\n", e.what());
    }
    return nullptr;
}

nvinfer1::IPluginV2* MoeFp16PluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept {
    try {
        return new MoeFp16Plugin(name, serialData, serialLength);
    } catch (std::exception const& e) {
        printf("Error deserializing MoE plugin: %s\n", e.what());
    }
    return nullptr;
}

} // namespace plugins
} // namespace trt_edgellm
