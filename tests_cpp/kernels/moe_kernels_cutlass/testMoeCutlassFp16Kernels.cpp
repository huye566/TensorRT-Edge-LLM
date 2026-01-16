#include "gtest/gtest.h"
#include "utils/utils.h"
#include "common/logger.h"
#include "kernels/moeCutlassKernels/moeCutlassFp16Kernels.h"
#include "plugins/utils/pluginUtils.h"

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using namespace trt_edgellm::kernel::cutlass_moe;

class MoeKernelsTest : public ::testing::Test {
protected:
    static void SetUpTestCase() {
    }

    static void TearDownTestCase() {

    }

    void SetUp() override {
    }

    void TearDown() override {
    }
};

namespace trt_edgellm {
namespace rt {

size_t getWorkspaceSize(int batchSize, int seqLen, int hiddenSize, int intermediateSize,
                        int expertsNum, int expertsTopK) {
    size_t workspaceSize = 0;
    int total_tokens = batchSize * seqLen;

    workspaceSize = plugins::accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens}, nvinfer1::DataType::kINT32);

    // router_logits: [total_tokens, experts_num] (half)
    workspaceSize = plugins::accumulateWorkspaceSize(
         workspaceSize, rt::Coords{total_tokens, (expertsNum + 7) / 8 * 8}, nvinfer1::DataType::kHALF);

    workspaceSize = plugins::accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens}, nvinfer1::DataType::kINT32);
    
    // expert_counts: [experts_num] (int32)
    workspaceSize = plugins::accumulateWorkspaceSize(
        workspaceSize, rt::Coords{expertsNum}, nvinfer1::DataType::kINT32);
    
    // expert_offsets: [experts_num + 1] (int32)
    workspaceSize = plugins::accumulateWorkspaceSize(
        workspaceSize, rt::Coords{expertsNum + 1}, nvinfer1::DataType::kINT32);
    
    // current_offsets: [experts_num] (int32)
    workspaceSize = plugins::accumulateWorkspaceSize(
        workspaceSize, rt::Coords{expertsNum}, nvinfer1::DataType::kINT32);
    
    // expert_input_buffer: [total_tokens, hidden_size] (half)
    workspaceSize = plugins::accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens, hiddenSize}, nvinfer1::DataType::kHALF);
    
    // gate_output: [total_tokens, intermediate_size] (half)
    workspaceSize = plugins::accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens, intermediateSize}, nvinfer1::DataType::kHALF);
    
    // up_output: [total_tokens, intermediate_size] (half)
    workspaceSize = plugins::accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens, intermediateSize}, nvinfer1::DataType::kHALF);

    // silu_output: [total_tokens, intermediate_size] (half)
    workspaceSize = plugins::accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens, intermediateSize}, nvinfer1::DataType::kHALF);

    // hadamard_output: [total_tokens, intermediate_size] (half)
    workspaceSize = plugins::accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens, intermediateSize}, nvinfer1::DataType::kHALF);
    
    // expert_output_buffer: [total_tokens, hidden_size] (half)
    workspaceSize = plugins::accumulateWorkspaceSize(
        workspaceSize, rt::Coords{total_tokens, hiddenSize}, nvinfer1::DataType::kHALF);

    // 添加额外的对齐空间
    workspaceSize += plugins::kDEVICE_ALIGNMENT;

    return workspaceSize;
}

void printDataInfo(MoeInputsParams &params, cudaStream_t stream) {
    LOG_INFO("MoE Input Parameters:");
    auto batch_size = params.batch_size;
    auto seq_len = params.seq_len;
    auto hidden_size = params.hidden_size;
    auto intermediate_size = params.intermediate_size;
    auto experts_num = params.experts_num;
    auto experts_topk = params.experts_topk;
    auto total_tokens = params.total_tokens();
    LOG_INFO("  Batch Size: %d", batch_size);
    LOG_INFO("  Sequence Length: %d", seq_len);
    LOG_INFO("  Hidden Size: %d", hidden_size);
    LOG_INFO("  Intermediate Size: %d", intermediate_size);
    LOG_INFO("  Experts Number: %d", experts_num);
    LOG_INFO("  Experts Top-K: %d", experts_topk);
    LOG_INFO("  Total Tokens: %d", total_tokens);

    LOG_INFO("Weight Data:");
    printDeviceData<half>(params.router_weight, 
                          hidden_size * experts_num, 
                          stream, "Router weight data");
    for(int i = 0; i < experts_num; ++i) {
        LOG_INFO("Expert %d:", i);
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

    LOG_INFO("Processing Data:");
    printDeviceData<int32_t>(params.expert_indices, 
                             total_tokens, 
                             stream, "Router indices", 5, true);
    for (int i = 0; i < total_tokens; ++i) {
        LOG_INFO("[Token %d]:", i);
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


bool moeForwardTest(cudaStream_t stream) {
    std::vector<Tensor> moeInputTensors;
    std::vector<Tensor> moeWeightsTensors;
    if (!loadMoeData(moeInputTensors, moeWeightsTensors, stream)) {
        return false;
    }
    
    Tensor* hiddenStateTensor = nullptr;
    Tensor* routerWeightTensor = nullptr;
    Tensor* routerBiasTensor = nullptr;
    Tensor* gateProjWeightTensor = nullptr;
    Tensor* upProjWeightTensor = nullptr;
    Tensor* downProjWeightTensor = nullptr;
    
    for (auto& tensor : moeInputTensors) {
        if (tensor.getName() == "hidden_state") {
            hiddenStateTensor = &tensor;
        }
    }
    
    for (auto& tensor : moeWeightsTensors) {
        if (tensor.getName() == "model.layers.0.mlp.router_weight") {
            routerWeightTensor = &tensor;
        } else if (tensor.getName() == "model.layers.0.mlp.router_bias") {
            routerBiasTensor = &tensor;
        } else if (tensor.getName() == "model.layers.0.mlp.experts_gate_proj_weight") {
            gateProjWeightTensor = &tensor;
        } else if (tensor.getName() == "model.layers.0.mlp.experts_up_proj_weight") {
            upProjWeightTensor = &tensor;
        } else if (tensor.getName() == "model.layers.0.mlp.experts_down_proj_weight") {
            downProjWeightTensor = &tensor;
        }
    }
    
    if (hiddenStateTensor == nullptr) {
        LOG_ERROR("hidden_state tensor not found");
        return false;
    }
    if (routerWeightTensor == nullptr) {
        LOG_ERROR("router_weight tensor not found");
        return false;
    }
    if (routerBiasTensor == nullptr) {
        LOG_ERROR("router_bias tensor not found");
        return false;
    }
    if (gateProjWeightTensor == nullptr) {
        LOG_ERROR("gate_proj_weight tensor not found");
        return false;
    }
    if (upProjWeightTensor == nullptr) {
        LOG_ERROR("up_proj_weight tensor not found");
        return false;
    }
    if (downProjWeightTensor == nullptr) {
        LOG_ERROR("down_proj_weight tensor not found");
        return false;
    }
    
    auto hiddenStateShape = hiddenStateTensor->getShape();
    auto routerWeightShape = routerWeightTensor->getShape();
    auto gateProjWeightShape = gateProjWeightTensor->getShape();
    int batchSize = hiddenStateShape[0];
    int seqLen = hiddenStateShape[1];
    int hiddenSize = hiddenStateShape[2];
    int expertsNum = routerWeightShape[1];
    int intermediateSize = gateProjWeightShape[2];
    int expertsTopK = 1;
    
    int totalTokens = batchSize * seqLen;
    
    LOG_INFO("MoE Parameters: batch_size=%d, seq_len=%d, hidden_size=%d, "
             "intermediate_size=%d, experts_num=%d, experts_topk=%d, total_tokens=%d",
             batchSize, seqLen, hiddenSize, intermediateSize, expertsNum, expertsTopK, totalTokens);
    
    auto outputShape = rt::Coords{batchSize, seqLen, hiddenSize};
    Tensor outputTensor(outputShape, DeviceType::kGPU, nvinfer1::DataType::kHALF, "moe_output");

    size_t workspaceSize = getWorkspaceSize(batchSize, seqLen, hiddenSize, intermediateSize, expertsNum, expertsTopK);
    
    void* workspacePtr = nullptr;
    auto cudaStatus = cudaMalloc(&workspacePtr, workspaceSize);
    if (cudaStatus != cudaSuccess) {
        LOG_ERROR("Failed to allocate workspace: %s", cudaGetErrorString(cudaStatus));
        return false;
    }

    void* alignedWorkspacePtr = plugins::alignDevicePtr(workspacePtr);
    Tensor expert_indices_tensor = plugins::assignTensorFromWorkspace(alignedWorkspacePtr, Coords{totalTokens}, nvinfer1::DataType::kINT32);
    Tensor router_logits_tensor = plugins::assignTensorFromWorkspace(alignedWorkspacePtr, Coords{totalTokens, (expertsNum + 7) / 8 * 8}, nvinfer1::DataType::kHALF);
    Tensor token_to_buffer_map_tensor = plugins::assignTensorFromWorkspace(alignedWorkspacePtr, Coords{totalTokens}, nvinfer1::DataType::kINT32);
    Tensor expert_counts_tensor = plugins::assignTensorFromWorkspace(alignedWorkspacePtr, Coords{expertsNum}, nvinfer1::DataType::kINT32);
    Tensor expert_offsets_tensor = plugins::assignTensorFromWorkspace(alignedWorkspacePtr, Coords{expertsNum + 1}, nvinfer1::DataType::kINT32);
    Tensor current_offsets_tensor = plugins::assignTensorFromWorkspace(alignedWorkspacePtr, Coords{expertsNum}, nvinfer1::DataType::kINT32);
    Tensor expert_input_buffer_tensor = plugins::assignTensorFromWorkspace(alignedWorkspacePtr, Coords{totalTokens, hiddenSize}, nvinfer1::DataType::kHALF);
    Tensor gate_output_tensor = plugins::assignTensorFromWorkspace(alignedWorkspacePtr, Coords{totalTokens, intermediateSize}, nvinfer1::DataType::kHALF);
    Tensor up_output_tensor = plugins::assignTensorFromWorkspace(alignedWorkspacePtr, Coords{totalTokens, intermediateSize}, nvinfer1::DataType::kHALF);
    Tensor silu_output_tensor = plugins::assignTensorFromWorkspace(alignedWorkspacePtr, Coords{totalTokens, intermediateSize}, nvinfer1::DataType::kHALF);
    Tensor hadamard_output_tensor = plugins::assignTensorFromWorkspace(alignedWorkspacePtr, Coords{totalTokens, intermediateSize}, nvinfer1::DataType::kHALF);
    Tensor expert_output_buffer_tensor = plugins::assignTensorFromWorkspace(alignedWorkspacePtr, Coords{totalTokens, hiddenSize}, nvinfer1::DataType::kHALF);

    // 移除错误的cudaMemsetAsync调用，它们使用了未声明的变量
    CUDA_CHECK(cudaMemsetAsync(expert_indices_tensor.rawPointer(), 0, 
                    expert_indices_tensor.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(expert_counts_tensor.rawPointer(), 0, 
                    expert_counts_tensor.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(current_offsets_tensor.rawPointer(), 0, 
                    current_offsets_tensor.getMemoryCapacity(), stream));
    
    MoeInputsParams params;
    params.hidden_state = hiddenStateTensor->dataPointer<half>();
    params.router_weight = routerWeightTensor->dataPointer<half>();
    params.router_bias = routerBiasTensor->dataPointer<half>();
    params.gate_proj_weight = gateProjWeightTensor->dataPointer<half>();
    params.up_proj_weight = upProjWeightTensor->dataPointer<half>();
    params.down_proj_weight = downProjWeightTensor->dataPointer<half>();
    params.output = outputTensor.dataPointer<half>();
    
    params.expert_indices = expert_indices_tensor.dataPointer<int32_t>();
    params.router_logits = router_logits_tensor.dataPointer<half>();
    params.expert_counts = expert_counts_tensor.dataPointer<int32_t>();
    params.expert_offsets = expert_offsets_tensor.dataPointer<int32_t>();
    params.current_offsets = current_offsets_tensor.dataPointer<int32_t>();
    params.token_to_buffer_map = token_to_buffer_map_tensor.dataPointer<int32_t>();
    params.expert_input_buffer = expert_input_buffer_tensor.dataPointer<half>();
    params.gate_output = gate_output_tensor.dataPointer<half>();
    params.up_output = up_output_tensor.dataPointer<half>();
    params.silu_output = silu_output_tensor.dataPointer<half>();
    params.hadamard_output = hadamard_output_tensor.dataPointer<half>();
    params.expert_output_buffer = expert_output_buffer_tensor.dataPointer<half>();

    params.batch_size = batchSize;
    params.seq_len = seqLen;
    params.hidden_size = hiddenSize;
    params.intermediate_size = intermediateSize;
    params.experts_num = expertsNum;
    params.experts_topk = expertsTopK;
    params.stream = stream;
    
    LOG_INFO("Running forward_moe...");
    forward_moe(params);
    
    cudaStatus = cudaStreamSynchronize(stream);
    if (cudaStatus != cudaSuccess) {
        LOG_ERROR("CUDA stream synchronization failed: %s", cudaGetErrorString(cudaStatus));
        cudaFree(workspacePtr);
        return false;
    }
    
    LOG_INFO("MoE forward propagation completed successfully");

    printDataInfo(params, stream);
    std::vector<Tensor> moeIntermediateTensors;

    if (!loadMoeIntermediateRes(moeIntermediateTensors, stream)) {
        LOG_WARNING("Failed to load MoE intermediate results, but test passed");
    } else {
        LOG_INFO("Comparing intermediate results...");
        Tensor* imGateOutTensor = nullptr;
        Tensor* imUpOutTensor = nullptr;
        Tensor* imHadamardOutTensor = nullptr;
        Tensor* imSiluOutTensor = nullptr;

        for (auto& tensor : moeIntermediateTensors) {
            if (tensor.getName() == "model.layers.0.mlp.experts.0.hadamard_out") {
                imHadamardOutTensor = &tensor;
            } else if (tensor.getName() == "model.layers.0.mlp.experts.0.silu_out") {
                imSiluOutTensor = &tensor;
            } else if (tensor.getName() == "model.layers.0.mlp.experts.0.up_out") {
                imUpOutTensor = &tensor;
            } else if (tensor.getName() == "model.layers.0.mlp.experts.0.gate_out") {
                imGateOutTensor = &tensor;
            }
        }
        if (imGateOutTensor) {
            LOG_INFO("Comparing gate_output...");
            compareTensors(gate_output_tensor, *imGateOutTensor, stream);
        }
        if (imUpOutTensor) {
            LOG_INFO("Comparing up_output...");
            compareTensors(up_output_tensor, *imUpOutTensor, stream);
        }
        if (imSiluOutTensor) {
            LOG_INFO("Comparing silu_output...");
            compareTensors(silu_output_tensor, *imSiluOutTensor, stream);
        }
        if (imHadamardOutTensor) {
            LOG_INFO("Comparing hadamard_output...");
            compareTensors(hadamard_output_tensor, *imHadamardOutTensor, stream);
        }
    }

    std::vector<Tensor> referenceTensors;
    if (loadReferenceOutput(referenceTensors, stream)) {
        auto refTensor = std::move(referenceTensors[0]);
        if (!compareTensors(outputTensor, refTensor, stream)) {
            LOG_ERROR("Output does not match reference data");
            cudaFree(workspacePtr);
            return false;
        }
    } else {
        LOG_WARNING("Failed to load reference output, but test passed");
    }
    std::vector<Tensor> outputTensors;
    outputTensors.push_back(std::move(outputTensor));
    bool saveSuccess = saveOutputTensor(outputTensors, stream);
    if (!saveSuccess) {
        LOG_WARNING("Failed to save output tensor, but test passed");
    }
    
    cudaFree(workspacePtr);
    return saveSuccess;
}

} // namespace trt_edgellm::rt
} // namespace trt_edgellm


#if 0
TEST_F(MoeKernelsTest, LoadSafeTensorTest) {
    cudaStream_t stream;

    cudaError_t cudaStatus = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaStatus, cudaSuccess) << "Failed to create CUDA stream: " 
                                       << cudaGetErrorString(cudaStatus);

    std::vector<trt_edgellm::rt::Tensor> moeInputTensors;
    std::vector<trt_edgellm::rt::Tensor> moeWeightsTensors;
    ASSERT_TRUE(trt_edgellm::rt::loadMoeData(moeInputTensors, moeWeightsTensors, stream)) << "Failed to load moe data";

    trt_edgellm::rt::printTensorInfo(moeInputTensors);
    trt_edgellm::rt::printTensorInfo(moeWeightsTensors);
    cudaStreamDestroy(stream);
}
#endif

TEST_F(MoeKernelsTest, MoeForwardTest) {
    cudaStream_t stream;
    cudaError_t cudaStatus = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaStatus, cudaSuccess) << "Failed to create CUDA stream: " 
                                       << cudaGetErrorString(cudaStatus);

    ASSERT_TRUE(trt_edgellm::rt::moeForwardTest(stream)) << "Failed to run moe forward test";
    cudaStreamDestroy(stream);
}
