#include <dlfcn.h>
#include <vector>
#include <numeric>
#include <cmath>
#include <NvInferPlugin.h>
#include <cuda_fp16.h>
#include <cstring>
#include "utils/utils.h"
#include "common/logger.h"
#include "plugins/utils/pluginUtils.h"
#include "plugins/moeNvfp4Plugin/moeNvfp4Plugin.h"

using namespace nvinfer1;

#define INPUT_TENSOR_NAME "hidden_state"
#define ROUTER_WEIGHT_NAME "model.layers.0.mlp.router_weight"
#define ROUTER_BIAS_NAME "model.layers.0.mlp.router_bias"
#define GATE_QWEIGHT_NAME "model.layers.0.mlp.experts.gate_qweight"
#define GATE_QSCALES_NAME "model.layers.0.mlp.experts.gate_qscales"
#define GATE_INPUT_GLOBAL_SCALE_NAME "model.layers.0.mlp.experts.gate_input_global_scale"
#define GATE_WEIGHT_GLOBAL_SCALE_NAME "model.layers.0.mlp.experts.gate_weight_global_scale"
#define GATE_ALPHA_NAME "model.layers.0.mlp.experts.gate_alpha"
#define UP_QWEIGHT_NAME "model.layers.0.mlp.experts.up_qweight"
#define UP_QSCALES_NAME "model.layers.0.mlp.experts.up_qscales"
#define UP_INPUT_GLOBAL_SCALE_NAME "model.layers.0.mlp.experts.up_input_global_scale"
#define UP_WEIGHT_GLOBAL_SCALE_NAME "model.layers.0.mlp.experts.up_weight_global_scale"
#define UP_ALPHA_NAME "model.layers.0.mlp.experts.up_alpha"
#define DOWN_QWEIGHT_NAME "model.layers.0.mlp.experts.down_qweight"
#define DOWN_QSCALES_NAME "model.layers.0.mlp.experts.down_qscales"
#define DOWN_INPUT_GLOBAL_SCALE_NAME "model.layers.0.mlp.experts.down_input_global_scale"
#define DOWN_WEIGHT_GLOBAL_SCALE_NAME "model.layers.0.mlp.experts.down_weight_global_scale"
#define DOWN_ALPHA_NAME "model.layers.0.mlp.experts.down_alpha"

namespace trt_edgellm {
namespace rt {
bool loadMoeNvfp4Data(
    std::unordered_map<std::string, Tensor>& tensors,
    std::vector<float>& gateAlpha,
    std::vector<float>& upAlpha,
    std::vector<float>& downAlpha,
    cudaStream_t stream) {

    std::filesystem::path inputPath = getSafetensorPath("moe_input.safetensors");
    std::filesystem::path weightPath = getSafetensorPath("moe_nvfp4_weights.safetensors");

    LOG_INFO("Loading NVFP4 MoE input data from: %s", inputPath.string().c_str());
    LOG_INFO("Loading NVFP4 MoE weight data from: %s", weightPath.string().c_str());

    std::vector<Tensor> inputTensors, weightTensors;
    if (!loadSafetensors(inputPath, inputTensors, stream)) {
        LOG_ERROR("Failed to load NVFP4 MoE input safetensors");
        return false;
    }
    if (!loadSafetensors(weightPath, weightTensors, stream)) {
        LOG_ERROR("Failed to load NVFP4 MoE weight safetensors");
        return false;
    }

    for (auto& t : inputTensors) {
        tensors[t.getName()] = std::move(t);
    }
    for (auto& t : weightTensors) {
        tensors[t.getName()] = std::move(t);
    }


    std::vector<std::string> requiredNames = {
        INPUT_TENSOR_NAME,
        ROUTER_WEIGHT_NAME,
        ROUTER_BIAS_NAME,
        GATE_QWEIGHT_NAME,
        GATE_QSCALES_NAME,
        GATE_INPUT_GLOBAL_SCALE_NAME,
        GATE_WEIGHT_GLOBAL_SCALE_NAME,
        UP_QWEIGHT_NAME,
        UP_QSCALES_NAME,
        UP_INPUT_GLOBAL_SCALE_NAME,
        UP_WEIGHT_GLOBAL_SCALE_NAME,
        DOWN_QWEIGHT_NAME,
        DOWN_QSCALES_NAME,
        DOWN_INPUT_GLOBAL_SCALE_NAME,
        DOWN_WEIGHT_GLOBAL_SCALE_NAME,
        GATE_ALPHA_NAME,
        UP_ALPHA_NAME,
        DOWN_ALPHA_NAME
    };
    for (const auto& name : requiredNames) {
        if (tensors.find(name) == tensors.end()) {
            LOG_ERROR("Required tensor '%s' not found", name.c_str());
            return false;
        }
    }

    auto loadAlpha = [&](const std::string& name, std::vector<float>& alphaVec) -> bool {
        const Tensor& t = tensors.at(name);
        if (t.getDataType() != DataType::kFLOAT) {
            LOG_ERROR("Alpha tensor '%s' must be of type FLOAT", name.c_str());
            return false;
        }
        size_t numElements = 3; // todo
        alphaVec.resize(numElements);
        if (t.getDeviceType() == DeviceType::kGPU) {
            CUDA_CHECK(cudaMemcpyAsync(alphaVec.data(), t.rawPointer(),
                                       numElements * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
            memcpy(alphaVec.data(), t.rawPointer(), numElements * sizeof(float));
        }
        return true;
    };

    if (!loadAlpha(GATE_ALPHA_NAME, gateAlpha)) return false;
    if (!loadAlpha(UP_ALPHA_NAME, upAlpha)) return false;
    if (!loadAlpha(DOWN_ALPHA_NAME, downAlpha)) return false;

    LOG_INFO("Successfully loaded all NVFP4 MoE tensors and alpha vectors.");
    return true;
}

bool loadNvfp4ReferenceOutput(std::vector<trt_edgellm::rt::Tensor>& referenceTensors, cudaStream_t stream) {
    std::filesystem::path referencePath = getSafetensorPath("moe_nvfp4_output_ref.safetensors");

    if (!loadSafetensors(referencePath, referenceTensors, stream)) {
        std::cerr << "Failed to load reference output from: " << referencePath.string() << std::endl;
        return false;
    }

    if (referenceTensors.empty()) {
        std::cerr << "No tensors found in reference output file" << std::endl;
        return false;
    }

    auto& refTensor = referenceTensors[0];
    auto shape = refTensor.getShape();
    std::cout << "Reference tensor loaded: " << refTensor.getName()
              << ", shape: " << shape[0] << "x" << shape[1] << "x" << shape[2] << std::endl;

    return true;
}

}
}

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

struct MoeNvfp4Parameters {
    int batchSize;
    int seqLen;
    int hiddenSize;
    int intermediateSize;
    int expertsNum;
    int expertsTopK;
    int hiddenHalf;           // hiddenSize / 2
    int intermediateHalf;     // intermediateSize / 2
    int gateScalePadSize;     // getScalePadSize(total_tokens, hiddenSize)
    int downScalePadSize;     // getScalePadSize(total_tokens, intermediateSize)
    
    MoeNvfp4Parameters() : batchSize(0), seqLen(0), hiddenSize(0),
        intermediateSize(0), expertsNum(0), expertsTopK(1),
        hiddenHalf(0), intermediateHalf(0), gateScalePadSize(0), downScalePadSize(0) {}
};

// NVFP4 测试数据容器，存储所有输入张量及参考输出
struct MoeNvfp4TestData {
    trt_edgellm::rt::Tensor hiddenState;               // FP16
    trt_edgellm::rt::Tensor routerWeight;              // FP16
    trt_edgellm::rt::Tensor routerBias;                // FP16
    trt_edgellm::rt::Tensor gateQWeight;               // INT8
    trt_edgellm::rt::Tensor gateQScales;               // INT8
    trt_edgellm::rt::Tensor gateInputGlobalScale;      // FLOAT
    trt_edgellm::rt::Tensor gateWeightGlobalScale;     // FLOAT
    trt_edgellm::rt::Tensor upQWeight;                 // INT8
    trt_edgellm::rt::Tensor upQScales;                 // INT8
    trt_edgellm::rt::Tensor upInputGlobalScale;        // FLOAT
    trt_edgellm::rt::Tensor upWeightGlobalScale;       // FLOAT
    trt_edgellm::rt::Tensor downQWeight;               // INT8
    trt_edgellm::rt::Tensor downQScales;               // INT8
    trt_edgellm::rt::Tensor downInputGlobalScale;      // FLOAT
    trt_edgellm::rt::Tensor downWeightGlobalScale;     // FLOAT

    std::vector<float> gateAlpha;
    std::vector<float> upAlpha;
    std::vector<float> downAlpha;
    
    std::vector<trt_edgellm::rt::Tensor> referenceOutputs;
    MoeNvfp4Parameters params;
};

struct TrtResources {
    IBuilder* builder = nullptr;
    IBuilderConfig* config = nullptr;
    INetworkDefinition* network = nullptr;
    IHostMemory* serializedEngine = nullptr;
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    IPluginV2* plugin = nullptr;
    void* pluginHandle = nullptr;

    std::vector<void*> devicePointers;  // 按顺序存储 15个输入 + 1个输出
    void* workspace = nullptr;          // plugin workspace
    size_t workspaceSize = 0;
    cudaStream_t stream = nullptr;
};

bool initializeCudaStream(cudaStream_t& stream) {
    cudaError_t cudaStatus = cudaStreamCreate(&stream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream: " 
                  << cudaGetErrorString(cudaStatus) << std::endl;
        return false;
    }
    return true;
}

std::pair<int, int> getScalePadShape(int M, int N) {
    int scale_n = N / 16;
    int rounded_m = ((M + 128 - 1) / 128) * 128;
    int rounded_n = ((scale_n + 4 - 1) / 4) * 4;
    return std::make_pair(rounded_m, rounded_n);
}

int getScalePadSize(int M, int N) {
    auto [rounded_m, rounded_n] = getScalePadShape(M, N);
    return rounded_m * rounded_n;
}



bool loadMoeNvfp4TestData(MoeNvfp4TestData& testData, cudaStream_t stream) {
    trt_edgellm::rt::setCustomPath("testdata", trt_edgellm::rt::getResourcesPath() / "seq1");

    std::unordered_map<std::string, trt_edgellm::rt::Tensor> tensorMap;
    if (!trt_edgellm::rt::loadMoeNvfp4Data(tensorMap,
                                           testData.gateAlpha,
                                           testData.upAlpha,
                                           testData.downAlpha,
                                           stream)) {
        std::cerr << "Failed to load MoE NVFP4 test data" << std::endl;
        return false;
    }
    
        auto moveTensor = [&](const std::string& name, auto& dest) {
        auto it = tensorMap.find(name);
        if (it == tensorMap.end()) return false;
        dest = std::move(it->second);
        return true;
    };

    if (!moveTensor(INPUT_TENSOR_NAME, testData.hiddenState)) return false;
    if (!moveTensor(ROUTER_WEIGHT_NAME, testData.routerWeight)) return false;
    if (!moveTensor(ROUTER_BIAS_NAME, testData.routerBias)) return false;
    if (!moveTensor(GATE_QWEIGHT_NAME, testData.gateQWeight)) return false;
    if (!moveTensor(GATE_QSCALES_NAME, testData.gateQScales)) return false;
    if (!moveTensor(GATE_INPUT_GLOBAL_SCALE_NAME, testData.gateInputGlobalScale)) return false;
    if (!moveTensor(GATE_WEIGHT_GLOBAL_SCALE_NAME, testData.gateWeightGlobalScale)) return false;
    if (!moveTensor(UP_QWEIGHT_NAME, testData.upQWeight)) return false;
    if (!moveTensor(UP_QSCALES_NAME, testData.upQScales)) return false;
    if (!moveTensor(UP_INPUT_GLOBAL_SCALE_NAME, testData.upInputGlobalScale)) return false;
    if (!moveTensor(UP_WEIGHT_GLOBAL_SCALE_NAME, testData.upWeightGlobalScale)) return false;
    if (!moveTensor(DOWN_QWEIGHT_NAME, testData.downQWeight)) return false;
    if (!moveTensor(DOWN_QSCALES_NAME, testData.downQScales)) return false;
    if (!moveTensor(DOWN_INPUT_GLOBAL_SCALE_NAME, testData.downInputGlobalScale)) return false;
    if (!moveTensor(DOWN_WEIGHT_GLOBAL_SCALE_NAME, testData.downWeightGlobalScale)) return false;

    auto hiddenShape = testData.hiddenState.getShape();
    auto routerWeightShape = testData.routerWeight.getShape();
    auto gateQWeightShape = testData.gateQWeight.getShape();
    
    testData.params.batchSize = hiddenShape[0];
    testData.params.seqLen = hiddenShape[1];
    testData.params.hiddenSize = hiddenShape[2];
    testData.params.expertsNum = routerWeightShape[1];
    testData.params.intermediateSize = gateQWeightShape[1];  // [E, I, H/2]
    testData.params.hiddenHalf = testData.params.hiddenSize / 2;
    testData.params.intermediateHalf = testData.params.intermediateSize / 2;
    testData.params.expertsTopK = 1;
    
    int total_tokens = testData.params.batchSize * testData.params.seqLen;
    testData.params.gateScalePadSize = getScalePadSize(total_tokens, testData.params.hiddenSize);
    testData.params.downScalePadSize = getScalePadSize(total_tokens, testData.params.intermediateSize);
    
    if (!trt_edgellm::rt::loadNvfp4ReferenceOutput(testData.referenceOutputs, stream)) {
        std::cout << "Warning: Failed to load reference output, validation will be skipped" << std::endl;
    }
    
    std::cout << "\n=== MoE NVFP4 Parameters ===" << std::endl;
    std::cout << "batch_size: " << testData.params.batchSize << std::endl;
    std::cout << "seq_len: " << testData.params.seqLen << std::endl;
    std::cout << "hidden_size: " << testData.params.hiddenSize << std::endl;
    std::cout << "intermediate_size: " << testData.params.intermediateSize << std::endl;
    std::cout << "experts_num: " << testData.params.expertsNum << std::endl;
    std::cout << "experts_topk: " << testData.params.expertsTopK << std::endl;
    return true;
}

bool createNvfp4Plugin(TrtResources& resources, const MoeNvfp4TestData& testData, Logger& logger) {
    // 加载 plugin 库
    resources.pluginHandle = dlopen("./build/libNvInfer_edgellm_plugin.so", RTLD_LAZY);
    if (!resources.pluginHandle) {
        std::cerr << "Error loading plugin: " << dlerror() << std::endl;
        return false;
    }

    initLibNvInferPlugins(&logger, "");
    
    // 获取 Plugin Creator
    auto plugin_creator = getPluginRegistry()->getPluginCreator("MoENvFp4Plugin", "1");
    if (!plugin_creator) {
        std::cerr << "Error: MoENvFp4Plugin creator not found" << std::endl;
        return false;
    }
    
    // 准备 PluginField
    int32_t expertsNumValue = testData.params.expertsNum;
    int32_t expertsTopKValue = testData.params.expertsTopK;
    
    // Alpha 向量必须作为 PluginField 传递，长度 = expertsNum
    std::vector<float> gateAlpha = testData.gateAlpha;
    std::vector<float> upAlpha = testData.upAlpha;
    std::vector<float> downAlpha = testData.downAlpha;
#if ENABLE_MOE_DEBUG
    for (auto& alpha : {gateAlpha, upAlpha, downAlpha}) {
        for (auto& a : alpha) {
            std::cout << a << " ";
        }
        std::cout << std::endl;
    }
#endif
    
    nvinfer1::PluginField fields[] = {
        {"experts_num", &expertsNumValue, nvinfer1::PluginFieldType::kINT32, 1},
        {"experts_topk", &expertsTopKValue, nvinfer1::PluginFieldType::kINT32, 1},
        {"experts_gate_alpha", gateAlpha.data(), nvinfer1::PluginFieldType::kFLOAT32, 
            static_cast<int32_t>(gateAlpha.size())},
        {"experts_up_alpha", upAlpha.data(), nvinfer1::PluginFieldType::kFLOAT32, 
            static_cast<int32_t>(upAlpha.size())},
        {"experts_down_alpha", downAlpha.data(), nvinfer1::PluginFieldType::kFLOAT32, 
            static_cast<int32_t>(downAlpha.size())}
    };
    
    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = 5;
    fc.fields = fields;
    
    resources.plugin = plugin_creator->createPlugin("moe_nvfp4_test_plugin", &fc);
    if (!resources.plugin) {
        std::cerr << "Error: Plugin creation failed" << std::endl;
        return false;
    }
    
    std::cout << "\n=== NVFP4 Plugin Created ===" << std::endl;
    std::cout << "Plugin type: " << resources.plugin->getPluginType() << std::endl;
    std::cout << "Plugin version: " << resources.plugin->getPluginVersion() << std::endl;
    
    return true;
}

bool buildNetwork(TrtResources& resources, const MoeNvfp4TestData& testData, Logger& logger) {
    // 创建 builder
    resources.builder = createInferBuilder(logger);
    if (!resources.builder) {
        std::cerr << "Failed to create builder" << std::endl;
        return false;
    }
    
    resources.config = resources.builder->createBuilderConfig();
    resources.config->setFlag(BuilderFlag::kFP16);
    resources.config->setFlag(BuilderFlag::kINT8);
    resources.config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30);
    
    resources.network = resources.builder->createNetworkV2(0);
    
    const MoeNvfp4Parameters& params = testData.params;
    ITensor* hiddenState = resources.network->addInput(
        "hidden_state", DataType::kHALF, Dims3{params.batchSize, params.seqLen, params.hiddenSize});

    ITensor* routerWeight = resources.network->addInput(
        "router_weight", DataType::kHALF, Dims2{params.hiddenSize, params.expertsNum});

    Dims routerBiasDims{};
    routerBiasDims.nbDims = 1;
    routerBiasDims.d[0] = params.expertsNum;
    ITensor* routerBias = resources.network->addInput(
        "router_bias", DataType::kHALF, routerBiasDims);

    ITensor* gateQWeight = resources.network->addInput(
        "gate_qweight", DataType::kINT8, 
        Dims3{params.expertsNum, params.intermediateSize, params.hiddenHalf});
    
    auto [gateScaleRows, gateScaleCols] = getScalePadShape(params.intermediateSize, params.hiddenSize);
    ITensor* gateQScales = resources.network->addInput(
        "gate_qscales", DataType::kINT8, 
        Dims3{params.expertsNum, gateScaleRows, gateScaleCols});

    Dims scale1d{};
    scale1d.nbDims = 1;
    scale1d.d[0] = params.expertsNum;
    ITensor* gateInputGlobalScale = resources.network->addInput(
        "gate_input_global_scale", DataType::kFLOAT, scale1d);

    ITensor* gateWeightGlobalScale = resources.network->addInput(
        "gate_weight_global_scale", DataType::kFLOAT, scale1d);

    ITensor* upQWeight = resources.network->addInput(
        "up_qweight", DataType::kINT8, 
        Dims3{params.expertsNum, params.intermediateSize, params.hiddenHalf});

    ITensor* upQScales = resources.network->addInput(
        "up_qscales", DataType::kINT8, 
        Dims3{params.expertsNum, gateScaleRows, gateScaleCols});
    
    ITensor* upInputGlobalScale = resources.network->addInput(
        "up_input_global_scale", DataType::kFLOAT, scale1d);
    
    ITensor* upWeightGlobalScale = resources.network->addInput(
        "up_weight_global_scale", DataType::kFLOAT, scale1d);
    
    ITensor* downQWeight = resources.network->addInput(
        "down_qweight", DataType::kINT8, 
        Dims3{params.expertsNum, params.hiddenSize, params.intermediateHalf});
    
    auto [downScaleRows, downScaleCols] = getScalePadShape(params.hiddenSize, params.intermediateSize);
    ITensor* downQScales = resources.network->addInput(
        "down_qscales", DataType::kINT8, 
        Dims3{params.expertsNum, downScaleRows, downScaleCols});
    
    ITensor* downInputGlobalScale = resources.network->addInput(
        "down_input_global_scale", DataType::kFLOAT, scale1d);
    
    ITensor* downWeightGlobalScale = resources.network->addInput(
        "down_weight_global_scale", DataType::kFLOAT, scale1d);
    
    auto setInt8Range = [](ITensor* tensor, float minVal, float maxVal) {
        tensor->setDynamicRange(minVal, maxVal);
    };

    setInt8Range(gateQWeight, -128.0f, 127.0f);
    setInt8Range(gateQScales, -128.0f, 127.0f);
    setInt8Range(upQWeight, -128.0f, 127.0f);
    setInt8Range(upQScales, -128.0f, 127.0f);
    setInt8Range(downQWeight, -128.0f, 127.0f);
    setInt8Range(downQScales, -128.0f, 127.0f);
    
    ITensor* pluginInputs[] = {
        hiddenState,
        routerWeight,
        routerBias,
        gateQWeight,
        gateQScales,
        gateInputGlobalScale,
        gateWeightGlobalScale,
        upQWeight,
        upQScales,
        upInputGlobalScale,
        upWeightGlobalScale,
        downQWeight,
        downQScales,
        downInputGlobalScale,
        downWeightGlobalScale
    };
    
    // 添加插件层
    IPluginV2Layer* pluginLayer = resources.network->addPluginV2(pluginInputs, 15, *resources.plugin);
    if (!pluginLayer) {
        std::cerr << "Failed to add NVFP4 plugin layer to network" << std::endl;
        return false;
    }
    pluginLayer->setName("moe_nvfp4_plugin_layer");
    
    // 标记输出
    ITensor* outputTensor = pluginLayer->getOutput(0);
    outputTensor->setName("moe_out");
    resources.network->markOutput(*outputTensor);
    resources.network->getOutput(0)->setType(DataType::kHALF);
    
    // 打印网络输入输出
    for (int i = 0; i < resources.network->getNbInputs(); ++i) {
        std::cout << "[NET INPUT]  " << resources.network->getInput(i)->getName() << std::endl;
    }
    for (int i = 0; i < resources.network->getNbOutputs(); ++i) {
        std::cout << "[NET OUTPUT] " << resources.network->getOutput(i)->getName() << std::endl;
    }
    
    return true;
}

bool buildEngine(TrtResources& resources) {
    std::cout << "\n=== Building TensorRT Engine ===" << std::endl;
    resources.serializedEngine = resources.builder->buildSerializedNetwork(*resources.network, *resources.config);
    if (!resources.serializedEngine) {
        std::cerr << "Failed to build serialized network" << std::endl;
        return false;
    }
    
    resources.runtime = createInferRuntime(*static_cast<Logger*>(resources.builder->getLogger()));
    resources.engine = resources.runtime->deserializeCudaEngine(
        resources.serializedEngine->data(), 
        resources.serializedEngine->size()
    );
    if (!resources.engine) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }
    
    resources.context = resources.engine->createExecutionContext();
    if (!resources.context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }

    std::cout << "\n=== Engine IO tensor names ===" << std::endl;
    for (int i = 0; i < resources.engine->getNbIOTensors(); ++i) {
        const char* name = resources.engine->getIOTensorName(i);
        auto mode = resources.engine->getTensorIOMode(name);
        std::cout << "  " << (mode == TensorIOMode::kINPUT ? "IN " : "OUT")
                  << "  '" << name << "'" << std::endl;
    }
    
    return true;
}

bool allocateWorkspace(TrtResources& resources, const MoeNvfp4TestData& testData,
                       nvinfer1::PluginTensorDesc const* inputDescs) {
    resources.workspaceSize = static_cast<trt_edgellm::plugins::MoeNvfp4Plugin*>(resources.plugin)
        ->getWorkspaceSize(inputDescs, 15, nullptr, 1);
    
    if (resources.workspaceSize > 0) {
        CUDA_CHECK(cudaMalloc(&resources.workspace, resources.workspaceSize));
        std::cout << "Allocated workspace: " << resources.workspaceSize << " bytes" << std::endl;
    }
    return true;
}


void bindDevicePointers(TrtResources& resources, const MoeNvfp4TestData& testData) {
    resources.devicePointers.resize(16, nullptr);
    
    resources.devicePointers[0]  = const_cast<void*>(testData.hiddenState.rawPointer());
    resources.devicePointers[1]  = const_cast<void*>(testData.routerWeight.rawPointer());
    resources.devicePointers[2]  = const_cast<void*>(testData.routerBias.rawPointer());
    resources.devicePointers[3]  = const_cast<void*>(testData.gateQWeight.rawPointer());
    resources.devicePointers[4]  = const_cast<void*>(testData.gateQScales.rawPointer());
    resources.devicePointers[5]  = const_cast<void*>(testData.gateInputGlobalScale.rawPointer());
    resources.devicePointers[6]  = const_cast<void*>(testData.gateWeightGlobalScale.rawPointer());
    resources.devicePointers[7]  = const_cast<void*>(testData.upQWeight.rawPointer());
    resources.devicePointers[8]  = const_cast<void*>(testData.upQScales.rawPointer());
    resources.devicePointers[9]  = const_cast<void*>(testData.upInputGlobalScale.rawPointer());
    resources.devicePointers[10] = const_cast<void*>(testData.upWeightGlobalScale.rawPointer());
    resources.devicePointers[11] = const_cast<void*>(testData.downQWeight.rawPointer());
    resources.devicePointers[12] = const_cast<void*>(testData.downQScales.rawPointer());
    resources.devicePointers[13] = const_cast<void*>(testData.downInputGlobalScale.rawPointer());
    resources.devicePointers[14] = const_cast<void*>(testData.downWeightGlobalScale.rawPointer());
    
    size_t outputSize = testData.params.batchSize * testData.params.seqLen * 
                        testData.params.hiddenSize * sizeof(half);
    void* outputPtr = nullptr;
    CUDA_CHECK(cudaMalloc(&outputPtr, outputSize));
    resources.devicePointers[15] = outputPtr;
    
    // 绑定所有输入输出
    auto bindAndCheck = [&](const char* name, void* ptr) {
        bool ok = resources.context->setTensorAddress(name, ptr);
        std::cout << "[BIND] '" << name << "' -> " << ptr
                  << (ok ? "  OK" : "  FAILED") << std::endl;
        if (!ok) {
            std::cerr << "ERROR: binding '" << name << "' failed!\n";
            std::exit(EXIT_FAILURE);
        }
    };
    
    bindAndCheck("hidden_state",               resources.devicePointers[0]);
    bindAndCheck("router_weight",              resources.devicePointers[1]);
    bindAndCheck("router_bias",                resources.devicePointers[2]);
    bindAndCheck("gate_qweight",               resources.devicePointers[3]);
    bindAndCheck("gate_qscales",               resources.devicePointers[4]);
    bindAndCheck("gate_input_global_scale",    resources.devicePointers[5]);
    bindAndCheck("gate_weight_global_scale",   resources.devicePointers[6]);
    bindAndCheck("up_qweight",                 resources.devicePointers[7]);
    bindAndCheck("up_qscales",                 resources.devicePointers[8]);
    bindAndCheck("up_input_global_scale",      resources.devicePointers[9]);
    bindAndCheck("up_weight_global_scale",     resources.devicePointers[10]);
    bindAndCheck("down_qweight",               resources.devicePointers[11]);
    bindAndCheck("down_qscales",               resources.devicePointers[12]);
    bindAndCheck("down_input_global_scale",    resources.devicePointers[13]);
    bindAndCheck("down_weight_global_scale",   resources.devicePointers[14]);
    bindAndCheck("moe_out",                    resources.devicePointers[15]);
}


bool runInference(TrtResources& resources) {
    std::cout << "\n=== Running Inference (enqueueV3) ===" << std::endl;
    
    bool status = resources.context->enqueueV3(resources.stream);
    if (!status) {
        std::cerr << "enqueueV3 failed\n";
        return false;
    }
    
    CUDA_CHECK(cudaStreamSynchronize(resources.stream));
    std::cout << "Inference completed successfully" << std::endl;
    return true;
}


void printOutput(const std::vector<half> &out, const std::vector<half> &ref, int n) { 
    for (int i = 0; i < n; i++) {
        std::cout << "out[" << i << "] = " << float(out[i]) << ", ref[" << i << "] = " << float(ref[i]) << std::endl;
    }
}

bool validateOutput(TrtResources& resources, const MoeNvfp4TestData& testData) {
    if (testData.referenceOutputs.empty()) {
        std::cout << "\n=== WARNING: No reference output found, skipping validation ===" << std::endl;
        return true;
    }
    
    std::cout << "\n=== Validating Output ===" << std::endl;
    
    const MoeNvfp4Parameters& params = testData.params;
    const trt_edgellm::rt::Tensor& referenceTensor = testData.referenceOutputs[0];
    
    size_t numElements = params.batchSize * params.seqLen * params.hiddenSize;
    size_t refNumElements = referenceTensor.getShape()[0] * 
                            referenceTensor.getShape()[1] * 
                            referenceTensor.getShape()[2];
    
    if (refNumElements != numElements) {
        std::cout << "Output size mismatch: actual=" << numElements 
                  << ", expected=" << refNumElements << std::endl;
        return false;
    }
    
    std::vector<half> h_output(numElements);
    CUDA_CHECK(cudaMemcpyAsync(h_output.data(), resources.devicePointers[15],
                               numElements * sizeof(half), cudaMemcpyDeviceToHost, resources.stream));
    
    std::vector<half> h_reference(refNumElements);
    if (referenceTensor.getDeviceType() == trt_edgellm::rt::DeviceType::kGPU) {
        CUDA_CHECK(cudaMemcpyAsync(h_reference.data(), referenceTensor.rawPointer(),
                                   refNumElements * sizeof(half),
                                   cudaMemcpyDeviceToHost, resources.stream));
    } else {
        memcpy(h_reference.data(), referenceTensor.rawPointer(),
               refNumElements * sizeof(half));
    }
    
    CUDA_CHECK(cudaStreamSynchronize(resources.stream));
    
    bool validationPassed = trt_edgellm::rt::compareHalfArrays(
        h_output.data(), h_reference.data(), numElements);
    printOutput(h_output, h_reference, 10);
    
    if (validationPassed) {
        std::cout << "VALIDATION PASSED!" << std::endl;
    } else {
        std::cout << "VALIDATION FAILED!" << std::endl;
    }
    
    return validationPassed;
}


void cleanupResources(TrtResources& resources) {
    std::cout << "\n=== Cleaning up resources ===" << std::endl;
    
    // 释放自己分配的输出内存（其他 Tensor 由 testData 管理）
    if (resources.devicePointers.size() > 15 && resources.devicePointers[15]) {
        cudaFree(resources.devicePointers[15]);
    }
    if (resources.workspace) {
        cudaFree(resources.workspace);
    }
    
    if (resources.context) delete resources.context;
    if (resources.engine) delete resources.engine;
    if (resources.runtime) delete resources.runtime;
    if (resources.serializedEngine) delete resources.serializedEngine;
    if (resources.network) delete resources.network;
    if (resources.config) delete resources.config;
    if (resources.builder) delete resources.builder;
    if (resources.plugin) delete resources.plugin;
    
    if (resources.pluginHandle) dlclose(resources.pluginHandle);
    if (resources.stream) CUDA_CHECK(cudaStreamDestroy(resources.stream));
    
    std::cout << "\n=== NVFP4 MoE Plugin Test completed ===" << std::endl;
}

bool runPerformanceTest(TrtResources& resources) {
    std::cout << "\n=== Performance Testing ===" << std::endl;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 预热运行
    for (int i = 0; i < 10; ++i) {
        resources.context->enqueueV3(resources.stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(resources.stream));

    // 正式计时
    int numRuns = 100;
    CUDA_CHECK(cudaEventRecord(start, resources.stream));
    for (int i = 0; i < numRuns; ++i) {
        resources.context->enqueueV3(resources.stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, resources.stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    std::cout << "Total time for " << numRuns << " runs: " << milliseconds << " ms" << std::endl;
    std::cout << "Average time per run: " << (milliseconds / numRuns) << " ms" << std::endl;
    std::cout << "Throughput: " << (1000.0 / (milliseconds / numRuns)) << " inferences/sec" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return true;
}

int main() {
    Logger logger;
    TrtResources resources;
    MoeNvfp4TestData testData;
    
    if (!initializeCudaStream(resources.stream)) {
        return -1;
    }

    if (!loadMoeNvfp4TestData(testData, resources.stream)) {
        cleanupResources(resources);
        return -1;
    }

    if (!createNvfp4Plugin(resources, testData, logger)) {
        cleanupResources(resources);
        return -1;
    }

    if (!buildNetwork(resources, testData, logger)) {
        cleanupResources(resources);
        return -1;
    }

    if (!buildEngine(resources)) {
        cleanupResources(resources);
        return -1;
    }

    std::vector<PluginTensorDesc> inputDescs;
    for (int i = 0; i < resources.network->getNbInputs(); ++i) {
        PluginTensorDesc desc;
        desc.dims = resources.network->getInput(i)->getDimensions();
        desc.type = resources.network->getInput(i)->getType();
        desc.format = TensorFormat::kLINEAR;
        inputDescs.push_back(desc);
    }

    if (!allocateWorkspace(resources, testData, inputDescs.data())) {
        cleanupResources(resources);
        return -1;
    }

    bindDevicePointers(resources, testData);

    if (!runInference(resources)) {
        cleanupResources(resources);
        return -1;
    }

    bool validationPassed = validateOutput(resources, testData);

    // runPerformanceTest(resources);

    cleanupResources(resources);
    return validationPassed ? 0 : 1;
}
