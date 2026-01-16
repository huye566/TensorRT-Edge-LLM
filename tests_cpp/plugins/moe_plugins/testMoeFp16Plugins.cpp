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

using namespace nvinfer1;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

struct MoeParameters {
    int batchSize;
    int seqLen;
    int hiddenSize;
    int intermediateSize;
    int expertsNum;
    int expertsTopK;
    
    MoeParameters() : batchSize(0), seqLen(0), hiddenSize(0), 
                     intermediateSize(0), expertsNum(0), expertsTopK(1) {}
};

struct MoeTestData {
    std::vector<trt_edgellm::rt::Tensor> inputTensors;
    std::vector<trt_edgellm::rt::Tensor> weightTensors;
    std::vector<trt_edgellm::rt::Tensor> referenceTensors;
    
    const trt_edgellm::rt::Tensor* hiddenStateTensor = nullptr;
    const trt_edgellm::rt::Tensor* routerWeightTensor = nullptr;
    const trt_edgellm::rt::Tensor* routerBiasTensor = nullptr;
    const trt_edgellm::rt::Tensor* gateProjWeightTensor = nullptr;
    const trt_edgellm::rt::Tensor* upProjWeightTensor = nullptr;
    const trt_edgellm::rt::Tensor* downProjWeightTensor = nullptr;
    
    MoeParameters params;
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
    
    std::vector<void*> devicePointers;
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

bool loadTestData(MoeTestData& testData, cudaStream_t stream) {
    if (!trt_edgellm::rt::loadMoeData(testData.inputTensors, testData.weightTensors, stream)) {
        std::cerr << "Failed to load MoE test data" << std::endl;
        return false;
    }

    for (const auto& tensor : testData.inputTensors) {
        if (tensor.getName() == "hidden_state") {
            testData.hiddenStateTensor = &tensor;
            break;
        }
    }
    
    for (const auto& tensor : testData.weightTensors) {
        std::string name = tensor.getName();
        if (name.find("router_weight") != std::string::npos) {
            testData.routerWeightTensor = &tensor;
        } else if (name.find("router_bias") != std::string::npos) {
            testData.routerBiasTensor = &tensor;
        } else if (name.find("gate_proj") != std::string::npos) {
            testData.gateProjWeightTensor = &tensor;
        } else if (name.find("up_proj") != std::string::npos) {
            testData.upProjWeightTensor = &tensor;
        } else if (name.find("down_proj") != std::string::npos) {
            testData.downProjWeightTensor = &tensor;
        }
    }

    if (!testData.hiddenStateTensor || !testData.routerWeightTensor || !testData.routerBiasTensor || 
        !testData.gateProjWeightTensor || !testData.upProjWeightTensor || !testData.downProjWeightTensor) {
        std::cerr << "Failed to find all required tensors" << std::endl;
        return false;
    }
    
    auto hiddenStateShape = testData.hiddenStateTensor->getShape();
    auto routerWeightShape = testData.routerWeightTensor->getShape();
    auto gateProjWeightShape = testData.gateProjWeightTensor->getShape();
    
    testData.params.batchSize = hiddenStateShape[0];
    testData.params.seqLen = hiddenStateShape[1];
    testData.params.hiddenSize = hiddenStateShape[2];
    testData.params.expertsNum = routerWeightShape[1];
    testData.params.intermediateSize = gateProjWeightShape[2];
    testData.params.expertsTopK = 1;
    
    std::cout << "\n=== MoE Parameters ===" << std::endl;
    std::cout << "batch_size: " << testData.params.batchSize << std::endl;
    std::cout << "seq_len: " << testData.params.seqLen << std::endl;
    std::cout << "hidden_size: " << testData.params.hiddenSize << std::endl;
    std::cout << "intermediate_size: " << testData.params.intermediateSize << std::endl;
    std::cout << "experts_num: " << testData.params.expertsNum << std::endl;
    std::cout << "experts_topk: " << testData.params.expertsTopK << std::endl;

    if (!trt_edgellm::rt::loadReferenceOutput(testData.referenceTensors, stream)) {
        std::cout << "Warning: Failed to load reference output, validation will be skipped" << std::endl;
    }
    
    return true;
}

bool createPlugin(TrtResources& resources, const MoeParameters& params, Logger& logger) {
    // 加载plugin库
    resources.pluginHandle = dlopen("./build/libNvInfer_edgellm_plugin.so", RTLD_LAZY);
    if (!resources.pluginHandle) {
        std::cerr << "Error loading plugin: " << dlerror() << std::endl;
        return false;
    }

    initLibNvInferPlugins(&logger, "");
    
    // 创建plugin
    auto plugin_creator = getPluginRegistry()->getPluginCreator("MoEFp16Plugin", "1");
    if (!plugin_creator) {
        std::cerr << "Error: MoEFp16Plugin creator not found" << std::endl;
        return false;
    }
    
    int32_t expertsNumValue = params.expertsNum;
    int32_t expertsTopKValue = params.expertsTopK;
    
    nvinfer1::PluginField expertsNumField{"experts_num", &expertsNumValue, 
                                          nvinfer1::PluginFieldType::kINT32, 1};
    nvinfer1::PluginField expertsTopKField{"experts_topk", &expertsTopKValue, 
                                           nvinfer1::PluginFieldType::kINT32, 1};
    
    std::vector<nvinfer1::PluginField> fields = {expertsNumField, expertsTopKField};
    nvinfer1::PluginFieldCollection fc{};
    fc.nbFields = fields.size();
    fc.fields = fields.data();
    
    resources.plugin = plugin_creator->createPlugin("moe_test_plugin", &fc);
    if (!resources.plugin) {
        std::cerr << "Error: Plugin creation failed" << std::endl;
        return false;
    }
    
    std::cout << "\n=== Plugin Created ===" << std::endl;
    std::cout << "Plugin type: " << resources.plugin->getPluginType() << std::endl;
    std::cout << "Plugin version: " << resources.plugin->getPluginVersion() << std::endl;
    
    return true;
}

bool buildNetwork(TrtResources& resources, const MoeTestData& testData, Logger& logger) {
    // 创建builder
    resources.builder = createInferBuilder(logger);
    if (!resources.builder) {
        std::cerr << "Failed to create builder" << std::endl;
        return false;
    }
    
    resources.config = resources.builder->createBuilderConfig();
    resources.config->setFlag(BuilderFlag::kFP16);
    resources.config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30);
    
    resources.network = resources.builder->createNetworkV2(0);
    
    // 创建输入张量
    const MoeParameters& params = testData.params;
    
    ITensor* hiddenStateITensor = resources.network->addInput(
        "hidden_state", 
        DataType::kHALF, 
        Dims3{params.batchSize, params.seqLen, params.hiddenSize}
    );
    
    ITensor* routerWeightITensor = resources.network->addInput(
        "router_weight", 
        DataType::kHALF, 
        Dims2{params.hiddenSize, params.expertsNum}
    );

    nvinfer1::Dims routerBiasDims;
    routerBiasDims.nbDims = 1;
    routerBiasDims.d[0] = params.expertsNum;
    ITensor* routerBiasITensor = resources.network->addInput(
        "router_bias", 
        DataType::kHALF, 
        routerBiasDims
    );

    ITensor* gateProjITensor = resources.network->addInput(
        "gate_proj_weight", 
        DataType::kHALF, 
        Dims3{params.expertsNum, params.hiddenSize, params.intermediateSize}
    );
    
    ITensor* upProjITensor = resources.network->addInput(
        "up_proj_weight", 
        DataType::kHALF, 
        Dims3{params.expertsNum, params.hiddenSize, params.intermediateSize}
    );
    
    ITensor* downProjITensor = resources.network->addInput(
        "down_proj_weight", 
        DataType::kHALF, 
        Dims3{params.expertsNum, params.intermediateSize, params.hiddenSize}
    );
    
    // 添加plugin层
    ITensor* pluginInputs[] = {
        hiddenStateITensor,
        routerWeightITensor,
        routerBiasITensor,
        gateProjITensor,
        upProjITensor,
        downProjITensor,
    };

    IPluginV2Layer* pluginLayer = resources.network->addPluginV2(pluginInputs, 6, *resources.plugin);
    if (!pluginLayer) {
        std::cerr << "Failed to add plugin layer to network" << std::endl;
        return false;
    }
    
    pluginLayer->setName("moe_plugin_layer");
    ITensor* outputTensor = pluginLayer->getOutput(0);
    outputTensor->setName("moe_out");
    resources.network->markOutput(*outputTensor);
    resources.network->getOutput(0)->setType(DataType::kHALF);

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

void bindDevicePointer(TrtResources& resources) {
    auto bindAndCheck = [&](const char* name, void* ptr) {
        bool ok = resources.context->setTensorAddress(name, ptr);
        std::cout << "[BIND] '" << name << "' -> " << ptr
                << (ok ? "  OK" : "  FAILED") << std::endl;
        if (!ok) {
            std::cerr << "ERROR: binding '" << name << "' failed!\n";
            std::exit(EXIT_FAILURE);
        }
    };

    bindAndCheck("hidden_state",     resources.devicePointers[0]);
    bindAndCheck("router_weight",    resources.devicePointers[1]);
    bindAndCheck("router_bias",      resources.devicePointers[2]);
    bindAndCheck("gate_proj_weight", resources.devicePointers[3]);
    bindAndCheck("up_proj_weight",   resources.devicePointers[4]);
    bindAndCheck("down_proj_weight", resources.devicePointers[5]);
    bindAndCheck("moe_out",          resources.devicePointers[6]);
}

bool allocateAndCopyData(TrtResources& resources, const MoeTestData& testData) {
    const MoeParameters& params = testData.params;
    
    // 计算各张量的大小
    size_t hiddenStateSize = params.batchSize * params.seqLen * params.hiddenSize * sizeof(half);
    size_t routerWeightSize = params.hiddenSize * params.expertsNum * sizeof(half);
    size_t routerBiasSize = params.expertsNum * sizeof(half);
    size_t gateProjSize = params.expertsNum * params.hiddenSize * params.intermediateSize * sizeof(half);
    size_t upProjSize = params.expertsNum * params.hiddenSize * params.intermediateSize * sizeof(half);
    size_t downProjSize = params.expertsNum * params.intermediateSize * params.hiddenSize * sizeof(half);
    size_t outputSize = hiddenStateSize; // 输出与hidden_state形状相同
    std::cout << "hiddenStateSize: " << hiddenStateSize << " bytes" << std::endl;
    std::cout << "routerWeightSize: " << routerWeightSize << " bytes" << std::endl;
    std::cout << "routerBiasSize: " << routerBiasSize << " bytes" << std::endl;
    std::cout << "gateProjSize: " << gateProjSize << " bytes" << std::endl;
    std::cout << "upProjSize: " << upProjSize << " bytes" << std::endl;
    std::cout << "downProjSize: " << downProjSize << " bytes" << std::endl;
    std::cout << "outputSize: " << outputSize << " bytes" << std::endl;
    
    for (void* ptr : resources.devicePointers) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    resources.devicePointers.clear();
    resources.devicePointers.resize(7, nullptr);
    CUDA_CHECK(cudaMalloc(&resources.devicePointers[0], hiddenStateSize));
    CUDA_CHECK(cudaMalloc(&resources.devicePointers[1], routerWeightSize));
    CUDA_CHECK(cudaMalloc(&resources.devicePointers[2], routerBiasSize));
    CUDA_CHECK(cudaMalloc(&resources.devicePointers[3], gateProjSize));
    CUDA_CHECK(cudaMalloc(&resources.devicePointers[4], upProjSize));
    CUDA_CHECK(cudaMalloc(&resources.devicePointers[5], downProjSize));
    CUDA_CHECK(cudaMalloc(&resources.devicePointers[6], outputSize));
    for(size_t i = 0; i < resources.devicePointers.size(); ++i) {
        std::cout << "Device Pointer [" << i << "]: " << resources.devicePointers[i] << std::endl;
    }

    auto copyTensorToDevice = [&](void* devicePtr, const trt_edgellm::rt::Tensor* tensor, size_t size) {
        if (tensor->getDeviceType() == trt_edgellm::rt::DeviceType::kGPU) {
            CUDA_CHECK(cudaMemcpyAsync(devicePtr, tensor->rawPointer(), 
                           size, cudaMemcpyDeviceToDevice, resources.stream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(devicePtr, tensor->rawPointer(), 
                           size, cudaMemcpyHostToDevice, resources.stream));
        }
    };
    
    // 拷贝数据到设备
    copyTensorToDevice(resources.devicePointers[0], testData.hiddenStateTensor, hiddenStateSize);
    copyTensorToDevice(resources.devicePointers[1], testData.routerWeightTensor, routerWeightSize);
    copyTensorToDevice(resources.devicePointers[2], testData.routerBiasTensor, routerBiasSize);
    copyTensorToDevice(resources.devicePointers[3], testData.gateProjWeightTensor, gateProjSize);
    copyTensorToDevice(resources.devicePointers[4], testData.upProjWeightTensor, upProjSize);
    copyTensorToDevice(resources.devicePointers[5], testData.downProjWeightTensor, downProjSize);

    CUDA_CHECK(cudaStreamSynchronize(resources.stream));
    // bindDevicePointer(resources);

    return true;
}

bool allocateAndCopyDataV2(TrtResources& resources, const MoeTestData& testData) {
    const MoeParameters& params = testData.params;
    size_t hiddenStateSize = params.batchSize * params.seqLen * params.hiddenSize * sizeof(half);
    size_t outputSize = hiddenStateSize;
    
    resources.devicePointers.resize(7);
    CUDA_CHECK(cudaMalloc(&resources.devicePointers[6], outputSize));
    resources.devicePointers[0] = const_cast<void*>(testData.hiddenStateTensor->rawPointer());
    resources.devicePointers[1] = const_cast<void*>(testData.routerWeightTensor->rawPointer());
    resources.devicePointers[2] = const_cast<void*>(testData.routerBiasTensor->rawPointer());
    resources.devicePointers[3] = const_cast<void*>(testData.gateProjWeightTensor->rawPointer());
    resources.devicePointers[4] = const_cast<void*>(testData.upProjWeightTensor->rawPointer());
    resources.devicePointers[5] = const_cast<void*>(testData.downProjWeightTensor->rawPointer());
    CUDA_CHECK(cudaStreamSynchronize(resources.stream));

    bindDevicePointer(resources);
    return true;
}

bool runInference(TrtResources& resources) {
    std::cout << "\n=== Running Inference ===" << std::endl;
    
    void* bindings[] = {resources.devicePointers[0], 
                       resources.devicePointers[1],
                       resources.devicePointers[2],
                       resources.devicePointers[3],
                       resources.devicePointers[4],
                       resources.devicePointers[5],
                       resources.devicePointers[6]};
    bool success = resources.context->executeV2(bindings);
    // bool success = resources.context->executeV2(resources.devicePointers.data());
    if (!success) {
        std::cerr << "Failed to execute inference" << std::endl;
        return false;
    }
    
    std::cout << "Inference completed successfully" << std::endl;
    return true;
}

bool runInferenceV3(TrtResources& resources) {
    std::cout << "\n=== Running Inference ===" << std::endl;
    
    bool status = resources.context->enqueueV3(resources.stream);
    if (!status) {
        std::cerr << "enqueueV3 failed\n";
        return false;
    }

    CUDA_CHECK(cudaStreamSynchronize(resources.stream));
    std::cout << "Inference completed successfully" << std::endl;
    return true;
}

bool validateOutput(TrtResources& resources, const MoeTestData& testData) {
    if (testData.referenceTensors.empty()) {
        std::cout << "\n=== WARNING: No reference output found, skipping validation ===" << std::endl;
        return true;
    }
    
    std::cout << "\n=== Validating Output ===" << std::endl;
    
    const MoeParameters& params = testData.params;
    const trt_edgellm::rt::Tensor& referenceTensor = testData.referenceTensors[0];
    
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
    CUDA_CHECK(cudaMemcpyAsync(h_output.data(), resources.devicePointers[6], 
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
    
    if (validationPassed) {
        std::cout << "VALIDATION PASSED!" << std::endl;
    } else {
        std::cout << "VALIDATION FAILED!" << std::endl;
    }
    
    return validationPassed;
}

bool runPerformanceTest(TrtResources& resources) {
    std::cout << "\n=== Performance Testing ===" << std::endl;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 预热运行
    for (int i = 0; i < 10; ++i) {
        resources.context->executeV2(resources.devicePointers.data());
    }
    CUDA_CHECK(cudaStreamSynchronize(resources.stream));

    // 正式计时
    int numRuns = 100;
    CUDA_CHECK(cudaEventRecord(start, resources.stream));
    for (int i = 0; i < numRuns; ++i) {
        resources.context->executeV2(resources.devicePointers.data());
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

void cleanupResources(TrtResources& resources) {
    std::cout << "\n=== Cleaning up resources ===" << std::endl;

    for (void* ptr : resources.devicePointers) {
        if (ptr) cudaFree(ptr);
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

    std::cout << "\n=== Test completed successfully ===" << std::endl;
}

int main() {
    Logger logger;
    TrtResources resources;
    MoeTestData testData;
    trt_edgellm::rt::setCustomPath("testdata", trt_edgellm::rt::getResourcesPath() / "seq530");

    if (!initializeCudaStream(resources.stream)) {
        return -1;
    }

    if (!loadTestData(testData, resources.stream)) {
        cleanupResources(resources);
        return -1;
    }

    if (!createPlugin(resources, testData.params, logger)) {
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

    if (!allocateAndCopyData(resources, testData)) {
        cleanupResources(resources);
        return -1;
    }

    if (!runInference(resources)) {
        cleanupResources(resources);
        return -1;
    }

    bool validationPassed = validateOutput(resources, testData);

    // runPerformanceTest(resources);
    
    cleanupResources(resources);
    return validationPassed ? 0 : 1;
}