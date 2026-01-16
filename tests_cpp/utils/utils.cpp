#include "utils/utils.h"
#include "common/logger.h"
#include "kernels/moeKernels/moeFp16Kernels.h"
#include "plugins/utils/pluginUtils.h"


namespace trt_edgellm {
namespace rt {

void printTensorInfo(const std::vector<Tensor>& tensors) {
    for (const auto& tensor : tensors) {
        std::cout << "Tensor Name: " << tensor.getName() << "\n";
        std::cout << "Shape: ";
        auto shape = tensor.getShape();
        auto dims = shape.getNumDims();
        for (int i = 0; i < dims; ++i) {
            std::cout << shape[i] << " ";
        }
        std::cout << "\nData Type: " << getDataTypeString(tensor.getDataType()) << "\n";
        std::cout << "Device Type: " << (tensor.getDeviceType() == DeviceType::kGPU ? "GPU" : "CPU") << "\n";
        std::cout << "------------------------\n";
    }
}

bool compareTensors(const Tensor& actual, const Tensor& reference, cudaStream_t stream, float tolerance) {
    if (actual.getDataType() != reference.getDataType()) {
        LOG_ERROR("Data type mismatch: actual=%s, reference=%s",
                 getDataTypeString(actual.getDataType()), 
                 getDataTypeString(reference.getDataType()));
        return false;
    }
    
    auto actualShape = actual.getShape();
    auto referenceShape = reference.getShape();
    
    if (actualShape.getNumDims() != referenceShape.getNumDims()) {
        LOG_ERROR("Dimension count mismatch: actual=%d, reference=%d",
                 actualShape.getNumDims(), referenceShape.getNumDims());
        return false;
    }
    
    size_t elementCount = 1;
    for (int d = 0; d < actualShape.getNumDims(); d++) {
        if (actualShape[d] != referenceShape[d]) {
            LOG_ERROR("Shape mismatch at dimension %d: actual=%lld, reference=%lld",
                     d, actualShape[d], referenceShape[d]);
            return false;
        }
        elementCount *= actualShape[d];
    }
    
    if (actual.getDataType() != nvinfer1::DataType::kHALF) {
        LOG_ERROR("Only half precision floating point comparison is supported");
        return false;
    }
    
    LOG_INFO("Comparing tensors with %zu elements", elementCount);
    
    std::vector<half> actualDataCPU(elementCount);
    std::vector<half> referenceDataCPU(elementCount);

    auto copyToCPU = [stream](std::vector<half>& cpuBuffer, const half* gpuPtr, size_t count) -> bool {
        cudaError_t cudaStatus = cudaMemcpyAsync(
            cpuBuffer.data(),
            gpuPtr,
            count * sizeof(half),
            cudaMemcpyDeviceToHost,
            stream
        );
        return cudaStatus == cudaSuccess;
    };
    
    if (actual.getDeviceType() == DeviceType::kGPU) {
        if (!copyToCPU(actualDataCPU, actual.dataPointer<half>(), elementCount)) {
            LOG_ERROR("Failed to copy actual tensor data from GPU to CPU");
            return false;
        }
    } else {
        const half* actualData = actual.dataPointer<half>();
        std::copy(actualData, actualData + elementCount, actualDataCPU.begin());
    }
    
    if (reference.getDeviceType() == DeviceType::kGPU) {
        if (!copyToCPU(referenceDataCPU, reference.dataPointer<half>(), elementCount)) {
            LOG_ERROR("Failed to copy reference tensor data from GPU to CPU");
            return false;
        }
    } else {
        const half* referenceData = reference.dataPointer<half>();
        std::copy(referenceData, referenceData + elementCount, referenceDataCPU.begin());
    }
    
    cudaStreamSynchronize(stream);
    
    return compareHalfArrays(actualDataCPU.data(), referenceDataCPU.data(), elementCount, tolerance);
}

bool compareHalfArrays(const half* actual, const half* expected, size_t numElements, float tolerance) {
    bool passed = true;
    size_t mismatchCount = 0;
    float maxDiff = 0.0f;
    float totalDiff = 0.0f;
    
    for (size_t i = 0; i < numElements; ++i) {
        float actualVal = __half2float(actual[i]);
        float expectedVal = __half2float(expected[i]);
        float diff = std::abs(actualVal - expectedVal);
        
        if (diff > tolerance) {
            if (mismatchCount < 5) {
                LOG_WARNING("Mismatch at position %zu: actual=%f, reference=%f, diff=%f", 
                          i, actualVal, expectedVal, diff);
            }
            mismatchCount++;
            if (diff > maxDiff) maxDiff = diff;
        }
        totalDiff += diff;
    }
    
    if (mismatchCount > 0) {
        LOG_ERROR("Found %zu mismatched elements out of %zu (%.2f%%)",
                 mismatchCount, numElements, (100.0f * mismatchCount / numElements));
        LOG_ERROR("Max difference: %f, Average difference: %f", 
                 maxDiff, totalDiff / numElements);
        passed = false;
    } else {
        LOG_INFO("Tensors match perfectly! All %zu elements within tolerance %f", 
                numElements, tolerance);
    }
    
    return passed;
}


bool loadSafetensors(const std::filesystem::path& filepath, std::vector<Tensor>& tensors, cudaStream_t stream) {
    if (!std::filesystem::exists(filepath)) {
        LOG_ERROR("File does not exist: %s", filepath.string().c_str());
        return false;
    }
    
    return safetensors::loadSafetensors(filepath.string(), tensors, stream);
}

bool saveSafetensors(const std::filesystem::path& filepath, const std::vector<Tensor>& tensors, cudaStream_t stream) {
    bool success = safetensors::saveSafetensors(filepath.string(), tensors, stream);
    if (success) {
        LOG_INFO("Saved tensors to: %s", filepath.string().c_str());
    } else {
        LOG_ERROR("Failed to save tensors to: %s", filepath.string().c_str());
    }
    return success;
}


bool loadMoeData(std::vector<Tensor>& moeInputTensors,
                 std::vector<Tensor>& moeWeightsTensors, 
                 cudaStream_t stream) {

    std::filesystem::path moeInputPath = getSafetensorPath("moe_input.safetensors");
    std::filesystem::path moeWeightPath = getSafetensorPath("moe_weights.safetensors");

    LOG_INFO("Loading moe input data from: %s", moeInputPath.string().c_str());
    LOG_INFO("Loading moe weight data from: %s", moeWeightPath.string().c_str());

    if (!loadSafetensors(moeInputPath, moeInputTensors, stream)) {
        LOG_ERROR("Failed to load moe input data");
        return false;
    }
    
    if (!loadSafetensors(moeWeightPath, moeWeightsTensors, stream)) {
        LOG_ERROR("Failed to load moe weight data");
        return false;
    }
    
    return true;
}

bool loadMoeIntermediateRes(std::vector<Tensor>& intermediateResTensors, cudaStream_t stream) {
    std::filesystem::path intermediateResPath = getSafetensorPath("moe_intermediate_results.safetensors");
    return loadSafetensors(intermediateResPath, intermediateResTensors, stream);
}

bool loadReferenceOutput(std::vector<trt_edgellm::rt::Tensor>& referenceTensors, cudaStream_t stream) {
    std::filesystem::path referencePath = getSafetensorPath("moe_output_ref.safetensors");
    
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

bool saveOutputTensor(std::vector<Tensor>& outputTensors, cudaStream_t stream) {
    std::filesystem::path outputPath = getSafetensorPath("moe_output.safetensors");
    return saveSafetensors(outputPath, outputTensors, stream);
}

} // namespace rt
} // namespace trt_edgellm