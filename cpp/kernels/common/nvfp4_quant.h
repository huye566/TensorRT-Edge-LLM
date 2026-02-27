#ifndef NVFP4_QUANT_H
#define NVFP4_QUANT_H

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>

namespace trt_edgellm {
namespace kernel {

template <typename T>
void invokeFP4Quantization(
    int m,
    int n,
    T const* input,
    float const* SFScale,
    int64_t* output,
    int32_t* SFOuput,
    bool useUE8M0,
    int multiProcessorCount,
    cudaStream_t stream);

void scaled_fp4_quant(
    int m,
    int n,
    void const* input,
    float const* SFScale,
    int64_t* output,
    int32_t* SFOuput,
    cudaStream_t stream,
    nvinfer1::DataType inputType);
}
}

#endif // NVFP4_QUANT_H