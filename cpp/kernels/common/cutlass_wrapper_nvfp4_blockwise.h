#ifndef CUTLASS_WRAPPER_NVFP4_BLOCKWISE_H
#define CUTLASS_WRAPPER_NVFP4_BLOCKWISE_H

#include <cuda_runtime.h>
#include <NvInfer.h>

namespace trt_edgellm {
namespace kernel {

template <bool kEnableSilu = false, bool kEnableBias = false>
void cutlass_scaled_nvfp4(
    void* output,
    const void* input_a,
    const void* input_b,
    const void* scales_a,
    const void* scales_b,
    const void* bias,
    float alpha,
    int M,
    int N,
    int K,
    cudaStream_t stream,
    nvinfer1::DataType outputType,
    bool use_cached = true
);

} // namespace kernel
} // namespace trt_edgellm

#endif // CUTLASS_WRAPPER_NVFP4_BLOCKWISE_H