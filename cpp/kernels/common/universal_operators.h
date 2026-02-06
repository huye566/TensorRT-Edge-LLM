#ifndef UNIVERSAL_OPERATORS_H
#define UNIVERSAL_OPERATORS_H

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace trt_edgellm {
namespace kernel {

bool apply_silu_scalar(void* data, int num_elements,
    nvinfer1::DataType output_type, cudaStream_t stream);
bool apply_add_bias_scalar(void* data, const void* bias, int m, int n,
    nvinfer1::DataType output_type, cudaStream_t stream);
bool apply_silu_vec(void* data, int num,
    nvinfer1::DataType dtype, cudaStream_t stream);
bool apply_add_bias_vec(void* data, const void* bias,
    int m, int n, nvinfer1::DataType dtype, cudaStream_t stream);
bool apply_silu_vec_optimized(void* data, int num,
    nvinfer1::DataType dtype, cudaStream_t stream);
bool apply_add_bias_vec_optimized(void* data, const void* bias, int m, int n,
    nvinfer1::DataType dtype, cudaStream_t stream);
bool apply_add_bias_silu_fused(void* data, const void* bias, int m, int n,
    nvinfer1::DataType dtype, cudaStream_t stream);
}
}

#endif // UNIVERSAL_OPERATORS_H