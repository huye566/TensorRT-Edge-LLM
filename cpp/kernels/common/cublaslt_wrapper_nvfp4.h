#ifndef CUBLASLT_WRAPPER_NVFP4_H
#define CUBLASLT_WRAPPER_NVFP4_H

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <NvInfer.h>
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>

namespace trt_edgellm {
namespace kernel {

enum class MemoryFormat {
    ROW_MAJOR,
    COL_MAJOR
};

enum class ComputeType {
    FLOAT,
    HALF,
    TFLOAT32,
    FP8_E4M3
};

enum class ScaleMode {
    NONE,
    SCALAR,
    VECTOR,
    BLOCK_SCALE,
    VEC16_UE4M3
};

enum class EpilogueMode {
    NONE,           // 不使用Epilogue
    BIAS,           // GEMM + Bias
    BIAS_RELU,      // GEMM + Bias + ReLU
    BIAS_GELU,      // GEMM + Bias + GELU
    CUSTOM          // 自定义Epilogue（需要后续处理）
};

#define __CUDA_FP8_E4M3__
#define __CUDA_FP4_E2M1__

// NVFP4类型别名（使用NVIDIA的FP4 E2M1格式）
#if defined(__CUDA_FP4_E2M1__)
using nv_fp4_e2m1 = __nv_fp4_e2m1;
#else
// 如果不支持FP4，定义占位符类型
struct nv_fp4_e2m1 {
    uint8_t data;
};
#endif

// FP8类型别名
#if defined(__CUDA_FP8_E4M3__)
using nv_fp8_e4m3 = __nv_fp8_e4m3;
#else
struct nv_fp8_e4m3 {
    uint8_t data;
};
#endif

class CublasLtNVFP4Wrapper {
public:
    static CublasLtNVFP4Wrapper& instance() {
        static CublasLtNVFP4Wrapper instance;
        return instance;
    }

    cublasLtHandle_t handle() const { return handle_; }
    bool initialize();
    void cleanup();

    void set_preference(int max_workspace_size = 32 * 1024 * 1024);

    void* workspace() const { return workspace_; }
    size_t max_workspace_size() const { return max_workspace_size_; }

    bool check_fp4_support() const { return fp4_supported_; }

private:
    CublasLtNVFP4Wrapper();
    ~CublasLtNVFP4Wrapper();

    CublasLtNVFP4Wrapper(const CublasLtNVFP4Wrapper&) = delete;
    CublasLtNVFP4Wrapper& operator=(const CublasLtNVFP4Wrapper&) = delete;

    cublasLtHandle_t handle_;
    bool initialized_;
    bool fp4_supported_;

    size_t max_workspace_size_;
    void* workspace_;
};

// NVFP4 GEMM参数结构体
struct NvFp4GemmParams {
    MemoryFormat format;
    ComputeType compute_type;
    cublasLtMatmulMatrixScale_t a_scale_mode;
    cublasLtMatmulMatrixScale_t b_scale_mode;
    cublasLtMatmulMatrixScale_t c_scale_mode;
    cublasLtMatmulMatrixScale_t d_scale_mode;
    cublasLtMatmulMatrixScale_t d_out_scale_mode;
    float alpha;
    float beta;
    cublasOperation_t trans_a;
    cublasOperation_t trans_b;
    EpilogueMode epilogue_mode;

    NvFp4GemmParams()
        : format(MemoryFormat::ROW_MAJOR)
        , compute_type(ComputeType::FLOAT)
        , a_scale_mode(CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3)
        , b_scale_mode(CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3)
        , c_scale_mode(CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F)
        , d_scale_mode(CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F)
        , d_out_scale_mode(CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F)
        , alpha(1.0f)
        , beta(0.0f)
        , trans_a(CUBLAS_OP_N)
        , trans_b(CUBLAS_OP_N)
        , epilogue_mode(EpilogueMode::NONE) {}
};

// 主要NVFP4 GEMM接口（A和B都是NVFP4）
bool cublaslt_gemm_nvfp4_standard(
    cublasLtHandle_t handle,
    int m, int n, int k,
    const void* A,                     // NVFP4输入A
    const void* B,                     // NVFP4输入B
    void* C,
    void* D,
    const void* bias,                  // 偏置（可选）
    const nv_fp8_e4m3* a_scale,       // A的缩放因子
    const nv_fp8_e4m3* b_scale,       // B的缩放因子
    const nv_fp8_e4m3* c_scale,       // C的缩放因子
    const nv_fp8_e4m3* d_scale,       // D的缩放因子
    const nv_fp8_e4m3* d_out_scale,   // 输出缩放因子
    const NvFp4GemmParams& params = NvFp4GemmParams(),
    cudaStream_t stream = 0);

template <bool kEnableSilu = false, bool kEnableBias = false>
bool cublaslt_gemm_nvfp4(
    cublasLtHandle_t handle,
    int m, int n, int k,
    const void* A,
    const void* B,
    const void* bias,
    void* output,
    const void* a_scale,
    const void* b_scale,
    float alpha,
    nvinfer1::DataType output_type,
    cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm

#endif // CUBLASLT_WRAPPER_NVFP4_H