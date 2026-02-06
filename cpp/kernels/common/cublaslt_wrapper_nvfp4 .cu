#include "cublaslt_wrapper_nvfp4.h"
#include "universal_operators.h"
#include <iostream>
#include <cassert>
#include <functional>
#include <cmath>
#include <unordered_map>
#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

inline const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "Unknown CUBLAS error";
    }
}

#define CUBLASLT_CHECK(status)                                             \
    do {                                                                   \
        if (status != CUBLAS_STATUS_SUCCESS) {                             \
            std::cerr << "CUBLASLT error at " << __FILE__ << ":" << __LINE__ \
                     << " code: " << status << " (" << cublasGetErrorString(status) << ")" << std::endl; \
            exit(EXIT_FAILURE);                                             \
        }                                                                  \
    } while(0)

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "CUDA error: " << cudaGetErrorString(error)          \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

namespace trt_edgellm {
namespace kernel {

bool check_nvfp4_hardware_support() {
    cudaDeviceProp prop;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // NVFP4需要Hopper架构（SM90）或更高
    return prop.major >= 9;
}

void printAlgo(const cublasLtMatmulAlgo_t &algo) {
    int algoId, tile, swizzle, customOption, numSplitsK, reductionScheme;

    CUBLASLT_CHECK(
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), NULL));
    CUBLASLT_CHECK(
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), NULL));
    CUBLASLT_CHECK(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK,
                                                           sizeof(numSplitsK), NULL));
    CUBLASLT_CHECK(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                                           &reductionScheme, sizeof(reductionScheme), NULL));
    CUBLASLT_CHECK(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle,
                                                           sizeof(swizzle), NULL));
    CUBLASLT_CHECK(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption,
                                                           sizeof(customOption), NULL));

    printf("algo={ Id=%d, tileIdx=%d splitK=%d reduc=%d swizzle=%d custom=%d }\n", algoId, tile, numSplitsK,
           reductionScheme, swizzle, customOption);
}

// CublasLtNVFP4Wrapper实现
CublasLtNVFP4Wrapper::CublasLtNVFP4Wrapper()
    : handle_(nullptr)
    , initialized_(false)
    , fp4_supported_(false)
    , max_workspace_size_(32 * 1024 * 1024)
    , workspace_(nullptr) {
    initialize();
}

CublasLtNVFP4Wrapper::~CublasLtNVFP4Wrapper() {
    cleanup();
}

bool CublasLtNVFP4Wrapper::initialize() {
    if (initialized_) return true;

    cublasStatus_t status = cublasLtCreate(&handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create cublasLt handle: "
                  << cublasGetErrorString(status) << std::endl;
        return false;
    }

    // 检查NVFP4支持
    fp4_supported_ = check_nvfp4_hardware_support();
    if (!fp4_supported_) {
        std::cout << "Warning: NVFP4 not supported on this hardware" << std::endl;
    }

    // 分配工作空间
    if (max_workspace_size_ > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_, max_workspace_size_));
    }

    initialized_ = true;
    std::cout << "CublasLtNVFP4Wrapper initialized successfully" << std::endl;
    return true;
}

void CublasLtNVFP4Wrapper::cleanup() {
    if (workspace_) {
        cudaFree(workspace_);
        workspace_ = nullptr;
    }

    if (handle_) {
        cublasLtDestroy(handle_);
        handle_ = nullptr;
    }

    initialized_ = false;
    fp4_supported_ = false;
}

void CublasLtNVFP4Wrapper::set_preference(int max_workspace_size) {
    max_workspace_size_ = max_workspace_size;

    if (workspace_) {
        cudaFree(workspace_);
        workspace_ = nullptr;
    }

    if (max_workspace_size_ > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_, max_workspace_size_));
    }
}

cublasLtEpilogue_t get_cublaslt_epilogue(EpilogueMode mode) {
    switch (mode) {
        case EpilogueMode::BIAS:
            return CUBLASLT_EPILOGUE_BIAS;
        case EpilogueMode::BIAS_RELU:
            return CUBLASLT_EPILOGUE_RELU_BIAS;
        case EpilogueMode::BIAS_GELU:
            return CUBLASLT_EPILOGUE_GELU_BIAS;
        case EpilogueMode::NONE:
        default:
            return CUBLASLT_EPILOGUE_DEFAULT;
    }
}

// https://github.com/NVIDIA/CUDALibrarySamples/issues/303
bool cublaslt_gemm_nvfp4_impl(
    cublasLtHandle_t handle,
    int m, int n, int k,
    const void* A,
    const void* B,
    void* C,
    void* D,
    const void* bias,
    const nv_fp8_e4m3* a_scale,
    const nv_fp8_e4m3* b_scale,
    const nv_fp8_e4m3* c_scale,
    const nv_fp8_e4m3* d_scale,
    const nv_fp8_e4m3* d_out_scale,
    const NvFp4GemmParams& params,
    cudaStream_t stream) {

    if (!check_nvfp4_hardware_support()) {
        std::cerr << "NVFP4 GEMM not supported on this hardware" << std::endl;
        return false;
    }
    // auto version = cublasLtGetVersion();
    // printf("cublasLt version: %zu\n", version);

    cublasLtMatmulDesc_t operation_desc = nullptr;
    cudaDataType_t nvfp4DataType = CUDA_R_4F_E2M1;
    cudaDataType_t d_type = CUDA_R_32F;
    if (params.compute_type == ComputeType::HALF) {
        d_type = CUDA_R_16F;
    }
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                           &params.trans_a, sizeof(params.trans_a)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                           &params.trans_b, sizeof(params.trans_b)));

    cublasLtEpilogue_t epilogue = get_cublaslt_epilogue(params.epilogue_mode);
    // cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operation_desc,
        CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (bias && (params.epilogue_mode == EpilogueMode::BIAS ||
                params.epilogue_mode == EpilogueMode::BIAS_RELU ||
                params.epilogue_mode == EpilogueMode::BIAS_GELU)) {
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operation_desc,
            CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operation_desc,
            CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &d_type, sizeof(d_type)));
    }

    void *scale_A_ptr = (void *)a_scale;
    void *scale_B_ptr = (void *)b_scale;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                           &params.a_scale_mode, sizeof(params.a_scale_mode)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                        &params.b_scale_mode, sizeof(params.b_scale_mode)));
    if (params.format == MemoryFormat::ROW_MAJOR) {
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                               &scale_B_ptr, sizeof(scale_B_ptr)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                               &scale_A_ptr, sizeof(scale_A_ptr)));
    } else {
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                               &scale_A_ptr, sizeof(scale_A_ptr)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                               &scale_B_ptr, sizeof(scale_B_ptr)));
    }

    if (c_scale) {
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_C_SCALE_MODE,
                                           &params.c_scale_mode, sizeof(params.c_scale_mode)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER,
                                               &c_scale, sizeof(c_scale)));
    }
    if (d_scale) {
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE,
                                           &params.d_scale_mode, sizeof(params.d_scale_mode)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
                                               &d_scale, sizeof(d_scale)));
    }
    if (d_out_scale) {
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE,
                                           &params.d_out_scale_mode, sizeof(params.d_out_scale_mode)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER,
                                               &d_out_scale, sizeof(d_out_scale)));
    }

    // 创建矩阵布局
    cublasLtMatrixLayout_t A_desc = nullptr, B_desc = nullptr, C_desc = nullptr, D_desc = nullptr;

    int rows_A, cols_A, lda;
    int rows_B, cols_B, ldb;
    int rows_D, cols_D, ldd;
    if (params.format == MemoryFormat::ROW_MAJOR) {
        rows_A = k;
        cols_A = m;
        lda = k;
        rows_B = k;
        cols_B = n;
        ldb = k;
        rows_D = n;
        cols_D = m;
        ldd = n;
        // rows_D = m;
        // cols_D = n;
        // ldd = m;
    } else {
        rows_A = k;
        cols_A = m;
        lda = m;
        rows_B = k;
        cols_B = n;
        ldb = n;
        rows_D = m;
        cols_D = n;
        ldd = m;
    }

    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&A_desc, nvfp4DataType,
                                       rows_A, cols_A, lda));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&B_desc, nvfp4DataType,
                                       rows_B, cols_B, ldb));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&C_desc, d_type, rows_D, cols_D, ldd));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&D_desc, d_type, rows_D, cols_D, ldd));

    auto out_order = CUBLASLT_ORDER_COL;
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        D_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &out_order, sizeof(out_order)));
    // size_t sizeWritten;
    // CUBLASLT_CHECK(cublasLtMatrixLayoutGetAttribute(D_desc, 
    //     CUBLASLT_MATRIX_LAYOUT_ORDER, &out_order, sizeof(out_order), &sizeWritten));
    // std::cout << "out_order: " << out_order << ", sizeWritten: " << sizeWritten << std::endl;

    // 设置偏好
    cublasLtMatmulPreference_t preference = nullptr;
    size_t workspace_size = 32 * 1024 * 1024;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
                                                 CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                 &workspace_size, sizeof(workspace_size)));

    // 获取启发式算法
    int returned_results = 1;
    cublasLtMatmulHeuristicResult_t heuristic_result = {};
    CublasLtNVFP4Wrapper& wrapper = CublasLtNVFP4Wrapper::instance();

    if (params.format == MemoryFormat::ROW_MAJOR) {
        CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
            handle, operation_desc, B_desc, A_desc, D_desc, D_desc,
            preference, 1, &heuristic_result, &returned_results));
        if (returned_results == 0) {
            std::cerr << "No valid algorithm found for NVFP4 GEMM" << std::endl;
        } else {
            CUBLASLT_CHECK(cublasLtMatmul(
                handle, operation_desc,
                &params.alpha,
                B, B_desc,
                A, A_desc,
                &params.beta,
                D, D_desc,
                D, D_desc,
                &heuristic_result.algo,
                wrapper.workspace(),
                wrapper.max_workspace_size(),
                stream));
        }
    } else {
        CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
            handle, operation_desc, A_desc, B_desc, D_desc, D_desc,
            preference, 1, &heuristic_result, &returned_results));
        if (returned_results == 0) {
            std::cerr << "No valid algorithm found for NVFP4 GEMM" << std::endl;
        } else {
            CUBLASLT_CHECK(cublasLtMatmul(
            handle, operation_desc,
            &params.alpha,
            A, A_desc,
            B, B_desc,
            &params.beta,
            D, D_desc,
            D, D_desc,
            &heuristic_result.algo,
            wrapper.workspace(),
            wrapper.max_workspace_size(),
            stream));
        }
    }

    // 清理资源
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (D_desc) cublasLtMatrixLayoutDestroy(D_desc);
    if (C_desc) cublasLtMatrixLayoutDestroy(C_desc);
    if (B_desc) cublasLtMatrixLayoutDestroy(B_desc);
    if (A_desc) cublasLtMatrixLayoutDestroy(A_desc);
    if (operation_desc) cublasLtMatmulDescDestroy(operation_desc);

    return (returned_results != 0);
}


template <bool kEnableSilu, bool kEnableBias>
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
    cudaStream_t stream) {

    NvFp4GemmParams params;
    params.alpha = alpha;
    params.beta = 0.0f;
    if (kEnableBias && bias != nullptr) {
        params.epilogue_mode = EpilogueMode::BIAS;
    } else {
        params.epilogue_mode = EpilogueMode::NONE;
    }

    switch (output_type) {
        case nvinfer1::DataType::kHALF:
            params.compute_type = ComputeType::HALF;
            break;
        case nvinfer1::DataType::kBF16:
            params.compute_type = ComputeType::FLOAT;
            break;
        case nvinfer1::DataType::kFLOAT:
            params.compute_type = ComputeType::FLOAT;
            break;
        default:
            std::cerr << "Unsupported output type for NVFP4 GEMM" << std::endl;
            return false;
    }

    params.format = MemoryFormat::ROW_MAJOR;
    params.trans_a = CUBLAS_OP_T;
    params.trans_b = CUBLAS_OP_N;

    bool success = cublaslt_gemm_nvfp4_impl(handle, m, n, k, A, B, nullptr, output,
                              bias,
                              static_cast<const nv_fp8_e4m3*>(a_scale),
                              static_cast<const nv_fp8_e4m3*>(b_scale),
                              nullptr, nullptr, nullptr, params, stream);

    if (!success) {
        return false;
    }

    if (!kEnableBias && bias != nullptr) {
        success = apply_add_bias_vec_optimized(output, bias, m, n, output_type, stream);
        if (!success) {
            return false;
        }
    }

    if (kEnableSilu) {
        int num_elements = m * n;
        success = apply_silu_vec_optimized(output, num_elements, output_type, stream);
        if (!success) {
            return false;
        }
    }

    return true;
}

template bool cublaslt_gemm_nvfp4<false, false>(
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

template bool cublaslt_gemm_nvfp4<true, false>(
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

template bool cublaslt_gemm_nvfp4<false, true>(
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

template bool cublaslt_gemm_nvfp4<true, true>(
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