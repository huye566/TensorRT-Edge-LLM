#include "cublaslt_wrapper.h"
#include <iostream>
#include <cassert>
#include <functional>

#define CUBLASLT_CHECK(status)                                             \
    do {                                                                   \
        if (status != CUBLAS_STATUS_SUCCESS) {                             \
            std::cerr << "CUBLAS error at " << __FILE__ << ":" << __LINE__ \
                     << " code: " << status << std::endl;                  \
            exit(EXIT_FAILURE);                                            \
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

#define WORKSPACE_SIZE (32 * 1024 * 1024)

namespace trt_edgellm {
namespace kernel {

// SiLU激活核函数
template<typename T>
__global__ void silu_kernel(T* data, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        T x = data[idx];
        // SiLU(x) = x * sigmoid(x)
        T sigmoid_x = T(1.0) / (T(1.0) + exp(-x));
        data[idx] = x * sigmoid_x;
    }
}

template<>
__global__ void silu_kernel<half>(half* data, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        half x = data[idx];
        float x_f = __half2float(x);
        float sigmoid_x = 1.0f / (1.0f + expf(-x_f));
        data[idx] = __float2half(x_f * sigmoid_x);
    }
}

// 偏置相加核函数
template<typename T>
__global__ void add_bias_kernel(T* C, const T* bias, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        int idx = row * n + col;
        C[idx] += bias[col];
    }
}

// CublasLtWrapper实现
CublasLtWrapper::CublasLtWrapper()
    : handle_(nullptr)
    , initialized_(false)
    , max_workspace_size_(WORKSPACE_SIZE)
    , workspace_(nullptr) {
    initialize();
}

CublasLtWrapper::~CublasLtWrapper() {
    cleanup();
}

bool CublasLtWrapper::initialize() {
    if (initialized_) return true;

    cublasStatus_t status = cublasLtCreate(&handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create cublasLt handle" << std::endl;
        return false;
    }

    // 分配工作空间
    if (max_workspace_size_ > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_, max_workspace_size_));
    }

    initialized_ = true;
    return true;
}

void CublasLtWrapper::cleanup() {
    if (workspace_) {
        cudaFree(workspace_);
        workspace_ = nullptr;
    }

    if (handle_) {
        cublasLtDestroy(handle_);
        handle_ = nullptr;
    }

    initialized_ = false;
}

void CublasLtWrapper::set_preference(int max_workspace_size) {
    max_workspace_size_ = max_workspace_size;

    if (workspace_) {
        cudaFree(workspace_);
        workspace_ = nullptr;
    }

    if (max_workspace_size_ > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_, max_workspace_size_));
    }
}

// GemmDescriptor实现
template<typename T>
void GemmDescriptor<T>::create(int m, int n, int k, MemoryFormat format, cublasLtHandle_t handle) {
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    cudaDataType_t scale_type = CUDA_R_32F;
    cudaDataType_t data_type = CUDA_R_32F;
    if constexpr (std::is_same_v<T, float>) {
        compute_type = CUBLAS_COMPUTE_32F;
        scale_type = CUDA_R_32F;
        data_type = CUDA_R_32F;
    } else if constexpr (std::is_same_v<T, half>) {
        compute_type = CUBLAS_COMPUTE_16F;
        scale_type = CUDA_R_16F;
        data_type = CUDA_R_16F;
    } else {
        std::cerr << "Unsupported data type" << std::endl;
        exit(EXIT_FAILURE);
    }

    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&A_desc, data_type,
                              (format == MemoryFormat::ROW_MAJOR) ? k : m,
                              (format == MemoryFormat::ROW_MAJOR) ? m : k,
                              (format == MemoryFormat::ROW_MAJOR) ? k : m));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&B_desc, data_type,
                              (format == MemoryFormat::ROW_MAJOR) ? n : k,
                              (format == MemoryFormat::ROW_MAJOR) ? k : n,
                              (format == MemoryFormat::ROW_MAJOR) ? n : k));
    // 精度影响很大，性能差不多，也支持bias
    // CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&B_desc, data_type,
    //                           (format == MemoryFormat::ROW_MAJOR) ? k : n,
    //                           (format == MemoryFormat::ROW_MAJOR) ? n : k,
    //                           (format == MemoryFormat::ROW_MAJOR) ? k : n));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&C_desc, data_type,
                              (format == MemoryFormat::ROW_MAJOR) ? n : m,
                              (format == MemoryFormat::ROW_MAJOR) ? m : n,
                              (format == MemoryFormat::ROW_MAJOR) ? n : m));

    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&operation_desc, compute_type, scale_type));

    cublasOperation_t trans_a = (format == MemoryFormat::ROW_MAJOR) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t trans_b = (format == MemoryFormat::ROW_MAJOR) ? CUBLAS_OP_N : CUBLAS_OP_T;
    // cublasOperation_t trans_a = (format == MemoryFormat::ROW_MAJOR) ? CUBLAS_OP_T : CUBLAS_OP_T;
    // cublasOperation_t trans_b = (format == MemoryFormat::ROW_MAJOR) ? CUBLAS_OP_N : CUBLAS_OP_T;

    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

    // 创建偏好描述符
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    size_t workspace_size = WORKSPACE_SIZE;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
                                        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                        &workspace_size,
                                        sizeof(workspace_size)));
}

template<typename T>
void GemmDescriptor<T>::destroy() {
    if (A_desc) CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(A_desc));
    if (B_desc) CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(B_desc));
    if (C_desc) CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(C_desc));
    if (operation_desc) CUBLASLT_CHECK(cublasLtMatmulDescDestroy(operation_desc));
    if (preference) CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(preference));

    A_desc = nullptr;
    B_desc = nullptr;
    C_desc = nullptr;
    operation_desc = nullptr;
    preference = nullptr;
}

// 选择最佳算法
template<typename T>
bool select_algorithm(cublasLtHandle_t handle,
                     GemmDescriptor<T>& desc,
                     cublasLtMatmulAlgo_t& algo,
                     MemoryFormat format,
                     void* workspace,
                     size_t workspace_size) {
    // int requested_algo_count = 1;
    int returned_algo_count = 0;
    cublasLtMatmulHeuristicResult_t heuristic_result = {};

    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        handle,
        desc.operation_desc,
        (format == MemoryFormat::ROW_MAJOR) ? desc.B_desc : desc.A_desc,
        (format == MemoryFormat::ROW_MAJOR) ? desc.A_desc : desc.B_desc,
        desc.C_desc,
        desc.C_desc,
        desc.preference,
        1,
        &heuristic_result,
        &returned_algo_count);

    if (status != CUBLAS_STATUS_SUCCESS || returned_algo_count == 0) {
        return false;
    }

    algo = heuristic_result.algo;
    return true;
}


template<typename T, bool kEnableSilu, bool kEnableBias>
bool cublaslt_gemm(cublasLtHandle_t handle,
                    int m, int n, int k,
                    const T* A,
                    const T* B,
                    const T* bias,
                    T* C,
                    MemoryFormat format,
                    ComputeType compute_type,
                    cudaStream_t stream) {

    GemmDescriptor<T> desc;
    desc.create(m, n, k, format, handle);
    float alpha_float = 1.0f;
    float beta_float  = 0.0f;
    half  alpha_half  = half(1.0f);
    half  beta_half   = half(0.0f);
    void* alpha_ptr = nullptr;
    void* beta_ptr = nullptr;
    if constexpr (std::is_same_v<T, float>) {
        alpha_ptr = &alpha_float;
        beta_ptr = &beta_float;
    } else if constexpr (std::is_same_v<T, half>) {
        alpha_ptr = &alpha_half;
        beta_ptr = &beta_half;
    } else {
        std::cerr << "Unsupported data type" << std::endl;
        return false;
    }

    cublasComputeType_t cublas_compute_type;
    cudaDataType_t scale_type;

    switch (compute_type) {
        case ComputeType::HALF:
            cublas_compute_type = CUBLAS_COMPUTE_16F;
            scale_type = CUDA_R_16F;
            break;
        case ComputeType::FLOAT:
            cublas_compute_type = CUBLAS_COMPUTE_32F;
            scale_type = CUDA_R_32F;
            break;
        case ComputeType::TFLOAT32:
            cublas_compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
            scale_type = CUDA_R_32F;
            break;
        default:
            cublas_compute_type = CUBLAS_COMPUTE_32F;
            scale_type = CUDA_R_32F;
    }

    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc.operation_desc,
                                  CUBLASLT_MATMUL_DESC_COMPUTE_TYPE,
                                  &cublas_compute_type,
                                  sizeof(cublas_compute_type)));

    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc.operation_desc,
                                  CUBLASLT_MATMUL_DESC_SCALE_TYPE,
                                  &scale_type,
                                  sizeof(scale_type)));
    if constexpr (kEnableBias) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc.operation_desc,
            CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc.operation_desc,
            CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc.operation_desc,
            CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &scale_type, sizeof(scale_type)));
    }

    // 选择算法
    cublasLtMatmulAlgo_t algo;
    CublasLtWrapper& wrapper = CublasLtWrapper::instance();
    bool algo_found = select_algorithm(handle, desc, algo, format, wrapper.workspace(), wrapper.max_workspace_size());

    cublasStatus_t status;
    if (algo_found) {
        status = cublasLtMatmul(handle,
                               desc.operation_desc,
                               alpha_ptr,
                               (format == MemoryFormat::ROW_MAJOR) ? B : A,
                               (format == MemoryFormat::ROW_MAJOR) ? desc.B_desc : desc.A_desc,
                               (format == MemoryFormat::ROW_MAJOR) ? A : B,
                               (format == MemoryFormat::ROW_MAJOR) ? desc.A_desc : desc.B_desc,
                               beta_ptr,
                               C,
                               desc.C_desc,
                               C,
                               desc.C_desc,
                               &algo,
                               wrapper.workspace(),
                               wrapper.max_workspace_size(),
                               stream);
    } else {
        status = cublasLtMatmul(handle,
                               desc.operation_desc,
                               alpha_ptr,
                               (format == MemoryFormat::ROW_MAJOR) ? B : A,
                               (format == MemoryFormat::ROW_MAJOR) ? desc.B_desc : desc.A_desc,
                               (format == MemoryFormat::ROW_MAJOR) ? A : B,
                               (format == MemoryFormat::ROW_MAJOR) ? desc.A_desc : desc.B_desc,
                               beta_ptr,
                               C,
                               desc.C_desc,
                               C,
                               desc.C_desc,
                               nullptr, // 使用默认算法
                               wrapper.workspace(),
                               wrapper.max_workspace_size(),
                               stream);
    }

    CUBLASLT_CHECK(status);
    desc.destroy();

    if constexpr (!kEnableBias) {
        if (bias != nullptr) {
            const int block_size = 16;
            dim3 block_dim(block_size, block_size);
            dim3 grid_size(
                (m + block_size - 1) / block_size,
                (n + block_size - 1) / block_size
            );

            add_bias_kernel<T><<<grid_size, block_dim, 0, stream>>>(C, bias, m, n);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    if constexpr (kEnableSilu) {
        int num_elements = m * n;
        const int block_size = 256;
        const int grid_size = (num_elements + block_size - 1) / block_size;

        silu_kernel<T><<<grid_size, block_size, 0, stream>>>(C, num_elements);
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}

template bool cublaslt_gemm<half, false, false>(
    cublasLtHandle_t handle, int m, int n, int k,
    const half* A, const half* B, const half* bias,
    half* C, MemoryFormat format, ComputeType compute_type, cudaStream_t stream);

template bool cublaslt_gemm<half, false, true>(
    cublasLtHandle_t handle, int m, int n, int k,
    const half* A, const half* B, const half* bias,
    half* C, MemoryFormat format, ComputeType compute_type, cudaStream_t stream);

template bool cublaslt_gemm<half, true, false>(
    cublasLtHandle_t handle, int m, int n, int k,
    const half* A, const half* B, const half* bias,
    half* C, MemoryFormat format, ComputeType compute_type, cudaStream_t stream);

template bool cublaslt_gemm<half, true, true>(
    cublasLtHandle_t handle, int m, int n, int k,
    const half* A, const half* B, const half* bias,
    half* C, MemoryFormat format, ComputeType compute_type, cudaStream_t stream);

template bool cublaslt_gemm<float, false, false>(
    cublasLtHandle_t handle, int m, int n, int k,
    const float* A, const float* B, const float* bias,
    float* C, MemoryFormat format, ComputeType compute_type, cudaStream_t stream);

template bool cublaslt_gemm<float, false, true>(
    cublasLtHandle_t handle, int m, int n, int k,
    const float* A, const float* B, const float* bias,
    float* C, MemoryFormat format, ComputeType compute_type, cudaStream_t stream);

template bool cublaslt_gemm<float, true, false>(
    cublasLtHandle_t handle, int m, int n, int k,
    const float* A, const float* B, const float* bias,
    float* C, MemoryFormat format, ComputeType compute_type, cudaStream_t stream);

template bool cublaslt_gemm<float, true, true>(
    cublasLtHandle_t handle, int m, int n, int k,
    const float* A, const float* B, const float* bias,
    float* C, MemoryFormat format, ComputeType compute_type, cudaStream_t stream);

template struct GemmDescriptor<half>;
template struct GemmDescriptor<float>;

} // namespace kernel
} // namespace trt_edgellm
