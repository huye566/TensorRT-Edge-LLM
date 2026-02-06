#include "cublas_wrapper.h"
#include <iostream>

#define CUBLAS_CHECK(status)                                               \
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

namespace trt_edgellm {
namespace kernel {

CublasWrapper::CublasWrapper()
    : handle_(nullptr)
    , use_tensor_core_(false)
    , initialized_(false) {
    initialize();
}

CublasWrapper::~CublasWrapper() {
    cleanup();
}

bool CublasWrapper::initialize() {
    if (initialized_) return true;

    cublasStatus_t status = cublasCreate(&handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create cublas handle" << std::endl;
        return false;
    }

    set_math_mode(true);

    initialized_ = true;
    return true;
}

void CublasWrapper::cleanup() {
    if (handle_) {
        cublasDestroy(handle_);
        handle_ = nullptr;
    }
    initialized_ = false;
}

void CublasWrapper::set_math_mode(bool use_tensor_core) {
    if (!initialized_) return;
    if (use_tensor_core) {
        cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH);
    } else {
        cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH);
    }
}


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

// https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasGemmEx#cublasgemmex
// https://zhuanlan.zhihu.com/p/666391239
template<>
void cublas_gemm<half>(cublasHandle_t handle,
                      int m, int n, int k,
                      const half* A,
                      const half* B,
                      half* C,
                      MemoryFormat format,
                      cudaStream_t stream) {

    cublasStatus_t status;
    half alpha = 1.0f;
    half beta = 0.0f;
    cudaDataType_t data_type = CUDA_R_16F;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    // cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;

    // cuBLAS GEMM参数（列主序视角）
    // 实际计算：C_col = alpha * A_col * B_col + beta * C_col
    int m_col, n_col, k_col;
    int lda, ldb, ldc;
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    half * A_ptr = const_cast<half*>(A);
    half * B_ptr = const_cast<half*>(B);

    if (format == MemoryFormat::ROW_MAJOR) { // out A * B = (B^T * A^T)^T
        m_col = n;
        n_col = m;
        k_col = k;

        lda = k;
        ldb = n;
        ldc = n;
        status = cublasGemmEx(handle,
                         transA,
                         transB,
                         m_col, n_col, k_col,
                         &alpha,
                         B_ptr, data_type, ldb,
                         A_ptr, data_type, lda,
                         &beta,
                         C, data_type, ldc,
                         compute_type,
                         CUBLAS_GEMM_DEFAULT); // CUBLAS_GEMM_DEFAULT_TENSOR_OP
    } else { // (A * B)^T, out col major
        transA = CUBLAS_OP_T;
        transB = CUBLAS_OP_T;
        m_col = m;
        n_col = n;
        k_col = k;
        lda = k; // A row
        ldb = n; // B row
        ldc = m;
        status = cublasGemmEx(handle,
                         transA,
                         transB,
                         m_col, n_col, k_col,
                         &alpha,
                         A_ptr, data_type, lda,
                         B_ptr, data_type, ldb,
                         &beta,
                         C, data_type, ldc,
                         compute_type,
                         CUBLAS_GEMM_DEFAULT); // CUBLAS_GEMM_DEFAULT_TENSOR_OP
    }
    CUBLAS_CHECK(status);
}

template<>
void cublas_gemm<float>(cublasHandle_t handle,
                       int m, int n, int k,
                       const float* A,
                       const float* B,
                       float* C,
                       MemoryFormat format,
                       cudaStream_t stream) {

    cublasStatus_t status;
    float alpha = 1.0f;
    float beta = 0.0f;

    // cuBLAS GEMM参数（列主序视角）
    int m_col, n_col, k_col;
    int lda, ldb, ldc;
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    float * A_ptr = const_cast<float*>(A);
    float * B_ptr = const_cast<float*>(B);

    if (format == MemoryFormat::ROW_MAJOR) {
        m_col = n;
        n_col = m;
        k_col = k;

        lda = k;
        ldb = n;
        ldc = n;

        status = cublasSgemm(handle,
                            transA,
                            transB,
                            m_col, n_col, k_col,
                            &alpha,
                            B_ptr, ldb,
                            A_ptr, lda,
                            &beta,
                            C, ldc);
    } else {
        transA = CUBLAS_OP_T;
        transB = CUBLAS_OP_T;
        m_col = m;
        n_col = n;
        k_col = k;
        lda = k;
        ldb = n;
        ldc = m;
        status = cublasSgemm(handle,
                            transA,
                            transB,
                            m_col, n_col, k_col,
                            &alpha,
                            A_ptr, lda,
                            B_ptr, ldb,
                            &beta,
                            C, ldc);
    }
    CUBLAS_CHECK(status);
}

template<>
void cublas_gemm_silu<half>(cublasHandle_t handle,
                           int m, int n, int k,
                           const half* A,
                           const half* B,
                           half* C,
                           MemoryFormat format,
                           cudaStream_t stream) {
    cublas_gemm<half>(handle, m, n, k, A, B, C, format, stream);

    int num_elements = m * n;
    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;

    silu_kernel<half><<<grid_size, block_size, 0, stream>>>(C, num_elements);
    CUDA_CHECK(cudaGetLastError());
}

template<>
void cublas_gemm_silu<float>(cublasHandle_t handle,
                            int m, int n, int k,
                            const float* A,
                            const float* B,
                            float* C,
                            MemoryFormat format,
                            cudaStream_t stream) {
    cublas_gemm<float>(handle, m, n, k, A, B, C, format, stream);

    int num_elements = m * n;
    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;

    silu_kernel<float><<<grid_size, block_size, 0, stream>>>(C, num_elements);
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
__global__ void add_bias_kernel(T* C, const T* bias, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        int idx = row * n + col;
        C[idx] += bias[col];  // bias按列广播
    }
}

template<>
void cublas_gemm_bias<half>(cublasHandle_t handle,
                           int m, int n, int k,
                           const half* A,
                           const half* B,
                           const half* bias,
                           half* C,
                           MemoryFormat format,
                           cudaStream_t stream) {
    // 计算: C = A * B + bias (广播bias到每行)
    cublas_gemm<half>(handle, m, n, k, A, B, C, format, stream);

    // 添加bias（广播到每行）
    const int block_size = 16;
    dim3 block_dim(block_size, block_size);
    dim3 grid_size(
        (m + block_size - 1) / block_size,
        (n + block_size - 1) / block_size
    );

    add_bias_kernel<half><<<grid_size, block_dim, 0, stream>>>(C, bias, m, n);
    CUDA_CHECK(cudaGetLastError());
}

template<>
void cublas_gemm_bias<float>(cublasHandle_t handle,
                            int m, int n, int k,
                            const float* A,
                            const float* B,
                            const float* bias,
                            float* C,
                            MemoryFormat format,
                            cudaStream_t stream) {
    cublas_gemm<float>(handle, m, n, k, A, B, C, format, stream);
    const int block_size = 16;
    dim3 block_dim(block_size, block_size);
    dim3 grid_size(
        (m + block_size - 1) / block_size,
        (n + block_size - 1) / block_size
    );

    add_bias_kernel<float><<<grid_size, block_dim, 0, stream>>>(C, bias, m, n);
    CUDA_CHECK(cudaGetLastError());
}

}
}