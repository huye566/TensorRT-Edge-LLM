#include <iostream>
#include <vector>
#include <chrono>
#include <map>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cmath>
#include <type_traits>
#include "check/err_analysis.h"
#include "utils/cuda_check.h"
#include "gemm_silu_bench.h"

// ==================== 通用辅助函数 ====================

// SiLU激活函数：x * sigmoid(x) = x / (1 + exp(-x))
template<typename T>
__host__ __device__ T silu_function(T x) {
    if constexpr (std::is_same<T, half>::value) {
        float x_f = __half2float(x);
        float result = x_f / (1.0f + expf(-x_f));
        return __float2half(result);
    } else if constexpr (std::is_same<T, float>::value) {
        return x / (1.0f + expf(-x));
    } else {
        return x / (1.0 + exp(-x));
    }
}

// 计算CPU参考结果：GEMM + SiLU
template<typename T>
void compute_gemm_silu_cpu_reference(
    std::vector<T>& h_ref,
    const std::vector<T>& h_A,
    const std::vector<T>& h_B,
    int M, int N, int K,
    T alpha,
    int max_elements = MAX_COMPARE_COUNT) {

    int total_elements = M * N;
    int verify_count = std::min(total_elements, max_elements);
    h_ref.resize(total_elements);

    for (int idx = 0; idx < verify_count; ++idx) {
        int i = idx / N;  // 行索引
        int j = idx % N;  // 列索引

        T accum = T(0);
        for (int k = 0; k < K; ++k) {
            accum += h_A[i * K + k] * h_B[k * N + j];
        }
        T gemm_result = alpha * accum;
        h_ref[idx] = silu_function(gemm_result);
    }
}

// 特化half版本的CPU参考计算
template<>
void compute_gemm_silu_cpu_reference<half>(
    std::vector<half>& h_ref,
    const std::vector<half>& h_A,
    const std::vector<half>& h_B,
    int M, int N, int K,
    half alpha,
    int max_elements) {

    int total_elements = M * N;
    int verify_count = std::min(total_elements, max_elements);
    h_ref.resize(total_elements);

    float alpha_f = __half2float(alpha);

    for (int idx = 0; idx < verify_count; ++idx) {
        int i = idx / N;  // 行索引
        int j = idx % N;  // 列索引

        float accum = 0.0f;
        for (int k = 0; k < K; ++k) {
            accum += __half2float(h_A[i * K + k]) * __half2float(h_B[k * N + j]);
        }
        float gemm_result = alpha_f * accum;
        float silu_result = gemm_result / (1.0f + expf(-gemm_result));
        h_ref[idx] = __float2half(silu_result);
    }
}

// ==================== 独立SiLU Kernel ====================

template<typename T>
__global__ void silu_kernel(T* data, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        data[idx] = silu_function(data[idx]);
    }
}

template<typename T>
void launch_silu_kernel(T* d_data, int num_elements, cudaStream_t stream = 0) {
    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;

    silu_kernel<T><<<grid_size, block_size, 0, stream>>>(d_data, num_elements);
}

// ==================== 融合GEMM+SiLU Kernel ====================

// // 简单的融合kernel实现（使用global memory）
// template<typename T>
// __global__ void gemm_silu_fused_kernel(
//     const T* __restrict__ A,
//     const T* __restrict__ B,
//     T* __restrict__ C,
//     int M, int N, int K,
//     T alpha) {

//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row < M && col < N) {
//         T accum = T(0);

//         // 简单的矩阵乘法
//         for (int k = 0; k < K; ++k) {
//             T a_val = A[row * K + k];
//             T b_val = B[k * N + col];
//             accum += a_val * b_val;
//         }

//         // 应用alpha
//         T gemm_result = alpha * accum;

//         // 应用SiLU激活函数
//         if constexpr (std::is_same<T, half>::value) {
//             C[row * N + col] = __float2half(silu_function(__half2float(gemm_result)));
//         } else {
//             C[row * N + col] = silu_function(gemm_result);
//         }
//     }
// }

// template<typename T>
// void launch_gemm_silu_fused_kernel(
//     const T* d_A, const T* d_B, T* d_C,
//     int M, int N, int K,
//     T alpha,
//     cudaStream_t stream = 0) {

//     // 设置block和grid大小
//     dim3 block(16, 16);
//     dim3 grid((N + block.x - 1) / block.x,
//               (M + block.y - 1) / block.y);

//     gemm_silu_fused_kernel<T><<<grid, block, 0, stream>>>(
//         d_A, d_B, d_C, M, N, K, alpha);
// }


// 使用FP32累加的融合kernel
template<typename T>
__global__ void gemm_silu_fused_kernel_fp32_accum(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int N, int K,
    T alpha) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // 使用FP32进行累加以保证精度
        float accum = 0.0f;

        // 简单的矩阵乘法
        for (int k = 0; k < K; ++k) {
            float a_val = 0.0f;
            float b_val = 0.0f;

            if constexpr (std::is_same<T, half>::value) {
                a_val = __half2float(A[row * K + k]);
                b_val = __half2float(B[k * N + col]);
            } else {
                a_val = static_cast<float>(A[row * K + k]);
                b_val = static_cast<float>(B[k * N + col]);
            }

            accum += a_val * b_val;
        }

        // 应用alpha
        float alpha_f = 0.0f;
        if constexpr (std::is_same<T, half>::value) {
            alpha_f = __half2float(alpha);
        } else {
            alpha_f = static_cast<float>(alpha);
        }

        float gemm_result = alpha_f * accum;

        // 应用SiLU激活函数，使用数值稳定的实现
        float silu_result = 0.0f;

        // 数值稳定的SiLU实现
        if (gemm_result >= 0.0f) {
            // 当x >= 0时，使用: x / (1 + exp(-x))
            silu_result = gemm_result / (1.0f + expf(-gemm_result));
        } else {
            // 当x < 0时，使用: x * sigmoid(x) = x / (1 + exp(-x))
            // 避免负大数的exp下溢
            float exp_x = expf(gemm_result);
            silu_result = gemm_result * exp_x / (1.0f + exp_x);
        }

        // 转换回目标类型
        if constexpr (std::is_same<T, half>::value) {
            C[row * N + col] = __float2half(silu_result);
        } else {
            C[row * N + col] = static_cast<T>(silu_result);
        }
    }
}

// 优化的融合kernel，使用共享内存
template<typename T>
__global__ void gemm_silu_fused_kernel_optimized(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int N, int K,
    T alpha) {

    // 使用2D block
    const int BLOCK_SIZE = 16;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 计算该线程处理的输出位置
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float accum = 0.0f;

    // 循环遍历所有tile
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // 加载A的tile
        int A_row = row;
        int A_col = t * BLOCK_SIZE + tx;

        if (A_row < M && A_col < K) {
            if constexpr (std::is_same<T, half>::value) {
                As[ty][tx] = __half2float(A[A_row * K + A_col]);
            } else {
                As[ty][tx] = static_cast<float>(A[A_row * K + A_col]);
            }
        } else {
            As[ty][tx] = 0.0f;
        }

        // 加载B的tile
        int B_row = t * BLOCK_SIZE + ty;
        int B_col = col;

        if (B_row < K && B_col < N) {
            if constexpr (std::is_same<T, half>::value) {
                Bs[ty][tx] = __half2float(B[B_row * N + B_col]);
            } else {
                Bs[ty][tx] = static_cast<float>(B[B_row * N + B_col]);
            }
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // 计算tile内的点积
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            accum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // 只让处理有效元素的线程写入结果
    if (row < M && col < N) {
        // 应用alpha
        float alpha_f = 0.0f;
        if constexpr (std::is_same<T, half>::value) {
            alpha_f = __half2float(alpha);
        } else {
            alpha_f = static_cast<float>(alpha);
        }

        float gemm_result = alpha_f * accum;

        // 应用SiLU激活函数（数值稳定版本）
        float silu_result = 0.0f;

        if (gemm_result >= 0.0f) {
            // x >= 0: x / (1 + exp(-x))
            float exp_neg_x = expf(-gemm_result);
            // 防止除零
            if (exp_neg_x < 1e-9f) {
                exp_neg_x = 1e-9f;
            }
            silu_result = gemm_result / (1.0f + exp_neg_x);
        } else {
            // x < 0: x * sigmoid(x)
            float exp_x = expf(gemm_result);
            // 防止数值下溢
            if (exp_x < 1e-9f) {
                exp_x = 1e-9f;
            }
            silu_result = gemm_result * exp_x / (1.0f + exp_x);
        }

        // 转换回目标类型
        if constexpr (std::is_same<T, half>::value) {
            C[row * N + col] = __float2half(silu_result);
        } else {
            C[row * N + col] = static_cast<T>(silu_result);
        }
    }
}

// 简单的融合kernel（用于调试）
template<typename T>
__global__ void gemm_silu_fused_kernel_simple(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int N, int K,
    T alpha) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < M * N) {
        int row = idx / N;
        int col = idx % N;

        // 使用FP32进行累加
        float accum = 0.0f;

        for (int k = 0; k < K; ++k) {
            float a_val = 0.0f;
            float b_val = 0.0f;

            if constexpr (std::is_same<T, half>::value) {
                a_val = __half2float(A[row * K + k]);
                b_val = __half2float(B[k * N + col]);
            } else {
                a_val = static_cast<float>(A[row * K + k]);
                b_val = static_cast<float>(B[k * N + col]);
            }

            accum += a_val * b_val;
        }

        // 应用alpha
        float alpha_f = 0.0f;
        if constexpr (std::is_same<T, half>::value) {
            alpha_f = __half2float(alpha);
        } else {
            alpha_f = static_cast<float>(alpha);
        }

        float gemm_result = alpha_f * accum;

        // 数值稳定的SiLU实现
        float silu_result = 0.0f;

        if (gemm_result >= 0.0f) {
            // 当x >= 0时，使用: x / (1 + exp(-x))
            silu_result = gemm_result / (1.0f + expf(-gemm_result));
        } else {
            // 当x < 0时，使用: x * sigmoid(x) = x / (1 + exp(-x))
            // 避免负大数的exp下溢
            float exp_x = expf(gemm_result);
            silu_result = gemm_result * exp_x / (1.0f + exp_x);
        }

        // 转换回目标类型
        if constexpr (std::is_same<T, half>::value) {
            C[idx] = __float2half(silu_result);
        } else {
            C[idx] = static_cast<T>(silu_result);
        }
    }
}

template<typename T>
void launch_gemm_silu_fused_kernel(
    const T* d_A, const T* d_B, T* d_C,
    int M, int N, int K,
    T alpha,
    cudaStream_t stream = 0,
    bool use_optimized = false) {

    if (use_optimized && M >= 16 && N >= 16 && K >= 16) {
        // 使用优化的kernel
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x,
                  (M + block.y - 1) / block.y);

        gemm_silu_fused_kernel_optimized<T><<<grid, block, 0, stream>>>(
            d_A, d_B, d_C, M, N, K, alpha);
    } else {
        // 使用简单的kernel（每个线程处理一个输出元素）
        const int block_size = 256;
        const int grid_size = (M * N + block_size - 1) / block_size;

        gemm_silu_fused_kernel_simple<T><<<grid_size, block_size, 0, stream>>>(
            d_A, d_B, d_C, M, N, K, alpha);
    }
}

// ==================== 分离版本测试 ====================

template<typename T>
TestResult benchmark_gemm_silu_separate_template(int M, int N, int K, int iterations = 10) {
    TestResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.operation = "GEMM_SiLU_SEPARATE";
    result.iterations = iterations;

    std::ostringstream oss;
    oss << "(" << M << "," << K << "," << N << ")";
    result.test_case = oss.str();

    // 设置数据类型字符串
    if constexpr (std::is_same<T, half>::value) {
        result.data_type = "FP16";
    } else if constexpr (std::is_same<T, float>::value) {
        result.data_type = "FP32";
    }

    std::cout << "运行" << result.data_type << " GEMM+SiLU(分离)测试: "
              << "(" << M << "," << K << ")*(" << K << "," << N << ")" << std::endl;

    // 创建主机端数据（行主序）
    std::vector<T> h_A(M * K);
    std::vector<T> h_B(K * N);
    std::vector<T> h_ref(M * N);

    // 初始化数据
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = random_value<T>();
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = random_value<T>();
    }

    // 设备端内存分配
    T* d_A = nullptr;
    T* d_B = nullptr;
    T* d_gemm_result = nullptr;
    T* d_final_result = nullptr;

    size_t size_A = sizeof(T) * M * K;
    size_t size_B = sizeof(T) * K * N;
    size_t size_C = sizeof(T) * M * N;

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_gemm_result, size_C));
    CUDA_CHECK(cudaMalloc(&d_final_result, size_C));

    // 将数据转换为列主序（cuBLAS默认使用列主序）
    std::vector<T> h_A_colmajor(M * K);
    std::vector<T> h_B_colmajor(K * N);

    // 将行主序A转换为列主序
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            h_A_colmajor[j * M + i] = h_A[i * K + j];
        }
    }

    // 将行主序B转换为列主序
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            h_B_colmajor[j * K + i] = h_B[i * N + j];
        }
    }

    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A_colmajor.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_colmajor.data(), size_B, cudaMemcpyHostToDevice));

    // cuBLAS句柄
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // 设置cuBLAS数学模式（对FP16使用Tensor Core）
    bool use_tensor_core = false;
    if constexpr (std::is_same<T, half>::value) {
        cudaDeviceProp prop;
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        if (prop.major >= 7) {
            CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
            use_tensor_core = true;
            result.data_type = "FP16-TC";
        }
    }

    // 设置GEMM参数
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;

    int m = M;
    int n = N;
    int k = K;

    int lda = m;
    int ldb = k;
    int ldc = m;

    T alpha = T(1.0);
    T beta = T(0.0);

    // 设置数据类型和计算类型
    cudaDataType_t data_type;
    cublasComputeType_t compute_type;

    if constexpr (std::is_same<T, half>::value) {
        data_type = CUDA_R_16F;
        compute_type = CUBLAS_COMPUTE_16F;
    } else if constexpr (std::is_same<T, float>::value) {
        data_type = CUDA_R_32F;
        compute_type = CUBLAS_COMPUTE_32F;
    }

    // 预热运行
    if constexpr (std::is_same<T, half>::value) {
        CUBLAS_CHECK(cublasGemmEx(handle,
                                 transA, transB,
                                 m, n, k,
                                 &alpha,
                                 d_A, data_type, lda,
                                 d_B, data_type, ldb,
                                 &beta,
                                 d_gemm_result, data_type, ldc,
                                 compute_type,
                                 CUBLAS_GEMM_DEFAULT));
        launch_silu_kernel(d_gemm_result, M * N);
    } else if constexpr (std::is_same<T, float>::value) {
        float alpha_f = 1.0f;
        float beta_f = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle,
                                transA, transB,
                                m, n, k,
                                &alpha_f,
                                reinterpret_cast<float*>(d_A), lda,
                                reinterpret_cast<float*>(d_B), ldb,
                                &beta_f,
                                reinterpret_cast<float*>(d_gemm_result), ldc));
        launch_silu_kernel(reinterpret_cast<float*>(d_gemm_result), M * N);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // 性能测试
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double total_time_ms = 0.0;
    double min_time_ms = std::numeric_limits<double>::max();
    double max_time_ms = 0.0;

    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start));

        // 执行GEMM
        if constexpr (std::is_same<T, half>::value) {
            CUBLAS_CHECK(cublasGemmEx(handle,
                                     transA, transB,
                                     m, n, k,
                                     &alpha,
                                     d_A, data_type, lda,
                                     d_B, data_type, ldb,
                                     &beta,
                                     d_gemm_result, data_type, ldc,
                                     compute_type,
                                     CUBLAS_GEMM_DEFAULT));
            // 执行SiLU
            launch_silu_kernel(d_gemm_result, M * N);
        } else if constexpr (std::is_same<T, float>::value) {
            float alpha_f = 1.0f;
            float beta_f = 0.0f;
            CUBLAS_CHECK(cublasSgemm(handle,
                                    transA, transB,
                                    m, n, k,
                                    &alpha_f,
                                    reinterpret_cast<float*>(d_A), lda,
                                    reinterpret_cast<float*>(d_B), ldb,
                                    &beta_f,
                                    reinterpret_cast<float*>(d_gemm_result), ldc));
            launch_silu_kernel(reinterpret_cast<float*>(d_gemm_result), M * N);
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaDeviceSynchronize());

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        total_time_ms += elapsed_ms;
        min_time_ms = std::min(min_time_ms, static_cast<double>(elapsed_ms));
        max_time_ms = std::max(max_time_ms, static_cast<double>(elapsed_ms));
    }

    result.avg_time_ms = total_time_ms / iterations;
    result.min_time_ms = min_time_ms;
    result.max_time_ms = max_time_ms;

    // 拷贝最终结果到主机
    CUDA_CHECK(cudaMemcpy(d_final_result, d_gemm_result, size_C,
                         cudaMemcpyDeviceToDevice));

    // 获取GPU结果（列主序）
    std::vector<T> h_result_colmajor(M * N);
    CUDA_CHECK(cudaMemcpy(h_result_colmajor.data(), d_final_result, size_C,
                         cudaMemcpyDeviceToHost));

    // 将列主序结果转换为行主序，以便与CPU参考结果比较
    std::vector<T> h_result(M * N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            h_result[i * N + j] = h_result_colmajor[j * M + i];
        }
    }

    // 计算CPU参考结果（行主序）
    compute_gemm_silu_cpu_reference<T>(h_ref, h_A, h_B, M, N, K, alpha, MAX_COMPARE_COUNT);

    // 误差分析
    double abs_tolerance = 1e-3;
    double rel_tolerance = 1e-3;

    if constexpr (std::is_same<T, half>::value) {
        abs_tolerance = 1e-2;
        rel_tolerance = 1e-2;
    }

    // 计算实际验证的元素数量
    int total_elements = M * N;
    result.verify_count = std::min(total_elements, MAX_COMPARE_COUNT);

    auto error_result = analyze_errors(h_result, h_ref, 0, result.verify_count,
                                      abs_tolerance, rel_tolerance);

    result.max_abs_error = error_result.max_abs_error;
    result.max_rel_error = error_result.max_rel_error;
    result.passed = error_result.passed;
    result.error_count = error_result.error_count;
    result.total_count = error_result.total_count;

    // 计算性能指标
    // GEMM的FLOPs计数: 2 * M * N * K (矩阵乘法) + 5 * M * N (SiLU近似: exp + 除法等)
    double flops = 2.0 * M * N * K + 5.0 * M * N;
    result.avg_tflops = (flops / result.avg_time_ms) / 1e9;
    result.min_tflops = (flops / result.max_time_ms) / 1e9;
    result.max_tflops = (flops / result.min_time_ms) / 1e9;

    // 带宽计算
    size_t bytes_transferred = (M * K + K * N + M * N) * sizeof(T);  // 读取A、B，写入结果
    result.avg_bandwidth_gbs = (bytes_transferred / result.avg_time_ms) / 1e6;

    // 清理事件
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 清理设备内存和cuBLAS句柄
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_gemm_result));
    CUDA_CHECK(cudaFree(d_final_result));
    CUBLAS_CHECK(cublasDestroy(handle));

    return result;
}

// ==================== 融合版本测试 ====================

template<typename T>
TestResult benchmark_gemm_silu_fused_template(int M, int N, int K, int iterations = 10) {
    TestResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.operation = "GEMM_SiLU_FUSED";
    result.iterations = iterations;

    std::ostringstream oss;
    oss << "(" << M << "," << K << "," << N << ")";
    result.test_case = oss.str();

    // 设置数据类型字符串
    if constexpr (std::is_same<T, half>::value) {
        result.data_type = "FP16";
    } else if constexpr (std::is_same<T, float>::value) {
        result.data_type = "FP32";
    }

    std::cout << "运行" << result.data_type << " GEMM+SiLU(融合)测试: "
              << "(" << M << "," << K << ")*(" << K << "," << N << ")" << std::endl;

    // 创建主机端数据（行主序）
    std::vector<T> h_A(M * K);
    std::vector<T> h_B(K * N);
    std::vector<T> h_ref(M * N);

    // 初始化数据
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = random_value<T>();
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = random_value<T>();
    }

    // 设备端内存分配
    T* d_A = nullptr;
    T* d_B = nullptr;
    T* d_C = nullptr;

    size_t size_A = sizeof(T) * M * K;
    size_t size_B = sizeof(T) * K * N;
    size_t size_C = sizeof(T) * M * N;

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // 拷贝数据到设备（行主序，因为融合kernel使用行主序）
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    // 设置参数
    T alpha = T(1.0);

    // 预热运行
    launch_gemm_silu_fused_kernel(d_A, d_B, d_C, M, N, K, alpha);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 性能测试
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double total_time_ms = 0.0;
    double min_time_ms = std::numeric_limits<double>::max();
    double max_time_ms = 0.0;

    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start));

        // 执行融合kernel
        launch_gemm_silu_fused_kernel(d_A, d_B, d_C, M, N, K, alpha);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaDeviceSynchronize());

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        total_time_ms += elapsed_ms;
        min_time_ms = std::min(min_time_ms, static_cast<double>(elapsed_ms));
        max_time_ms = std::max(max_time_ms, static_cast<double>(elapsed_ms));
    }

    result.avg_time_ms = total_time_ms / iterations;
    result.min_time_ms = min_time_ms;
    result.max_time_ms = max_time_ms;

    // 获取GPU结果（行主序）
    std::vector<T> h_result(M * N);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    // 计算CPU参考结果（行主序）
    compute_gemm_silu_cpu_reference<T>(h_ref, h_A, h_B, M, N, K, alpha, MAX_COMPARE_COUNT);

    // 误差分析
    double abs_tolerance = 1e-3;
    double rel_tolerance = 1e-3;

    if constexpr (std::is_same<T, half>::value) {
        abs_tolerance = 1e-2;
        rel_tolerance = 1e-2;
    }

    // 计算实际验证的元素数量
    int total_elements = M * N;
    result.verify_count = std::min(total_elements, MAX_COMPARE_COUNT);

    auto error_result = analyze_errors(h_result, h_ref, 0, result.verify_count,
                                      abs_tolerance, rel_tolerance);

    result.max_abs_error = error_result.max_abs_error;
    result.max_rel_error = error_result.max_rel_error;
    result.passed = error_result.passed;
    result.error_count = error_result.error_count;
    result.total_count = error_result.total_count;

    // 计算性能指标
    double flops = 2.0 * M * N * K + 5.0 * M * N;
    result.avg_tflops = (flops / result.avg_time_ms) / 1e9;
    result.min_tflops = (flops / result.max_time_ms) / 1e9;
    result.max_tflops = (flops / result.min_time_ms) / 1e9;

    // 带宽计算
    size_t bytes_transferred = (M * K + K * N + M * N) * sizeof(T);
    result.avg_bandwidth_gbs = (bytes_transferred / result.avg_time_ms) / 1e6;

    // 清理事件
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 清理设备内存
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return result;
}

// ==================== 具体实现函数 ====================

TestResult benchmark_gemm_silu_separate_half(int M, int N, int K, int iterations) {
    return benchmark_gemm_silu_separate_template<half>(M, N, K, iterations);
}

TestResult benchmark_gemm_silu_separate_float(int M, int N, int K, int iterations) {
    return benchmark_gemm_silu_separate_template<float>(M, N, K, iterations);
}

TestResult benchmark_gemm_silu_fused_half(int M, int N, int K, int iterations) {
    return benchmark_gemm_silu_fused_template<half>(M, N, K, iterations);
}

TestResult benchmark_gemm_silu_fused_float(int M, int N, int K, int iterations) {
    return benchmark_gemm_silu_fused_template<float>(M, N, K, iterations);
}