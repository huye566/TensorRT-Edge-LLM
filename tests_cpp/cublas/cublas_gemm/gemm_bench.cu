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
#include "gemm_bench.h"

template<typename T>
void compute_gemm_cpu_reference(
    std::vector<T>& h_ref,
    const std::vector<T>& h_A,
    const std::vector<T>& h_B,
    const std::vector<T>& h_C,
    int M, int N, int K,
    T alpha, T beta,
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
        h_ref[idx] = alpha * accum + beta * h_C[idx];
    }
}

// 特化half版本
template<>
void compute_gemm_cpu_reference<half>(
    std::vector<half>& h_ref,
    const std::vector<half>& h_A,
    const std::vector<half>& h_B,
    const std::vector<half>& h_C,
    int M, int N, int K,
    half alpha, half beta,
    int max_elements) {

    int total_elements = M * N;
    int verify_count = std::min(total_elements, max_elements);
    h_ref.resize(total_elements);

    float alpha_f = __half2float(alpha);
    float beta_f = __half2float(beta);

    for (int idx = 0; idx < verify_count; ++idx) {
        int i = idx / N;  // 行索引
        int j = idx % N;  // 列索引

        float accum = 0.0f;
        for (int k = 0; k < K; ++k) {
            accum += __half2float(h_A[i * K + k]) * __half2float(h_B[k * N + j]);
        }
        float result = alpha_f * accum + beta_f * __half2float(h_C[idx]);
        h_ref[idx] = __float2half(result);
    }
}


template<typename T>
TestResult benchmark_gemm_template(int M, int N, int K, int iterations = 10) {
    TestResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.operation = "GEMM";
    result.iterations = iterations;

    std::ostringstream oss;
    oss << "(" << M << "," << K << "," << N << ")";
    result.test_case = oss.str();

    // 设置数据类型字符串
    if constexpr (std::is_same<T, half>::value) {
        result.data_type = "FP16";
    } else if constexpr (std::is_same<T, float>::value) {
        result.data_type = "FP32";
    } else if constexpr (std::is_same<T, double>::value) {
        result.data_type = "FP64";
    }

    std::cout << "运行" << result.data_type << " GEMM测试: "
              << "(" << M << "," << K << ")*(" << K << "," << N << ")" << std::endl;

    // 创建主机端数据（行主序）
    std::vector<T> h_A(M * K);
    std::vector<T> h_B(K * N);
    std::vector<T> h_C(M * N);
    std::vector<T> h_ref(M * N);

    // 初始化数据
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = random_value<T>();
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = random_value<T>();
    }
    for (int i = 0; i < M * N; ++i) {
        h_C[i] = random_value<T>();
    }

    // 设备端内存分配
    T* d_A = nullptr;
    T* d_B = nullptr;
    T* d_C = nullptr;
    T* d_result = nullptr;

    size_t size_A = sizeof(T) * M * K;
    size_t size_B = sizeof(T) * K * N;
    size_t size_C = sizeof(T) * M * N;

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMalloc(&d_result, size_C));

    // 将数据转换为列主序（cuBLAS默认使用列主序）
    std::vector<T> h_A_colmajor(M * K);
    std::vector<T> h_B_colmajor(K * N);
    std::vector<T> h_C_colmajor(M * N);

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

    // 将行主序C转换为列主序
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            h_C_colmajor[j * M + i] = h_C[i * N + j];
        }
    }

    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A_colmajor.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_colmajor.data(), size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C_colmajor.data(), size_C, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_result, h_C_colmajor.data(), size_C, cudaMemcpyHostToDevice));

    // cuBLAS句柄
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // 设置cuBLAS数学模式（对FP16使用Tensor Core）
    bool use_tensor_core = false;
    if constexpr (std::is_same<T, half>::value) {
        // 检查是否支持Tensor Core
        cudaDeviceProp prop;
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        if (prop.major >= 7) {
            CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
            use_tensor_core = true;
            result.data_type = "FP16-TC";  // 标记使用了Tensor Core
        }
    }

    // 设置GEMM参数
    // 计算 C = alpha * A * B + beta * C
    // 其中 A: MxK, B: KxN, C: MxN

    cublasOperation_t transA = CUBLAS_OP_N;  // 不转置
    cublasOperation_t transB = CUBLAS_OP_N;  // 不转置

    int m = M;      // 输出矩阵C的列数（因为B是KxN，不转置）
    int n = N;      // 输出矩阵C的行数（因为A是MxK，不转置）
    int k = K;      // 内积维度

    int lda = m;    // 列主序A的主维度（列数）
    int ldb = k;    // 列主序B的主维度（列数）
    int ldc = m;    // 列主序C的主维度（列数）

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
    } else if constexpr (std::is_same<T, double>::value) {
        data_type = CUDA_R_64F;
        compute_type = CUBLAS_COMPUTE_64F;
    }

    // 预热运行
    if constexpr (std::is_same<T, half>::value) {
        // 对于FP16，使用cublasGemmEx
        CUBLAS_CHECK(cublasGemmEx(handle,
                                 transA, transB,
                                 m, n, k,
                                 &alpha,
                                 d_A, data_type, lda,
                                 d_B, data_type, ldb,
                                 &beta,
                                 d_result, data_type, ldc,
                                 compute_type,
                                 CUBLAS_GEMM_DEFAULT));
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
                                reinterpret_cast<float*>(d_result), ldc));
    } else if constexpr (std::is_same<T, double>::value) {
        double alpha_d = 1.0;
        double beta_d = 0.0;
        CUBLAS_CHECK(cublasDgemm(handle,
                                transA, transB,
                                m, n, k,
                                &alpha_d,
                                reinterpret_cast<double*>(d_A), lda,
                                reinterpret_cast<double*>(d_B), ldb,
                                &beta_d,
                                reinterpret_cast<double*>(d_result), ldc));
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

        if constexpr (std::is_same<T, half>::value) {
            CUBLAS_CHECK(cublasGemmEx(handle,
                                     transA, transB,
                                     m, n, k,
                                     &alpha,
                                     d_A, data_type, lda,
                                     d_B, data_type, ldb,
                                     &beta,
                                     d_result, data_type, ldc,
                                     compute_type,
                                     CUBLAS_GEMM_DEFAULT));
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
                                    reinterpret_cast<float*>(d_result), ldc));
        } else if constexpr (std::is_same<T, double>::value) {
            double alpha_d = 1.0;
            double beta_d = 0.0;
            CUBLAS_CHECK(cublasDgemm(handle,
                                    transA, transB,
                                    m, n, k,
                                    &alpha_d,
                                    reinterpret_cast<double*>(d_A), lda,
                                    reinterpret_cast<double*>(d_B), ldb,
                                    &beta_d,
                                    reinterpret_cast<double*>(d_result), ldc));
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

    // 计算性能指标
    // GEMM的FLOPs计数: 2 * M * N * K
    double flops = 2.0 * M * N * K;
    result.avg_tflops = (flops / result.avg_time_ms) / 1e9;
    result.min_tflops = (flops / result.max_time_ms) / 1e9;  // 最小时间对应最大性能
    result.max_tflops = (flops / result.min_time_ms) / 1e9;  // 最大时间对应最小性能

    // 带宽计算
    // 读取: A (M*K), B (K*N), C (M*N)
    // 写入: 结果 (M*N)
    size_t bytes_transferred = (M * K + K * N + 2 * M * N) * sizeof(T);
    result.avg_bandwidth_gbs = (bytes_transferred / result.avg_time_ms) / 1e6;

    // 清理事件
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 获取GPU结果（列主序）
    std::vector<T> h_result_colmajor(M * N);
    CUDA_CHECK(cudaMemcpy(h_result_colmajor.data(), d_result, size_C,
                         cudaMemcpyDeviceToHost));

    // 将列主序结果转换为行主序，以便与CPU参考结果比较
    std::vector<T> h_result(M * N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            h_result[i * N + j] = h_result_colmajor[j * M + i];
        }
    }

    // 计算CPU参考结果（行主序）
    compute_gemm_cpu_reference<T>(h_ref, h_A, h_B, h_C, M, N, K,
                                  alpha, beta, MAX_COMPARE_COUNT);

    // 误差分析
    double abs_tolerance = 1e-3;
    double rel_tolerance = 1e-3;

    if constexpr (std::is_same<T, half>::value) {
        abs_tolerance = 1e-2;
        rel_tolerance = 1e-2;
    } else if constexpr (std::is_same<T, double>::value) {
        abs_tolerance = 1e-12;
        rel_tolerance = 1e-12;
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

    // 清理设备内存和cuBLAS句柄
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_result));
    CUBLAS_CHECK(cublasDestroy(handle));

    return result;
}

// 具体的GEMM测试函数
TestResult benchmark_gemm_half(int M, int N, int K, int iterations) {
    return benchmark_gemm_template<half>(M, N, K, iterations);
}

TestResult benchmark_gemm_float(int M, int N, int K, int iterations) {
    return benchmark_gemm_template<float>(M, N, K, iterations);
}

TestResult benchmark_gemm_double(int M, int N, int K, int iterations) {
    return benchmark_gemm_template<double>(M, N, K, iterations);
}