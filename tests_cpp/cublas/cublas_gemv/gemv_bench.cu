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
#include "gemv_bench.h"

template<typename T>
void compute_gemv_cpu_reference_cublas(
    std::vector<T>& h_ref,
    const std::vector<T>& h_A,
    const std::vector<T>& h_x,
    const std::vector<T>& h_y,
    int M, int K,
    T alpha, T beta,
    int max_elements = MAX_COMPARE_COUNT) {

    int verify_count = std::min(M, max_elements);
    h_ref.resize(M);

    for (int m = 0; m < verify_count; ++m) {
        T accum = T(0);
        for (int k = 0; k < K; ++k) {
            accum += h_A[m * K + k] * h_x[k];
        }
        h_ref[m] = alpha * accum + beta * h_y[m];
    }
}

// 特化half版本
template<>
void compute_gemv_cpu_reference_cublas<half>(
    std::vector<half>& h_ref,
    const std::vector<half>& h_A,
    const std::vector<half>& h_x,
    const std::vector<half>& h_y,
    int M, int K,
    half alpha, half beta,
    int max_elements) {

    int verify_count = std::min(M, max_elements);
    h_ref.resize(M);

    float alpha_f = __half2float(alpha);
    float beta_f = __half2float(beta);

    for (int m = 0; m < verify_count; ++m) {
        float accum = 0.0f;
        for (int k = 0; k < K; ++k) {
            accum += __half2float(h_A[m * K + k]) * __half2float(h_x[k]);
        }
        float result = alpha_f * accum + beta_f * __half2float(h_y[m]);
        h_ref[m] = __float2half(result);
    }
}

TestResult benchmark_gemv_half(int M, int N, int iterations) {
    TestResult result;
    result.M = M;
    result.N = 1;
    result.K = N;
    result.operation = "GEMV";
    result.data_type = "FP16";
    result.iterations = iterations;

    // 创建数据
    std::vector<half> h_A(M * N);
    std::vector<half> h_x(N);
    std::vector<half> h_y(M);
    std::vector<half> h_ref(M);

    for (int i = 0; i < M * N; ++i) h_A[i] = random_value<half>();
    for (int i = 0; i < N; ++i) h_x[i] = random_value<half>();
    for (int i = 0; i < M; ++i) h_y[i] = random_value<half>();

    // 设备内存
    half *d_A, *d_x, *d_y, *d_result;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(half) * M * N));
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(half) * N));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(half) * M));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(half) * M));

    // 参考代码中直接将行主序A拷贝到设备，没有转换
    // 但注意：cublasGemmEx默认使用列主序，所以我们需要正确处理
    // 方法1：使用CUBLAS_OP_T（转置A）
    // 方法2：将A转换为列主序再拷贝

    // 使用方法2：将A转换为列主序
    std::vector<half> h_A_colmajor(M * N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            h_A_colmajor[j * M + i] = h_A[i * N + j];  // 行主序 -> 列主序
        }
    }

    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A_colmajor.data(), sizeof(half) * M * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), sizeof(half) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), sizeof(half) * M, cudaMemcpyHostToDevice));

    // cuBLAS句柄
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // 根据参考代码，对于FP16，使用half类型的alpha和beta
    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);

    // 使用cublasGemmEx实现GEMV
    // GEMV: y = alpha * A * x + beta * y
    // 转换为GEMM: [Mx1] = alpha * [MxK] * [Kx1] + beta * [Mx1]

    // 根据参考代码设置参数：
    cublasOperation_t transA = CUBLAS_OP_N;  // A已经是列主序，不需要转置
    cublasOperation_t transB = CUBLAS_OP_N;  // x不转置

    // 注意：在参考代码中，m、n、k对应的是C = A * B的维度
    // 对于GEMV: y = A * x，其中A是MxN，x是Nx1，y是Mx1
    // 所以：m = M（输出行数），n = 1（输出列数），k = N（内积维度）
    int m = M;      // 输出行数
    int n = 1;      // 输出列数
    int k = N;      // 内积维度

    int lda = M;    // 列主序A的主维度
    int ldb = k;    // 列向量x的步长
    int ldc = M;    // 结果向量的步长

    // 根据参考代码，对于FP16：
    // AType = BType = CType = ComputeType = CUDA_R_16F
    cudaDataType_t AType = CUDA_R_16F;
    cudaDataType_t BType = CUDA_R_16F;
    cudaDataType_t CType = CUDA_R_16F;
    cublasComputeType_t computeType = CUBLAS_COMPUTE_16F;

    // 使用CUBLAS_GEMM_DEFAULT算法
    // cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    // 预热
    CUBLAS_CHECK(cublasGemmEx(handle,
                             transA,
                             transB,
                             m, n, k,
                             &alpha,          // alpha指针
                             d_A, AType, lda,
                             d_x, BType, ldb,
                             &beta,           // beta指针
                             d_result, CType, ldc,
                             computeType,
                             algo));
    CUDA_CHECK(cudaDeviceSynchronize());

    // 性能测试
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double total_ms = 0.0;
    double min_ms = 1e9;
    double max_ms = 0.0;

    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        CUBLAS_CHECK(cublasGemmEx(handle,
                                 transA,
                                 transB,
                                 m, n, k,
                                 &alpha,
                                 d_A, AType, lda,
                                 d_x, BType, ldb,
                                 &beta,
                                 d_result, CType, ldc,
                                 computeType,
                                 algo));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaDeviceSynchronize());

        float elapsed;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        total_ms += elapsed;
        min_ms = std::min(min_ms, static_cast<double>(elapsed));
        max_ms = std::max(max_ms, static_cast<double>(elapsed));
    }

    result.avg_time_ms = total_ms / iterations;
    result.min_time_ms = min_ms;
    result.max_time_ms = max_ms;

    // 性能指标
    double flops = 2.0 * M * N;
    result.avg_tflops = (flops / result.avg_time_ms) / 1e9;
    result.max_tflops = (flops / result.min_time_ms) / 1e9;

    // 带宽
    size_t bytes = (M * N + N + 2 * M) * sizeof(half);
    result.avg_bandwidth_gbs = (bytes / result.avg_time_ms) / 1e6;

    // 验证
    std::vector<half> h_result(M);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, sizeof(half) * M,
                         cudaMemcpyDeviceToHost));

    // CPU参考计算
    compute_gemv_cpu_reference_cublas<half>(h_ref, h_A, h_x, h_y,
                                           M, N, alpha, beta, MAX_COMPARE_COUNT);

    result.verify_count = std::min(M, MAX_COMPARE_COUNT);
    auto error_result = analyze_errors(h_result, h_ref, 0, result.verify_count,
                                       1e-2, 1e-2);

    result.max_abs_error = error_result.max_abs_error;
    result.passed = error_result.passed;
    result.error_count = error_result.error_count;
    result.total_count = error_result.total_count;

    // 清理
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));

    return result;
}

template<typename T>
TestResult benchmark_cublas_gemv_standard(int M, int N, int iterations = 100,
                                         double alpha_val = 1.0, double beta_val = 0.0) {

    TestResult result;
    result.M = M;
    result.N = 1;
    result.K = N;
    result.operation = "GEMV";
    result.iterations = iterations;

    // 设置数据类型字符串
    if constexpr (std::is_same<T, float>::value) {
        result.data_type = "FP32";
    } else if constexpr (std::is_same<T, double>::value) {
        result.data_type = "FP64";
    }

    // cuBLAS句柄
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // 创建主机端数据
    std::vector<T> h_A(M * N);
    std::vector<T> h_x(N);
    std::vector<T> h_y(M);
    std::vector<T> h_ref(M);

    // 初始化数据
    for (int i = 0; i < M * N; ++i) {
        h_A[i] = random_value<T>();
    }
    for (int i = 0; i < N; ++i) {
        h_x[i] = random_value<T>();
    }
    for (int i = 0; i < M; ++i) {
        h_y[i] = random_value<T>();
    }

    // 设备端内存分配
    T* d_A = nullptr;
    T* d_x = nullptr;
    T* d_y = nullptr;
    T* d_result = nullptr;

    size_t size_A = sizeof(T) * M * N;
    size_t size_x = sizeof(T) * N;
    size_t size_y = sizeof(T) * M;

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_x, size_x));
    CUDA_CHECK(cudaMalloc(&d_y, size_y));
    CUDA_CHECK(cudaMalloc(&d_result, size_y));

    // 注意：cuBLAS默认使用列主序
    // 对于cublasSgemv/cublasDgemv，如果A是行主序，需要设置lda = N
    // 但我们通常按列主序存储

    std::vector<T> h_A_colmajor(M * N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            h_A_colmajor[j * M + i] = h_A[i * N + j];  // 转置：行主序 -> 列主序
        }
    }

    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A_colmajor.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), size_x, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), size_y, cudaMemcpyHostToDevice));

    // 预热运行
    if constexpr (std::is_same<T, float>::value) {
        float alpha = static_cast<float>(alpha_val);
        float beta = static_cast<float>(beta_val);
        // cublasSgemv使用列主序A
        CUBLAS_CHECK(cublasSgemv(handle,
                                CUBLAS_OP_N,
                                M, N,
                                &alpha,
                                d_A, M,  // 列主序：lda = M
                                d_x, 1,
                                &beta,
                                d_result, 1));
    } else if constexpr (std::is_same<T, double>::value) {
        double alpha = alpha_val;
        double beta = beta_val;
        CUBLAS_CHECK(cublasDgemv(handle,
                                CUBLAS_OP_N,
                                M, N,
                                &alpha,
                                d_A, M,  // 列主序：lda = M
                                d_x, 1,
                                &beta,
                                d_result, 1));
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

        if constexpr (std::is_same<T, float>::value) {
            float alpha = static_cast<float>(alpha_val);
            float beta = static_cast<float>(beta_val);
            CUBLAS_CHECK(cublasSgemv(handle,
                                    CUBLAS_OP_N,
                                    M, N,
                                    &alpha,
                                    d_A, M,
                                    d_x, 1,
                                    &beta,
                                    d_result, 1));
        } else if constexpr (std::is_same<T, double>::value) {
            double alpha = alpha_val;
            double beta = beta_val;
            CUBLAS_CHECK(cublasDgemv(handle,
                                    CUBLAS_OP_N,
                                    M, N,
                                    &alpha,
                                    d_A, M,
                                    d_x, 1,
                                    &beta,
                                    d_result, 1));
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
    double flops = 2.0 * M * N;
    result.avg_tflops = (flops / result.avg_time_ms) / 1e9;
    result.min_tflops = (flops / result.max_time_ms) / 1e9;
    result.max_tflops = (flops / result.min_time_ms) / 1e9;

    // 带宽计算
    size_t bytes_transferred = (M * N + N + 2 * M) * sizeof(T);
    result.avg_bandwidth_gbs = (bytes_transferred / result.avg_time_ms) / 1e6;

    // 清理事件
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 获取GPU结果
    std::vector<T> h_result(M);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, size_y, cudaMemcpyDeviceToHost));

    // 计算CPU参考结果（使用原始行主序A）
    if constexpr (std::is_same<T, float>::value) {
        T alpha = static_cast<T>(alpha_val);
        T beta = static_cast<T>(beta_val);
        compute_gemv_cpu_reference_cublas<T>(
            h_ref, h_A, h_x, h_y, M, N, alpha, beta, MAX_COMPARE_COUNT);
    } else if constexpr (std::is_same<T, double>::value) {
        T alpha = static_cast<T>(alpha_val);
        T beta = static_cast<T>(beta_val);
        compute_gemv_cpu_reference_cublas<T>(
            h_ref, h_A, h_x, h_y, M, N, alpha, beta, MAX_COMPARE_COUNT);
    }

    // 误差分析
    double abs_tolerance = 1e-3;
    double rel_tolerance = 1e-3;

    if constexpr (std::is_same<T, double>::value) {
        abs_tolerance = 1e-12;
        rel_tolerance = 1e-12;
    }

    result.verify_count = std::min(M, MAX_COMPARE_COUNT);

    auto error_result = analyze_errors(h_result, h_ref, 0, result.verify_count,
                                      abs_tolerance, rel_tolerance);

    result.max_abs_error = error_result.max_abs_error;
    result.max_rel_error = error_result.max_rel_error;
    result.passed = error_result.passed;
    result.error_count = error_result.error_count;
    result.total_count = error_result.total_count;

    // 清理
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_result));
    CUBLAS_CHECK(cublasDestroy(handle));

    return result;
}

TestResult benchmark_gemv_float(int M, int N, int iterations) {
    return benchmark_cublas_gemv_standard<float>(M, N, iterations, 1.0, 0.0);
}

TestResult benchmark_gemv_double(int M, int N, int iterations) {
    return benchmark_cublas_gemv_standard<double>(M, N, iterations, 1.0, 0.0);
}