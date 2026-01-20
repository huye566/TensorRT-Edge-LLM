#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include "cublas_wrapper_bench.h"
#include "check/err_analysis.h"


template<typename T>
std::vector<T> generate_random_vector(size_t size) {
    std::vector<T> result(size);
    for (size_t i = 0; i < size; ++i) {
        result[i] = random_value<T>();
    }
    return result;
}

// 参考计算函数实现
template<>
void compute_reference_gemm<float>(std::vector<float>& h_ref,
                                  const std::vector<float>& h_A,
                                  const std::vector<float>& h_B,
                                  int M, int N, int K) {
    h_ref.resize(M * N);
    int total_elements = M * N;
    int verify_count = std::min(total_elements, MAX_COMPARE_COUNT);
    
    for (int idx = 0; idx < verify_count; ++idx) {
        int m = idx / N;
        int n = idx % N;
        float accum = 0.0f;
        
        for (int k = 0; k < K; ++k) {
            // A行主序: A(m, k) = h_A[m * K + k]
            // B行主序: B(k, n) = h_B[k * N + n]
            float a_val = h_A[m * K + k];
            float b_val = h_B[k * N + n];
            accum += a_val * b_val;
        }
        
        h_ref[m * N + n] = accum;
    }
}

template<>
void compute_reference_gemm<__half>(std::vector<__half>& h_ref,
                                   const std::vector<__half>& h_A,
                                   const std::vector<__half>& h_B,
                                   int M, int N, int K) {
    h_ref.resize(M * N);
    int total_elements = M * N;
    int verify_count = std::min(total_elements, MAX_COMPARE_COUNT);
    
    for (int idx = 0; idx < verify_count; ++idx) {
        int m = idx / N;
        int n = idx % N;
        float accum = 0.0f;
        
        for (int k = 0; k < K; ++k) {
            float a_val = __half2float(h_A[m * K + k]);
            float b_val = __half2float(h_B[k * N + n]);
            accum += a_val * b_val;
        }
        
        h_ref[m * N + n] = __float2half(accum);
    }
}

template<>
void compute_reference_gemm_silu<float>(std::vector<float>& h_ref,
                                       const std::vector<float>& h_A,
                                       const std::vector<float>& h_B,
                                       int M, int N, int K) {
    h_ref.resize(M * N);
    int total_elements = M * N;
    int verify_count = std::min(total_elements, MAX_COMPARE_COUNT);
    
    for (int idx = 0; idx < verify_count; ++idx) {
        int m = idx / N;
        int n = idx % N;
        float accum = 0.0f;
        
        for (int k = 0; k < K; ++k) {
            float a_val = h_A[m * K + k];
            float b_val = h_B[k * N + n];
            accum += a_val * b_val;
        }
        
        // SiLU激活函数: x * sigmoid(x)
        float silu = accum * (1.0f / (1.0f + expf(-accum)));
        h_ref[m * N + n] = silu;
    }
}

template<>
void compute_reference_gemm_silu<__half>(std::vector<__half>& h_ref,
                                        const std::vector<__half>& h_A,
                                        const std::vector<__half>& h_B,
                                        int M, int N, int K) {
    h_ref.resize(M * N);
    int total_elements = M * N;
    int verify_count = std::min(total_elements, MAX_COMPARE_COUNT);
    
    for (int idx = 0; idx < verify_count; ++idx) {
        int m = idx / N;
        int n = idx % N;
        float accum = 0.0f;
        
        for (int k = 0; k < K; ++k) {
            float a_val = __half2float(h_A[m * K + k]);
            float b_val = __half2float(h_B[k * N + n]);
            accum += a_val * b_val;
        }
        
        // SiLU激活函数: x * sigmoid(x)
        float silu = accum * (1.0f / (1.0f + expf(-accum)));
        h_ref[m * N + n] = __float2half(silu);
    }
}

template<>
void compute_reference_gemm_bias<float>(std::vector<float>& h_ref,
                                       const std::vector<float>& h_A,
                                       const std::vector<float>& h_B,
                                       const std::vector<float>& h_bias,
                                       int M, int N, int K) {
    h_ref.resize(M * N);
    int total_elements = M * N;
    int verify_count = std::min(total_elements, MAX_COMPARE_COUNT);
    
    for (int idx = 0; idx < verify_count; ++idx) {
        int m = idx / N;
        int n = idx % N;
        float accum = 0.0f;
        
        for (int k = 0; k < K; ++k) {
            float a_val = h_A[m * K + k];
            float b_val = h_B[k * N + n];
            accum += a_val * b_val;
        }
        
        // 加上偏置
        accum += h_bias[n];
        h_ref[m * N + n] = accum;
    }
}

template<>
void compute_reference_gemm_bias<__half>(std::vector<__half>& h_ref,
                                        const std::vector<__half>& h_A,
                                        const std::vector<__half>& h_B,
                                        const std::vector<__half>& h_bias,
                                        int M, int N, int K) {
    h_ref.resize(M * N);
    int total_elements = M * N;
    int verify_count = std::min(total_elements, MAX_COMPARE_COUNT);
    
    for (int idx = 0; idx < verify_count; ++idx) {
        int m = idx / N;
        int n = idx % N;
        float accum = 0.0f;
        
        for (int k = 0; k < K; ++k) {
            float a_val = __half2float(h_A[m * K + k]);
            float b_val = __half2float(h_B[k * N + n]);
            accum += a_val * b_val;
        }
        
        // 加上偏置
        accum += __half2float(h_bias[n]);
        h_ref[m * N + n] = __float2half(accum);
    }
}

// cuBLAS GEMM运行函数
enum class CublasGemmType {
    GEMM,
    GEMM_SiLU,
    GEMM_Bias
};

template<typename T>
void cublas_gemm_run(T* d_C,
                    const T* d_A,
                    const T* d_B,
                    const T* d_bias,
                    int M,
                    int N,
                    int K,
                    CublasGemmType gemm_type,
                    cublasHandle_t handle,
                    cudaStream_t stream = 0) {
    switch (gemm_type) {
        case CublasGemmType::GEMM:
            trt_edgellm::kernel::cublas_gemm<T>(handle, M, N, K, d_A, d_B, d_C);
            break;
        case CublasGemmType::GEMM_SiLU:
            trt_edgellm::kernel::cublas_gemm_silu<T>(handle, M, N, K, d_A, d_B, d_C);
            break;
        case CublasGemmType::GEMM_Bias:
            trt_edgellm::kernel::cublas_gemm_bias<T>(handle, M, N, K, d_A, d_B, d_bias, d_C);
            break;
    }
}

// 测试基本GEMM（模板函数）
template<typename T>
TestResult benchmark_cublas_gemm(int M, int N, int K, int iterations) {
    TestResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    
    // 设置数据类型标签
    if constexpr (std::is_same<T, float>::value) {
        result.data_type = "FP32";
        result.operation = "cuBLAS_GEMM_FP32";
    } else if constexpr (std::is_same<T, __half>::value) {
        result.data_type = "FP16";
        result.operation = "cuBLAS_GEMM_FP16";
    } else {
        result.data_type = "UNKNOWN";
        result.operation = "cuBLAS_GEMM";
    }
    
    result.iterations = iterations;
    
    std::ostringstream oss;
    oss << "(" << M << "," << K << "," << N << ")";
    result.test_case = oss.str();

    // 初始化cuBLAS
    auto& cublas_wrapper = trt_edgellm::kernel::CublasWrapper::instance();
    cublasHandle_t handle = cublas_wrapper.handle();
    
    // 生成随机数据
    auto h_A = generate_random_vector<T>(M * K);
    auto h_B = generate_random_vector<T>(K * N);
    
    // 分配设备内存
    T* d_A = nullptr;
    T* d_B = nullptr;
    T* d_C = nullptr;
    
    size_t size_A = M * K * sizeof(T);
    size_t size_B = K * N * sizeof(T);
    size_t size_C = M * N * sizeof(T);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    
    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    
    // 预热
    cublas_gemm_run<T>(d_C, d_A, d_B, nullptr, M, N, K, CublasGemmType::GEMM, handle);
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
        cublas_gemm_run<T>(d_C, d_A, d_B, nullptr, M, N, K, CublasGemmType::GEMM, handle);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
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
    double flops = 2.0 * M * N * K;
    result.avg_tflops = (flops / result.avg_time_ms) / 1e9;
    result.min_tflops = (flops / result.max_time_ms) / 1e9;
    result.max_tflops = (flops / result.min_time_ms) / 1e9;
    
    // 计算带宽
    size_t bytes_transferred = (M * K + K * N + M * N) * sizeof(T);
    result.avg_bandwidth_gbs = (bytes_transferred / result.avg_time_ms) / 1e6;
    
    // 清理事件
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    // 验证结果
    std::vector<T> h_C(M * N);
    std::vector<T> h_ref;
    
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    
    // 计算CPU参考结果
    compute_reference_gemm<T>(h_ref, h_A, h_B, M, N, K);
    
    // 误差分析
    result.verify_count = std::min(M * N, MAX_COMPARE_COUNT);
    double abs_tolerance, rel_tolerance;
    
    if constexpr (std::is_same<T, float>::value) {
        abs_tolerance = 1e-5;
        rel_tolerance = 1e-5;
    } else if constexpr (std::is_same<T, __half>::value) {
        abs_tolerance = 1e-2;
        rel_tolerance = 1e-2;
    } else {
        abs_tolerance = 1e-3;
        rel_tolerance = 1e-3;
    }
    
    auto error_result = analyze_errors<T>(h_C, h_ref, 0, result.verify_count, 
                                         abs_tolerance, rel_tolerance);
    
    result.max_abs_error = error_result.max_abs_error;
    result.max_rel_error = error_result.max_rel_error;
    result.passed = error_result.passed;
    result.error_count = error_result.error_count;
    result.total_count = error_result.total_count;
    
    // 清理设备内存
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    return result;
}

// 测试GEMM+SiLU（模板函数）
template<typename T>
TestResult benchmark_cublas_gemm_silu(int M, int N, int K, int iterations) {
    TestResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    
    // 设置数据类型标签
    if constexpr (std::is_same<T, float>::value) {
        result.data_type = "FP32";
        result.operation = "cuBLAS_GEMM_SiLU_FP32";
    } else if constexpr (std::is_same<T, __half>::value) {
        result.data_type = "FP16";
        result.operation = "cuBLAS_GEMM_SiLU_FP16";
    } else {
        result.data_type = "UNKNOWN";
        result.operation = "cuBLAS_GEMM_SiLU";
    }
    
    result.iterations = iterations;
    
    std::ostringstream oss;
    oss << "(" << M << "," << K << "," << N << ")";
    result.test_case = oss.str();

    // 初始化cuBLAS
    auto& cublas_wrapper = trt_edgellm::kernel::CublasWrapper::instance();
    cublasHandle_t handle = cublas_wrapper.handle();
    
    // 生成随机数据
    auto h_A = generate_random_vector<T>(M * K);
    auto h_B = generate_random_vector<T>(K * N);
    
    // 分配设备内存
    T* d_A = nullptr;
    T* d_B = nullptr;
    T* d_C = nullptr;
    
    size_t size_A = M * K * sizeof(T);
    size_t size_B = K * N * sizeof(T);
    size_t size_C = M * N * sizeof(T);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    
    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    
    // 预热
    cublas_gemm_run<T>(d_C, d_A, d_B, nullptr, M, N, K, CublasGemmType::GEMM_SiLU, handle);
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
        cublas_gemm_run<T>(d_C, d_A, d_B, nullptr, M, N, K, CublasGemmType::GEMM_SiLU, handle);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
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
    double flops = 2.0 * M * N * K;
    result.avg_tflops = (flops / result.avg_time_ms) / 1e9;
    result.min_tflops = (flops / result.max_time_ms) / 1e9;
    result.max_tflops = (flops / result.min_time_ms) / 1e9;
    
    // 计算带宽
    size_t bytes_transferred = (M * K + K * N + M * N) * sizeof(T);
    result.avg_bandwidth_gbs = (bytes_transferred / result.avg_time_ms) / 1e6;
    
    // 清理事件
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    // 验证结果
    std::vector<T> h_C(M * N);
    std::vector<T> h_ref;
    
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    
    // 计算CPU参考结果（包含SiLU激活）
    compute_reference_gemm_silu<T>(h_ref, h_A, h_B, M, N, K);
    
    // 误差分析
    result.verify_count = std::min(M * N, MAX_COMPARE_COUNT);
    double abs_tolerance, rel_tolerance;
    
    if constexpr (std::is_same<T, float>::value) {
        abs_tolerance = 1e-5;
        rel_tolerance = 1e-5;
    } else if constexpr (std::is_same<T, __half>::value) {
        abs_tolerance = 1e-2;
        rel_tolerance = 1e-2;
    } else {
        abs_tolerance = 1e-3;
        rel_tolerance = 1e-3;
    }
    
    auto error_result = analyze_errors<T>(h_C, h_ref, 0, result.verify_count, 
                                         abs_tolerance, rel_tolerance);
    
    result.max_abs_error = error_result.max_abs_error;
    result.max_rel_error = error_result.max_rel_error;
    result.passed = error_result.passed;
    result.error_count = error_result.error_count;
    result.total_count = error_result.total_count;
    
    // 清理设备内存
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    return result;
}

// 测试GEMM+Bias（模板函数）
template<typename T>
TestResult benchmark_cublas_gemm_bias(int M, int N, int K, int iterations) {
    TestResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    
    // 设置数据类型标签
    if constexpr (std::is_same<T, float>::value) {
        result.data_type = "FP32";
        result.operation = "cuBLAS_GEMM_Bias_FP32";
    } else if constexpr (std::is_same<T, __half>::value) {
        result.data_type = "FP16";
        result.operation = "cuBLAS_GEMM_Bias_FP16";
    } else {
        result.data_type = "UNKNOWN";
        result.operation = "cuBLAS_GEMM_Bias";
    }
    
    result.iterations = iterations;
    
    std::ostringstream oss;
    oss << "(" << M << "," << K << "," << N << ")";
    result.test_case = oss.str();

    // 初始化cuBLAS
    auto& cublas_wrapper = trt_edgellm::kernel::CublasWrapper::instance();
    cublasHandle_t handle = cublas_wrapper.handle();
    
    // 生成随机数据
    auto h_A = generate_random_vector<T>(M * K);
    auto h_B = generate_random_vector<T>(K * N);
    auto h_bias = generate_random_vector<T>(N);
    
    // 分配设备内存
    T* d_A = nullptr;
    T* d_B = nullptr;
    T* d_bias = nullptr;
    T* d_C = nullptr;
    
    size_t size_A = M * K * sizeof(T);
    size_t size_B = K * N * sizeof(T);
    size_t size_bias = N * sizeof(T);
    size_t size_C = M * N * sizeof(T);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_bias, size_bias));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    
    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), size_bias, cudaMemcpyHostToDevice));
    
    // 预热
    cublas_gemm_run<T>(d_C, d_A, d_B, d_bias, M, N, K, CublasGemmType::GEMM_Bias, handle);
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
        cublas_gemm_run<T>(d_C, d_A, d_B, d_bias, M, N, K, CublasGemmType::GEMM_Bias, handle);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
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
    double flops = 2.0 * M * N * K;
    result.avg_tflops = (flops / result.avg_time_ms) / 1e9;
    result.min_tflops = (flops / result.max_time_ms) / 1e9;
    result.max_tflops = (flops / result.min_time_ms) / 1e9;
    
    // 计算带宽
    size_t bytes_transferred = (M * K + K * N + N + M * N) * sizeof(T);
    result.avg_bandwidth_gbs = (bytes_transferred / result.avg_time_ms) / 1e6;
    
    // 清理事件
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    // 验证结果
    std::vector<T> h_C(M * N);
    std::vector<T> h_ref;
    
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    
    // 计算CPU参考结果（包含bias）
    compute_reference_gemm_bias<T>(h_ref, h_A, h_B, h_bias, M, N, K);
    
    // 误差分析
    result.verify_count = std::min(M * N, MAX_COMPARE_COUNT);
    double abs_tolerance, rel_tolerance;
    
    if constexpr (std::is_same<T, float>::value) {
        abs_tolerance = 1e-5;
        rel_tolerance = 1e-5;
    } else if constexpr (std::is_same<T, __half>::value) {
        abs_tolerance = 1e-2;
        rel_tolerance = 1e-2;
    } else {
        abs_tolerance = 1e-3;
        rel_tolerance = 1e-3;
    }
    
    auto error_result = analyze_errors<T>(h_C, h_ref, 0, result.verify_count, 
                                         abs_tolerance, rel_tolerance);
    
    result.max_abs_error = error_result.max_abs_error;
    result.max_rel_error = error_result.max_rel_error;
    result.passed = error_result.passed;
    result.error_count = error_result.error_count;
    result.total_count = error_result.total_count;
    
    // 清理设备内存
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_C));
    
    return result;
}

// 显式实例化模板函数
template TestResult benchmark_cublas_gemm<float>(int M, int N, int K, int iterations);
template TestResult benchmark_cublas_gemm<__half>(int M, int N, int K, int iterations);

template TestResult benchmark_cublas_gemm_silu<float>(int M, int N, int K, int iterations);
template TestResult benchmark_cublas_gemm_silu<__half>(int M, int N, int K, int iterations);

template TestResult benchmark_cublas_gemm_bias<float>(int M, int N, int K, int iterations);
template TestResult benchmark_cublas_gemm_bias<__half>(int M, int N, int K, int iterations);
