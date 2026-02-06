#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <chrono>
#include "cutlass_wrapper_blackwell_bench.h"
#include "check/err_analysis.h"

// 生成随机向量
template<typename T>
std::vector<T> generate_random_vector(size_t size) {
    std::vector<T> result(size);
    for (size_t i = 0; i < size; ++i) {
        result[i] = random_value<T>();
    }
    return result;
}

// 参考计算函数实现
void compute_reference_blackwell_gemm(std::vector<cutlass::half_t>& h_ref,
                                     const std::vector<cutlass::half_t>& h_A,
                                     const std::vector<cutlass::half_t>& h_B,
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
            float a_val = static_cast<float>(h_A[m * K + k]);
            float b_val = static_cast<float>(h_B[k * N + n]);
            accum += a_val * b_val;
        }

        h_ref[m * N + n] = cutlass::half_t(accum);
    }
}

void compute_reference_blackwell_gemm_silu(std::vector<cutlass::half_t>& h_ref,
                                          const std::vector<cutlass::half_t>& h_A,
                                          const std::vector<cutlass::half_t>& h_B,
                                          int M, int N, int K) {
    h_ref.resize(M * N);
    int total_elements = M * N;
    int verify_count = std::min(total_elements, MAX_COMPARE_COUNT);

    for (int idx = 0; idx < verify_count; ++idx) {
        int m = idx / N;
        int n = idx % N;
        float accum = 0.0f;

        for (int k = 0; k < K; ++k) {
            float a_val = static_cast<float>(h_A[m * K + k]);
            float b_val = static_cast<float>(h_B[k * N + n]);
            accum += a_val * b_val;
        }

        // SiLU激活函数: x * sigmoid(x)
        float silu = accum * (1.0f / (1.0f + expf(-accum)));
        h_ref[m * N + n] = cutlass::half_t(silu);
    }
}

void compute_reference_blackwell_gemm_bias(std::vector<cutlass::half_t>& h_ref,
                                          const std::vector<cutlass::half_t>& h_A,
                                          const std::vector<cutlass::half_t>& h_B,
                                          const std::vector<cutlass::half_t>& h_bias,
                                          int M, int N, int K) {
    h_ref.resize(M * N);
    int total_elements = M * N;
    int verify_count = std::min(total_elements, MAX_COMPARE_COUNT);

    for (int idx = 0; idx < verify_count; ++idx) {
        int m = idx / N;
        int n = idx % N;
        float accum = 0.0f;

        for (int k = 0; k < K; ++k) {
            float a_val = static_cast<float>(h_A[m * K + k]);
            float b_val = static_cast<float>(h_B[k * N + n]);
            accum += a_val * b_val;
        }

        // 加上偏置
        accum += static_cast<float>(h_bias[n]);
        h_ref[m * N + n] = cutlass::half_t(accum);
    }
}

// Blackwell GEMM运行函数
enum class BlackwellGemmType {
    GEMM,
    GEMM_SiLU,
    GEMM_Bias
};

void blackwell_gemm_half_run(cutlass::half_t* d_C,
                             const cutlass::half_t* d_A,
                             const cutlass::half_t* d_B,
                             const cutlass::half_t* d_bias,
                             int M,
                             int N,
                             int K,
                             BlackwellGemmType gemm_type,
                             cudaStream_t stream = 0,
                             bool use_cached = true) {
    int M_ = M;
    int N_ = N;
    int K_ = K;

    switch (gemm_type) {
        case BlackwellGemmType::GEMM:
            trt_edgellm::kernel::cutlass_gemm_blackwell<false, false>(
                d_C, d_A, d_B, nullptr, M_, N_, K_, stream, use_cached);
            break;
        case BlackwellGemmType::GEMM_SiLU:
            trt_edgellm::kernel::cutlass_gemm_blackwell<true, false>(
                d_C, d_A, d_B, nullptr, M_, N_, K_, stream, use_cached);
            break;
        case BlackwellGemmType::GEMM_Bias:
            trt_edgellm::kernel::cutlass_gemm_blackwell<false, true>(
                d_C, d_A, d_B, d_bias, M_, N_, K_, stream, use_cached);
            break;
    }
}

// 测试基本GEMM
TestResult benchmark_blackwell_gemm_half(int M, int N, int K, int iterations) {
    TestResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.data_type = "FP16";
    result.operation = "Blackwell_GEMM";
    result.iterations = iterations;

    std::ostringstream oss;
    oss << "(" << M << "," << K << "," << N << ")";
    result.test_case = oss.str();

    // 生成随机数据
    auto h_A = generate_random_vector<cutlass::half_t>(M * K);
    auto h_B = generate_random_vector<cutlass::half_t>(K * N);

    // 分配设备内存
    cutlass::half_t* d_A = nullptr;
    cutlass::half_t* d_B = nullptr;
    cutlass::half_t* d_C = nullptr;

    size_t size_A = M * K * sizeof(cutlass::half_t);
    size_t size_B = K * N * sizeof(cutlass::half_t);
    size_t size_C = M * N * sizeof(cutlass::half_t);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    // 预热
    blackwell_gemm_half_run(d_C, d_A, d_B, nullptr, M, N, K, BlackwellGemmType::GEMM);
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
        blackwell_gemm_half_run(d_C, d_A, d_B, nullptr, M, N, K, BlackwellGemmType::GEMM);
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
    size_t bytes_transferred = (M * K + K * N + M * N) * sizeof(cutlass::half_t);
    result.avg_bandwidth_gbs = (bytes_transferred / result.avg_time_ms) / 1e6;

    // 清理事件
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 验证结果
    std::vector<cutlass::half_t> h_C(M * N);
    std::vector<cutlass::half_t> h_ref;

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    // 计算CPU参考结果
    compute_reference_blackwell_gemm(h_ref, h_A, h_B, M, N, K);

    // 误差分析
    result.verify_count = std::min(M * N, MAX_COMPARE_COUNT);
    double abs_tolerance = 1e-2;
    double rel_tolerance = 1e-2;

    auto error_result = analyze_errors(h_C, h_ref, 0, result.verify_count,
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

// 测试GEMM+SiLU
TestResult benchmark_blackwell_gemm_silu_half(int M, int N, int K, int iterations) {
    TestResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.data_type = "FP16";
    result.operation = "Blackwell_GEMM_SiLU";
    result.iterations = iterations;

    std::ostringstream oss;
    oss << "(" << M << "," << K << "," << N << ")";
    result.test_case = oss.str();

    // 生成随机数据
    auto h_A = generate_random_vector<cutlass::half_t>(M * K);
    auto h_B = generate_random_vector<cutlass::half_t>(K * N);

    // 分配设备内存
    cutlass::half_t* d_A = nullptr;
    cutlass::half_t* d_B = nullptr;
    cutlass::half_t* d_C = nullptr;

    size_t size_A = M * K * sizeof(cutlass::half_t);
    size_t size_B = K * N * sizeof(cutlass::half_t);
    size_t size_C = M * N * sizeof(cutlass::half_t);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    // 预热
    blackwell_gemm_half_run(d_C, d_A, d_B, nullptr, M, N, K, BlackwellGemmType::GEMM_SiLU);
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
        blackwell_gemm_half_run(d_C, d_A, d_B, nullptr, M, N, K, BlackwellGemmType::GEMM_SiLU);
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
    size_t bytes_transferred = (M * K + K * N + M * N) * sizeof(cutlass::half_t);
    result.avg_bandwidth_gbs = (bytes_transferred / result.avg_time_ms) / 1e6;

    // 清理事件
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 验证结果
    std::vector<cutlass::half_t> h_C(M * N);
    std::vector<cutlass::half_t> h_ref;

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    // 计算CPU参考结果（包含SiLU激活）
    compute_reference_blackwell_gemm_silu(h_ref, h_A, h_B, M, N, K);

    // 误差分析
    result.verify_count = std::min(M * N, MAX_COMPARE_COUNT);
    double abs_tolerance = 1e-2;
    double rel_tolerance = 1e-2;

    auto error_result = analyze_errors(h_C, h_ref, 0, result.verify_count,
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

// 测试GEMM+Bias
TestResult benchmark_blackwell_gemm_bias_half(int M, int N, int K, int iterations) {
    TestResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.data_type = "FP16";
    result.operation = "Blackwell_GEMM_Bias";
    result.iterations = iterations;

    std::ostringstream oss;
    oss << "(" << M << "," << K << "," << N << ")";
    result.test_case = oss.str();

    // 生成随机数据
    auto h_A = generate_random_vector<cutlass::half_t>(M * K);
    auto h_B = generate_random_vector<cutlass::half_t>(K * N);
    auto h_bias = generate_random_vector<cutlass::half_t>(N);

    // 分配设备内存
    cutlass::half_t* d_A = nullptr;
    cutlass::half_t* d_B = nullptr;
    cutlass::half_t* d_bias = nullptr;
    cutlass::half_t* d_C = nullptr;

    size_t size_A = M * K * sizeof(cutlass::half_t);
    size_t size_B = K * N * sizeof(cutlass::half_t);
    size_t size_bias = N * sizeof(cutlass::half_t);
    size_t size_C = M * N * sizeof(cutlass::half_t);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_bias, size_bias));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), size_bias, cudaMemcpyHostToDevice));

    // 预热
    blackwell_gemm_half_run(d_C, d_A, d_B, d_bias, M, N, K, BlackwellGemmType::GEMM_Bias);
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
        blackwell_gemm_half_run(d_C, d_A, d_B, d_bias, M, N, K, BlackwellGemmType::GEMM_Bias);
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
    size_t bytes_transferred = (M * K + K * N + N + M * N) * sizeof(cutlass::half_t);
    result.avg_bandwidth_gbs = (bytes_transferred / result.avg_time_ms) / 1e6;

    // 清理事件
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 验证结果
    std::vector<cutlass::half_t> h_C(M * N);
    std::vector<cutlass::half_t> h_ref;

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    // 计算CPU参考结果（包含bias）
    compute_reference_blackwell_gemm_bias(h_ref, h_A, h_B, h_bias, M, N, K);

    // 误差分析
    result.verify_count = std::min(M * N, MAX_COMPARE_COUNT);
    double abs_tolerance = 1e-2;
    double rel_tolerance = 1e-2;

    auto error_result = analyze_errors(h_C, h_ref, 0, result.verify_count,
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
