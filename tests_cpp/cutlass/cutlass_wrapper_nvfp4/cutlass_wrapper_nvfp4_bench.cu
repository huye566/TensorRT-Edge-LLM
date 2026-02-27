#include "cutlass_wrapper_nvfp4_bench.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <chrono>
#include "check/err_analysis.h"

// ==================== Utility Functions ====================
template<typename T>
std::vector<T> generate_random_vector(size_t size, float min_val = -1.0f, float max_val = 1.0f) {
    std::vector<T> result(size);
    for (size_t i = 0; i < size; ++i) {
        float val = min_val + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX/(max_val - min_val)));
        if constexpr (std::is_same<T, cutlass::half_t>::value) {
            result[i] = cutlass::half_t(val);
        } else {
            result[i] = static_cast<T>(val);
        }
    }
    return result;
}

std::vector<float> generate_random_scales(size_t size) {
    std::vector<float> result(size);
    for (size_t i = 0; i < size; ++i) {
        result[i] = 0.5f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX/(2.0f)));
    }
    return result;
}

// ==================== Reference Implementation ====================
float e2m1_to_float_cpu(uint8_t e2m1) {
    const float e2m1_table[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return e2m1_table[e2m1 & 0xF];
}

void dequantize_nvfp4(std::vector<float>& dequantized,
                     const std::vector<uint8_t>& quantized,
                     const std::vector<float>& scales,
                     int M, int N, int block_size = 16) {
    int packed_n = (N + 1) / 2;
    int scale_cols = (N + block_size - 1) / block_size;

    dequantized.resize(M * N);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int packed_idx = n / 2;
            uint8_t packed = quantized[m * packed_n + packed_idx];

            uint8_t e2m1;
            if (n % 2 == 0) {
                e2m1 = packed & 0x0F;
            } else {
                e2m1 = (packed >> 4) & 0x0F;
            }

            float val = e2m1_to_float_cpu(e2m1);
            int scale_idx = m * scale_cols + (n / block_size);
            float scale = scales[scale_idx];

            dequantized[m * N + n] = val * scale;
        }
    }
}

void compute_reference_nvfp4_gemm(std::vector<cutlass::half_t>& h_ref,
                                 const std::vector<cutlass::half_t>& h_A,
                                 const std::vector<cutlass::half_t>& h_B,
                                 const std::vector<float>& h_B_scales,
                                 int M, int N, int K, int block_size) {

    // Dequantize B
    std::vector<uint8_t> quantized_B(N * ((K + 1) / 2));
    std::vector<float> dequantized_B(N * K);

    // For simplicity, use existing B as is (in reality we'd quantize it)
    // This is just for testing framework

    h_ref.resize(M * N);
    int total_elements = M * N;
    int verify_count = std::min(total_elements, MAX_COMPARE_COUNT);

    for (int idx = 0; idx < verify_count; ++idx) {
        int m = idx / N;
        int n = idx % N;
        float accum = 0.0f;

        for (int k = 0; k < K; ++k) {
            float a_val = static_cast<float>(h_A[m * K + k]);
            float b_val = static_cast<float>(h_B[n * K + k]);  // B is transposed
            accum += a_val * b_val;
        }

        h_ref[m * N + n] = cutlass::half_t(accum);
    }
}

// ==================== Benchmark Functions ====================
TestResult benchmark_nvfp4_gemm_half(int M, int N, int K, int iterations) {
    TestResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.data_type = "FP16/NVFP4";
    result.operation = "NVFP4_GEMM";
    result.iterations = iterations;

    std::ostringstream oss;
    oss << "(" << M << "," << K << "," << N << ")";
    result.test_case = oss.str();

    const int block_size = 16;

    // Generate random data
    auto h_A = generate_random_vector<cutlass::half_t>(M * K);
    auto h_B = generate_random_vector<cutlass::half_t>(N * K);  // B is stored transposed

    // Quantize B to NVFP4
    size_t quantized_B_size = N * ((K + 1) / 2);
    size_t scales_B_size = N * ((K + block_size - 1) / block_size);

    std::vector<uint8_t> h_quantized_B(quantized_B_size);
    std::vector<scale_type> h_scales_B(scales_B_size);

    // Note: In real implementation, we would call quantize_fp16_to_nvfp4 here
    // For testing, we'll use the existing B as is

    // Allocate device memory
    cutlass::half_t* d_A = nullptr;
    cutlass::half_t* d_B_quantized = nullptr;
    scale_type* d_B_scales = nullptr;
    cutlass::half_t* d_C = nullptr;

    size_t size_A = M * K * sizeof(cutlass::half_t);
    size_t size_B_quantized = quantized_B_size * sizeof(uint8_t);
    size_t size_B_scales = scales_B_size * sizeof(scale_type);
    size_t size_C = M * N * sizeof(cutlass::half_t);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B_quantized, size_B_quantized));
    CUDA_CHECK(cudaMalloc(&d_B_scales, size_B_scales));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_quantized, h_quantized_B.data(), size_B_quantized, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_scales, h_scales_B.data(), size_B_scales, cudaMemcpyHostToDevice));

    // Alpha scaling factor
    float alpha = 1.0f;

    // Warmup
    trt_edgellm::kernel::cutlass_nvfp4_gemm(d_C, d_A, d_B_quantized, d_B_scales,
                                           alpha, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Performance test
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double total_time_ms = 0.0;
    double min_time_ms = std::numeric_limits<double>::max();
    double max_time_ms = 0.0;

    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        trt_edgellm::kernel::cutlass_nvfp4_gemm(d_C, d_A, d_B_quantized, d_B_scales,
                                               alpha, M, N, K);
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

    // Calculate performance metrics
    double flops = 2.0 * M * N * K;
    result.avg_tflops = (flops / result.avg_time_ms) / 1e9;
    result.min_tflops = (flops / result.max_time_ms) / 1e9;
    result.max_tflops = (flops / result.min_time_ms) / 1e9;

    // Calculate bandwidth (approximate)
    size_t bytes_transferred = (M * K * sizeof(cutlass::half_t)) +  // A
                               (N * K / 2 * sizeof(uint8_t)) +      // B quantized (4-bit)
                               (N * ((K + 15) / 16) * sizeof(scale_type)) +  // B scales
                               (M * N * sizeof(cutlass::half_t));   // C
    result.avg_bandwidth_gbs = (bytes_transferred / result.avg_time_ms) / 1e6;

    // Cleanup events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Verify results
    std::vector<cutlass::half_t> h_C(M * N);
    std::vector<cutlass::half_t> h_ref;

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    // Compute CPU reference (simplified - using original B)
    std::vector<float> dummy_scales(scales_B_size, 1.0f);
    compute_reference_nvfp4_gemm(h_ref, h_A, h_B, dummy_scales, M, N, K, block_size);

    // Error analysis
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

    // Cleanup device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B_quantized));
    CUDA_CHECK(cudaFree(d_B_scales));
    CUDA_CHECK(cudaFree(d_C));

    return result;
}

TestResult benchmark_nvfp4_gemm_bias_half(int M, int N, int K, int iterations) {
    TestResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.data_type = "FP16/NVFP4";
    result.operation = "NVFP4_GEMM_Bias";
    result.iterations = iterations;

    std::ostringstream oss;
    oss << "(" << M << "," << K << "," << N << ")";
    result.test_case = oss.str();

    const int block_size = 16;

    // Generate random data
    auto h_A = generate_random_vector<cutlass::half_t>(M * K);
    auto h_B = generate_random_vector<cutlass::half_t>(N * K);
    auto h_bias = generate_random_vector<cutlass::half_t>(N);

    // Allocate device memory
    cutlass::half_t* d_A = nullptr;
    cutlass::half_t* d_B_quantized = nullptr;
    scale_type* d_B_scales = nullptr;
    cutlass::half_t* d_bias = nullptr;
    cutlass::half_t* d_C = nullptr;

    size_t size_A = M * K * sizeof(cutlass::half_t);
    size_t size_B_quantized = N * ((K + 1) / 2) * sizeof(uint8_t);
    size_t size_B_scales = N * ((K + block_size - 1) / block_size) * sizeof(scale_type);
    size_t size_bias = N * sizeof(cutlass::half_t);
    size_t size_C = M * N * sizeof(cutlass::half_t);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B_quantized, size_B_quantized));
    CUDA_CHECK(cudaMalloc(&d_B_scales, size_B_scales));
    CUDA_CHECK(cudaMalloc(&d_bias, size_bias));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), size_bias, cudaMemcpyHostToDevice));

    // Note: Need to quantize B on device - for now use dummy data
    std::vector<uint8_t> h_quantized_B(N * ((K + 1) / 2), 0);
    std::vector<scale_type> h_scales_B(N * ((K + block_size - 1) / block_size), scale_type(1.0f));

    CUDA_CHECK(cudaMemcpy(d_B_quantized, h_quantized_B.data(), size_B_quantized, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_scales, h_scales_B.data(), size_B_scales, cudaMemcpyHostToDevice));

    float alpha = 1.0f;

    // Warmup
    // Note: The bias version is not implemented yet in the wrapper
    // For now, use the regular gemm
    trt_edgellm::kernel::cutlass_nvfp4_gemm(d_C, d_A, d_B_quantized, d_B_scales,
                                           alpha, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Performance test
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double total_time_ms = 0.0;
    double min_time_ms = std::numeric_limits<double>::max();
    double max_time_ms = 0.0;

    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        trt_edgellm::kernel::cutlass_nvfp4_gemm(d_C, d_A, d_B_quantized, d_B_scales,
                                               alpha, M, N, K);
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

    // Calculate performance metrics
    double flops = 2.0 * M * N * K;
    result.avg_tflops = (flops / result.avg_time_ms) / 1e9;
    result.min_tflops = (flops / result.max_time_ms) / 1e9;
    result.max_tflops = (flops / result.min_time_ms) / 1e9;

    // Calculate bandwidth
    size_t bytes_transferred = (M * K * sizeof(cutlass::half_t)) +
                               (N * K / 2 * sizeof(uint8_t)) +
                               (N * ((K + 15) / 16) * sizeof(scale_type)) +
                               (N * sizeof(cutlass::half_t)) +
                               (M * N * sizeof(cutlass::half_t));
    result.avg_bandwidth_gbs = (bytes_transferred / result.avg_time_ms) / 1e6;

    // Cleanup events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Cleanup device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B_quantized));
    CUDA_CHECK(cudaFree(d_B_scales));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_C));

    return result;
}