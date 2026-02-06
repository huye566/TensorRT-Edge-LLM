#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <fstream>
#include "cublaslt_wrapper_nvfp4_bench.h"
#include "check/err_analysis.h"

using nv_fp8_e4m3 = trt_edgellm::kernel::nv_fp8_e4m3;

inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

inline void apply_bias_to_reference(std::vector<__half>& reference, const std::vector<__half>& bias, int M, int N, int start_idx, int count) {
    int end_idx = std::min(start_idx + count, static_cast<int>(reference.size()));

    for (int idx = start_idx; idx < end_idx; ++idx) {
        int row = idx / N;
        int col = idx % N;
        if (row < M && col < N) {
            float val = __half2float(reference[idx - start_idx]);
            val += __half2float(bias[col]);
            reference[idx - start_idx] = __float2half(val);
        }
    }
}

inline void apply_silu_to_reference(std::vector<__half>& reference, int start_idx, int count) {
    int end_idx = std::min(start_idx + count, static_cast<int>(reference.size()));

    for (int idx = start_idx; idx < end_idx; ++idx) {
        float val = __half2float(reference[idx - start_idx]);
        val = silu(val);
        reference[idx - start_idx] = __float2half(val);
    }
}

struct CublasLtNVFP4Data {
    int M, N, K;
    std::vector<__half> A_fp16;
    std::vector<__half> B_fp16;
    std::vector<int64_t> A_fp4;
    std::vector<int64_t> B_fp4;
    std::vector<int32_t> A_scales;
    std::vector<int32_t> B_scales;
    std::vector<__half> output;
    std::vector<__half> output_ref;
    float a_global_scale;
    float b_global_scale;
    float alpha;
    std::vector<__half> bias;  // 用于测试偏置
};

const float E2M1_TO_FLOAT32[16] = {
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
};

float fp8_e4m3_to_fp32(__nv_fp8_e4m3 value) {
    return static_cast<float>(value);
}

std::vector<__nv_fp8_e4m3> convert_float_to_fp8_e4m3(const std::vector<float>& float_scales) {
    std::vector<__nv_fp8_e4m3> fp8_scales(float_scales.size());

    for (size_t i = 0; i < float_scales.size(); i++) {
        fp8_scales[i] = __nv_fp8_e4m3(float_scales[i]);
    }

    return fp8_scales;
}

// FP4索引转浮点数
float fp4_idx_to_float(uint8_t fp4_idx) {
    if (fp4_idx < 16) {
        return E2M1_TO_FLOAT32[fp4_idx];
    }
    return 0.0f;
}

void decode_fp4_to_float(const int64_t* fp4_data, float* float_data, int m, int n) {
    int total_elements = m * n;
    int packed_elements = total_elements / 8;

    for (int i = 0; i < packed_elements; i++) {
        uint32_t packed_val = reinterpret_cast<const uint32_t*>(fp4_data)[i];

        // 每个uint32_t包含8个FP4值
        for (int j = 0; j < 8; j++) {
            int element_idx = i * 8 + j;
            if (element_idx < total_elements) {
                // 提取每个4位的FP4值
                uint8_t fp4_val = (packed_val >> (j * 4)) & 0x0F;
                float_data[element_idx] = E2M1_TO_FLOAT32[fp4_val];
            }
        }
    }
}

void fp4_to_float(const uint8_t* fp4_data, float* float_data, int total_elements) {
    for (int i = 0; i < total_elements; i++) {
        float_data[i] = fp4_idx_to_float(fp4_data[i]);
    }
}

void recover_swizzled_scales(const int32_t* swizzled_scales, float* recovered_scales, int m, int n) {
    int scale_n = n / BLOCK_SIZE;  // 缩放因子矩阵的列数
    int rounded_m = ((m + 128 - 1) / 128) * 128;
    int rounded_scale_n = ((scale_n + 4 - 1) / 4) * 4;

    // 创建临时数组
    std::vector<uint8_t> tmp_data(rounded_m * rounded_scale_n);

    // 将int32_t转换为uint8_t数组
    int total_int32 = (rounded_m * rounded_scale_n + 3) / 4;
    for (int i = 0; i < total_int32; i++) {
        uint32_t val = reinterpret_cast<const uint32_t*>(swizzled_scales)[i];
        if (i * 4 < tmp_data.size()) tmp_data[i * 4] = val & 0xFF;
        if (i * 4 + 1 < tmp_data.size()) tmp_data[i * 4 + 1] = (val >> 8) & 0xFF;
        if (i * 4 + 2 < tmp_data.size()) tmp_data[i * 4 + 2] = (val >> 16) & 0xFF;
        if (i * 4 + 3 < tmp_data.size()) tmp_data[i * 4 + 3] = (val >> 24) & 0xFF;
    }

    // 重新排列数据（按照Python中的permute: (0,1,4,3,2,5)）
    std::vector<uint8_t> reordered(rounded_m * rounded_scale_n);

    int M_tile = rounded_m / 128;
    int K_tile = rounded_scale_n / 4;

    for (int m_tile = 0; m_tile < M_tile; m_tile++) {
        for (int k_tile = 0; k_tile < K_tile; k_tile++) {
            for (int outerM = 0; outerM < 32; outerM++) {
                for (int innerM = 0; innerM < 4; innerM++) {
                    for (int innerK = 0; innerK < 4; innerK++) {
                        // 源位置：按照 [M_tile, K_tile, 32, 4, 4] 布局
                        int src_idx = (((m_tile * K_tile + k_tile) * 32 + outerM) * 4 + innerM) * 4 + innerK;

                        // 目标位置：按照Python的permute (0,1,4,3,2,5)
                        // 这意味着原来的维度2（32）变成了维度3，维度3（4）变成了维度2，维度4（4）变成了维度5
                        int dst_row = m_tile * 128 + innerM * 32 + outerM;
                        int dst_col = k_tile * 4 + innerK;
                        int dst_idx = dst_row * rounded_scale_n + dst_col;

                        if (src_idx < tmp_data.size() && dst_idx < reordered.size()) {
                            reordered[dst_idx] = tmp_data[src_idx];
                        }
                    }
                }
            }
        }
    }

    // 转换为浮点数
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < scale_n; j++) {
            int idx = i * rounded_scale_n + j;
            if (idx < reordered.size()) {
                uint8_t fp8_val = reordered[idx];
                __nv_fp8_e4m3 fp8;
                fp8.__x = fp8_val;
                recovered_scales[i * scale_n + j] = fp8_e4m3_to_fp32(fp8);
            } else {
                recovered_scales[i * scale_n + j] = 0.0f;
            }
        }
    }
}

// 将FP4数据解码为浮点数矩阵
void dequantize_fp4_matrix(
    float* output,
    const int64_t* fp4_data,
    const int32_t* scale_interleaved,
    float global_scale,
    int m, int n) {

    int packed_n = n / 16;  // 每16个元素一个块
    int total_elements = m * n;

    // 1. 解码FP4数据为uint8数组
    std::vector<float> fp4_float(total_elements);
    decode_fp4_to_float(fp4_data, fp4_float.data(), m, n);

    // 2. 恢复缩放因子
    std::vector<float> scales(m * packed_n);
    recover_swizzled_scales(scale_interleaved, scales.data(), m, n);
    // for (int i = 0; i < scales.size(); i++) {
    //     std::cout << "scales[" << i << "] = " << scales[i] << std::endl;
    // }

    // 3. 将FP4转换为浮点数并应用缩放
    for (int idx = 0; idx < total_elements; idx++) {
        int i = idx / n;
        int j = idx % n;

        // 找到对应的块和缩放因子
        int block_idx = j / BLOCK_SIZE;

        float fp4_val = fp4_float[idx];
        float scale = scales[i * packed_n + block_idx];

        // 应用缩放：fp4_value * (scale / global_scale)
        output[idx] = fp4_val * (scale / global_scale);
    }
}

// NVFP4参考计算：模拟Python中的get_ref_results
std::vector<__half> compute_nvfp4_reference(
    const int64_t* a_fp4, const int64_t* b_fp4,
    const int32_t* a_scale_interleaved, const int32_t* b_scale_interleaved,
    float a_global_scale, float b_global_scale,
    float alpha, int M, int N, int K,
    int verify_count) {

    verify_count = std::min(verify_count, M * N);
    std::vector<__half> result(verify_count);

    // 解码A矩阵
    std::vector<float> a_float(M * K);
    dequantize_fp4_matrix(a_float.data(), a_fp4, a_scale_interleaved,
                         a_global_scale, M, K);

    // 解码B矩阵（注意B是转置的，所以维度是N×K）
    std::vector<float> b_float(N * K);
    dequantize_fp4_matrix(b_float.data(), b_fp4, b_scale_interleaved,
                         b_global_scale, N, K);

    // 计算矩阵乘法（只计算前verify_count个元素）
    #pragma omp parallel for
    for (int idx = 0; idx < verify_count; idx++) {
        int m = idx / N;
        int n = idx % N;
        if (m < M && n < N) {
            float accum = 0.0f;
            #pragma omp simd reduction(+:accum)
            for (int k = 0; k < K; ++k) {
                float a_val = a_float[m * K + k];
                float b_val = b_float[n * K + k];  // B是列主序
                accum += a_val * b_val;
            }
            result[idx] = __float2half(accum * alpha);
        }
    }

    return result;
}

void compute_reference_fp16_gemm(
    std::vector<__half>& h_ref,
    const std::vector<__half>& h_A,
    const std::vector<__half>& h_B,
    float alpha,
    int M, int N, int K,
    int verify_count) {

    // 只分配验证需要的空间
    int total_elements = M * N;
    verify_count = std::min(verify_count, total_elements);
    h_ref.resize(verify_count);

    #pragma omp parallel for collapse(2)
    for (int idx = 0; idx < verify_count; ++idx) {
        int m = idx / N;
        int n = idx % N;

        if (m < M && n < N) {
            float accum = 0.0f;
            #pragma omp simd reduction(+:accum)
            for (int k = 0; k < K; ++k) {
                float a_val = __half2float(h_A[m * K + k]);
                float b_val = __half2float(h_B[n * K + k]);  // B是列主序
                accum += a_val * b_val;
            }
            h_ref[idx] = __float2half(accum * alpha);
        }
    }
}

std::vector<__half> generate_random_half_vector(size_t size) {
    std::vector<__half> result(size);
    for (size_t i = 0; i < size; ++i) {
        float val = static_cast<float>(rand() % 1000) / 100.0f - 5.0f;  // [-0.5, 0.5]
        result[i] = __float2half(val);
    }
    return result;
}

// 计算全局缩放因子
float compute_global_scale(const std::vector<__half>& data) {
    float max_val = 0.0f;
    for (const auto& val : data) {
        float fval = __half2float(val);
        max_val = std::max(max_val, std::abs(fval));
    }
    // 防止除0
    if (max_val < 1e-9f) max_val = 1.0f;
    return (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / max_val;
}

static size_t get_scale_pad_size(int M, int N) {
    int scale_n = N / BLOCK_SIZE;
    int rounded_m = ((M + 128 - 1) / 128) * 128;
    int rounded_n = ((scale_n + 4 - 1) / 4) * 4;
    return rounded_m * rounded_n;
}


bool cublaslt_gemm_nvfp4_run(
    cublasLtHandle_t handle,
    const void* d_A_fp4,
    const void* d_B_fp4,
    const void* d_bias,
    void* d_output,
    const void* d_A_scales,
    const void* d_B_scales,
    float alpha,
    nvinfer1::DataType output_type,
    cudaStream_t stream,
    const CublasLtTestParams &params) {
    bool success = false;
    if (params.use_silu && params.use_bias) {
        success = trt_edgellm::kernel::cublaslt_gemm_nvfp4<true, true>(
            handle,
            params.M, params.N, params.K,
            d_A_fp4,
            d_B_fp4,
            d_bias,
            d_output,
            d_A_scales,
            d_B_scales,
            alpha,
            output_type,
            stream);
    } else if (params.use_silu) {
        success = trt_edgellm::kernel::cublaslt_gemm_nvfp4<true, false>(
            handle,
            params.M, params.N, params.K,
            d_A_fp4,
            d_B_fp4,
            d_bias,
            d_output,
            d_A_scales,
            d_B_scales,
            alpha,
            output_type,
            stream);
    } else if (params.use_bias) {
        // false, false
        success = trt_edgellm::kernel::cublaslt_gemm_nvfp4<false, true>(
            handle,
            params.M, params.N, params.K,
            d_A_fp4,
            d_B_fp4,
            d_bias,
            d_output,
            d_A_scales,
            d_B_scales,
            alpha,
            output_type,
            stream);
    } else {
        success = trt_edgellm::kernel::cublaslt_gemm_nvfp4<false, false>(
            handle,
            params.M, params.N, params.K,
            d_A_fp4,
            d_B_fp4,
            d_bias,
            d_output,
            d_A_scales,
            d_B_scales,
            alpha,
            output_type,
            stream);
    }
    return success;
}


// 主测试函数（FP16输出）
TestResult benchmark_cublaslt_nvfp4_gemm_half(const CublasLtTestParams &params) {
    int iterations = params.iterations;

    CublasLtNVFP4Data data;

    // 初始化cublasLt
    auto& cublaslt_wrapper = trt_edgellm::kernel::CublasLtNVFP4Wrapper::instance();
    if (!cublaslt_wrapper.initialize()) {
        std::cerr << "Failed to initialize cublasLt wrapper" << std::endl;
        TestResult failed_result;
        failed_result.passed = false;
        failed_result.error_count = 1;
        return failed_result;
    }

    if (!cublaslt_wrapper.check_fp4_support()) {
        std::cout << "Warning: NVFP4 not supported on this hardware, skipping test" << std::endl;
        TestResult result;
        result.M = params.M;
        result.N = params.N;
        result.K = params.K;
        result.passed = true;  // 硬件不支持，不算失败
        result.data_type = "NVFP4->FP16";
        result.operation = "CUBLASLT_NVFP4_GEMM";
        return result;
    }

    std::cout << "Generating random data..." << std::endl;
    data.M = params.M;
    data.N = params.N;
    data.K = params.K;
    data.A_fp16 = generate_random_half_vector(data.M * data.K);
    data.B_fp16 = generate_random_half_vector(data.N * data.K);

    data.a_global_scale = compute_global_scale(data.A_fp16);
    data.b_global_scale = compute_global_scale(data.B_fp16);

    data.alpha = 1.0f / (data.a_global_scale * data.b_global_scale);
    std::cout << "Generated data: M=" << data.M << ", N=" << data.N << ", K=" << data.K << std::endl;
    std::cout << "Global scales: A=" << data.a_global_scale
                << ", B=" << data.b_global_scale
                << ", Alpha=" << data.alpha << std::endl;

    TestResult result;
    result.M = data.M;
    result.N = data.N;
    result.K = data.K;
    result.data_type = "NVFP4->FP16";
    result.operation = params.use_silu ? "CUBLASLT_NVFP4_GEMM_SiLU" :
                      params.use_bias ? "CUBLASLT_NVFP4_GEMM_Bias" :
                      "CUBLASLT_NVFP4_GEMM";
    result.iterations = iterations;

    std::ostringstream oss;
    oss << "M" << result.M << "_N" << result.N << "_K" << result.K;
    result.test_case = oss.str();

    // 3. 分配设备内存
    __half* d_A_fp16 = nullptr;
    __half* d_B_fp16 = nullptr;
    float* d_a_global_scale = nullptr;
    float* d_b_global_scale = nullptr;

    size_t size_A = data.M * data.K * sizeof(__half);
    size_t size_B = data.N * data.K * sizeof(__half);

    CUDA_CHECK(cudaMalloc(&d_A_fp16, size_A));
    CUDA_CHECK(cudaMalloc(&d_B_fp16, size_B));
    CUDA_CHECK(cudaMalloc(&d_a_global_scale, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b_global_scale, sizeof(float)));

    // 4. 创建stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 5. 拷贝数据到设备
    CUDA_CHECK(cudaMemcpyAsync(d_A_fp16, data.A_fp16.data(), size_A, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_fp16, data.B_fp16.data(), size_B, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_a_global_scale, &data.a_global_scale, sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b_global_scale, &data.b_global_scale, sizeof(float), cudaMemcpyHostToDevice, stream));

    // 6. 量化参数分配
    int64_t* d_A_fp4 = nullptr;
    int64_t* d_B_fp4 = nullptr;
    int32_t* d_A_scales = nullptr;
    int32_t* d_B_scales = nullptr;

    // 计算量化数据大小
    size_t size_A_fp4 = (data.M * data.K + 7) / 8 * sizeof(int64_t);
    size_t size_B_fp4 = (data.N * data.K + 7) / 8 * sizeof(int64_t);
    size_t size_A_scales = get_scale_pad_size(data.M, data.K);
    size_t size_B_scales = get_scale_pad_size(data.N, data.K);

    CUDA_CHECK(cudaMalloc(&d_A_fp4, size_A_fp4));
    CUDA_CHECK(cudaMalloc(&d_B_fp4, size_B_fp4));
    CUDA_CHECK(cudaMalloc(&d_A_scales, size_A_scales));
    CUDA_CHECK(cudaMalloc(&d_B_scales, size_B_scales));

    // 7. 执行量化
    std::cout << "Quantizing matrices..." << std::endl;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 量化矩阵A
    trt_edgellm::kernel::scaled_fp4_quant(
        data.M, data.K,
        d_A_fp16,
        d_a_global_scale,
        d_A_fp4,
        d_A_scales,
        stream,
        nvinfer1::DataType::kHALF);

    // 量化矩阵B
    trt_edgellm::kernel::scaled_fp4_quant(
        data.N, data.K,
        d_B_fp16,
        d_b_global_scale,
        d_B_fp4,
        d_B_scales,
        stream,
        nvinfer1::DataType::kHALF);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 8. 输出矩阵分配
    __half* d_output = nullptr;
    size_t size_output = data.M * data.N * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&d_output, size_output));

    // 10. 分配偏置内存（如果需要）
    __half* d_bias = nullptr;
    if (params.use_bias) {
        data.bias = generate_random_half_vector(data.N);
        CUDA_CHECK(cudaMalloc(&d_bias, data.N * sizeof(__half)));
        CUDA_CHECK(cudaMemcpyAsync(d_bias, data.bias.data(), data.N * sizeof(__half),
                                   cudaMemcpyHostToDevice, stream));
    }

    __nv_fp8_e4m3* d_A_scales_fp8 = nullptr;
    __nv_fp8_e4m3* d_B_scales_fp8 = nullptr;
    {
        int scale_n_a = data.K / BLOCK_SIZE;
        int scale_n_b = data.K / BLOCK_SIZE;
        size_t size_A_scales_fp8 = data.M * scale_n_a * sizeof(__nv_fp8_e4m3);
        size_t size_B_scales_fp8 = data.N * scale_n_b * sizeof(__nv_fp8_e4m3);

        CUDA_CHECK(cudaMalloc(&d_A_scales_fp8, size_A_scales_fp8));
        CUDA_CHECK(cudaMalloc(&d_B_scales_fp8, size_B_scales_fp8));

        std::vector<int32_t> h_A_scales_int32(size_A_scales / sizeof(int32_t));
        std::vector<int32_t> h_B_scales_int32(size_B_scales / sizeof(int32_t));
        CUDA_CHECK(cudaMemcpyAsync(h_A_scales_int32.data(), d_A_scales, size_A_scales,
                                cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_B_scales_int32.data(), d_B_scales, size_B_scales,
                                cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<float> a_scales_float(data.M * (data.K / 16));
        recover_swizzled_scales(h_A_scales_int32.data(), a_scales_float.data(), data.M, data.K);

        std::vector<float> b_scales_float(data.N * (data.K / 16));
        recover_swizzled_scales(h_B_scales_int32.data(), b_scales_float.data(), data.N, data.K);

        // 转换为FP8 E4M3
        auto h_A_scales_fp8 = convert_float_to_fp8_e4m3(a_scales_float);
        auto h_B_scales_fp8 = convert_float_to_fp8_e4m3(b_scales_float);
        // for (int i = 0; i < a_scales_float.size(); i++) {
        //     std::cout << "a_scales_float[" << i << "] = " << a_scales_float[i] << std::endl;
        // }

        // for (int i = 0; i < b_scales_float.size(); i++) {
        //     std::cout << "b_scales_float[" << i << "] = " << b_scales_float[i] << std::endl;
        // }

        CUDA_CHECK(cudaMemcpyAsync(d_A_scales_fp8, h_A_scales_fp8.data(), size_A_scales_fp8,
                                cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B_scales_fp8, h_B_scales_fp8.data(), size_B_scales_fp8,
                                cudaMemcpyHostToHost, stream));
    }

    // 11. 预热
    std::cout << "预热..." << std::endl;
    bool success = false;
    success = cublaslt_gemm_nvfp4_run(
        cublaslt_wrapper.handle(),
        d_A_fp4,
        d_B_fp4,
        d_bias,
        d_output,
        d_A_scales,
        d_B_scales,
        // d_A_scales_fp8,
        // d_B_scales_fp8,
        data.alpha,
        nvinfer1::DataType::kHALF,
        stream,
        params
    );

    if (!success) {
        std::cerr << "Failed to execute cublasLt NVFP4 GEMM" << std::endl;
        CUDA_CHECK(cudaStreamDestroy(stream));
        TestResult failed_result;
        failed_result.passed = false;
        return failed_result;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 12. 性能测试
    double total_time_ms = 0.0;
    double min_time_ms = std::numeric_limits<double>::max();
    double max_time_ms = 0.0;

    std::cout << "开始性能测试 (" << iterations << " 次迭代)..." << std::endl;
    GPUTimer gpu_timer;
    for (int i = 0; i < iterations; ++i) {
        gpu_timer.start(stream);

        success = cublaslt_gemm_nvfp4_run(
            cublaslt_wrapper.handle(),
            d_A_fp4,
            d_B_fp4,
            d_bias,
            d_output,
            d_A_scales,
            d_B_scales,
            // d_A_scales_fp8,
            // d_B_scales_fp8,
            data.alpha,
            nvinfer1::DataType::kHALF,
            stream,
            params
        );

        if (!success) {
            std::cerr << "Failed to execute cublasLt NVFP4 GEMM in iteration " << i << std::endl;
            break;
        }

        gpu_timer.stop(stream);
        float elapsed_ms = gpu_timer.elapsed();
        CUDA_CHECK(cudaStreamSynchronize(stream));

        total_time_ms += elapsed_ms;
        min_time_ms = std::min(min_time_ms, static_cast<double>(elapsed_ms));
        max_time_ms = std::max(max_time_ms, static_cast<double>(elapsed_ms));

        if ((iterations >= 10 && i % (iterations / 10) == 0) || iterations < 10) {
            std::cout << "  迭代 " << i << ": " << elapsed_ms << " ms" << std::endl;
        }
    }

    result.avg_time_ms = total_time_ms / iterations;
    result.min_time_ms = min_time_ms;
    result.max_time_ms = max_time_ms;

    // 13. 计算性能指标
    double flops = 2.0 * data.M * data.N * data.K;
    result.avg_tflops = (flops / result.avg_time_ms) / 1e9;
    result.min_tflops = (flops / result.max_time_ms) / 1e9;
    result.max_tflops = (flops / result.min_time_ms) / 1e9;

    // 计算带宽
    size_t bytes_transferred = size_A_fp4 + size_B_fp4 + size_A_scales + size_B_scales + size_output;
    result.avg_bandwidth_gbs = (bytes_transferred / result.avg_time_ms) / 1e6;

    // 14. 获取GPU结果
    data.output.resize(data.M * data.N);
    CUDA_CHECK(cudaMemcpyAsync(data.output.data(), d_output, size_output, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    result.verify_count = std::min(data.M * data.N, MAX_COMPARE_COUNT);

    // 15. 计算CPU参考结果
    std::cout << "计算CPU参考结果（限制数量）..." << std::endl;
    std::vector<__half> h_ref;

    if (params.nvfp4_ref) {
        // 读取量化数据到主机
        std::vector<int64_t> h_A_fp4(size_A_fp4 / sizeof(int64_t));
        std::vector<int64_t> h_B_fp4(size_B_fp4 / sizeof(int64_t));
        std::vector<int32_t> h_A_scales(size_A_scales / sizeof(int32_t));
        std::vector<int32_t> h_B_scales(size_B_scales / sizeof(int32_t));

        CUDA_CHECK(cudaMemcpyAsync(h_A_fp4.data(), d_A_fp4, size_A_fp4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_B_fp4.data(), d_B_fp4, size_B_fp4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_A_scales.data(), d_A_scales, size_A_scales, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_B_scales.data(), d_B_scales, size_B_scales, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        h_ref = compute_nvfp4_reference(
            h_A_fp4.data(), h_B_fp4.data(),
            h_A_scales.data(), h_B_scales.data(),
            data.a_global_scale, data.b_global_scale,
            1.0, data.M, data.N, data.K,
            result.verify_count);

    } else {
        compute_reference_fp16_gemm(h_ref, data.A_fp16, data.B_fp16, 1.0, data.M, data.N, data.K, result.verify_count);
    }

    if (params.use_bias && data.bias.size() > 0) {
        apply_bias_to_reference(h_ref, data.bias, data.M, data.N, 0, result.verify_count);
    }

    if (params.use_silu) {
        apply_silu_to_reference(h_ref, 0, result.verify_count);
    }

    // 16. 误差分析
    double abs_tolerance = 5e-2;  // 4-bit量化的误差容忍度
    double rel_tolerance = 5e-2;

    auto error_result = analyze_errors(data.output, h_ref, 0, result.verify_count,
                                      abs_tolerance, rel_tolerance);

    result.max_abs_error = error_result.max_abs_error;
    result.max_rel_error = error_result.max_rel_error;
    result.passed = error_result.passed;
    result.error_count = error_result.error_count;
    result.total_count = error_result.total_count;

    std::cout << "误差分析: max_abs_error=" << result.max_abs_error
              << ", max_rel_error=" << result.max_rel_error
              << ", errors=" << result.error_count << "/" << result.verify_count
              << " (" << (result.passed ? "PASS" : "FAIL") << ")" << std::endl;

    // 17. 清理
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_A_fp16));
    CUDA_CHECK(cudaFree(d_B_fp16));
    CUDA_CHECK(cudaFree(d_a_global_scale));
    CUDA_CHECK(cudaFree(d_b_global_scale));
    CUDA_CHECK(cudaFree(d_A_fp4));
    CUDA_CHECK(cudaFree(d_B_fp4));
    CUDA_CHECK(cudaFree(d_A_scales));
    CUDA_CHECK(cudaFree(d_B_scales));
    CUDA_CHECK(cudaFree(d_A_scales_fp8));
    CUDA_CHECK(cudaFree(d_B_scales_fp8));
    CUDA_CHECK(cudaFree(d_output));
    if (params.use_bias) {
        CUDA_CHECK(cudaFree(d_bias));
    }

    return result;
}
