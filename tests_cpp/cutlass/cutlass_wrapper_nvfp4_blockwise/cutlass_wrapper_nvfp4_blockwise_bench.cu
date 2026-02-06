#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include "cutlass_wrapper_nvfp4_blockwise_bench.h"
#include "check/err_analysis.h"


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

struct NVFP4Data {
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
    std::vector<__half> bias;
};

void save_nvfp4_data(const NVFP4Data& data, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    // 写入维度信息
    file.write(reinterpret_cast<const char*>(&data.M), sizeof(int));
    file.write(reinterpret_cast<const char*>(&data.N), sizeof(int));
    file.write(reinterpret_cast<const char*>(&data.K), sizeof(int));

    // 写入全局缩放因子
    file.write(reinterpret_cast<const char*>(&data.a_global_scale), sizeof(float));
    file.write(reinterpret_cast<const char*>(&data.b_global_scale), sizeof(float));
    file.write(reinterpret_cast<const char*>(&data.alpha), sizeof(float));

    // 写入数据大小
    size_t size_A_fp16 = data.A_fp16.size();
    size_t size_B_fp16 = data.B_fp16.size();
    size_t size_A_fp4 = data.A_fp4.size();
    size_t size_B_fp4 = data.B_fp4.size();
    size_t size_A_scales = data.A_scales.size();
    size_t size_B_scales = data.B_scales.size();
    size_t size_output = data.output.size();
    size_t size_output_ref = data.output_ref.size();

    file.write(reinterpret_cast<const char*>(&size_A_fp16), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&size_B_fp16), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&size_A_fp4), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&size_B_fp4), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&size_A_scales), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&size_B_scales), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&size_output), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&size_output_ref), sizeof(size_t));

    // 写入数据
    if (!data.A_fp16.empty())
        file.write(reinterpret_cast<const char*>(data.A_fp16.data()), size_A_fp16 * sizeof(__half));
    if (!data.B_fp16.empty())
        file.write(reinterpret_cast<const char*>(data.B_fp16.data()), size_B_fp16 * sizeof(__half));
    if (!data.A_fp4.empty())
        file.write(reinterpret_cast<const char*>(data.A_fp4.data()), size_A_fp4 * sizeof(int64_t));
    if (!data.B_fp4.empty())
        file.write(reinterpret_cast<const char*>(data.B_fp4.data()), size_B_fp4 * sizeof(int64_t));
    if (!data.A_scales.empty())
        file.write(reinterpret_cast<const char*>(data.A_scales.data()), size_A_scales * sizeof(int32_t));
    if (!data.B_scales.empty())
        file.write(reinterpret_cast<const char*>(data.B_scales.data()), size_B_scales * sizeof(int32_t));
    if (!data.output.empty())
        file.write(reinterpret_cast<const char*>(data.output.data()), size_output * sizeof(__half));
    if (!data.output_ref.empty())
        file.write(reinterpret_cast<const char*>(data.output_ref.data()), size_output_ref * sizeof(__half));

    std::cout << "Data saved to " << filename << std::endl;
}

// 从文件加载数据
NVFP4Data load_nvfp4_data(const std::string& filename) {
    NVFP4Data data;
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return data;
    }

    // 读取维度信息
    file.read(reinterpret_cast<char*>(&data.M), sizeof(int));
    file.read(reinterpret_cast<char*>(&data.N), sizeof(int));
    file.read(reinterpret_cast<char*>(&data.K), sizeof(int));

    // 读取全局缩放因子
    file.read(reinterpret_cast<char*>(&data.a_global_scale), sizeof(float));
    file.read(reinterpret_cast<char*>(&data.b_global_scale), sizeof(float));
    file.read(reinterpret_cast<char*>(&data.alpha), sizeof(float));

    // 读取数据大小
    size_t size_A_fp16, size_B_fp16, size_A_fp4, size_B_fp4;
    size_t size_A_scales, size_B_scales, size_output, size_output_ref;

    file.read(reinterpret_cast<char*>(&size_A_fp16), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&size_B_fp16), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&size_A_fp4), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&size_B_fp4), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&size_A_scales), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&size_B_scales), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&size_output), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&size_output_ref), sizeof(size_t));

    // 调整vector大小并读取数据
    data.A_fp16.resize(size_A_fp16);
    data.B_fp16.resize(size_B_fp16);
    data.A_fp4.resize(size_A_fp4);
    data.B_fp4.resize(size_B_fp4);
    data.A_scales.resize(size_A_scales);
    data.B_scales.resize(size_B_scales);
    data.output.resize(size_output);
    data.output_ref.resize(size_output_ref);

    if (!data.A_fp16.empty())
        file.read(reinterpret_cast<char*>(data.A_fp16.data()), size_A_fp16 * sizeof(__half));
    if (!data.B_fp16.empty())
        file.read(reinterpret_cast<char*>(data.B_fp16.data()), size_B_fp16 * sizeof(__half));
    if (!data.A_fp4.empty())
        file.read(reinterpret_cast<char*>(data.A_fp4.data()), size_A_fp4 * sizeof(int64_t));
    if (!data.B_fp4.empty())
        file.read(reinterpret_cast<char*>(data.B_fp4.data()), size_B_fp4 * sizeof(int64_t));
    if (!data.A_scales.empty())
        file.read(reinterpret_cast<char*>(data.A_scales.data()), size_A_scales * sizeof(int32_t));
    if (!data.B_scales.empty())
        file.read(reinterpret_cast<char*>(data.B_scales.data()), size_B_scales * sizeof(int32_t));
    if (!data.output.empty())
        file.read(reinterpret_cast<char*>(data.output.data()), size_output * sizeof(__half));
    if (!data.output_ref.empty())
        file.read(reinterpret_cast<char*>(data.output_ref.data()), size_output_ref * sizeof(__half));

    std::cout << "Data loaded from " << filename << std::endl;
    std::cout << "M=" << data.M << ", N=" << data.N << ", K=" << data.K << std::endl;
    std::cout << "Global scales: A=" << data.a_global_scale
                  << ", B=" << data.b_global_scale
                  << ", Alpha=" << data.alpha << std::endl;
    return data;
}

// E2M1转浮点数查找表
const float E2M1_TO_FLOAT32[16] = {
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
};

float fp8_e4m3_to_fp32(__nv_fp8_e4m3 value) {
    return static_cast<float>(value);
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

// 将uint8_t数组中的FP4值转换为浮点数
void fp4_to_float(const uint8_t* fp4_data, float* float_data, int total_elements) {
    for (int i = 0; i < total_elements; i++) {
        float_data[i] = fp4_idx_to_float(fp4_data[i]);
    }
}

// 恢复Swizzled布局的缩放因子
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

// CPU参考计算：直接FP16矩阵乘，只计算部分元素用于验证
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

static size_t get_scale_pad_size(int M, int N) {
    int scale_n = N / BLOCK_SIZE;
    int rounded_m = ((M + 128 - 1) / 128) * 128;
    int rounded_n = ((scale_n + 4 - 1) / 4) * 4;
    return rounded_m * rounded_n;
}

void cutlass_gemm_nvfp4_run(
    const void* d_A_fp4,
    const void* d_B_fp4,
    const void* d_bias,
    void* d_output,
    const void* d_A_scales,
    const void* d_B_scales,
    float alpha,
    nvinfer1::DataType output_type,
    cudaStream_t stream,
    const TestParams &params,
    const NVFP4Data &data,
    bool use_cached = true) {

    if (params.use_silu && params.use_bias) {
        trt_edgellm::kernel::cutlass_scaled_nvfp4<true, true>(
            d_output,
            d_A_fp4,
            d_B_fp4,
            d_A_scales,
            d_B_scales,
            d_bias,
            data.alpha,
            data.M, data.N, data.K,
            stream,
            output_type,
            use_cached);
    } else if (params.use_silu) {
        trt_edgellm::kernel::cutlass_scaled_nvfp4<true, false>(
            d_output,
            d_A_fp4,
            d_B_fp4,
            d_A_scales,
            d_B_scales,
            d_bias,
            data.alpha,
            data.M, data.N, data.K,
            stream,
            output_type,
            use_cached);
    } else if (params.use_bias) {
        trt_edgellm::kernel::cutlass_scaled_nvfp4<false, true>(
            d_output,
            d_A_fp4,
            d_B_fp4,
            d_A_scales,
            d_B_scales,
            d_bias,
            data.alpha,
            data.M, data.N, data.K,
            stream,
            output_type,
            use_cached);
    } else {
        trt_edgellm::kernel::cutlass_scaled_nvfp4<false, false>(
            d_output,
            d_A_fp4,
            d_B_fp4,
            d_A_scales,
            d_B_scales,
            d_bias,
            data.alpha,
            data.M, data.N, data.K,
            stream,
            output_type,
            use_cached);
    }
}

// 主测试函数
TestResult benchmark_nvfp4_gemm_half(const TestParams &params) {
    int iterations = params.iterations;

    NVFP4Data data;
    bool load_from_file = !params.input_file.empty();
    if (load_from_file) {
        std::cout << "Loading data from file: " << params.input_file << std::endl;
        data = load_nvfp4_data(params.input_file);
    } else {
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
    }

    TestResult result;
    result.M = data.M;
    result.N = data.N;
    result.K = data.K;
    result.data_type = "NVFP4->FP16";
    result.operation = params.use_silu ? params.use_bias ? "NVFP4_GEMM_Bias_SiLU" : "NVFP4_GEMM_SiLU" :
                      params.use_bias ? "NVFP4_GEMM_Bias" :
                      "NVFP4_GEMM";
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

    // 5. 拷贝数据到设备（使用同一个stream）
    CUDA_CHECK(cudaMemcpyAsync(d_A_fp16, data.A_fp16.data(), size_A, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_fp16, data.B_fp16.data(), size_B, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_a_global_scale, &data.a_global_scale, sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b_global_scale, &data.b_global_scale, sizeof(float), cudaMemcpyHostToDevice, stream));

    // 6. 量化参数分配
    int64_t* d_A_fp4 = nullptr;
    int64_t* d_B_fp4 = nullptr;
    int32_t* d_A_scales = nullptr;
    int32_t* d_B_scales = nullptr;

    // NVFP4每2个FP16元素打包成1个byte，但scaled_fp4_quant输出int64_t
    // 缩放因子：每16个元素一个缩放因子
    size_t size_A_fp4 = (data.M * data.K + 8 - 1) / 8 * sizeof(int64_t);
    size_t size_B_fp4 = (data.N * data.K + 8 - 1) / 8 * sizeof(int64_t);
    // size_t size_A_scales = (data.M * data.K + 16 - 1) / 16 * sizeof(int32_t);
    // size_t size_B_scales = (data.N * data.K + 16 - 1) / 16 * sizeof(int32_t);
    size_t size_A_scales = get_scale_pad_size(data.M, data.K);
    size_t size_B_scales = get_scale_pad_size(data.N, data.K);

    CUDA_CHECK(cudaMalloc(&d_A_fp4, size_A_fp4));
    CUDA_CHECK(cudaMalloc(&d_B_fp4, size_B_fp4));
    CUDA_CHECK(cudaMalloc(&d_A_scales, size_A_scales));
    CUDA_CHECK(cudaMalloc(&d_B_scales, size_B_scales));

    // 7. 执行量化（使用同一个stream）
    std::cout << "Quantizing matrices..." << std::endl;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 量化矩阵A
    trt_edgellm::kernel::scaled_fp4_quant(
        data.M, data.K,  // m, n
        d_A_fp16,  // input
        d_a_global_scale,  // SFScale
        d_A_fp4,  // output (int64_t*)
        d_A_scales,  // SFOuput (int32_t*)
        stream,
        nvinfer1::DataType::kHALF);

    // 量化矩阵B
    trt_edgellm::kernel::scaled_fp4_quant(
        data.N, data.K,  // m, n (注意B是列主序)
        d_B_fp16,  // input
        d_b_global_scale,  // SFScale
        d_B_fp4,  // output (int64_t*)
        d_B_scales,  // SFOuput (int32_t*)
        stream,
        nvinfer1::DataType::kHALF);

    // 8. 输出矩阵分配
    __half* d_output = nullptr;
    size_t size_output = data.M * data.N * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&d_output, size_output));

    __half* d_bias = nullptr;
    if (params.use_bias) {
        data.bias = generate_random_half_vector(data.N);
        CUDA_CHECK(cudaMalloc(&d_bias, data.N * sizeof(__half)));
        CUDA_CHECK(cudaMemcpyAsync(d_bias, data.bias.data(), data.N * sizeof(__half),
                                   cudaMemcpyHostToDevice, stream));
    }

    // 9. 预热（使用同一个stream）
    std::cout << "预热..." << std::endl;
    cutlass_gemm_nvfp4_run(
        d_A_fp4,
        d_B_fp4,
        d_bias,
        d_output,
        d_A_scales,
        d_B_scales,
        data.alpha,
        nvinfer1::DataType::kHALF,
        stream,
        params,
        data
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 10. 性能测试
    double total_time_ms = 0.0;
    double min_time_ms = std::numeric_limits<double>::max();
    double max_time_ms = 0.0;

    std::cout << "开始性能测试 (" << iterations << " 次迭代)..." << std::endl;
    GPUTimer gpu_timer;
    for (int i = 0; i < iterations; ++i) {
        gpu_timer.start(stream);
        cutlass_gemm_nvfp4_run(
            d_A_fp4,
            d_B_fp4,
            d_bias,
            d_output,
            d_A_scales,
            d_B_scales,
            data.alpha,
            nvinfer1::DataType::kHALF,
            stream,
            params,
            data
        );
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

    // 11. 计算性能指标
    double flops = 2.0 * data.M * data.N * data.K;
    result.avg_tflops = (flops / result.avg_time_ms) / 1e9;
    result.min_tflops = (flops / result.max_time_ms) / 1e9;
    result.max_tflops = (flops / result.min_time_ms) / 1e9;

    // 计算带宽
    size_t bytes_transferred = size_A_fp4 + size_B_fp4 + size_A_scales + size_B_scales + size_output;
    result.avg_bandwidth_gbs = (bytes_transferred / result.avg_time_ms) / 1e6;

    // 12. 获取GPU结果（使用同一个stream）
    data.output.resize(data.M * data.N);
    CUDA_CHECK(cudaMemcpyAsync(data.output.data(), d_output, size_output, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    result.verify_count = std::min(data.M * data.N, MAX_COMPARE_COUNT);

    // 13. 计算CPU参考结果（只计算部分元素）
    std::cout << "计算CPU参考结果（限制数量）..." << std::endl;
    std::vector<__half> h_ref;

    if (params.nvfp4_ref) {
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

    // 14. 误差分析
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

    if (!params.output_file.empty()) {
        data.A_fp4.resize(size_A_fp4 / sizeof(int64_t));
        data.B_fp4.resize(size_B_fp4 / sizeof(int64_t));
        data.A_scales.resize(size_A_scales / sizeof(int32_t));
        data.B_scales.resize(size_B_scales / sizeof(int32_t));
        // std::cout << "size_A_scales: " << size_A_scales << " size_B_scales: " << size_B_scales << std::endl;

        CUDA_CHECK(cudaMemcpyAsync(data.A_fp4.data(), d_A_fp4, size_A_fp4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(data.B_fp4.data(), d_B_fp4, size_B_fp4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(data.A_scales.data(), d_A_scales, size_A_scales, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(data.B_scales.data(), d_B_scales, size_B_scales, cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));
        save_nvfp4_data(data, params.output_file);
    }

    // 15. 清理
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_A_fp16));
    CUDA_CHECK(cudaFree(d_B_fp16));
    CUDA_CHECK(cudaFree(d_a_global_scale));
    CUDA_CHECK(cudaFree(d_b_global_scale));
    CUDA_CHECK(cudaFree(d_A_fp4));
    CUDA_CHECK(cudaFree(d_B_fp4));
    CUDA_CHECK(cudaFree(d_A_scales));
    CUDA_CHECK(cudaFree(d_B_scales));
    CUDA_CHECK(cudaFree(d_output));

    return result;
}
