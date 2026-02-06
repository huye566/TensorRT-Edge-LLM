#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <NvInfer.h>
#include "kernels/common/nvfp4_quant.h"
#include "utils/cuda_check.h"

// 配置开关
const bool PRINT_FIRST_5_COMPARISONS = true;

// 常量定义
const float FLOAT4_E2M1_MAX = 6.0f;
const float FLOAT8_E4M3_MAX = 448.0f; // torch.finfo(torch.float8_e4m3fn).max
const int BLOCK_SIZE = 16;

const float E2M1_TO_FLOAT32[16] = {
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
};

__nv_fp8_e4m3 fp32_to_fp8_e4m3(float value) {
    return __nv_fp8_e4m3(value);
}

float fp8_e4m3_to_fp32(__nv_fp8_e4m3 value) {
    return static_cast<float>(value);
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

float fp4_idx_to_float(uint8_t fp4_idx) {
    if (fp4_idx >= 0 && fp4_idx < 16) {
        return E2M1_TO_FLOAT32[fp4_idx];
    }
    return 0.0f;
}

uint8_t float_to_fp4_idx(float x) {
    // 处理符号
    uint8_t sign_bit = (x < 0) ? 0x08 : 0x00;
    float abs_x = std::abs(x);

    // 根据Python实现中的逻辑进行映射
    if (abs_x <= 0.25f) {
        return sign_bit | 0x0;
    } else if (abs_x > 0.25f && abs_x < 0.75f) {
        return sign_bit | 0x1;
    } else if (abs_x >= 0.75f && abs_x <= 1.25f) {
        return sign_bit | 0x2;
    } else if (abs_x > 1.25f && abs_x < 1.75f) {
        return sign_bit | 0x3;
    } else if (abs_x >= 1.75f && abs_x <= 2.5f) {
        return sign_bit | 0x4;
    } else if (abs_x > 2.5f && abs_x < 3.5f) {
        return sign_bit | 0x5;
    } else if (abs_x >= 3.5f && abs_x <= 5.0f) {
        return sign_bit | 0x6;
    } else {
        return sign_bit | 0x7;
    }
}

void fp4_to_uint8_array(const int64_t* fp4_data, uint8_t* uint8_data, int m, int n) {
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
                uint8_data[element_idx] = fp4_val;
            }
        }
    }
}

// 计算倒数
float get_reciprocal(float x) {
    return (x == 0.0f) ? 0.0f : 1.0f / x;
}

// 获取int32_t中的uint8_t值
uint8_t get_uint8_from_int32(const int32_t* data, int index) {
    const uint8_t* byte_ptr = reinterpret_cast<const uint8_t*>(data);
    return byte_ptr[index];
}

void ref_nvfp4_quant(const half* input, int m, int n, float global_scale,
                     std::vector<uint8_t>& out_ref, std::vector<float>& scale_ref) {
    int groups_k = n / BLOCK_SIZE;

    out_ref.resize(m * n);
    scale_ref.resize(m * groups_k);

    // 转换为float数组
    std::vector<float> input_float(m * n);
    for (int i = 0; i < m * n; i++) {
        input_float[i] = __half2float(input[i]);
    }

    // 处理每个16元素组
    for (int i = 0; i < m; i++) {
        for (int g = 0; g < groups_k; g++) {
            // 计算组内最大值
            float vec_max = 0.0f;
            for (int k = 0; k < BLOCK_SIZE; k++) {
                int idx = i * n + g * BLOCK_SIZE + k;
                float abs_val = std::abs(input_float[idx]);
                if (abs_val > vec_max) {
                    vec_max = abs_val;
                }
            }

            // 计算scale
            float scale = global_scale * (vec_max * get_reciprocal(FLOAT4_E2M1_MAX));

            // FP8 E4M3量化
            __nv_fp8_e4m3 fp8_scale = fp32_to_fp8_e4m3(scale);
            scale = fp8_e4m3_to_fp32(fp8_scale);
            scale_ref[i * groups_k + g] = scale;

            // 计算output_scale
            float output_scale = get_reciprocal(scale * get_reciprocal(global_scale));

            // 对组内每个元素进行缩放和量化
            for (int k = 0; k < BLOCK_SIZE; k++) {
                int idx = i * n + g * BLOCK_SIZE + k;
                float scaled_x = input_float[idx] * output_scale;
                // 裁剪到[-6, 6]
                if (scaled_x > 6.0f) scaled_x = 6.0f;
                if (scaled_x < -6.0f) scaled_x = -6.0f;
                out_ref[idx] = float_to_fp4_idx(scaled_x);
            }
        }
    }
}

void recover_swizzled_scales(const int32_t* swizzled_scales, float* recovered_scales, int m, int n) {
    int scale_n = n / BLOCK_SIZE;
    int rounded_m = ((m + 128 - 1) / 128) * 128;
    int rounded_n = ((scale_n + 4 - 1) / 4) * 4;

    // 创建临时数组
    std::vector<uint8_t> tmp_data(rounded_m * rounded_n);

    // 将int32_t转换为uint8_t
    for (int i = 0; i < (rounded_m * rounded_n) / 4; i++) {
        uint32_t val = reinterpret_cast<const uint32_t*>(swizzled_scales)[i];
        tmp_data[i * 4] = val & 0xFF;
        tmp_data[i * 4 + 1] = (val >> 8) & 0xFF;
        tmp_data[i * 4 + 2] = (val >> 16) & 0xFF;
        tmp_data[i * 4 + 3] = (val >> 24) & 0xFF;
    }

    // 重新排列数据
    std::vector<uint8_t> reordered(rounded_m * rounded_n);

    int M_tile = rounded_m / 128;
    int K_tile = rounded_n / 4;

    // 关键：理解数据是如何存储的
    // 在内核中，数据是按照 [M_tile, K_tile, 32, 4, 4] 存储的
    // 我们需要重新排列为 [rounded_m, rounded_n]
    // 其中每128行对应一个M_tile，每4列对应一个K_tile

    for (int m_tile = 0; m_tile < M_tile; m_tile++) {
        for (int k_tile = 0; k_tile < K_tile; k_tile++) {
            for (int outerM = 0; outerM < 32; outerM++) {
                for (int innerM = 0; innerM < 4; innerM++) {
                    for (int innerK = 0; innerK < 4; innerK++) {
                        // 源位置
                        int src_idx = (((m_tile * K_tile + k_tile) * 32 + outerM) * 4 + innerM) * 4 + innerK;

                        // 目标位置
                        // 按照Python的permute: (0,1,4,3,2,5)
                        // 这意味着:
                        // - 原来的维度2（32）变成了维度3
                        // - 原来的维度3（4）变成了维度2
                        // - 原来的维度4（K_tile）变成了维度4
                        int dst_row = m_tile * 128 + innerM * 32 + outerM;
                        int dst_col = k_tile * 4 + innerK;
                        int dst_idx = dst_row * rounded_n + dst_col;

                        if (src_idx < (int)tmp_data.size() && dst_idx < (int)reordered.size()) {
                            reordered[dst_idx] = tmp_data[src_idx];
                        }
                    }
                }
            }
        }
    }

    // 转换为float
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < scale_n; j++) {
            int idx = i * rounded_n + j;
            if (idx < (int)reordered.size()) {
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

// 比对结果函数
bool compare_results(const std::vector<uint8_t>& ref_result,
                    const std::vector<uint8_t>& test_result,
                    const std::vector<float>& ref_scale,
                    const std::vector<float>& test_scale,
                    int m, int n,
                    float tolerance = 1e-5) {

    int total_elements = m * n;
    int total_scales = m * (n / BLOCK_SIZE);

    bool all_passed = true;
    int mismatch_count = 0;

    // 比对量化结果
    for (int i = 0; i < total_elements; i++) {
        if (ref_result[i] != test_result[i]) {
            mismatch_count++;
            if (PRINT_FIRST_5_COMPARISONS && mismatch_count <= 5) {
                float ref_val = fp4_idx_to_float(ref_result[i]);
                float test_val = fp4_idx_to_float(test_result[i]);
                std::cout << "FP4 mismatch at element " << i
                          << " (row=" << i/n << ", col=" << i%n << ")"
                          << ": ref=" << ref_val
                          << " (idx=" << (int)ref_result[i] << ")"
                          << ", test=" << test_val
                          << " (idx=" << (int)test_result[i] << ")" << std::endl;
            }
            all_passed = false;
        }
    }

    // 比对scale因子 - 使用相对误差
    int scale_mismatch_count = 0;
    for (int i = 0; i < total_scales; i++) {
        float ref = ref_scale[i];
        float test = test_scale[i];
        float abs_diff = std::abs(ref - test);

        // 计算相对误差，避免除以零
        float rel_error = 0.0f;
        if (ref != 0.0f) {
            rel_error = abs_diff / std::abs(ref);
        } else if (test != 0.0f) {
            rel_error = 1.0f; // ref=0, test!=0
        }

        // 如果绝对误差大于1.0，或者相对误差大于10%，则认为不匹配
        if (abs_diff > 1.0f || rel_error > 0.1f) {
            scale_mismatch_count++;
            if (PRINT_FIRST_5_COMPARISONS && scale_mismatch_count <= 5) {
                std::cout << "Scale mismatch at index " << i
                          << " (row=" << i/(n/BLOCK_SIZE) << ", col=" << i%(n/BLOCK_SIZE) << ")"
                          << ": ref=" << ref_scale[i]
                          << ", test=" << test_scale[i]
                          << ", abs_diff=" << abs_diff
                          << ", rel_error=" << rel_error << std::endl;
            }
            all_passed = false;
        }
    }

    if (!all_passed) {
        std::cout << "Total FP4 mismatches: " << mismatch_count << "/" << total_elements
                  << " (" << (100.0f * mismatch_count / total_elements) << "%)" << std::endl;
        std::cout << "Total scale mismatches: " << scale_mismatch_count << "/" << total_scales
                  << " (" << (100.0f * scale_mismatch_count / total_scales) << "%)" << std::endl;
    } else {
        std::cout << "All FP4 values matched!" << std::endl;
        std::cout << "All scale values matched!" << std::endl;
    }

    return all_passed;
}

// 生成随机half数据
void generate_random_half(half* data, int size, float min_val = -3.0f, float max_val = 3.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (int i = 0; i < size; i++) {
        data[i] = __float2half(dis(gen));
    }
}

float get_tenor_max(const half* data, int size) {
    float max_val = 0.0f;
    for (int i = 0; i < size; i++) {
        float val = std::abs(__half2float(data[i]));
        if (val > max_val) {
            max_val = val;
        }
    }
    return max_val;
}

// 打印帮助信息
void print_help() {
    std::cout << "FP4 Quantization Test" << std::endl;
    std::cout << "=====================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  PRINT_FIRST_5_COMPARISONS: " << (PRINT_FIRST_5_COMPARISONS ? "ON" : "OFF") << std::endl;
    std::cout << "  FLOAT4_E2M1_MAX: " << FLOAT4_E2M1_MAX << std::endl;
    std::cout << "  FLOAT8_E4M3_MAX: " << FLOAT8_E4M3_MAX << std::endl;
    std::cout << "  BLOCK_SIZE: " << BLOCK_SIZE << std::endl;
    std::cout << std::endl;
}

// 检查CUDA错误
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        return false; \
    } \
}

// 测试函数
bool test_fp4_quant_half(int m, int n) {
    if (n % BLOCK_SIZE != 0) {
        std::cerr << "Error: n must be multiple of " << BLOCK_SIZE << std::endl;
        return false;
    }

    int total_elements = m * n;
    int groups_k = n / BLOCK_SIZE;
    int sf_size = m * groups_k;

    // 分配主机内存
    std::vector<half> h_input(total_elements);
    generate_random_half(h_input.data(), total_elements, -2.0f, 2.0f);

    // 计算全局scale
    float tensor_amax = get_tenor_max(h_input.data(), total_elements);
    float global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax;

    std::vector<float> h_scale_factor(1, global_scale);
    std::vector<int64_t> h_output(total_elements / 8);  // 每8个元素打包为一个int64_t
    std::vector<int32_t> h_sf_output(sf_size);  // scale因子

    // 分配设备内存
    half* d_input = nullptr;
    float* d_scale = nullptr;
    int64_t* d_output = nullptr;
    int32_t* d_sf_output = nullptr;

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, total_elements * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_scale, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, (total_elements / 8) * sizeof(int64_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_sf_output, sf_size * sizeof(int32_t)));

    // 拷贝数据到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), total_elements * sizeof(half),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_scale, h_scale_factor.data(), sizeof(float),
                                cudaMemcpyHostToDevice));

    // 创建CUDA流
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // 创建GPU计时器
    GPUTimer gpu_timer;

    // 预热一次，确保GPU稳定
    for (int warmup = 0; warmup < 3; ++warmup) {
        try {
            trt_edgellm::kernel::scaled_fp4_quant(m, n, d_input, d_scale, d_output, d_sf_output,
                                                  stream, nvinfer1::DataType::kHALF);
            cudaStreamSynchronize(stream);
        } catch (...) {
            // 忽略预热错误
        }
    }

    // 开始计时
    gpu_timer.start(stream);

    // 调用量化函数
    try {
        trt_edgellm::kernel::scaled_fp4_quant(m, n, d_input, d_scale, d_output, d_sf_output,
                                              stream, nvinfer1::DataType::kHALF);
    } catch (const std::exception& e) {
        std::cerr << "Error during quantization: " << e.what() << std::endl;

        // 清理资源
        cudaStreamDestroy(stream);
        cudaFree(d_input);
        cudaFree(d_scale);
        cudaFree(d_output);
        cudaFree(d_sf_output);

        return false;
    }

    // 停止计时
    gpu_timer.stop(stream);

    // 同步流，确保所有操作完成
    cudaStreamSynchronize(stream);

    // 获取GPU执行时间
    float gpu_ms = gpu_timer.elapsed();

    // 拷贝结果回主机
    CHECK_CUDA_ERROR(cudaMemcpy(h_output.data(), d_output, (total_elements / 8) * sizeof(int64_t),
                                cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_sf_output.data(), d_sf_output, sf_size * sizeof(int32_t),
                                cudaMemcpyDeviceToHost));

    // 解码FP4数据为uint8_t数组
    std::vector<uint8_t> h_test_result(total_elements);
    fp4_to_uint8_array(h_output.data(), h_test_result.data(), m, n);

    // 恢复scale因子布局
    std::vector<float> h_recovered_scales(sf_size);
    recover_swizzled_scales(h_sf_output.data(), h_recovered_scales.data(), m, n);

    // 生成参考结果
    std::vector<uint8_t> h_ref_result(total_elements);
    std::vector<float> h_ref_scales(sf_size);
    ref_nvfp4_quant(h_input.data(), m, n, global_scale, h_ref_result, h_ref_scales);

    // 打印前几个scale值用于调试
    if (PRINT_FIRST_5_COMPARISONS && m > 0 && groups_k > 0) {
        std::cout << "First 5 reference scales: ";
        for (int i = 0; i < std::min(5, sf_size); i++) {
            std::cout << h_ref_scales[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "First 5 recovered scales: ";
        for (int i = 0; i < std::min(5, sf_size); i++) {
            std::cout << h_recovered_scales[i] << " ";
        }
        std::cout << std::endl;
    }

    // 比对结果
    bool test_passed = compare_results(h_ref_result, h_test_result,
                                       h_ref_scales, h_recovered_scales,
                                       m, n, 1e-5);

    // 输出测试结果
    std::cout << "Test m=" << m << ", n=" << n << ":" << std::endl;
    std::cout << "  Input tensor amax: " << tensor_amax << std::endl;
    std::cout << "  Global scale: " << global_scale << std::endl;
    std::cout << "  Total elements: " << total_elements << std::endl;
    std::cout << "  GPU execution time: " << gpu_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << (total_elements * sizeof(half) / (gpu_ms / 1000.0f) / 1e9)
              << " GB/s" << std::endl;
    std::cout << "  Test " << (test_passed ? "PASSED" : "FAILED") << std::endl;
    std::cout << std::endl;

    // 清理资源
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_scale);
    cudaFree(d_output);
    cudaFree(d_sf_output);

    return test_passed;
}

int main() {
    print_help();

    std::cout << "Testing FP4 quantization for half data type..." << std::endl;
    std::cout << "==============================================" << std::endl;

    // 检查CUDA设备
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global memory: " << prop.totalGlobalMem / (1024 * 1024 * 1024.0) << " GB" << std::endl;
    std::cout << std::endl;

    // 定义测试形状
    std::vector<std::pair<int, int>> test_shapes = {
        {128, 64},
        {128, 128},
        {256, 64},
        {256, 128},
        {512, 256},
        {1024, 512}
    };

    bool all_tests_passed = true;

    // 运行所有测试
    for (const auto& shape : test_shapes) {
        int m = shape.first;
        int n = shape.second;

        std::cout << "Running test with shape [" << m << ", " << n << "]" << std::endl;
        std::cout << "----------------------------------------------" << std::endl;

        bool passed = test_fp4_quant_half(m, n);
        if (!passed) {
            all_tests_passed = false;
        }
    }

    // 测试结果汇总
    std::cout << "==============================================" << std::endl;
    if (all_tests_passed) {
        std::cout << "All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED!" << std::endl;
        return 1;
    }
}