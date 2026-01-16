#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <cmath>
#include <type_traits>
#include <iomanip>
#include <sstream>

#include "utils/cuda_check.h"
#include "check/err_analysis.h"
#include "elementwise_mul_bench.h"

// ==================== CUDA核函数实现 ====================

// 模板化的CUDA核函数 - 支持float和__half
template<typename T>
__global__ void elementwise_mul_kernel_template(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ c,
    int total_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < total_elements; i += stride) {
        if constexpr (std::is_same<T, float>::value) {
            c[i] = a[i] * b[i];
        } else if constexpr (std::is_same<T, __half>::value) {
            c[i] = __hmul(a[i], b[i]);
        }
    }
}

// 二维版本的CUDA核函数
template<typename T>
__global__ void elementwise_mul_2d_kernel_template(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ c,
    int rows,
    int cols) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        if constexpr (std::is_same<T, float>::value) {
            c[idx] = a[idx] * b[idx];
        } else if constexpr (std::is_same<T, __half>::value) {
            c[idx] = __hmul(a[idx], b[idx]);
        }
    }
}

// ==================== Thrust实现 ====================

// Thrust乘法仿函数模板
template<typename T>
struct thrust_multiplies {
    __host__ __device__ T operator()(const T& a, const T& b) const {
        return a * b;
    }
};

// __half特化的Thrust乘法仿函数
template<>
struct thrust_multiplies<__half> {
    __host__ __device__ __half operator()(const __half& a, const __half& b) const {
        return __hmul(a, b);
    }
};

// ==================== CPU参考实现 ====================

template<typename T>
void compute_cpu_reference_elementwise(
    std::vector<T>& h_ref,
    const std::vector<T>& h_a,
    const std::vector<T>& h_b,
    int total_elements,
    int max_elements = MAX_COMPARE_COUNT) {
    
    int verify_count = std::min(total_elements, max_elements);
    h_ref.resize(total_elements);
    
    for (int i = 0; i < verify_count; ++i) {
        if constexpr (std::is_same<T, float>::value) {
            h_ref[i] = h_a[i] * h_b[i];
        } else if constexpr (std::is_same<T, __half>::value) {
            // 简化处理，实际应该用__hmul
            float a_float = __half2float(h_a[i]);
            float b_float = __half2float(h_b[i]);
            h_ref[i] = __float2half(a_float * b_float);
        }
    }
}

// ==================== 模板测试函数 ====================

template<typename T>
TestResult benchmark_elementwise_mul_template(
    int rows, int cols, int iterations,
    bool use_thrust, bool use_2d_kernel) {
    
    TestResult result;
    result.operation = "ElementwiseMul";
    
    if constexpr (std::is_same<T, float>::value) {
        result.data_type = "FP32";
    } else if constexpr (std::is_same<T, __half>::value) {
        result.data_type = "FP16";
    }
    
    std::ostringstream oss;
    oss << "(" << rows << "," << cols << ")";
    if (use_thrust) {
        oss << "_Thrust";
    } else {
        oss << "_CUDA";
        if (use_2d_kernel) {
            oss << "_2D";
        }
    }
    result.test_case = oss.str();
    
    // 创建主机端数据
    int total_elements = rows * cols;
    std::vector<T> h_a(total_elements);
    std::vector<T> h_b(total_elements);
    std::vector<T> h_c(total_elements, T(0));
    std::vector<T> h_ref(total_elements);
    
    // 初始化数据
    for (int i = 0; i < total_elements; ++i) {
        if constexpr (std::is_same<T, float>::value) {
            h_a[i] = static_cast<float>(rand() % 100) / 100.0f;
            h_b[i] = static_cast<float>(rand() % 100) / 100.0f;
        } else if constexpr (std::is_same<T, __half>::value) {
            float val_a = static_cast<float>(rand() % 100) / 100.0f;
            float val_b = static_cast<float>(rand() % 100) / 100.0f;
            h_a[i] = __float2half(val_a);
            h_b[i] = __float2half(val_b);
        }
    }
    
    // 设备端内存分配
    T* d_a = nullptr;
    T* d_b = nullptr;
    T* d_c = nullptr;
    
    size_t size = sizeof(T) * total_elements;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));
    
    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice));
    
    // 预热运行
    if (!use_thrust) {
        // CUDA实现
        if (use_2d_kernel && rows > 1) {
            dim3 blockSize(16, 16);
            dim3 gridSize(
                (cols + blockSize.x - 1) / blockSize.x,
                (rows + blockSize.y - 1) / blockSize.y
            );
            elementwise_mul_2d_kernel_template<T><<<gridSize, blockSize>>>(
                d_a, d_b, d_c, rows, cols);
        } else {
            int blockSize = 256;
            int gridSize = (total_elements + blockSize - 1) / blockSize;
            elementwise_mul_kernel_template<T><<<gridSize, blockSize>>>(
                d_a, d_b, d_c, total_elements);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        // Thrust实现
        thrust::device_ptr<T> thrust_a(d_a);
        thrust::device_ptr<T> thrust_b(d_b);
        thrust::device_ptr<T> thrust_c(d_c);
        
        thrust::transform(thrust_a, thrust_a + total_elements,
                         thrust_b, thrust_c,
                         thrust_multiplies<T>());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // 性能测试
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    double total_time_ms = 0.0;
    double min_time_ms = std::numeric_limits<double>::max();
    double max_time_ms = 0.0;
    
    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        
        if (!use_thrust) {
            // CUDA实现
            if (use_2d_kernel && rows > 1) {
                dim3 blockSize(16, 16);
                dim3 gridSize(
                    (cols + blockSize.x - 1) / blockSize.x,
                    (rows + blockSize.y - 1) / blockSize.y
                );
                elementwise_mul_2d_kernel_template<T><<<gridSize, blockSize>>>(
                    d_a, d_b, d_c, rows, cols);
            } else {
                int blockSize = 256;
                int gridSize = (total_elements + blockSize - 1) / blockSize;
                elementwise_mul_kernel_template<T><<<gridSize, blockSize>>>(
                    d_a, d_b, d_c, total_elements);
            }
        } else {
            // Thrust实现
            thrust::device_ptr<T> thrust_a(d_a);
            thrust::device_ptr<T> thrust_b(d_b);
            thrust::device_ptr<T> thrust_c(d_c);
            
            thrust::transform(thrust_a, thrust_a + total_elements,
                            thrust_b, thrust_c,
                            thrust_multiplies<T>());
        }
        
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
    result.iterations = iterations;
    
    // 计算性能指标
    // Element-wise乘法: 每个元素1次乘法 = 总元素数 FLOPs
    double flops = total_elements;
    double avg_gflops = (flops / result.avg_time_ms) / 1e6;  // 转换为GFLOPS
    double max_gflops = (flops / min_time_ms) / 1e6;
    double min_gflops = (flops / max_time_ms) / 1e6;
    
    // 使用tflops字段来存储GFLOPS（需要调整比例）
    result.avg_tflops = avg_gflops / 1000.0;  // 转换为TFLOPS
    result.max_tflops = max_gflops / 1000.0;
    result.min_tflops = min_gflops / 1000.0;
    
    // 带宽计算: 读取a和b，写入c = 3 * 数据量
    size_t bytes_transferred = 3 * size;
    result.avg_bandwidth_gbs = (bytes_transferred / result.avg_time_ms) / 1e6;
    
    // 清理事件
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    // 计算CPU参考结果
    compute_cpu_reference_elementwise<T>(h_ref, h_a, h_b, total_elements);
    
    // 获取GPU结果
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost));
    
    // 误差分析
    double abs_tolerance = 1e-3;
    double rel_tolerance = 1e-3;
    
    if constexpr (std::is_same<T, __half>::value) {
        abs_tolerance = 1e-2;
        rel_tolerance = 1e-2;
    }
    
    result.verify_count = std::min(total_elements, MAX_COMPARE_COUNT);
    
    // 将__half转换为float进行误差分析
    if constexpr (std::is_same<T, __half>::value) {
        std::vector<float> h_c_float(h_c.size());
        std::vector<float> h_ref_float(h_ref.size());
        
        for (size_t i = 0; i < h_c.size(); ++i) {
            h_c_float[i] = __half2float(h_c[i]);
            h_ref_float[i] = __half2float(h_ref[i]);
        }
        
        auto error_result = analyze_errors<float>(
            h_c_float, h_ref_float, 0, result.verify_count, 
            abs_tolerance, rel_tolerance);
        
        result.max_abs_error = error_result.max_abs_error;
        result.max_rel_error = error_result.max_rel_error;
        result.passed = error_result.passed;
        result.error_count = error_result.error_count;
        result.total_count = error_result.total_count;
    } else {
        auto error_result = analyze_errors<T>(
            h_c, h_ref, 0, result.verify_count, 
            abs_tolerance, rel_tolerance);
        
        result.max_abs_error = error_result.max_abs_error;
        result.max_rel_error = error_result.max_rel_error;
        result.passed = error_result.passed;
        result.error_count = error_result.error_count;
        result.total_count = error_result.total_count;
    }
    
    // 清理设备内存
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    
    return result;
}

// ==================== 具体测试函数实现 ====================

TestResult benchmark_elementwise_mul_cuda_float(int rows, int cols, int iterations) {
    return benchmark_elementwise_mul_template<float>(
        rows, cols, iterations, false, (rows > 1));
}

TestResult benchmark_elementwise_mul_cuda_half(int rows, int cols, int iterations) {
    return benchmark_elementwise_mul_template<__half>(
        rows, cols, iterations, false, (rows > 1));
}

TestResult benchmark_elementwise_mul_thrust_float(int rows, int cols, int iterations) {
    return benchmark_elementwise_mul_template<float>(
        rows, cols, iterations, true, false);
}

TestResult benchmark_elementwise_mul_thrust_half(int rows, int cols, int iterations) {
    return benchmark_elementwise_mul_template<__half>(
        rows, cols, iterations, true, false);
}