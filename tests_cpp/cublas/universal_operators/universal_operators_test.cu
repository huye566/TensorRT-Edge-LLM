#include "universal_operators_test.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
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

// FP32参考计算
void compute_reference_silu(std::vector<float>& h_ref, const std::vector<float>& h_input) {
    h_ref.resize(h_input.size());
    for (size_t i = 0; i < h_input.size(); ++i) {
        float x = h_input[i];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        h_ref[i] = x * sigmoid;
    }
}

void compute_reference_add_bias(std::vector<float>& h_ref, const std::vector<float>& h_input,
                                const std::vector<float>& h_bias, int M, int N) {
    size_t total_elements = M * N;
    h_ref.resize(total_elements);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int idx = m * N + n;
            h_ref[idx] = h_input[idx] + h_bias[n];
        }
    }
}

void compute_reference_add_bias_silu(std::vector<float>& h_ref, const std::vector<float>& h_input,
                                     const std::vector<float>& h_bias, int M, int N) {
    size_t total_elements = M * N;
    h_ref.resize(total_elements);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int idx = m * N + n;
            float x = h_input[idx] + h_bias[n];
            float sigmoid = 1.0f / (1.0f + expf(-x));
            h_ref[idx] = x * sigmoid;
        }
    }
}

// FP16参考计算
void compute_reference_silu(std::vector<half>& h_ref, const std::vector<half>& h_input) {
    h_ref.resize(h_input.size());
    for (size_t i = 0; i < h_input.size(); ++i) {
        float x = static_cast<float>(h_input[i]);
        float sigmoid = 1.0f / (1.0f + expf(-x));
        h_ref[i] = half(x * sigmoid);
    }
}

void compute_reference_add_bias(std::vector<half>& h_ref, const std::vector<half>& h_input,
                                const std::vector<half>& h_bias, int M, int N) {
    size_t total_elements = M * N;
    h_ref.resize(total_elements);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int idx = m * N + n;
            float sum = static_cast<float>(h_input[idx]) + static_cast<float>(h_bias[n]);
            h_ref[idx] = half(sum);
        }
    }
}

void compute_reference_add_bias_silu(std::vector<half>& h_ref, const std::vector<half>& h_input,
                                     const std::vector<half>& h_bias, int M, int N) {
    size_t total_elements = M * N;
    h_ref.resize(total_elements);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int idx = m * N + n;
            float x = static_cast<float>(h_input[idx]) + static_cast<float>(h_bias[n]);
            float sigmoid = 1.0f / (1.0f + expf(-x));
            h_ref[idx] = half(x * sigmoid);
        }
    }
}

// 通用测试模板
template<typename T>
TestResult benchmark_universal_operator_template(
    const std::string& operation_name,
    nvinfer1::DataType dtype,
    int M, int N, int iterations,
    std::function<bool(void*, int, nvinfer1::DataType, cudaStream_t)> scalar_func,
    std::function<bool(void*, const void*, int, int, nvinfer1::DataType, cudaStream_t)> bias_func,
    std::function<bool(void*, const void*, int, int, nvinfer1::DataType, cudaStream_t)> fused_func,
    bool test_bias = false,
    bool test_fused = false,
    bool test_scalar = false) {

    TestResult result;
    result.M = M;
    result.N = N;
    result.K = 0; // Not applicable for element-wise ops
    result.data_type = (dtype == nvinfer1::DataType::kHALF) ? "FP16" : "FP32";
    result.operation = operation_name;
    result.iterations = iterations;

    std::ostringstream oss;
    oss << "(" << M << "x" << N << ")";
    result.test_case = oss.str();

    // 生成随机数据
    size_t total_elements = M * N;
    std::vector<T> h_input = generate_random_vector<T>(total_elements);
    std::vector<T> h_bias;
    if (test_bias || test_fused) {
        h_bias = generate_random_vector<T>(N);
    }

    // 分配设备内存
    T* d_input = nullptr;
    T* d_bias = nullptr;
    T* d_workspace = nullptr;

    size_t input_size = total_elements * sizeof(T);
    size_t bias_size = (test_bias || test_fused) ? N * sizeof(T) : 0;

    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    if (test_bias || test_fused) {
        CUDA_CHECK(cudaMalloc(&d_bias, bias_size));
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), bias_size, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&d_workspace, input_size));

    // 拷贝输入数据
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_workspace, h_input.data(), input_size, cudaMemcpyHostToDevice));

    // 预热
    cudaStream_t stream = 0;
    if (test_scalar && scalar_func) {
        scalar_func(d_workspace, total_elements, dtype, stream);
    } else if (test_bias && bias_func) {
        bias_func(d_workspace, d_bias, M, N, dtype, stream);
    } else if (test_fused && fused_func) {
        fused_func(d_workspace, d_bias, M, N, dtype, stream);
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
        // 每次测试前恢复原始数据
        CUDA_CHECK(cudaMemcpy(d_workspace, h_input.data(), input_size, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(start));

        if (test_scalar && scalar_func) {
            scalar_func(d_workspace, total_elements, dtype, stream);
        } else if (test_bias && bias_func) {
            bias_func(d_workspace, d_bias, M, N, dtype, stream);
        } else if (test_fused && fused_func) {
            fused_func(d_workspace, d_bias, M, N, dtype, stream);
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

    // 计算带宽（近似）
    size_t bytes_transferred = 0;
    if (test_scalar) {
        bytes_transferred = total_elements * sizeof(T) * 2; // 读+写
    } else if (test_bias) {
        bytes_transferred = (total_elements + N + total_elements) * sizeof(T); // 输入+偏置+输出
    } else if (test_fused) {
        bytes_transferred = (total_elements + N + total_elements) * sizeof(T);
    }

    result.avg_bandwidth_gbs = (bytes_transferred / result.avg_time_ms) / 1e6;

    // 验证结果
    std::vector<T> h_output(total_elements);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_workspace, input_size, cudaMemcpyDeviceToHost));

    std::vector<T> h_ref;
    result.verify_count = std::min(static_cast<int>(total_elements), MAX_COMPARE_COUNT);

    double abs_tolerance = (dtype == nvinfer1::DataType::kHALF) ? 1e-2 : 1e-5;
    double rel_tolerance = (dtype == nvinfer1::DataType::kHALF) ? 1e-2 : 1e-5;

    // 计算参考结果
    if (test_scalar) {
        compute_reference_silu(h_ref, h_input);
    } else if (test_bias) {
        compute_reference_add_bias(h_ref, h_input, h_bias, M, N);
    } else if (test_fused) {
        compute_reference_add_bias_silu(h_ref, h_input, h_bias, M, N);
    }

    // 误差分析
    auto error_result = analyze_errors(h_output, h_ref, 0, result.verify_count,
                                      abs_tolerance, rel_tolerance);

    result.max_abs_error = error_result.max_abs_error;
    result.max_rel_error = error_result.max_rel_error;
    result.passed = error_result.passed;
    result.error_count = error_result.error_count;
    result.total_count = error_result.total_count;

    // TFLOPS（对于逐元素操作不适用，设置为0）
    result.avg_tflops = 0.0;
    result.min_tflops = 0.0;
    result.max_tflops = 0.0;

    // 清理
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_workspace));
    if (test_bias || test_fused) {
        CUDA_CHECK(cudaFree(d_bias));
    }

    return result;
}

// 具体测试函数实现
TestResult benchmark_universal_silu_half(int M, int N, int iterations) {
    return benchmark_universal_operator_template<half>(
        "Vec_SiLU(FP16)",
        nvinfer1::DataType::kHALF,
        M, N, iterations,
        [](void* data, int num, nvinfer1::DataType dtype, cudaStream_t stream) {
            return trt_edgellm::kernel::apply_silu_vec(reinterpret_cast<half*>(data),
                                                      num, dtype, stream);
        },
        nullptr, // bias_func
        nullptr, // fused_func
        false, false, true
    );
}

TestResult benchmark_universal_add_bias_half(int M, int N, int iterations) {
    return benchmark_universal_operator_template<half>(
        "Vec_AddBias(FP16)",
        nvinfer1::DataType::kHALF,
        M, N, iterations,
        nullptr, // scalar_func
        [](void* data, const void* bias, int m, int n, nvinfer1::DataType dtype, cudaStream_t stream) {
            return trt_edgellm::kernel::apply_add_bias_vec(reinterpret_cast<half*>(data),
                                                         reinterpret_cast<const half*>(bias),
                                                         m, n, dtype, stream);
        },
        nullptr, // fused_func
        true, false, false
    );
}

TestResult benchmark_universal_add_bias_silu_fused_half(int M, int N, int iterations) {
    return benchmark_universal_operator_template<half>(
        "Vec_AddBiasSiLU_Fused(FP16)",
        nvinfer1::DataType::kHALF,
        M, N, iterations,
        nullptr, // scalar_func
        nullptr, // bias_func
        [](void* data, const void* bias, int m, int n, nvinfer1::DataType dtype, cudaStream_t stream) {
            return trt_edgellm::kernel::apply_add_bias_silu_fused(reinterpret_cast<half*>(data),
                                                                reinterpret_cast<const half*>(bias),
                                                                m, n, dtype, stream);
        },
        false, true, false
    );
}

TestResult benchmark_universal_silu_float(int M, int N, int iterations) {
    return benchmark_universal_operator_template<float>(
        "Vec_SiLU(FP32)",
        nvinfer1::DataType::kFLOAT,
        M, N, iterations,
        [](void* data, int num, nvinfer1::DataType dtype, cudaStream_t stream) {
            return trt_edgellm::kernel::apply_silu_vec(reinterpret_cast<float*>(data),
                                                      num, dtype, stream);
        },
        nullptr, // bias_func
        nullptr, // fused_func
        false, false, true
    );
}

TestResult benchmark_universal_add_bias_float(int M, int N, int iterations) {
    return benchmark_universal_operator_template<float>(
        "Vec_AddBias(FP32)",
        nvinfer1::DataType::kFLOAT,
        M, N, iterations,
        nullptr, // scalar_func
        [](void* data, const void* bias, int m, int n, nvinfer1::DataType dtype, cudaStream_t stream) {
            return trt_edgellm::kernel::apply_add_bias_vec(reinterpret_cast<float*>(data),
                                                         reinterpret_cast<const float*>(bias),
                                                         m, n, dtype, stream);
        },
        nullptr, // fused_func
        true, false, false
    );
}

TestResult benchmark_universal_add_bias_silu_fused_float(int M, int N, int iterations) {
    return benchmark_universal_operator_template<float>(
        "Vec_AddBiasSiLU_Fused(FP32)",
        nvinfer1::DataType::kFLOAT,
        M, N, iterations,
        nullptr, // scalar_func
        nullptr, // bias_func
        [](void* data, const void* bias, int m, int n, nvinfer1::DataType dtype, cudaStream_t stream) {
            return trt_edgellm::kernel::apply_add_bias_silu_fused(reinterpret_cast<float*>(data),
                                                                reinterpret_cast<const float*>(bias),
                                                                m, n, dtype, stream);
        },
        false, true, false
    );
}

TestResult benchmark_universal_silu_scalar_half(int M, int N, int iterations) {
    return benchmark_universal_operator_template<half>(
        "Scalar_SiLU(FP16)",
        nvinfer1::DataType::kHALF,
        M, N, iterations,
        [](void* data, int num, nvinfer1::DataType dtype, cudaStream_t stream) {
            return trt_edgellm::kernel::apply_silu_scalar(data, num, dtype, stream);
        },
        nullptr, // bias_func
        nullptr, // fused_func
        false, false, true
    );
}

TestResult benchmark_universal_add_bias_scalar_half(int M, int N, int iterations) {
    return benchmark_universal_operator_template<half>(
        "Scalar_AddBias(FP16)",
        nvinfer1::DataType::kHALF,
        M, N, iterations,
        nullptr, // scalar_func
        [](void* data, const void* bias, int m, int n, nvinfer1::DataType dtype, cudaStream_t stream) {
            return trt_edgellm::kernel::apply_add_bias_scalar(data, bias, m, n, dtype, stream);
        },
        nullptr, // fused_func
        true, false, false
    );
}

// 标量接口测试函数（FP32）
TestResult benchmark_universal_silu_scalar_float(int M, int N, int iterations) {
    return benchmark_universal_operator_template<float>(
        "Scalar_SiLU(FP32)",
        nvinfer1::DataType::kFLOAT,
        M, N, iterations,
        [](void* data, int num, nvinfer1::DataType dtype, cudaStream_t stream) {
            return trt_edgellm::kernel::apply_silu_scalar(data, num, dtype, stream);
        },
        nullptr, // bias_func
        nullptr, // fused_func
        false, false, true
    );
}

TestResult benchmark_universal_add_bias_scalar_float(int M, int N, int iterations) {
    return benchmark_universal_operator_template<float>(
        "Scalar_AddBias(FP32)",
        nvinfer1::DataType::kFLOAT,
        M, N, iterations,
        nullptr, // scalar_func
        [](void* data, const void* bias, int m, int n, nvinfer1::DataType dtype, cudaStream_t stream) {
            return trt_edgellm::kernel::apply_add_bias_scalar(data, bias, m, n, dtype, stream);
        },
        nullptr, // fused_func
        true, false, false
    );
}

TestResult benchmark_universal_silu_vec_optimized_half(int M, int N, int iterations) {
    return benchmark_universal_operator_template<half>(
        "VecOptimized_SiLU(FP16)",
        nvinfer1::DataType::kHALF,
        M, N, iterations,
        [](void* data, int num, nvinfer1::DataType dtype, cudaStream_t stream) {
            return trt_edgellm::kernel::apply_silu_vec_optimized(data, num, dtype, stream);
        },
        nullptr, // bias_func
        nullptr, // fused_func
        false, false, true
    );
}

TestResult benchmark_universal_add_bias_vec_optimized_half(int M, int N, int iterations) {
    return benchmark_universal_operator_template<half>(
        "VecOptimized_AddBias(FP16)",
        nvinfer1::DataType::kHALF,
        M, N, iterations,
        nullptr, // scalar_func
        [](void* data, const void* bias, int m, int n, nvinfer1::DataType dtype, cudaStream_t stream) {
            return trt_edgellm::kernel::apply_add_bias_vec_optimized(data, bias, m, n, dtype, stream);
        },
        nullptr, // fused_func
        true, false, false
    );
}

// 优化向量接口测试函数（FP32）
TestResult benchmark_universal_silu_vec_optimized_float(int M, int N, int iterations) {
    return benchmark_universal_operator_template<float>(
        "VecOptimized_SiLU(FP32)",
        nvinfer1::DataType::kFLOAT,
        M, N, iterations,
        [](void* data, int num, nvinfer1::DataType dtype, cudaStream_t stream) {
            return trt_edgellm::kernel::apply_silu_vec_optimized(data, num, dtype, stream);
        },
        nullptr, // bias_func
        nullptr, // fused_func
        false, false, true
    );
}

TestResult benchmark_universal_add_bias_vec_optimized_float(int M, int N, int iterations) {
    return benchmark_universal_operator_template<float>(
        "VecOptimized_AddBias(FP32)",
        nvinfer1::DataType::kFLOAT,
        M, N, iterations,
        nullptr, // scalar_func
        [](void* data, const void* bias, int m, int n, nvinfer1::DataType dtype, cudaStream_t stream) {
            return trt_edgellm::kernel::apply_add_bias_vec_optimized(data, bias, m, n, dtype, stream);
        },
        nullptr, // fused_func
        true, false, false
    );
}