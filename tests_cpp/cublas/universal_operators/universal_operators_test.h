#ifndef UNIVERSAL_OPERATORS_TEST_H
#define UNIVERSAL_OPERATORS_TEST_H

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "kernels/common/universal_operators.h"
#include "utils/cuda_check.h"
#include "utils/results_info.h"

// 测试函数声明
TestResult benchmark_universal_silu_half(int M, int N, int iterations = 10);
TestResult benchmark_universal_add_bias_half(int M, int N, int iterations = 10);
TestResult benchmark_universal_add_bias_silu_fused_half(int M, int N, int iterations = 10);
TestResult benchmark_universal_silu_float(int M, int N, int iterations = 10);
TestResult benchmark_universal_add_bias_float(int M, int N, int iterations = 10);
TestResult benchmark_universal_add_bias_silu_fused_float(int M, int N, int iterations = 10);

// 标量接口测试
TestResult benchmark_universal_silu_scalar_half(int M, int N, int iterations = 10);
TestResult benchmark_universal_add_bias_scalar_half(int M, int N, int iterations = 10);
TestResult benchmark_universal_silu_scalar_float(int M, int N, int iterations = 10);
TestResult benchmark_universal_add_bias_scalar_float(int M, int N, int iterations = 10);

// 优化向量接口测试
TestResult benchmark_universal_silu_vec_optimized_half(int M, int N, int iterations = 10);
TestResult benchmark_universal_add_bias_vec_optimized_half(int M, int N, int iterations = 10);
TestResult benchmark_universal_silu_vec_optimized_float(int M, int N, int iterations = 10);
TestResult benchmark_universal_add_bias_vec_optimized_float(int M, int N, int iterations = 10);

// 参考计算函数声明
void compute_reference_add_bias(std::vector<float>& h_ref, const std::vector<float>& h_input,
                                const std::vector<float>& h_bias, int M, int N);
void compute_reference_add_bias(std::vector<half>& h_ref, const std::vector<half>& h_input,
                                const std::vector<half>& h_bias, int M, int N);
void compute_reference_silu(std::vector<float>& h_ref, const std::vector<float>& h_input);
void compute_reference_silu(std::vector<half>& h_ref, const std::vector<half>& h_input);
void compute_reference_add_bias_silu(std::vector<float>& h_ref, const std::vector<float>& h_input,
                                     const std::vector<float>& h_bias, int M, int N);
void compute_reference_add_bias_silu(std::vector<half>& h_ref, const std::vector<half>& h_input,
                                     const std::vector<half>& h_bias, int M, int N);

#endif // UNIVERSAL_OPERATORS_TEST_H