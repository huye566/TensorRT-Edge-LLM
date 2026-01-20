#ifndef BENCHMARK_CUTLASS_WRAPPER_TEST_H
#define BENCHMARK_CUTLASS_WRAPPER_TEST_H

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "kernels/common/cutlass_wrapper.h"
#include "utils/cuda_check.h"
#include "utils/results_info.h"

// 声明测试函数
TestResult benchmark_wrapper_gemm_half(int M, int N, int K, int iterations = 10);
TestResult benchmark_wrapper_gemm_silu_half(int M, int N, int K, int iterations = 10);
TestResult benchmark_wrapper_gemm_bias_half(int M, int N, int K, int iterations = 10);

void compute_reference_gemm(std::vector<cutlass::half_t>& h_ref,
                           const std::vector<cutlass::half_t>& h_A,
                           const std::vector<cutlass::half_t>& h_B,
                           int M, int N, int K);
void compute_reference_gemm_silu(std::vector<cutlass::half_t>& h_ref,
                                const std::vector<cutlass::half_t>& h_A,
                                const std::vector<cutlass::half_t>& h_B,
                                int M, int N, int K);
void compute_reference_gemm_bias(std::vector<cutlass::half_t>& h_ref,
                                const std::vector<cutlass::half_t>& h_A,
                                const std::vector<cutlass::half_t>& h_B,
                                const std::vector<cutlass::half_t>& h_bias,
                                int M, int N, int K);

#endif // BENCHMARK_CUTLASS_WRAPPER_TEST_H