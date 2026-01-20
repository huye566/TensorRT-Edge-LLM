#ifndef CUTLASS_WRAPPER_BLACKWELL_BENCH_H
#define CUTLASS_WRAPPER_BLACKWELL_BENCH_H

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "kernels/common/cutlass_wrapper_blackwell.h"
#include "utils/cuda_check.h"
#include "utils/results_info.h"

// 声明测试函数
TestResult benchmark_blackwell_gemm_half(int M, int N, int K, int iterations = 10);
TestResult benchmark_blackwell_gemm_silu_half(int M, int N, int K, int iterations = 10);
TestResult benchmark_blackwell_gemm_bias_half(int M, int N, int K, int iterations = 10);

// 参考计算函数声明
void compute_reference_blackwell_gemm(std::vector<cutlass::half_t>& h_ref,
                                     const std::vector<cutlass::half_t>& h_A,
                                     const std::vector<cutlass::half_t>& h_B,
                                     int M, int N, int K);
void compute_reference_blackwell_gemm_silu(std::vector<cutlass::half_t>& h_ref,
                                          const std::vector<cutlass::half_t>& h_A,
                                          const std::vector<cutlass::half_t>& h_B,
                                          int M, int N, int K);
void compute_reference_blackwell_gemm_bias(std::vector<cutlass::half_t>& h_ref,
                                          const std::vector<cutlass::half_t>& h_A,
                                          const std::vector<cutlass::half_t>& h_B,
                                          const std::vector<cutlass::half_t>& h_bias,
                                          int M, int N, int K);

#endif // CUTLASS_WRAPPER_BLACKWELL_BENCH_H