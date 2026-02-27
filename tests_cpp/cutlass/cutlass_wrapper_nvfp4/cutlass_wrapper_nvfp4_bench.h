#ifndef CUTLASS_WRAPPER_NVFP4_BENCH_H
#define CUTLASS_WRAPPER_NVFP4_BENCH_H

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "cutlass_wrapper_nvfp4.h"
#include "utils/cuda_check.h"
#include "utils/results_info.h"

// 测试函数声明
TestResult benchmark_nvfp4_gemm_half(int M, int N, int K, int iterations = 10);
TestResult benchmark_nvfp4_gemm_bias_half(int M, int N, int K, int iterations = 10);

// 参考计算函数声明
void compute_reference_nvfp4_gemm(std::vector<cutlass::half_t>& h_ref,
                                 const std::vector<cutlass::half_t>& h_A,
                                 const std::vector<cutlass::half_t>& h_B,
                                 const std::vector<float>& h_B_scales,
                                 int M, int N, int K, int block_size = 16);

#endif // CUTLASS_WRAPPER_NVFP4_BENCH_H