#ifndef BENCHMARK_CUBLAS_GEMM_SILU_H
#define BENCHMARK_CUBLAS_GEMM_SILU_H

#include "utils/results_info.h"

// 分离版本：cuBLAS GEMM + 独立的SiLU kernel
TestResult benchmark_gemm_silu_separate_half(int M, int N, int K, int iterations = 10);
TestResult benchmark_gemm_silu_separate_float(int M, int N, int K, int iterations = 10);

// 融合版本：自定义融合kernel
TestResult benchmark_gemm_silu_fused_half(int M, int N, int K, int iterations = 10);
TestResult benchmark_gemm_silu_fused_float(int M, int N, int K, int iterations = 10);

#endif // BENCHMARK_CUBLAS_GEMM_SILU_H