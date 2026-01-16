#ifndef BENCHMARK_CUTLASS_GEMM_SILU_H
#define BENCHMARK_CUTLASS_GEMM_SILU_H

#include "utils/results_info.h"

TestResult benchmark_gemm_silu_half(int M, int N, int K, int iterations = 10);
TestResult benchmark_gemm_silu_float(int M, int N, int K, int iterations = 10);

#endif // BENCHMARK_CUTLASS_GEMM_SILU_H