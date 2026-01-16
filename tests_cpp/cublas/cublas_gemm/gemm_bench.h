#ifndef BENCHMARK_CUBLAS_GEMM_H
#define BENCHMARK_CUBLAS_GEMM_H
#include "utils/results_info.h"

TestResult benchmark_gemm_half(int M, int N, int K, int iterations = 10);
TestResult benchmark_gemm_float(int M, int N, int K, int iterations = 10);

#endif // BENCHMARK_CUBLAS_GEMM_H