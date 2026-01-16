#ifndef BENCHMARK_CUBLAS_GEMV_H
#define BENCHMARK_CUBLAS_GEMV_H

#include "utils/results_info.h"

TestResult benchmark_gemv_half(int M, int N, int iterations = 100);
TestResult benchmark_gemv_float(int M, int N, int iterations = 100);
TestResult benchmark_gemv_double(int M, int N, int iterations = 100);

#endif