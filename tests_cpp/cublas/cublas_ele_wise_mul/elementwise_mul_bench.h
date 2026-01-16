#ifndef BENCHMARK_ELEMENTWISE_MUL_H
#define BENCHMARK_ELEMENTWISE_MUL_H

#include <string>
#include <vector>
#include "utils/results_info.h"

// 纯CUDA实现的测试函数
TestResult benchmark_elementwise_mul_cuda_float(int rows, int cols, int iterations = 100);
TestResult benchmark_elementwise_mul_cuda_half(int rows, int cols, int iterations = 100);

// Thrust实现的测试函数
TestResult benchmark_elementwise_mul_thrust_float(int rows, int cols, int iterations = 100);
TestResult benchmark_elementwise_mul_thrust_half(int rows, int cols, int iterations = 100);

#endif // BENCHMARK_ELEMENTWISE_MUL_H