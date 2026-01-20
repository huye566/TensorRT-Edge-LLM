#ifndef CUBLAS_WRAPPER_BENCH_H
#define CUBLAS_WRAPPER_BENCH_H

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "kernels/common/cublas_wrapper.h"
#include "utils/cuda_check.h"
#include "utils/results_info.h"

// 声明测试函数（模板化）
template<typename T>
TestResult benchmark_cublas_gemm(int M, int N, int K, int iterations = 10);

template<typename T>
TestResult benchmark_cublas_gemm_silu(int M, int N, int K, int iterations = 10);

template<typename T>
TestResult benchmark_cublas_gemm_bias(int M, int N, int K, int iterations = 10);

// 参考计算函数声明
template<typename T>
void compute_reference_gemm(std::vector<T>& h_ref,
                           const std::vector<T>& h_A,
                           const std::vector<T>& h_B,
                           int M, int N, int K);

template<typename T>
void compute_reference_gemm_silu(std::vector<T>& h_ref,
                                const std::vector<T>& h_A,
                                const std::vector<T>& h_B,
                                int M, int N, int K);

template<typename T>
void compute_reference_gemm_bias(std::vector<T>& h_ref,
                                const std::vector<T>& h_A,
                                const std::vector<T>& h_B,
                                const std::vector<T>& h_bias,
                                int M, int N, int K);

#endif // CUBLAS_WRAPPER_BENCH_H