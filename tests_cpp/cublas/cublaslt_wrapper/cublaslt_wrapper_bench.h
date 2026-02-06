#ifndef CUBLASLT_WRAPPER_BENCH_H
#define CUBLASLT_WRAPPER_BENCH_H

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include "kernels/common/cublaslt_wrapper.h"
#include "utils/cuda_check.h"
#include "utils/results_info.h"

// 声明测试函数（模板化）
template<typename T>
TestResult benchmark_cublaslt_gemm(int M, int N, int K, 
                                  trt_edgellm::kernel::ComputeType compute_type,
                                  int iterations = 10);

template<typename T>
TestResult benchmark_cublaslt_gemm_silu(int M, int N, int K, 
                                       trt_edgellm::kernel::ComputeType compute_type,
                                       int iterations = 10);

template<typename T>
TestResult benchmark_cublaslt_gemm_bias(int M, int N, int K, 
                                       trt_edgellm::kernel::ComputeType compute_type,
                                       int iterations = 10);

// 辅助函数：计算类型转字符串
inline std::string compute_type_to_string(trt_edgellm::kernel::ComputeType type) {
    switch(type) {
        case trt_edgellm::kernel::ComputeType::FLOAT:
            return "FP32";
        case trt_edgellm::kernel::ComputeType::HALF:
            return "FP16";
        case trt_edgellm::kernel::ComputeType::TFLOAT32:
            return "TF32";
        default:
            return "UNKNOWN";
    }
}

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

#endif // CUBLASLT_WRAPPER_BENCH_H