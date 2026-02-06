#ifndef CUTLASS_WRAPPER_NVFP4_BLOCKWISE_BENCH_H
#define CUTLASS_WRAPPER_NVFP4_BLOCKWISE_BENCH_H

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "kernels/common/cutlass_wrapper_nvfp4_blockwise.h"
#include "kernels/common/nvfp4_quant.h"
#include "utils/cuda_check.h"
#include "utils/results_info.h"

// 常量定义
constexpr float FLOAT8_E4M3_MAX = 448.0f;  // E4M3的最大值
constexpr float FLOAT4_E2M1_MAX = 6.0f;    // E2M1的最大值
constexpr int BLOCK_SIZE = 16;  // NVFP4 block size

struct TestParams {
    int M;
    int N;
    int K;
    int iterations{10};
    bool use_bias{false};
    bool use_silu{false};
    bool nvfp4_ref{true};
    std::string input_file;
    std::string output_file;
};

// 测试函数声明
TestResult benchmark_nvfp4_gemm_half(const TestParams &params);

#endif // CUTLASS_WRAPPER_NVFP4_BLOCKWISE_BENCH_H