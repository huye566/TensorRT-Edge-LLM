#ifndef CUTLASS_WRAPPER_H
#define CUTLASS_WRAPPER_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include "cutlass/half.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/layout.h"

namespace trt_edgellm {
namespace kernel {

// 基元类型别名
using cutlass_half_t = cutlass::half_t;
using ElementAccumulator = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA = cutlass_half_t;
using ElementInputB = cutlass_half_t;
using ElementOutput = cutlass_half_t;

// 内存布局
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// 计算架构配置
using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;

// 线程块和warp配置
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

template <int Ma = 8, int Na = 8, int Ca = 8, int Stages = 3>
void cutlass_gemm(ElementOutput* output,
                  const ElementInputA* input,
                  const ElementInputB* weights,
                  int M, int N, int K,
                  cudaStream_t stream = 0);

template <int Ma = 8, int Na = 8, int Ca = 8, int Stages = 3>
void cutlass_gemm_silu(ElementOutput* output,
                       const ElementInputA* input,
                       const ElementInputB* weights,
                       int M, int N, int K,
                       cudaStream_t stream = 0);

template <int Ma = 8, int Na = 8, int Ca = 8, int Stages = 3>
void cutlass_gemm_bias(ElementOutput* output,
                       const ElementInputA* input,
                       const ElementInputB* weights,
                       const ElementOutput* bias,
                       int M, int N, int K,
                       cudaStream_t stream = 0);

template <int M1 = 128, int N1 = 128, int K1 = 32, int M2 = 64, int N2 = 64, int K2 = 32, bool kEnableSilu = false>
void cutlass_gemm_grouped(ElementOutput* output,
                          ElementInputA* input,
                          ElementInputB* weight,
                          std::vector<cutlass::gemm::GemmCoord>& problem_sizes,
                          cudaStream_t stream = 0);

} // namespace kernel
} // namespace trt_edgellm

#endif // CUTLASS_WRAPPER_H