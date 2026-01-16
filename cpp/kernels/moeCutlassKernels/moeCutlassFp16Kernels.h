#pragma once
#include <cutlass/half.h>
#include "moeInputsParams.h"

namespace trt_edgellm
{
namespace kernel
{
namespace cutlass_moe
{

void elementwise_multiply(cutlass::half_t* C,
                         const cutlass::half_t* A,
                         const cutlass::half_t* B,
                         int M, int N,
                         cudaStream_t stream);

void forward_mlp(cutlass::half_t* output,
                  const cutlass::half_t* input,
                  const cutlass::half_t* gate_weight,
                  const cutlass::half_t* up_weight,
                  const cutlass::half_t* down_weight,
                  cutlass::half_t* gate_output,
                  cutlass::half_t* up_output,
                  cutlass::half_t* gate_mul_up_output,
                  int batch_size,
                  int seq_len,
                  int hidden_size,
                  int intermediate_size,
                  cudaStream_t stream);

void forward_mlp_grouped(cutlass::half_t* output,
                  cutlass::half_t* input,
                  cutlass::half_t* gate_weight,
                  cutlass::half_t* up_weight,
                  cutlass::half_t* down_weight,
                  cutlass::half_t* gate_silu_output,
                  cutlass::half_t* up_output,
                  cutlass::half_t* gate_mul_up_output,
                  std::vector<int32_t> experts_tokens_count,
                  int hidden_size,
                  int intermediate_size,
                  cudaStream_t stream);

void forward_moe(const MoeInputsParams& params);

void context_moe_fp16_forward_cuda(const MoeInputsParams& params);

void decode_moe_fp16_forward_cuda(const MoeInputsParams& params);

} // cutlass_moe
} // namespace kernel
} // namespace trt_edgellm