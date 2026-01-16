
#include "moeCutlassFp16Kernels.h"

#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <assert.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"
#include "kernels/common/cublas_wrapper.h"
#include "kernels/common/cutlass_wrapper.h"

#ifdef ENABLE_MOE_DEBUG
    #define MOE_PRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
    #define MOE_PRINT(fmt, ...) do {} while(0)
#endif


namespace trt_edgellm {
namespace kernel {
namespace cutlass_moe {

template <typename T>
__global__ void elementwise_multiply_kernel(
    T* C,
    const T* A,
    const T* B,
    int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        int idx = row * N + col;
        if constexpr (std::is_same<T, half>::value) {
            C[idx] = __hmul(A[idx], B[idx]);
        } else {
            C[idx] = A[idx] * B[idx];
        }
    }
}

void elementwise_multiply(cutlass::half_t* C,
                         const cutlass::half_t* A,
                         const cutlass::half_t* B,
                         int M, int N,
                         cudaStream_t stream) {
    dim3 block_size(16, 16);
    dim3 grid_size(
        (N + block_size.x - 1) / block_size.x,
        (M + block_size.y - 1) / block_size.y
    );
    
    elementwise_multiply_kernel<cutlass::half_t><<<grid_size, block_size, 0, stream>>>(
        C, A, B, M, N);

    CUDA_CHECK(cudaGetLastError());
}

void elementwise_multiply_half(half* C,
                              const half* A,
                              const half* B,
                              int M, int N,
                              cudaStream_t stream) {
    
    dim3 block_size(16, 16);
    dim3 grid_size(
        (N + block_size.x - 1) / block_size.x,
        (M + block_size.y - 1) / block_size.y
    );
    
    elementwise_multiply_kernel<half><<<grid_size, block_size, 0, stream>>>(
        C, A, B, M, N);

    CUDA_CHECK(cudaGetLastError());
}

__global__ void find_row_max_kernel(
    const cutlass::half_t* matrix,
    int* max_indices,
    int M, int N, int real_N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        const cutlass::half_t* row_ptr = matrix + row * N;

        cutlass::half_t max_val = row_ptr[0];
        int max_idx = 0;
        
        #pragma unroll
        for (int i = 1; i < real_N; i++) {
            if (row_ptr[i] > max_val) {
                max_val = row_ptr[i];
                max_idx = i;
            }
        }
        max_indices[row] = max_idx;
    }
}

// -------------------- 统计每个 expert 多少 token --------------------
__global__ void kernel_count_tokens(const int32_t* expert_idx,
                                    int32_t* expert_count,   // 长度 = num_experts
                                    int num_tokens) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_tokens) {
    int e = expert_idx[idx];
    atomicAdd(expert_count + e, 1);
  }
}

// -------------------- 重排 + scatter --------------------
// 先把 token 按 expert 连续写进 `sorted_x`，记录 `token2pos` 方便 scatter 回去
__global__ void kernel_reorder(const cutlass::half_t* x,
                               const int* expert_idx,
                               cutlass::half_t* sorted_x,       // 输出缓存
                               int32_t* token2pos,              // 原 token_id -> 在 sorted 里的偏移
                               int32_t* expert_offset,          // 每个 expert 在 sorted 里的起始偏移
                               int num_tokens,
                               int hidden_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_tokens) return;
  int e = expert_idx[idx];

  int pos;

 // 只有 threadIdx.y == 0 的线程执行原子操作
  if (threadIdx.y == 0) {
    pos = atomicAdd(expert_offset + e, 1);
    token2pos[idx] = pos;
  }
  
  // 同步线程块，确保 pos 值已写入
  __syncthreads();

  // 所有线程都读取 pos 值（从共享内存或全局内存）
  pos = token2pos[idx];

  // 拷贝 hidden 维
  for (int h = threadIdx.y; h < hidden_size; h += blockDim.y) 
  {
    sorted_x[pos * hidden_size + h] = x[idx * hidden_size + h];
  }
}

// 把专家结果写回原位
__global__ void kernel_scatter(const cutlass::half_t* sorted_out,
                               const int32_t* token2pos,
                               cutlass::half_t* out,
                               int num_tokens,
                               int hidden_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_tokens) return;
  int pos = token2pos[idx];
  for (int h = threadIdx.y; h < hidden_size; h += blockDim.y) {
    out[idx * hidden_size + h] = sorted_out[pos * hidden_size + h];
  }
}

__global__ void compute_max_logit_kernel(
    const half* router_logits,  // [num_tokens, experts_num]
    int* expert_indices,        // [num_tokens]
    int num_tokens,
    int experts_num) {
    
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;
    
    const half* token_logits = router_logits + token_idx * experts_num;
    float max_logit = -INFINITY;
    int best_expert = 0;
    
    for (int e = 0; e < experts_num; ++e) {
        float logit = __half2float(token_logits[e]);
        if (logit > max_logit) {
            max_logit = logit;
            best_expert = e;
        }
    }
    
    expert_indices[token_idx] = best_expert;
}


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
                  cudaStream_t stream) {
  if (!input || !gate_weight || !up_weight || !down_weight || !output 
      || !gate_silu_output || !up_output || !gate_mul_up_output) {
      return;
  }

  int num_tokens = 0;
  std::vector<cutlass::gemm::GemmCoord> problem_sizes_gate_up;
  std::vector<cutlass::gemm::GemmCoord> problem_sizes_down;
  for (size_t e = 0; e < experts_tokens_count.size(); e++) {
    int tokens_this_e = experts_tokens_count[e];
    cutlass::gemm::GemmCoord problem_gate_up(tokens_this_e, intermediate_size, hidden_size);
    cutlass::gemm::GemmCoord problem_down(tokens_this_e, hidden_size, intermediate_size);
    problem_sizes_gate_up.push_back(problem_gate_up);
    problem_sizes_down.push_back(problem_down);
    num_tokens += tokens_this_e;
  }

  cutlass_gemm_grouped<128, 128, 32, 64, 64, 32, true>(gate_silu_output, input, gate_weight, problem_sizes_gate_up, stream);

  cutlass_gemm_grouped<128, 128, 32, 64, 64, 32, false>(up_output, input, up_weight, problem_sizes_gate_up, stream);
 
  elementwise_multiply( gate_mul_up_output, 
                        gate_silu_output, 
                        up_output, 
                        num_tokens, 
                        intermediate_size, 
                        stream);

  cutlass_gemm_grouped<128, 128, 32, 64, 64, 32, false>(output, gate_mul_up_output, down_weight, problem_sizes_down, stream);
}

void forward_mlp(cutlass::half_t* output,
                  const cutlass::half_t* input,
                  const cutlass::half_t* gate_weight,
                  const cutlass::half_t* up_weight,
                  const cutlass::half_t* down_weight,
                  cutlass::half_t* gate_silu_output,
                  cutlass::half_t* up_output,
                  cutlass::half_t* gate_mul_up_output,
                  int batch_size,
                  int seq_len,
                  int hidden_size,
                  int intermediate_size,
                  cudaStream_t stream) {
    if (!input || !gate_weight || !up_weight || !down_weight || !output 
        || !gate_silu_output || !up_output || !gate_mul_up_output) {
        return;
    }

    int num_tokens = batch_size * seq_len;

    cutlass_gemm_silu(
        gate_silu_output,
        input,
        gate_weight,
        num_tokens,
        intermediate_size,
        hidden_size,
        stream);

    cutlass_gemm(
        reinterpret_cast<cutlass::half_t*>(up_output),
        reinterpret_cast<const cutlass::half_t*>(input),
        reinterpret_cast<const cutlass::half_t*>(up_weight),
        num_tokens,
        intermediate_size,
        hidden_size,
        stream);

    elementwise_multiply(
        gate_mul_up_output,
        gate_silu_output,
        up_output,
        num_tokens,
        intermediate_size,
        stream);

    cutlass_gemm(
        reinterpret_cast<cutlass::half_t*>(output),
        reinterpret_cast<const cutlass::half_t*>(gate_mul_up_output),
        reinterpret_cast<const cutlass::half_t*>(down_weight),
        num_tokens,
        hidden_size,
        intermediate_size,
        stream);
}

void prefill_compute_router_logits_cutlass_v1(const MoeInputsParams& params) {
  int num_tokens = params.total_tokens();
  int num_experts = params.experts_num;
  int num_experts_padded = 8;
  //todo: 在外面做 
  half *router_weight_padded, *router_bias_padded;
  size_t weight_padded_size = num_experts_padded * params.hidden_size * sizeof(half);
  size_t bias_padded_size = num_experts_padded * sizeof(half);
  cudaMalloc(reinterpret_cast<void**>(&router_weight_padded), weight_padded_size);
  cudaMalloc(reinterpret_cast<void**>(&router_bias_padded), bias_padded_size);

  cudaMemsetAsync(router_weight_padded, 0, weight_padded_size, params.stream);
  cudaMemsetAsync(router_bias_padded, 0, bias_padded_size, params.stream);

  cudaStreamSynchronize(params.stream); 

  cudaMemcpyAsync(router_bias_padded,
              params.router_bias,
              num_experts * sizeof(half),
              cudaMemcpyDeviceToDevice,
              params.stream);

  cudaMemcpy2DAsync(
      router_weight_padded,
      num_experts_padded * sizeof(half),
      params.router_weight,
      num_experts * sizeof(half),
      num_experts * sizeof(half),
      params.hidden_size,
      cudaMemcpyDeviceToDevice,
      params.stream);

  const cutlass::half_t* input_router =  reinterpret_cast<cutlass::half_t*>(params.hidden_state);
  const cutlass::half_t* weight_router =  reinterpret_cast<cutlass::half_t*>(router_weight_padded);
  const cutlass::half_t* bias_router =  reinterpret_cast<cutlass::half_t*>(router_bias_padded);

  cutlass_gemm_bias<8, 8, 8, 3>(reinterpret_cast<cutlass::half_t*>(params.router_logits),
            input_router,
            weight_router,
            bias_router,
            num_tokens,
            num_experts_padded,
            params.hidden_size,
            params.stream);
  CUDA_CHECK(cudaGetLastError());

  dim3 block_router(256);
  dim3 grid_router((num_tokens + block_router.x - 1) / block_router.x);
  find_row_max_kernel<<<grid_router, block_router, 0, params.stream>>>(
      reinterpret_cast<const cutlass::half_t*>(params.router_logits),
      params.expert_indices,
      num_tokens, num_experts_padded, num_experts);
  CUDA_CHECK(cudaGetLastError());

  //todo: 在外面做 
  cudaFree(router_weight_padded);
  cudaFree(router_bias_padded);
}

void prefill_compute_router_logits_cutlass_v2(const MoeInputsParams& params) {
  int num_tokens = params.total_tokens();
  int num_experts = params.experts_num;

  cutlass_gemm_bias<8, 1, 1, 2>(reinterpret_cast<cutlass::half_t*>(params.router_logits),
            reinterpret_cast<const cutlass::half_t*>(params.hidden_state),
            reinterpret_cast<const cutlass::half_t*>(params.router_weight),
            reinterpret_cast<const cutlass::half_t*>(params.router_bias),
            num_tokens,
            num_experts,
            params.hidden_size,
            params.stream);
  CUDA_CHECK(cudaGetLastError());

  dim3 block_router(256);
  dim3 grid_router((num_tokens + block_router.x - 1) / block_router.x);
  find_row_max_kernel<<<grid_router, block_router, 0, params.stream>>>(
      reinterpret_cast<const cutlass::half_t*>(params.router_logits),
      params.expert_indices,
      num_tokens, num_experts, num_experts);
  CUDA_CHECK(cudaGetLastError());
}

void prefill_compute_router_logits_cutlass(const MoeInputsParams& params) {
  int num_tokens = params.total_tokens();
  int num_experts = params.experts_num;
  int num_experts_padded = 8;
  const cutlass::half_t* input_router =  reinterpret_cast<cutlass::half_t*>(params.hidden_state);

  cutlass_gemm_bias<8, 8, 8, 3>(reinterpret_cast<cutlass::half_t*>(params.router_logits),
            input_router,
            reinterpret_cast<cutlass::half_t*>(params.router_weights_padded),
            reinterpret_cast<cutlass::half_t*>(params.router_bias_padded),
            num_tokens,
            num_experts_padded,
            params.hidden_size,
            params.stream);
  CUDA_CHECK(cudaGetLastError());

  dim3 block_router(256);
  dim3 grid_router((num_tokens + block_router.x - 1) / block_router.x);
  find_row_max_kernel<<<grid_router, block_router, 0, params.stream>>>(
      reinterpret_cast<const cutlass::half_t*>(params.router_logits),
      params.expert_indices,
      num_tokens, num_experts_padded, num_experts);
  CUDA_CHECK(cudaGetLastError());
}

void prefill_compute_router_logits_cublas(const MoeInputsParams& params) {
  int num_tokens = params.total_tokens();
  int num_experts = params.experts_num;

  auto& cublas_wrapper = CublasWrapper::instance();
  cublasSetStream(cublas_wrapper.handle(), params.stream);
  
  cublas_gemm_bias<half>(cublas_wrapper.handle(),
                        num_tokens,
                        num_experts,
                        params.hidden_size,
                        reinterpret_cast<const half*>(params.hidden_state), 
                        reinterpret_cast<const half*>(params.router_weight), 
                        reinterpret_cast<const half*>(params.router_bias),
                        reinterpret_cast<half*>(params.router_logits));

  dim3 block_router(256);
  dim3 grid_router((num_tokens + block_router.x - 1) / block_router.x);
  find_row_max_kernel<<<grid_router, block_router, 0, params.stream>>>(
      reinterpret_cast<const cutlass::half_t*>(params.router_logits),
      params.expert_indices,
      num_tokens, num_experts, num_experts);
  CUDA_CHECK(cudaGetLastError());
}

void decode_compute_router_logits_cublas(const MoeInputsParams& params) {
    int num_tokens = params.total_tokens();
    int hidden_size = params.hidden_size;
    int experts_num = params.experts_num;

    auto& cublas_wrapper = CublasWrapper::instance();
    cublasSetStream(cublas_wrapper.handle(), params.stream);
    
    cublas_gemm_bias<half>(cublas_wrapper.handle(),
                          num_tokens,
                          experts_num,
                          hidden_size,
                          reinterpret_cast<const half*>(params.hidden_state), 
                          reinterpret_cast<const half*>(params.router_weight), 
                          reinterpret_cast<const half*>(params.router_bias),
                          reinterpret_cast<half*>(params.router_logits));

    dim3 block_router(256);
    dim3 grid_router((num_tokens + block_router.x - 1) / block_router.x);
    compute_max_logit_kernel<<<grid_router, block_router, 0, params.stream>>>(
        reinterpret_cast<const half*>(params.router_logits), 
        params.expert_indices, 
        num_tokens, 
        experts_num);
    
    CUDA_CHECK(cudaGetLastError());
}

void forward_mlp_cublas(half* output,
                        const half* input,
                        const half* gate_weight,
                        const half* up_weight,
                        const half* down_weight,
                        half* gate_output,
                        half* up_output,
                        half* gate_mul_up_output,
                        int batch_size,
                        int seq_len,
                        int hidden_size,
                        int intermediate_size,
                        cudaStream_t stream) {
    
    int num_tokens = batch_size * seq_len;

    auto& cublas_wrapper = CublasWrapper::instance();
    cublasSetStream(cublas_wrapper.handle(), stream);

    cublas_gemm_silu<half>(cublas_wrapper.handle(),
                          num_tokens, intermediate_size, hidden_size,
                          input,
                          gate_weight,
                          gate_output);
    cublas_gemm<half>(cublas_wrapper.handle(),
                     num_tokens, intermediate_size, hidden_size,
                     input,
                     up_weight,
                     up_output);
    elementwise_multiply_half(gate_mul_up_output, gate_output, up_output,
                              num_tokens, intermediate_size, stream);
    cublas_gemm<half>(cublas_wrapper.handle(),
                     num_tokens, hidden_size, intermediate_size,
                     gate_mul_up_output,
                     down_weight,
                     output);
}

void forward_moe_single_token_cublas(const MoeInputsParams& params) {
    MOE_PRINT("[HOST INFO] Single token path with cuBLAS\n");
    
    int hidden_size = params.hidden_size;
    int intermediate_size = params.intermediate_size;
    
    decode_compute_router_logits_cublas(params);

    int best_expert = 0;
    cudaMemcpyAsync(&best_expert, params.expert_indices, sizeof(int), 
                   cudaMemcpyDeviceToHost, params.stream);
    cudaStreamSynchronize(params.stream);
    
    MOE_PRINT("[HOST INFO] Single token assigned to expert %d\n", best_expert);
    
    const half* mlp_gate_weight = reinterpret_cast<const half*>(params.gate_proj_weight) + 
                                 best_expert * hidden_size * intermediate_size;
    const half* mlp_up_weight = reinterpret_cast<const half*>(params.up_proj_weight) + 
                               best_expert * hidden_size * intermediate_size;
    const half* mlp_down_weight = reinterpret_cast<const half*>(params.down_proj_weight) + 
                                 best_expert * intermediate_size * hidden_size;
    
    forward_mlp_cublas(
        reinterpret_cast<half*>(params.output),
        reinterpret_cast<const half*>(params.hidden_state),
        mlp_gate_weight,
        mlp_up_weight,
        mlp_down_weight,
        reinterpret_cast<half*>(params.silu_output),
        reinterpret_cast<half*>(params.up_output),
        reinterpret_cast<half*>(params.hadamard_output),
        1, 1,
        hidden_size,
        intermediate_size,
        params.stream);
}

void forward_moe(const MoeInputsParams& params) {
    MOE_PRINT("[HOST INFO] Starting cutlass forward_moe\n");
    
    if (params.hidden_state == nullptr ||
        params.router_weight == nullptr ||
        params.router_bias == nullptr ||
        params.output == nullptr ||
        params.gate_proj_weight == nullptr ||
        params.up_proj_weight == nullptr ||
        params.down_proj_weight == nullptr) {
        MOE_PRINT("[HOST ERROR] Required parameters are null\n");
        return;
    }

    if (params.expert_input_buffer == nullptr ||
        params.expert_output_buffer == nullptr ||
        params.silu_output == nullptr ||
        params.up_output == nullptr ||
        params.hadamard_output == nullptr ||
        params.expert_counts == nullptr ||
        params.expert_offsets == nullptr ||
        params.token_to_buffer_map == nullptr ||
        params.expert_indices == nullptr ||
        params.router_logits == nullptr) {
        MOE_PRINT("[HOST ERROR] Buffer parameters are null\n");
        return;
    }
    
    int num_tokens = params.total_tokens();
    MOE_PRINT("[HOST INFO] Total tokens: %d, hidden_size: %d, intermediate_size: %d, experts_num: %d\n",
              num_tokens, params.hidden_size, params.intermediate_size, params.experts_num);
    
    if (num_tokens == 1) {
        forward_moe_single_token_cublas(params);
        return;
    }

    int num_experts = params.experts_num;
    int hidden_size = params.hidden_size;
    int intermediate_size = params.intermediate_size;

    cutlass::half_t *input = reinterpret_cast<cutlass::half_t*>(params.hidden_state);
    cutlass::half_t *output = reinterpret_cast<cutlass::half_t*>(params.output);
    cutlass::half_t *gate_proj_weight = reinterpret_cast<cutlass::half_t*>(params.gate_proj_weight);
    cutlass::half_t *up_proj_weight = reinterpret_cast<cutlass::half_t*>(params.up_proj_weight);
    cutlass::half_t *down_proj_weight = reinterpret_cast<cutlass::half_t*>(params.down_proj_weight);

    cutlass::half_t *d_sorted_x = reinterpret_cast<cutlass::half_t*>(params.expert_input_buffer);
    cutlass::half_t *d_sorted_out = reinterpret_cast<cutlass::half_t*>(params.expert_output_buffer);
    cutlass::half_t *silu_output = reinterpret_cast<cutlass::half_t*>(params.silu_output);
    cutlass::half_t *up_output = reinterpret_cast<cutlass::half_t*>(params.up_output);
    cutlass::half_t *hadamard_output = reinterpret_cast<cutlass::half_t*>(params.hadamard_output);

    int32_t *d_expert_count = params.expert_counts;
    int32_t *d_expert_offset = params.expert_offsets;
    int32_t *d_token2pos = params.token_to_buffer_map;

    prefill_compute_router_logits_cutlass(params);

    cudaMemsetAsync(d_expert_count, 0, num_experts * sizeof(int32_t), params.stream);
    kernel_count_tokens<<<(num_tokens + 255) / 256, 256, 0, params.stream>>>(params.expert_indices, d_expert_count, num_tokens);
    CUDA_CHECK(cudaGetLastError());

    // ---------- 把 count 拷到 host 算 offset ----------
    std::vector<int32_t> h_count(num_experts);
    cudaMemcpyAsync(h_count.data(), d_expert_count, num_experts * sizeof(int32_t), cudaMemcpyDeviceToHost, params.stream);
    CUDA_CHECK(cudaGetLastError());
    cudaStreamSynchronize(params.stream);
    CUDA_CHECK(cudaGetLastError());

    std::vector<int32_t> h_offset(num_experts + 1);
    h_offset[0] = 0;
    for (int i = 0; i < num_experts; ++i) h_offset[i + 1] = h_offset[i] + h_count[i];
    cudaMemcpyAsync(d_expert_offset, h_offset.data(), num_experts * sizeof(int32_t), cudaMemcpyHostToDevice, params.stream);
    CUDA_CHECK(cudaGetLastError());

    // ---------- reorder ----------
    dim3 reorder_block(8, 128);  
    kernel_reorder<<<(num_tokens + reorder_block.x - 1) / reorder_block.x, reorder_block, 0, params.stream>>>(
        input, params.expert_indices, d_sorted_x, d_token2pos, d_expert_offset, num_tokens, hidden_size);
    CUDA_CHECK(cudaGetLastError());

    // ---------- 逐个 expert 计算 MLP ----------
    for (int e = 0; e < num_experts; ++e) {
      int tokens_this_e = h_count[e];
      if (tokens_this_e == 0) {
        continue;
      }
      
      cutlass::half_t *mlp_output = d_sorted_out + h_offset[e] * hidden_size;
      const cutlass::half_t *mlp_input = d_sorted_x + h_offset[e] * hidden_size;
      const cutlass::half_t *mlp_gate_weight = gate_proj_weight + e * hidden_size * intermediate_size;
      const cutlass::half_t *mlp_up_weight = up_proj_weight + e * hidden_size * intermediate_size;
      const cutlass::half_t *mlp_down_weight = down_proj_weight + e * intermediate_size * hidden_size;

      forward_mlp(mlp_output,
                mlp_input,
                mlp_gate_weight,
                mlp_up_weight,
                mlp_down_weight,
                silu_output,
                up_output,
                hadamard_output,
                params.batch_size,
                tokens_this_e,
                params.hidden_size,
                params.intermediate_size,
                params.stream); 
      CUDA_CHECK(cudaGetLastError());
    }

    // forward_mlp_grouped(d_sorted_out,
    //               d_sorted_x,
    //               gate_proj_weight,
    //               up_proj_weight,
    //               down_proj_weight,
    //               silu_output,
    //               up_output,
    //               hadamard_output,
    //               h_count,
    //               params.hidden_size,
    //               params.intermediate_size,
    //               params.stream);
    // CUDA_CHECK(cudaGetLastError());  

    // ---------- scatter 回原位 ----------
    kernel_scatter<<<(num_tokens + reorder_block.x - 1) / reorder_block.x, reorder_block, 0, params.stream>>>(
        d_sorted_out, d_token2pos, output, num_tokens, hidden_size);
    CUDA_CHECK(cudaGetLastError());

    MOE_PRINT("[HOST INFO] cutlass forward_moe prefill completed successfully\n");
}

void context_moe_fp16_forward_cuda(const MoeInputsParams& params) {
  MOE_PRINT("[HOST INFO] Starting context_moe_fp16_forward_cuda\n");
  forward_moe(params);
}

void decode_moe_fp16_forward_cuda(const MoeInputsParams& params) {
  MOE_PRINT("[HOST INFO] Starting decode_moe_fp16_forward_cuda\n");
  forward_moe(params);  
}

} // cutlass_moe
} // namespace kernel
} // namespace trt_edgellm