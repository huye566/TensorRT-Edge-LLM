#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

namespace trt_edgellm
{
namespace kernel
{
namespace cuda
{

struct MoeInputsParams
{
    // 输入输出指针
    half* hidden_state;          // [batch_size, seq_len, hidden_size]
    half* router_weight;         // [hidden_size, experts_num]
    half* router_bias;           // [experts_num]
    half* gate_proj_weight;      // [experts_num, hidden_size, intermediate_size]
    half* up_proj_weight;        // [experts_num, hidden_size, intermediate_size]
    half* down_proj_weight;      // [experts_num, intermediate_size, hidden_size]
    half* output;                // [batch_size, seq_len, hidden_size]
    half* router_weights_padded; // [hidden_size, (experts_num + 7) / 8 * 8]
    half* router_bias_padded;    // [(experts_num + 7) / 8 * 8]

    // Workspace分配的缓冲区
    int32_t* expert_indices;     // [batch_size * seq_len]
    half* router_logits;         // [batch_size * seq_len, (experts_num + 7) / 8 * 8]
    int32_t* expert_counts;      // [experts_num]
    int32_t* expert_offsets;     // [experts_num + 1]
    int32_t* current_offsets;    // [experts_num]
    int32_t* token_to_buffer_map; // [total_tokens]
    
    half* expert_input_buffer;   // [total_tokens_selected, hidden_size]
    half* gate_output;           // [total_tokens_selected, intermediate_size]
    half* up_output;             // [total_tokens_selected, intermediate_size]
    half* silu_output;           // [total_tokens_selected, intermediate_size]
    half* hadamard_output;       // [total_tokens_selected, intermediate_size]
    half* expert_output_buffer;  // [total_tokens_selected, hidden_size]
    
    // 形状参数
    int batch_size;
    int seq_len;
    int hidden_size;
    int intermediate_size;
    int experts_num;
    int experts_topk;
    
    cudaStream_t stream;
    
    // Helper function
    __host__ __device__ int total_tokens() const { return batch_size * seq_len; }
};

// 修改函数声明，使用MoeInputsParams
void context_moe_fp16_forward_cuda(const MoeInputsParams& params);

void decode_moe_fp16_forward_cuda(const MoeInputsParams& params);

} // cuda
} // namespace kernel
} // namespace trt_edgellm