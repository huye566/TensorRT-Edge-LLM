#ifndef MOE_NVFP4_INPUTS_PARAMS_H__
#define MOE_NVFP4_INPUTS_PARAMS_H__

#include <cuda_fp16.h>
#include <stdint.h>

namespace trt_edgellm {
namespace kernel {
namespace moe_nvfp4 {

struct MoeNvfp4InputsParams {
    half* hidden_state;                    // [batch_size, seq_len, hidden_size]
    half* output;                          // [batch_size, seq_len, hidden_size]

    half* router_weight;                   // [hidden_size, experts_num]
    half* router_bias;                     // [experts_num]
    half* router_weights_padded;           // [hidden_size, (experts_num + 7) / 8 * 8]
    half* router_bias_padded;              // [(experts_num + 7) / 8 * 8]

    uint8_t* gate_qweight;                 // [experts_num, intermediate_size, hidden_size // 2]
    uint8_t* gate_qscales;                 // [experts_num, padd, padd]
    float* gate_input_global_scale;        // [experts_num]
    float* gate_weight_global_scale;       // [experts_num]
    float* gate_alpha;                     // [experts_num]

    uint8_t* up_qweight;                   // [experts_num, intermediate_size, hidden_size // 2]
    uint8_t* up_qscales;                   // [experts_num, padd, padd]
    float* up_input_global_scale;          // [experts_num]
    float* up_weight_global_scale;         // [experts_num]
    float* up_alpha;                       // [experts_num]

    uint8_t* down_qweight;                 // [experts_num, hidden_size, intermediate_size // 2]
    uint8_t* down_qscales;                 // [experts_num, padd, padd]
    float* down_input_global_scale;        // [experts_num]
    float* down_weight_global_scale;       // [experts_num]
    float* down_alpha;                     // [experts_num]

    // Workspace buffers
    int32_t* expert_indices;               // [batch_size * seq_len]
    int32_t* expert_counts;                // [experts_num]
    int32_t* expert_offsets;               // [experts_num + 1]
    int32_t* current_offsets;              // [experts_num]
    int32_t* token_to_buffer_map;          // [total_tokens]
    half* router_logits;                   // [batch_size * seq_len, (experts_num + 7) / 8 * 8]

    half* expert_input_buffer;             // [total_tokens_selected, hidden_size]
    half* expert_output_buffer;            // [total_tokens_selected, hidden_size]
    half* silu_output;                     // [total_tokens_selected, intermediate_size]
    half* up_output;                       // [total_tokens_selected, intermediate_size]
    half* hadamard_output;                 // [total_tokens_selected, intermediate_size]

    uint8_t* gate_input_quant;             // [total_tokens_selected, hidden_size // 2]
    uint8_t* gate_input_qscales;           // [total_tokens_selected, padd]
    uint8_t* up_input_quant;               // [total_tokens_selected, hidden_size // 2]
    uint8_t* up_input_qscales;             // [total_tokens_selected, padd]
    uint8_t* down_input_quant;             // [total_tokens_selected, intermediate_size // 2]
    uint8_t* down_input_qscales;           // [total_tokens_selected, padd]

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

} // moe_nvfp4
} // namespace kernel
} // namespace trt_edgellm

#endif // MOE_NVFP4_INPUTS_PARAMS_H__