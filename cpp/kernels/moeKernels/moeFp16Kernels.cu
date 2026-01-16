#include "moeFp16Kernels.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

#ifdef ENABLE_MOE_DEBUG
    #define MOE_PRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
    #define MOE_PRINT(fmt, ...) do {} while(0)
#endif

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

namespace trt_edgellm
{
namespace kernel
{
namespace cuda
{

__device__ half silu(half x) {
    float x_f = __half2float(x);
    float sigmoid = 1.0f / (1.0f + expf(-x_f));
    return __float2half(x_f * sigmoid);
}

__global__ void compute_token_expert_mlp_parallel_kernel(
    const half* hidden_state,
    const half* router_weight,  // [hidden_size, experts_num]
    const half* router_bias,    // [experts_num]
    const half* gate_proj_weight,  // [experts_num, hidden_size, intermediate_size]
    const half* up_proj_weight,    // [experts_num, hidden_size, intermediate_size]
    const half* down_proj_weight,  // [experts_num, intermediate_size, hidden_size]
    half* output,
    int32_t* expert_indices,
    int total_tokens,
    int hidden_size,
    int intermediate_size,
    int experts_num) {
    
    int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;
    
    const half* token_vec = hidden_state + token_idx * hidden_size;
    
    extern __shared__ char shared_mem[];
    half* intermediate_results = (half*)shared_mem;
    int* p_best_expert = (int*)(shared_mem + intermediate_size * sizeof(half));
    
    // 使用线程块内的第一个线程计算router logits
    if (threadIdx.x == 0) {
        half max_logit = __float2half(-1e30f);
        int best_expert = 0;
        
        // 计算每个expert的logit
        // router_weight: [hidden_size, experts_num]
        // 计算: token_vec (1×hidden_size) × router_weight (hidden_size×experts_num) -> (1×experts_num)
        for (int expert = 0; expert < experts_num; ++expert) {
            float logit = 0.0f;
            
            // 对于每个expert，我们需要访问router_weight的第expert列
            // 由于C/C++是行优先存储，所以第expert列的元素分布在：
            // router_weight[0*experts_num + expert], router_weight[1*experts_num + expert], ...
            for (int i = 0; i < hidden_size; ++i) {
                logit += __half2float(token_vec[i]) * 
                         __half2float(router_weight[i * experts_num + expert]);
            }
            
            logit += __half2float(router_bias[expert]);
            
            if (expert == 0 || logit > __half2float(max_logit)) {
                max_logit = __float2half(logit);
                best_expert = expert;
            }
        }
        
        if (expert_indices != nullptr) {
            expert_indices[token_idx] = best_expert;
        }
        
        // 将选择的专家索引存储到共享内存
        *p_best_expert = best_expert;
    }
    
    __syncthreads();
    
    int best_expert = *p_best_expert;
    
    // 获取当前token对应专家的权重
    // 每个专家权重的大小: hidden_size * intermediate_size
    size_t expert_offset = best_expert * hidden_size * intermediate_size;
    const half* expert_gate_weight = gate_proj_weight + expert_offset;
    const half* expert_up_weight = up_proj_weight + expert_offset;
    
    // down_proj权重的偏移量不同，因为它的形状是 [experts_num, intermediate_size, hidden_size]
    size_t down_expert_offset = best_expert * intermediate_size * hidden_size;
    const half* expert_down_weight = down_proj_weight + down_expert_offset;
    
    // 并行计算gate_proj和up_proj
    for (int elem_idx = threadIdx.x; elem_idx < intermediate_size; elem_idx += blockDim.x) {
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        
        // gate_proj_weight和up_proj_weight形状: [hidden_size, intermediate_size]
        // 对于每个中间层元素elem_idx，我们需要访问权重矩阵的第elem_idx列
        for (int i = 0; i < hidden_size; ++i) {
            float token_val = __half2float(token_vec[i]);
            // gate_proj_weight[i, elem_idx]
            gate_sum += token_val * __half2float(expert_gate_weight[i * intermediate_size + elem_idx]);
            // up_proj_weight[i, elem_idx]
            up_sum += token_val * __half2float(expert_up_weight[i * intermediate_size + elem_idx]);
        }
        
        half gate_val = __float2half(gate_sum);
        half up_val = __float2half(up_sum);
        
        // 计算silu(gate) * up
        float gate_f = __half2float(gate_val);
        float sigmoid = 1.0f / (1.0f + expf(-gate_f));
        half hadamard_val = __float2half(gate_f * sigmoid * __half2float(up_val));
        
        intermediate_results[elem_idx] = hadamard_val;
    }
    
    __syncthreads();
    
    // 并行计算down_proj
    half* final_output = output + token_idx * hidden_size;
    
    for (int hidden_idx = threadIdx.x; hidden_idx < hidden_size; hidden_idx += blockDim.x) {
        float sum = 0.0f;
        
        // down_proj权重形状为 [intermediate_size, hidden_size]
        // 对于每个隐藏层元素hidden_idx，我们需要访问权重矩阵的第hidden_idx列
        for (int i = 0; i < intermediate_size; ++i) {
            sum += __half2float(intermediate_results[i]) * 
                   __half2float(expert_down_weight[i * hidden_size + hidden_idx]);
        }
        
        final_output[hidden_idx] = __float2half(sum);
    }
}

__global__ void compute_router_logits_kernel(
    const MoeInputsParams params) {
    
    int total_tokens = params.total_tokens();
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (token_idx < total_tokens) {
        // 检查指针是否有效
        if (params.hidden_state == nullptr || params.router_weight == nullptr || 
            params.router_bias == nullptr || params.expert_indices == nullptr) {
            return;
        }
        
        const half* token_vec = params.hidden_state + token_idx * params.hidden_size;
        half max_logit = __float2half(-1e30f);
        int best_expert = 0;
        
        // 计算每个expert的logit
        // router_weight: [hidden_size, experts_num]
        for (int expert = 0; expert < params.experts_num; ++expert) {
            float logit = 0.0f;
            
            // 计算token_vec与router_weight的第expert列的点积
            for (int i = 0; i < params.hidden_size; ++i) {
                logit += __half2float(token_vec[i]) * 
                         __half2float(params.router_weight[i * params.experts_num + expert]);
            }
            
            logit += __half2float(params.router_bias[expert]);
            
            if (expert == 0 || logit > __half2float(max_logit)) {
                max_logit = __float2half(logit);
                best_expert = expert;
            }
        }
        
        params.expert_indices[token_idx] = best_expert;
    }
}

__global__ void compute_expert_mlp_kernel(
    const half* input,  // [num_tokens, hidden_size]
    const half* gate_proj_weight,  // [hidden_size, intermediate_size]
    const half* up_proj_weight,    // [hidden_size, intermediate_size]
    half* gate_output,
    half* up_output,
    half* silu_output,
    half* hadamard_output,
    int num_tokens,
    int hidden_size,
    int intermediate_size) {
    
    // 使用二维网格：x维度处理token，y维度处理中间层元素
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elem_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (token_idx < num_tokens && elem_idx < intermediate_size) {
        // 计算gate_proj
        float gate_sum = 0.0f;
        const half* token_vec = input + token_idx * hidden_size;
        
        // 权重形状为 [hidden_size, intermediate_size]
        // 对于中间层元素elem_idx，我们需要访问权重矩阵的第elem_idx列
        for (int i = 0; i < hidden_size; ++i) {
            // gate_proj_weight[i, elem_idx]
            gate_sum += __half2float(token_vec[i]) * 
                       __half2float(gate_proj_weight[i * intermediate_size + elem_idx]);
        }
        half gate_val = __float2half(gate_sum);
        gate_output[token_idx * intermediate_size + elem_idx] = gate_val;
        
        // 应用SiLU激活
        half silu_val = silu(gate_val);
        silu_output[token_idx * intermediate_size + elem_idx] = silu_val;
        
        // 计算up_proj
        float up_sum = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            // up_proj_weight[i, elem_idx]
            up_sum += __half2float(token_vec[i]) * 
                     __half2float(up_proj_weight[i * intermediate_size + elem_idx]);
        }
        half up_val = __float2half(up_sum);
        up_output[token_idx * intermediate_size + elem_idx] = up_val;
        
        // 计算Hadamard乘积 (silu(gate) * up)
        half hadamard_val = __float2half(__half2float(silu_val) * __half2float(up_val));
        hadamard_output[token_idx * intermediate_size + elem_idx] = hadamard_val;
    }
}

__global__ void compute_down_proj_kernel(
    const half* hadamard_output,  // [num_tokens, intermediate_size]
    const half* down_proj_weight, // [intermediate_size, hidden_size]
    half* output,                 // [num_tokens, hidden_size]
    int num_tokens,
    int intermediate_size,
    int hidden_size) {
    
    // 使用二维网格：x维度处理token，y维度处理隐藏层元素
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hidden_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (token_idx < num_tokens && hidden_idx < hidden_size) {
        float sum = 0.0f;
        
        // 权重形状为 [intermediate_size, hidden_size]
        // 对于隐藏层元素hidden_idx，我们需要访问权重矩阵的第hidden_idx列
        for (int i = 0; i < intermediate_size; ++i) {
            half val = hadamard_output[token_idx * intermediate_size + i];
            // down_proj_weight[i, hidden_idx]
            sum += __half2float(val) * 
                   __half2float(down_proj_weight[i * hidden_size + hidden_idx]);
        }
        output[token_idx * hidden_size + hidden_idx] = __float2half(sum);
    }
}

void context_moe_fp16_forward_cuda(const MoeInputsParams& params) {
    
    MOE_PRINT("[HOST INFO] Starting context_moe_fp16_forward_cuda_fused\n");
    
    // 检查必要参数
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
    
    int total_tokens = params.total_tokens();
    MOE_PRINT("[HOST INFO] Total tokens: %d, hidden_size: %d, intermediate_size: %d, experts_num: %d\n",
              total_tokens, params.hidden_size, params.intermediate_size, params.experts_num);
    
    // 方法1: 使用优化的并行kernel，每个线程块处理一个token
    if (total_tokens > 1) {
        // 每个线程块处理一个token
        dim3 block(256);  // 每个token使用256个线程
        dim3 grid(total_tokens);
        
        // 计算共享内存大小：中间结果 + 专家索引
        size_t shared_mem_size = params.intermediate_size * sizeof(half) + sizeof(int);
        
        // 检查共享内存限制（通常48KB）
        if (shared_mem_size > 48 * 1024) {
            MOE_PRINT("[HOST WARNING] Shared memory requirement (%zu bytes) exceeds typical limit. Adjusting...\n", 
                     shared_mem_size);
            
            // 如果中间结果太大，可以分批处理
            // 这里我们减少线程数并调整策略
            block.x = 512;
            
            // 或者使用另一种策略：每个线程处理多个中间元素
            // 这需要修改kernel实现
        }
        
        MOE_PRINT("[HOST INFO] Using parallel fused kernel with %d threads per block, %d blocks, %zu bytes shared memory\n", 
                  block.x, grid.x, shared_mem_size);
        
        compute_token_expert_mlp_parallel_kernel<<<grid, block, shared_mem_size, params.stream>>>(
            params.hidden_state,
            params.router_weight,
            params.router_bias,
            params.gate_proj_weight,
            params.up_proj_weight,
            params.down_proj_weight,
            params.output,
            params.expert_indices,
            total_tokens,
            params.hidden_size,
            params.intermediate_size,
            params.experts_num);
    } else {
        // 简化的实现：每个线程处理一个token
        dim3 block(128);
        dim3 grid((total_tokens + block.x - 1) / block.x);
        
        MOE_PRINT("[HOST INFO] Using simplified kernel with %d threads per block, %d blocks\n", 
                  block.x, grid.x);
        
        // 计算每个token的MLP
        const int block_x = 32;
        const int block_y = 32;
        dim3 block_mlp(block_x, block_y, 1);
        dim3 grid_mlp(
            (total_tokens + block_x - 1) / block_x,
            (params.intermediate_size + block_y - 1) / block_y,
            1
        );
        
        // 首先计算router logits
        dim3 block_router(256);
        dim3 grid_router((total_tokens + block_router.x - 1) / block_router.x);
        compute_router_logits_kernel<<<grid_router, block_router, 0, params.stream>>>(params);
        
        // 然后计算MLP
        compute_expert_mlp_kernel<<<grid_mlp, block_mlp, 0, params.stream>>>(
            params.hidden_state,
            params.gate_proj_weight,
            params.up_proj_weight,
            params.gate_output,
            params.up_output,
            params.silu_output,
            params.hadamard_output,
            total_tokens,
            params.hidden_size,
            params.intermediate_size);
        
        // 计算down_proj
        dim3 grid_down(
            (total_tokens + block_x - 1) / block_x,
            (params.hidden_size + block_y - 1) / block_y,
            1
        );
        
        compute_down_proj_kernel<<<grid_down, block_mlp, 0, params.stream>>>(
            params.hadamard_output,
            params.down_proj_weight,
            params.output,
            total_tokens,
            params.intermediate_size,
            params.hidden_size);
    }
    
    CUDA_CHECK(cudaGetLastError());
    MOE_PRINT("[HOST INFO] context_moe_fp16_forward_cuda_fused completed successfully\n");
}

void decode_moe_fp16_forward_cuda(const MoeInputsParams& params) {
    MOE_PRINT("[HOST INFO] Starting decode_moe_fp16_forward_cuda\n");
    context_moe_fp16_forward_cuda(params);
}

} // cuda
} // namespace kernel
} // namespace trt_edgellm
