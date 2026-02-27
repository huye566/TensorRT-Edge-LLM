#include "moeNvfp4Kernels.h"

#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <assert.h>

#include "kernels/common/nvfp4_quant.h"
#include "kernels/common/cublas_wrapper.h"
#include "kernels/common/cublaslt_wrapper_nvfp4.h"

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

#ifdef ENABLE_MOE_DEBUG
    #define MOE_PRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
    #define MOE_PRINT(fmt, ...) do {} while(0)
#endif


namespace trt_edgellm {
namespace kernel {
namespace moe_nvfp4 {

struct MlpNvfp4InputsParams {
    half* input;
    half* output;
    half* gate_output;
    half* up_output;
    half* hadamard_output;
    const uint8_t* gate_qweight;
    const uint8_t* gate_qscales;
    const float* gate_input_global_scale;
    const float* gate_weight_global_scale;
    float gate_alpha;
    const uint8_t* up_qweight;
    const uint8_t* up_qscales;
    const float* up_input_global_scale;
    const float* up_weight_global_scale;
    float up_alpha;
    const uint8_t* down_qweight;
    const uint8_t* down_qscales;
    const float* down_input_global_scale;
    const float* down_weight_global_scale;
    float down_alpha;

    uint8_t* gate_input_quant;
    uint8_t* gate_input_qscales;
    uint8_t* up_input_quant;
    uint8_t* up_input_qscales;
    uint8_t* down_input_quant;
    uint8_t* down_input_qscales;

    friend std::ostream& operator<<(std::ostream& os, const MlpNvfp4InputsParams& params) {
        auto print_ptr = [&os](const char* name, const void* ptr, 
                               size_t elem_size, const char* type_name,
                               std::function<void(const void* host_buf)> print_values) {
            os << name << ": ";
            if (!ptr) {
                os << "nullptr";
                return;
            }

            cudaPointerAttributes attrs;
            cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
            if (err != cudaSuccess) {
                os << "address=" << ptr << " (无法获取指针属性: " << cudaGetErrorString(err) << ")";
                return;
            }

            bool is_device = (attrs.type == cudaMemoryTypeDevice);
            os << "address=" << ptr;

            if (is_device) {
                os << " [device] ";
                char host_buf[3 * elem_size];
                cudaError_t copy_err = cudaMemcpy(host_buf, ptr, 3 * elem_size, cudaMemcpyDeviceToHost);
                if (copy_err == cudaSuccess) {
                    os << "values=";
                    print_values(host_buf);
                } else {
                    os << "拷贝失败: " << cudaGetErrorString(copy_err);
                }
            } else {
                os << " [host] values=";
                print_values(ptr);
            }
        };

        auto print_half_ptr = [&](const char* name, half* ptr) {
            print_ptr(name, ptr, sizeof(half), "half", [&](const void* buf) {
                const half* hbuf = static_cast<const half*>(buf);
                os << "[" << __half2float(hbuf[0]) << ", "
                   << __half2float(hbuf[1]) << ", "
                   << __half2float(hbuf[2]) << "]";
            });
        };

        auto print_const_uint8_ptr = [&](const char* name, const uint8_t* ptr) {
            print_ptr(name, ptr, sizeof(uint8_t), "uint8_t", [&](const void* buf) {
                const uint8_t* ubuf = static_cast<const uint8_t*>(buf);
                os << "[" << static_cast<int>(ubuf[0]) << ", "
                   << static_cast<int>(ubuf[1]) << ", "
                   << static_cast<int>(ubuf[2]) << "]";
            });
        };

        auto print_float_ptr = [&](const char* name, const float* ptr) {
            print_ptr(name, ptr, sizeof(float), "float", [&](const void* buf) {
                const float* fbuf = static_cast<const float*>(buf);
                os << "[" << fbuf[0] << ", " << fbuf[1] << ", " << fbuf[2] << "]";
            });
        };

        auto print_uint8_ptr = [&](const char* name, uint8_t* ptr) {
            print_ptr(name, ptr, sizeof(uint8_t), "uint8_t", [&](const void* buf) {
                const uint8_t* ubuf = static_cast<const uint8_t*>(buf);
                os << "[" << static_cast<int>(ubuf[0]) << ", "
                   << static_cast<int>(ubuf[1]) << ", "
                   << static_cast<int>(ubuf[2]) << "]";
            });
        };

        // 依次打印所有字段
        print_half_ptr("input", params.input); os << "\n";
        print_half_ptr("output", params.output); os << "\n";
        print_half_ptr("gate_output", params.gate_output); os << "\n";
        print_half_ptr("up_output", params.up_output); os << "\n";
        print_half_ptr("hadamard_output", params.hadamard_output); os << "\n";
        print_const_uint8_ptr("gate_qweight", params.gate_qweight); os << "\n";
        print_const_uint8_ptr("gate_qscales", params.gate_qscales); os << "\n";
        print_float_ptr("gate_input_global_scale", params.gate_input_global_scale); os << "\n";
        print_float_ptr("gate_weight_global_scale", params.gate_weight_global_scale); os << "\n";
        os << "gate_alpha: " << params.gate_alpha << "\n";
        print_const_uint8_ptr("up_qweight", params.up_qweight); os << "\n";
        print_const_uint8_ptr("up_qscales", params.up_qscales); os << "\n";
        print_float_ptr("up_input_global_scale", params.up_input_global_scale); os << "\n";
        print_float_ptr("up_weight_global_scale", params.up_weight_global_scale); os << "\n";
        os << "up_alpha: " << params.up_alpha << "\n";
        print_const_uint8_ptr("down_qweight", params.down_qweight); os << "\n";
        print_const_uint8_ptr("down_qscales", params.down_qscales); os << "\n";
        print_float_ptr("down_input_global_scale", params.down_input_global_scale); os << "\n";
        print_float_ptr("down_weight_global_scale", params.down_weight_global_scale); os << "\n";
        os << "down_alpha: " << params.down_alpha << "\n";
        print_uint8_ptr("gate_input_quant", params.gate_input_quant); os << "\n";
        print_uint8_ptr("gate_input_qscales", params.gate_input_qscales); os << "\n";
        print_uint8_ptr("up_input_quant", params.up_input_quant); os << "\n";
        print_uint8_ptr("up_input_qscales", params.up_input_qscales); os << "\n";
        print_uint8_ptr("down_input_quant", params.down_input_quant); os << "\n";
        print_uint8_ptr("down_input_qscales", params.down_input_qscales);

        return os;
    }
};

static size_t get_scale_pad_size(int M, int N) {
    int scale_n = N / 16;
    int rounded_m = ((M + 128 - 1) / 128) * 128;
    int rounded_n = ((scale_n + 4 - 1) / 4) * 4;
    return rounded_m * rounded_n;
}

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

void elementwise_multiply(half* C,
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
    const half* matrix,
    int* max_indices,
    int M, int N, int real_N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        const half* row_ptr = matrix + row * N;

        half max_val = row_ptr[0];
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

__global__ void kernel_reorder(const half* x,
                               const int* expert_idx,
                               half* sorted_x,       // 输出缓存
                               int32_t* token2pos,              // 原 token_id -> 在 sorted 里的偏移
                               int32_t* expert_offset,          // 每个 expert 在 sorted 里的起始偏移
                               int num_tokens,
                               int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pos;
    if (idx < num_tokens) {
        // 全部逻辑放在条件内
        int e = expert_idx[idx];
        if (threadIdx.y == 0) {
            pos = atomicAdd(expert_offset + e, 1);
            token2pos[idx] = pos;
        }
        __syncthreads();   // 同一 block 内所有活线程同步，无死锁
        pos = token2pos[idx];
        for (int h = threadIdx.y; h < hidden_size; h += blockDim.y) {
            sorted_x[pos * hidden_size + h] = x[idx * hidden_size + h];
        }
    }
}

// 把专家结果写回原位
__global__ void kernel_scatter(const half* sorted_out,
                               const int32_t* token2pos,
                               half* out,
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

void prefill_compute_router_logits_cublas(const MoeNvfp4InputsParams& params) {
    int num_tokens = params.total_tokens();
    int num_experts = params.experts_num;

    auto& cublas_wrapper = CublasWrapper::instance();
    cublasSetStream(cublas_wrapper.handle(), params.stream);
    // printf("num_tokens: %d, num_experts: %d\n", num_tokens, num_experts);
    // printf("hidden_size: %d\n", params.hidden_size);
    // printf("intermediate_size: %d\n", params.intermediate_size);
    // printf("params.router_weight: %p\n", params.router_weight);
    // printf("params.router_bias: %p\n", params.router_bias);

    cublas_gemm_bias<half>(cublas_wrapper.handle(),
                            num_tokens,
                            num_experts,
                            params.hidden_size,
                            reinterpret_cast<const half*>(params.hidden_state),
                            reinterpret_cast<const half*>(params.router_weight),
                            reinterpret_cast<const half*>(params.router_bias),
                            reinterpret_cast<half*>(params.router_logits),
                            MemoryFormat::ROW_MAJOR,
                            params.stream);

    dim3 block_router(256);
    dim3 grid_router((num_tokens + block_router.x - 1) / block_router.x);
    find_row_max_kernel<<<grid_router, block_router, 0, params.stream>>>(
        reinterpret_cast<const half*>(params.router_logits),
        params.expert_indices,
        num_tokens, num_experts, num_experts);
    CUDA_CHECK(cudaGetLastError());
}

void decode_compute_router_logits_cublas(const MoeNvfp4InputsParams& params) {
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
                          reinterpret_cast<half*>(params.router_logits),
                          MemoryFormat::ROW_MAJOR,
                          params.stream);

    dim3 block_router(256);
    dim3 grid_router((num_tokens + block_router.x - 1) / block_router.x);
    compute_max_logit_kernel<<<grid_router, block_router, 0, params.stream>>>(
        reinterpret_cast<const half*>(params.router_logits),
        params.expert_indices,
        num_tokens,
        experts_num);

    CUDA_CHECK(cudaGetLastError());
}

void forward_mlp(const MoeNvfp4InputsParams& moe_params, 
                 const MlpNvfp4InputsParams mlp_params,
                 int num_tokens) {

    auto& cublaslt_wrapper = trt_edgellm::kernel::CublasLtNVFP4Wrapper::instance();
    if (!cublaslt_wrapper.initialize()) {
        std::cerr << "Failed to initialize cublasLt wrapper" << std::endl;
        return;
    }

    int M = num_tokens;
    int K = moe_params.hidden_size;
    int N = moe_params.intermediate_size;
    trt_edgellm::kernel::scaled_fp4_quant(
        M, K,
        mlp_params.input,
        mlp_params.gate_input_global_scale,
        reinterpret_cast<int64_t*>(mlp_params.gate_input_quant),
        reinterpret_cast<int32_t*>(mlp_params.gate_input_qscales),
        moe_params.stream,
        nvinfer1::DataType::kHALF);
    CUDA_CHECK(cudaGetLastError());
    bool success = trt_edgellm::kernel::cublaslt_gemm_nvfp4<true, false>(
        cublaslt_wrapper.handle(),
        M, N, K,
        mlp_params.gate_input_quant,
        mlp_params.gate_qweight,
        nullptr,
        mlp_params.gate_output,
        mlp_params.gate_input_qscales,
        mlp_params.gate_qscales,
        mlp_params.gate_alpha,
        nvinfer1::DataType::kHALF,
        moe_params.stream);
    if (!success) {
        std::cerr << "cublaslt_gemm_nvfp4 failed for gate projection" << std::endl;
        return;
    }
    CUDA_CHECK(cudaGetLastError());

    trt_edgellm::kernel::scaled_fp4_quant(
        M, K,
        mlp_params.input,
        mlp_params.up_input_global_scale,
        reinterpret_cast<int64_t*>(mlp_params.up_input_quant),
        reinterpret_cast<int32_t*>(mlp_params.up_input_qscales),
        moe_params.stream,
        nvinfer1::DataType::kHALF);
    CUDA_CHECK(cudaGetLastError());
    success = trt_edgellm::kernel::cublaslt_gemm_nvfp4<false, false>(
        cublaslt_wrapper.handle(),
        M, N, K,
        mlp_params.up_input_quant,
        mlp_params.up_qweight,
        nullptr,
        mlp_params.up_output,
        mlp_params.up_input_qscales,
        mlp_params.up_qscales,
        mlp_params.up_alpha,
        nvinfer1::DataType::kHALF,
        moe_params.stream);
    if (!success) {
        std::cerr << "cublaslt_gemm_nvfp4 failed for up projection" << std::endl;
        return;
    }

    elementwise_multiply(mlp_params.hadamard_output, mlp_params.gate_output, 
                              mlp_params.up_output, M, N, moe_params.stream);
    CUDA_CHECK(cudaGetLastError());
    
    K = moe_params.intermediate_size;
    N = moe_params.hidden_size;
    trt_edgellm::kernel::scaled_fp4_quant(
        M, K,
        mlp_params.hadamard_output,
        mlp_params.down_input_global_scale,
        reinterpret_cast<int64_t*>(mlp_params.down_input_quant),
        reinterpret_cast<int32_t*>(mlp_params.down_input_qscales),
        moe_params.stream,
        nvinfer1::DataType::kHALF);
    CUDA_CHECK(cudaGetLastError());
    success = trt_edgellm::kernel::cublaslt_gemm_nvfp4<false, false>(
        cublaslt_wrapper.handle(),
        M, N, K,
        mlp_params.down_input_quant,
        mlp_params.down_qweight,
        nullptr,
        mlp_params.output,
        mlp_params.down_input_qscales,
        mlp_params.down_qscales,
        mlp_params.down_alpha,
        nvinfer1::DataType::kHALF,
        moe_params.stream);
    if (!success) {
        std::cerr << "cublaslt_gemm_nvfp4 failed for down projection" << std::endl;
        return;
    }
    CUDA_CHECK(cudaGetLastError());
}


void forward_moe_decode(const MoeNvfp4InputsParams& params) {
    MOE_PRINT("[HOST INFO] Single token path with cuBLAS\n");

    int hidden_size = params.hidden_size;
    int intermediate_size = params.intermediate_size;

    decode_compute_router_logits_cublas(params);

    int best_expert = 0;
    cudaMemcpyAsync(&best_expert, params.expert_indices, sizeof(int),
                   cudaMemcpyDeviceToHost, params.stream);
    cudaStreamSynchronize(params.stream);

    MOE_PRINT("[HOST INFO] Single token assigned to expert %d\n", best_expert);
    MlpNvfp4InputsParams mlp_params = {
        .input = reinterpret_cast<half*>(params.hidden_state),
        .output = reinterpret_cast<half*>(params.output),
        .gate_output = reinterpret_cast<half*>(params.silu_output),
        .up_output = reinterpret_cast<half*>(params.up_output),
        .hadamard_output = reinterpret_cast<half*>(params.hadamard_output),
        .gate_qweight = reinterpret_cast<uint8_t*>(params.gate_qweight) +
                                 best_expert * intermediate_size * hidden_size / 2,
        .gate_qscales = reinterpret_cast<uint8_t*>(params.gate_qscales) +
                                 best_expert * get_scale_pad_size(intermediate_size, hidden_size),
        .gate_input_global_scale = reinterpret_cast<float*>(params.gate_input_global_scale) +
                                 best_expert,
        .gate_weight_global_scale = reinterpret_cast<float*>(params.gate_weight_global_scale) +
                                 best_expert,
        .gate_alpha = params.gate_alpha[best_expert],
        .up_qweight = reinterpret_cast<uint8_t*>(params.up_qweight) +
                                 best_expert * intermediate_size * hidden_size / 2,
        .up_qscales = reinterpret_cast<uint8_t*>(params.up_qscales) +
                                 best_expert * get_scale_pad_size(intermediate_size, hidden_size),
        .up_input_global_scale = reinterpret_cast<float*>(params.up_input_global_scale) +
                                 best_expert,
        .up_weight_global_scale = reinterpret_cast<float*>(params.up_weight_global_scale) +
                                 best_expert,
        .up_alpha = params.up_alpha[best_expert],
        .down_qweight = reinterpret_cast<uint8_t*>(params.down_qweight) +
                                 best_expert * hidden_size * intermediate_size / 2,
        .down_qscales = reinterpret_cast<uint8_t*>(params.down_qscales) +
                                 best_expert * get_scale_pad_size(hidden_size, intermediate_size),
        .down_input_global_scale = reinterpret_cast<float*>(params.down_input_global_scale) +
                                 best_expert,
        .down_weight_global_scale = reinterpret_cast<float*>(params.down_weight_global_scale) +
                                 best_expert,
        .down_alpha = params.down_alpha[best_expert],
        .gate_input_quant = params.gate_input_quant,
        .gate_input_qscales = params.gate_input_qscales,
        .up_input_quant = params.up_input_quant,
        .up_input_qscales = params.up_input_qscales,
        .down_input_quant = params.down_input_quant,
        .down_input_qscales = params.down_input_qscales
    };

    forward_mlp(params, mlp_params, 1);
}

void forward_moe_context(const MoeNvfp4InputsParams& params) {
    int num_tokens = params.total_tokens();
    int num_experts = params.experts_num;
    int hidden_size = params.hidden_size;
    int intermediate_size = params.intermediate_size;

    half *d_sorted_x = reinterpret_cast<half*>(params.expert_input_buffer);
    half *d_sorted_out = reinterpret_cast<half*>(params.expert_output_buffer);

    int32_t *d_expert_count = params.expert_counts;
    int32_t *d_expert_offset = params.expert_offsets;
    int32_t *d_token2pos = params.token_to_buffer_map;

    prefill_compute_router_logits_cublas(params);

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
        params.hidden_state, params.expert_indices, d_sorted_x, d_token2pos, d_expert_offset, num_tokens, hidden_size);
    CUDA_CHECK(cudaGetLastError());

    for (int e = 0; e < num_experts; ++e) {
        int tokens_this_e = h_count[e];
        if (tokens_this_e == 0) {
            continue;
        }

        MlpNvfp4InputsParams mlp_params = {
            .input = d_sorted_x + h_offset[e] * hidden_size,
            .output = d_sorted_out + h_offset[e] * hidden_size,
            .gate_output = reinterpret_cast<half*>(params.silu_output),
            .up_output = reinterpret_cast<half*>(params.up_output),
            .hadamard_output = reinterpret_cast<half*>(params.hadamard_output),
            .gate_qweight = reinterpret_cast<uint8_t*>(params.gate_qweight) +
                                    e * intermediate_size * hidden_size / 2,
            .gate_qscales = reinterpret_cast<uint8_t*>(params.gate_qscales) +
                                    e * get_scale_pad_size(intermediate_size, hidden_size),
            .gate_input_global_scale = reinterpret_cast<float*>(params.gate_input_global_scale) +
                                    e,
            .gate_weight_global_scale = reinterpret_cast<float*>(params.gate_weight_global_scale) +
                                    e,
            .gate_alpha = params.gate_alpha[e],
            .up_qweight = reinterpret_cast<uint8_t*>(params.up_qweight) +
                                    e * intermediate_size * hidden_size / 2,
            .up_qscales = reinterpret_cast<uint8_t*>(params.up_qscales) +
                                    e * get_scale_pad_size(intermediate_size, hidden_size),
            .up_input_global_scale = reinterpret_cast<float*>(params.up_input_global_scale) +
                                    e,
            .up_weight_global_scale = reinterpret_cast<float*>(params.up_weight_global_scale) +
                                    e,
            .up_alpha = params.up_alpha[e],
            .down_qweight = reinterpret_cast<uint8_t*>(params.down_qweight) +
                                    e * hidden_size * intermediate_size / 2,
            .down_qscales = reinterpret_cast<uint8_t*>(params.down_qscales) +
                                    e * get_scale_pad_size(hidden_size, intermediate_size),
            .down_input_global_scale = reinterpret_cast<float*>(params.down_input_global_scale) +
                                    e,
            .down_weight_global_scale = reinterpret_cast<float*>(params.down_weight_global_scale) +
                                    e,
            .down_alpha = params.down_alpha[e],
            .gate_input_quant = params.gate_input_quant,
            .gate_input_qscales = params.gate_input_qscales,
            .up_input_quant = params.up_input_quant,
            .up_input_qscales = params.up_input_qscales,
            .down_input_quant = params.down_input_quant,
            .down_input_qscales = params.down_input_qscales
        };
#if ENABLE_MOE_DEBUG
        std::cout << "Expert " << e << ": " << tokens_this_e << " tokens" << std::endl;
        std::cout << mlp_params << std::endl;
#endif

        forward_mlp(params, mlp_params, tokens_this_e);
        CUDA_CHECK(cudaGetLastError());
    }

    kernel_scatter<<<(num_tokens + reorder_block.x - 1) / reorder_block.x, reorder_block, 0, params.stream>>>(
        d_sorted_out, d_token2pos, params.output, num_tokens, hidden_size);
    CUDA_CHECK(cudaGetLastError());

    MOE_PRINT("[HOST INFO] cutlass forward_moe prefill completed successfully\n");
}

void forward_moe(const MoeNvfp4InputsParams& params) {
    MOE_PRINT("[HOST INFO] Starting forward_moe\n");

    if (params.hidden_state == nullptr ||
        params.output == nullptr ||
        params.router_weight == nullptr ||
        params.router_bias == nullptr ||
        params.gate_qweight == nullptr ||
        params.gate_qscales == nullptr ||
        params.gate_input_global_scale == nullptr ||
        params.gate_weight_global_scale == nullptr ||
        params.up_qweight == nullptr ||
        params.up_qscales == nullptr ||
        params.up_input_global_scale == nullptr ||
        params.up_weight_global_scale == nullptr ||
        params.down_qweight == nullptr ||
        params.down_qscales == nullptr ||
        params.down_input_global_scale == nullptr ||
        params.down_weight_global_scale == nullptr) {
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
        params.router_logits == nullptr ||
        params.gate_input_quant == nullptr ||
        params.gate_input_qscales == nullptr ||
        params.up_input_quant == nullptr ||
        params.up_input_qscales == nullptr ||
        params.down_input_quant == nullptr ||
        params.down_input_qscales == nullptr) {
        MOE_PRINT("[HOST ERROR] Buffer parameters are null\n");
        return;
    }

    int num_tokens = params.total_tokens();
    MOE_PRINT("[HOST INFO] Total tokens: %d, hidden_size: %d, intermediate_size: %d, experts_num: %d\n",
              num_tokens, params.hidden_size, params.intermediate_size, params.experts_num);

    if (num_tokens == 1) {
        forward_moe_decode(params);
    } else {
        forward_moe_context(params);
    }
}

void moe_nvfp4_forward_cuda(const MoeNvfp4InputsParams& params) {
    MOE_PRINT("[HOST INFO] Starting moe_nvfp4_forward_cuda\n");
    forward_moe(params);
}

} // moe_nvfp4
} // namespace kernel
} // namespace trt_edgellm
