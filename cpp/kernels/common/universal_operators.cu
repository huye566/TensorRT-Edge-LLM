#include "universal_operators.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>


#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "CUDA error: " << cudaGetErrorString(error)          \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

namespace trt_edgellm {
namespace kernel {

template<typename Func>
bool launch_with_type(nvinfer1::DataType output_type, Func&& func,
                     const char* operation_name = "operation") {
    switch (output_type) {
        case nvinfer1::DataType::kHALF:
            func(half{});
            break;
        case nvinfer1::DataType::kBF16:
            func(__nv_bfloat16{});
            break;
        case nvinfer1::DataType::kFLOAT:
            func(float{});
            break;
        default:
            std::cerr << "Unsupported output type for " << operation_name
                      << ": " << static_cast<int>(output_type) << std::endl;
            return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

template<typename T>
__global__ static void silu_kernel_impl(T* data, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        T x = data[idx];
        // SiLU(x) = x * sigmoid(x)
        if constexpr (std::is_same<T, float>::value) {
            float sigmoid_x = 1.0f / (1.0f + expf(-x));
            data[idx] = x * sigmoid_x;
        } else if constexpr (std::is_same<T, half>::value) {
            float x_f = __half2float(x);
            float sigmoid_x = 1.0f / (1.0f + expf(-x_f));
            data[idx] = __float2half(x_f * sigmoid_x);
        } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            float x_f = __bfloat162float(x);
            float sigmoid_x = 1.0f / (1.0f + expf(-x_f));
            data[idx] = __float2bfloat16(x_f * sigmoid_x);
        }
    }
}

bool apply_silu_scalar(void* data, int num_elements,
    nvinfer1::DataType output_type, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;

    auto launch_kernel = [&](auto type_tag) {
        using T = decltype(type_tag);
        silu_kernel_impl<T><<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<T*>(data), num_elements);
    };
    return launch_with_type(output_type, launch_kernel, "SiLU activation");
}

template<typename T>
__global__ static void add_bias_kernel_impl(T* data, const T* bias, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        int idx = row * n + col;
        data[idx] += bias[col];
    }
}

bool apply_add_bias_scalar(void* data, const void* bias, int m, int n,
    nvinfer1::DataType output_type, cudaStream_t stream) {
    const int block_size = 16;
    dim3 block_dim(block_size, block_size);
    dim3 grid_size(
        (m + block_size - 1) / block_size,
        (n + block_size - 1) / block_size
    );

    auto launch_kernel = [&](auto type_tag) {
        using T = decltype(type_tag);
        add_bias_kernel_impl<T><<<grid_size, block_dim, 0, stream>>>(
            reinterpret_cast<T*>(data),
            reinterpret_cast<const T*>(bias),
            m, n);
    };
    return launch_with_type(output_type, launch_kernel, "bias addition");
}

template<typename T>
struct VecType {
    using type = T;
    static constexpr int kElem = 1;
};
template<>
struct VecType<float> {
    using type = float4;
    static constexpr int kElem = 4;
};
template<>
struct VecType<half> {
    using type = half2;
    static constexpr int kElem = 2;
};
template<>
struct VecType<__nv_bfloat16> {
    using type = __nv_bfloat162;
    static constexpr int kElem = 2;
};

template<typename T>
using VecType_t = typename VecType<T>::type;

/* ----------  标量 silu  ---------- */
__device__ __forceinline__ float silu(float x) {
    float s = 1.0f / (1.0f + expf(-x));   // σ(x)
    return x * s;
}
__device__ __forceinline__ half silu(half x) {
    half s = hrcp(half(1) + hexp(-x));
    return x * s;
}
__device__ __forceinline__ __nv_bfloat16 silu(__nv_bfloat16 x) {
    float xf = __bfloat162float(x);
    float sf = 1.0f / (1.0f + expf(-xf));
    return __float2bfloat16(xf * sf);
}

/* ----------  向量 silu  ---------- */
__device__ __forceinline__ float4 silu_vec(float4 v) {
    return make_float4(silu(v.x), silu(v.y), silu(v.z), silu(v.w));
}
__device__ __forceinline__ half2 silu_vec(half2 v) {
    half2 neg = __hneg2(v);                 // -x
    half2 e   = h2exp(neg);                 // exp(-x)
    half2 s   = __h2div(__float2half2_rn(1.0f), __hadd2(__float2half2_rn(1.0f), e));
    return __hmul2(v, s);
}
__device__ __forceinline__ __nv_bfloat162 silu_vec(__nv_bfloat162 v) {
    __nv_bfloat162 neg = __hneg2(v);
    __nv_bfloat162 e   = h2exp(neg);
    __nv_bfloat162 one = __halves2bfloat162(__float2bfloat16(1.f),
                                        __float2bfloat16(1.f));
    __nv_bfloat162 s   = __h2div(one, __hadd2(one, e));
    return __hmul2(v, s);
}

/* ----------  SiLU kernel  ---------- */
template<typename T>
__global__ static void __launch_bounds__(256, 4)
silu_kernel_vec(T* __restrict__ data, int num) {
    using VecT = VecType_t<T>;
    const int vec_num = num / VecType<T>::kElem;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= vec_num) return;

    VecT* vec_ptr = reinterpret_cast<VecT*>(data);
    VecT reg = vec_ptr[tid];
    reg = silu_vec(reg);
    vec_ptr[tid] = reg;
}

/* ----------  AddBias kernel  ---------- */
template<typename T>
__global__ static void __launch_bounds__(256, 4)
add_bias_kernel_vec(T* __restrict__ data,
                    const T* __restrict__ bias,
                    int64_t total,
                    int64_t n) {
    constexpr int vec_elems = VecType<T>::kElem;
    const int64_t vec_total = total / vec_elems;

    int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= vec_total) return;

    int64_t col = (gid * vec_elems) % n;
    using VecT = VecType_t<T>;
    VecT* vec_data = reinterpret_cast<VecT*>(data);
    const VecT* vec_bias = reinterpret_cast<const VecT*>(bias + col);

    VecT b = __ldg(vec_bias);
    VecT d = vec_data[gid];

    if constexpr (std::is_same_v<T, float>) {          // float4 路径
        d.x += b.x;
        d.y += b.y;
        d.z += b.z;
        d.w += b.w;
    } else if constexpr (std::is_same_v<T, half>) {    // half2 路径
        d = __hadd2(d, b);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) { // bfloat162 路径
        d = __hadd2(d, b);
    }
    vec_data[gid] = d;
}


template<typename KernelPtr, typename... Args>
int get_opt_block_size(KernelPtr kernel, Args... args) {
    int min_grid = 0, best_block = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &best_block, kernel, 0, 0);
    return best_block;
}

bool apply_silu_vec(void* data, int num, nvinfer1::DataType dtype, cudaStream_t stream) {
    auto launch_kernel = [&](auto type_tag) {
        using T = decltype(type_tag);
        auto kernel = silu_kernel_vec<T>;
        auto* typed_ptr = reinterpret_cast<T*>(data);
        int block = get_opt_block_size(kernel);
        int vec_num = num / VecType<std::decay_t<decltype(*typed_ptr)>>::kElem;
        int grid = (vec_num + block - 1) / block;
        kernel<<<grid, block, 0, stream>>>(typed_ptr, num);
        // std::cout << "grid: " << grid << " block: " << block << std::endl;
    };
    return launch_with_type(dtype, launch_kernel, "vec SiLU activation");
}

bool apply_add_bias_vec(void* data, const void* bias, int m, int n,
                    nvinfer1::DataType dtype, cudaStream_t stream) {
    int64_t total = int64_t(m) * n;
    auto launch_kernel = [&](auto type_tag) {
        using T = decltype(type_tag);
        auto kernel = add_bias_kernel_vec<T>;
        auto* dptr = reinterpret_cast<T*>(data);
        auto* bptr = reinterpret_cast<const T*>(bias);
        int block = get_opt_block_size(kernel);
        int vec_total = total / VecType<std::decay_t<decltype(*dptr)>>::kElem;
        int grid = (vec_total + block - 1) / block;
        kernel<<<grid, block, 0, stream>>>(dptr, bptr, total, n);
        // std::cout << "grid: " << grid << " block: " << block << std::endl;
    };
    return launch_with_type(dtype, launch_kernel, "vec bias addition");
}

__device__ __forceinline__ float fast_silu(float x) {
    // SiLU(x) = x / (1.0 + exp(-x))
    return x * __fdividef(1.0f, 1.0f + __expf(-x));
}

__device__ __forceinline__ half2 fast_silu2(half2 x) {
    half2 one = __float2half2_rn(1.0f);
    // 使用硬件级 h2exp 和 h2rcp (近似倒数)
    return __hmul2(x, h2rcp(__hadd2(one, h2exp(__hneg2(x)))));
}

__device__ __forceinline__ __nv_bfloat162 fast_silu2(__nv_bfloat162 x) {
    __nv_bfloat162 one = __float2bfloat162_rn(1.0f);
    return __hmul2(x, h2rcp(__hadd2(one, h2exp(__hneg2(x)))));
}

template<typename T, int VecSize>
__global__ void silu_kernel_v2(T* data, int num) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = tid * VecSize;

    if (offset + VecSize <= num) {
        if constexpr (std::is_same_v<T, float> && VecSize == 4) {
            float4* ptr = reinterpret_cast<float4*>(data + offset);
            float4 v = *ptr;
            v.x = fast_silu(v.x); v.y = fast_silu(v.y);
            v.z = fast_silu(v.z); v.w = fast_silu(v.w);
            *ptr = v;
        } else if constexpr ((std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>) && VecSize == 8) {
            // 处理 8 个 half/bf16 (128-bit)
            using T2 = std::conditional_t<std::is_same_v<T, half>, half2, __nv_bfloat162>;
            T2* ptr = reinterpret_cast<T2*>(data + offset);
            #pragma unroll
            for (int i = 0; i < 4; ++i) ptr[i] = fast_silu2(ptr[i]);
        }
    } else {
        // Tail handling
        for (int i = offset; i < num; ++i) {
            if constexpr (std::is_same_v<T, float>) data[i] = fast_silu(data[i]);
            else data[i] = T(fast_silu(float(data[i])));
        }
    }
}

template<typename T, int VecSize>
__global__ void silu_kernel_v3(T* __restrict__ input, int num) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    const int v_stride = stride * VecSize;
    int offset = tid * VecSize;

    for (; offset + VecSize <= num; offset += v_stride) {
        if constexpr (std::is_same_v<T, float>) {
            float4 v = reinterpret_cast<const float4*>(input + offset)[0];
            v.x = v.x * __fdividef(1.0f, 1.0f + __expf(-v.x));
            v.y = v.y * __fdividef(1.0f, 1.0f + __expf(-v.y));
            v.z = v.z * __fdividef(1.0f, 1.0f + __expf(-v.z));
            v.w = v.w * __fdividef(1.0f, 1.0f + __expf(-v.w));
            reinterpret_cast<float4*>(input + offset)[0] = v;
        } else {
            using T2 = std::conditional_t<std::is_same_v<T, half>, half2, __nv_bfloat162>;
            
            T2 one;
            if constexpr (std::is_same_v<T, half>) {
                one = __float2half2_rn(1.0f);
            } else {
                one = __float2bfloat162_rn(1.0f);
            }

            float4 raw_data = reinterpret_cast<const float4*>(input + offset)[0];
            T2* regs = reinterpret_cast<T2*>(&raw_data);

            #pragma unroll
            for(int i = 0; i < 4; i++) {
                // __h2div, h2rcp
                regs[i] = __hmul2(regs[i], h2rcp(__hadd2(one, h2exp(__hneg2(regs[i])))));
            }

            // 128-bit 矢量化写入
            reinterpret_cast<float4*>(input + offset)[0] = raw_data;
        }
    }

    // Tail handling
    for (int i = offset + tid; i < num; i += stride) {
        float val = (float)input[i];
        input[i] = (T)(val / (1.0f + __expf(-val)));
    }
}


template<typename T, int VecSize>
__global__ void add_bias_kernel_v2(T* __restrict__ data, const T* __restrict__ bias, int m, int n) {
    // 使用 2D 索引避免执行过程中计算 offset % n
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_vec = blockIdx.x * blockDim.x + threadIdx.x;
    int col = col_vec * VecSize;

    if (row < m && col < n) {
        int64_t data_offset = (int64_t)row * n + col;

        if constexpr (std::is_same_v<T, float> && VecSize == 4) {
            float4 d = *reinterpret_cast<float4*>(data + data_offset);
            float4 b = __ldg(reinterpret_cast<const float4*>(bias + col));
            d.x += b.x; d.y += b.y; d.z += b.z; d.w += b.w;
            *reinterpret_cast<float4*>(data + data_offset) = d;
        } else if constexpr ((std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>) && VecSize == 8) {
            using T2 = std::conditional_t<std::is_same_v<T, half>, half2, __nv_bfloat162>;
            T2* d_ptr = reinterpret_cast<T2*>(data + data_offset);
            const T2* b_ptr = reinterpret_cast<const T2*>(bias + col);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                d_ptr[i] = __hadd2(d_ptr[i], __ldg(b_ptr + i));
            }
        }
    }
}

template<typename T, int VecSize>
__global__ void add_bias_kernel_v3(T* __restrict__ data, const T* __restrict__ bias, int m, int n) {
    // 使用 2D 索引避免执行过程中计算 offset % n
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_vec = blockIdx.x * blockDim.x + threadIdx.x;
    int col = col_vec * VecSize;

    if (row < m && col < n) {
        int64_t data_offset = (int64_t)row * n + col;

        if constexpr (std::is_same_v<T, float> && VecSize == 4) {
            float4 d = *reinterpret_cast<float4*>(data + data_offset);
            float4 b = __ldg(reinterpret_cast<const float4*>(bias + col));
            d.x += b.x; d.y += b.y; d.z += b.z; d.w += b.w;
            *reinterpret_cast<float4*>(data + data_offset) = d;
        } else if constexpr ((std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>) && VecSize == 8) {
            using T2 = std::conditional_t<std::is_same_v<T, half>, half2, __nv_bfloat162>;
            const T2* b_ptr = reinterpret_cast<const T2*>(bias + col);
            float4 raw_data_in = reinterpret_cast<const float4*>(data + data_offset)[0];
            T2* regs_in = reinterpret_cast<T2*>(&raw_data_in);

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                regs_in[i] = __hadd2(regs_in[i], __ldg(b_ptr + i));
            }
            reinterpret_cast<float4*>(data + data_offset)[0] = raw_data_in;
        }
    }
}


bool apply_silu_vec_optimized(void* data, int num, nvinfer1::DataType dtype, cudaStream_t stream) {
    auto launch = [&](auto type_tag) {
        using T = decltype(type_tag);
        constexpr int VecSize = (sizeof(T) == 4) ? 4 : 8;
        int threads = 256;
        int blocks = (num / VecSize + threads - 1) / threads;
        // silu_kernel_v2<T, VecSize><<<blocks, threads, 0, stream>>>(reinterpret_cast<T*>(data), num);
        // size_t shared_mem_size = threads * VecSize * sizeof(T);
        silu_kernel_v3<T, VecSize><<<blocks, threads, 0, stream>>>(reinterpret_cast<T*>(data), num);
    };
    return launch_with_type(dtype, launch, "Optimized SiLU");
}

bool apply_add_bias_vec_optimized(void* data, const void* bias, int m, int n,
                             nvinfer1::DataType dtype, cudaStream_t stream) {
    auto launch = [&](auto type_tag) {
        using T = decltype(type_tag);
        constexpr int VecSize = (sizeof(T) == 4) ? 4 : 8;

        if (n % VecSize != 0) {
            // 回退到简单版本或处理 Tail，这里简单演示逻辑
            dim3 block(16, 16);
            dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
            add_bias_kernel_impl<T><<<grid, block, 0, stream>>>(
                reinterpret_cast<T*>(data), reinterpret_cast<const T*>(bias), m, n);
            return;
        }

        dim3 block(32, 8); // 优化线程块形状
        dim3 grid((n / VecSize + block.x - 1) / block.x, (m + block.y - 1) / block.y);

        // add_bias_kernel_v2<T, VecSize><<<grid, block, 0, stream>>>(
        //     reinterpret_cast<T*>(data), reinterpret_cast<const T*>(bias), m, n);
        add_bias_kernel_v3<T, VecSize><<<grid, block, 0, stream>>>(
            reinterpret_cast<T*>(data), reinterpret_cast<const T*>(bias), m, n);
    };
    return launch_with_type(dtype, launch, "Optimized AddBias");
}

__device__ __forceinline__ float fast_silu_fused(float x, float b) {
    float sum = x + b;
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    return sum * __fdividef(1.0f, 1.0f + __expf(-sum));
}

__device__ __forceinline__ half2 fast_silu_fused2(half2 x, half2 b) {
    half2 sum = __hadd2(x, b);
    half2 one = __float2half2_rn(1.0f);
    // 使用 h2rcp(近似倒数) 和 h2exp(近似指数)
    return __hmul2(sum, h2rcp(__hadd2(one, h2exp(__hneg2(sum)))));
}

__device__ __forceinline__ __nv_bfloat162 fast_silu_fused2(__nv_bfloat162 x, __nv_bfloat162 b) {
    __nv_bfloat162 sum = __hadd2(x, b);
    __nv_bfloat162 one = __float2bfloat162_rn(1.0f);
    return __hmul2(sum, h2rcp(__hadd2(one, h2exp(__hneg2(sum)))));
}

template<typename T, int VecSize>
__global__ void __launch_bounds__(256)
add_bias_silu_fused_vec_kernel(T* __restrict__ data, const T* __restrict__ bias, int m, int n) {
    // blockIdx.y -> 行, blockIdx.x -> 列向量
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_vec = blockIdx.x * blockDim.x + threadIdx.x;
    int col = col_vec * VecSize;

    if (row < m && col < n) {
        int64_t offset = (int64_t)row * n + col;

        if constexpr (std::is_same_v<T, float> && VecSize == 4) {
            float4 d = *reinterpret_cast<float4*>(data + offset);
            float4 b = __ldg(reinterpret_cast<const float4*>(bias + col));

            d.x = fast_silu_fused(d.x, b.x);
            d.y = fast_silu_fused(d.y, b.y);
            d.z = fast_silu_fused(d.z, b.z);
            d.w = fast_silu_fused(d.w, b.w);
            *reinterpret_cast<float4*>(data + offset) = d;
        }
        else if constexpr ((std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>) && VecSize == 8) {
            using T2 = std::conditional_t<std::is_same_v<T, half>, half2, __nv_bfloat162>;
            const T2* b_ptr = reinterpret_cast<const T2*>(bias + col);
            float4 raw_data_in = reinterpret_cast<const float4*>(data + offset)[0];
            T2* regs_in = reinterpret_cast<T2*>(&raw_data_in);

            #pragma unroll
            for (int i = 0; i < 4; ++i) { // 4 * T2 = 8 elements
                regs_in[i] = fast_silu_fused2(regs_in[i], __ldg(b_ptr + i));
            }

            reinterpret_cast<float4*>(data + offset)[0] = raw_data_in;
        }
    }
}

template<typename T>
__global__ void add_bias_silu_fused_scalar_kernel(T* __restrict__ data, const T* __restrict__ bias, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int64_t offset = (int64_t)row * n + col;
        float x = (float)data[offset];
        float b = (float)__ldg(bias + col);
        data[offset] = (T)fast_silu_fused(x, b);
    }
}

bool apply_add_bias_silu_fused(void* data, const void* bias, int m, int n,
                               nvinfer1::DataType dtype, cudaStream_t stream) {
    auto launch = [&](auto type_tag) {
        using T = decltype(type_tag);
        // FP32: float4(4元素), FP16/BF16: 128bit对应8元素
        constexpr int VecSize = (sizeof(T) == 4) ? 4 : 8;

        // 对齐判定：n 必须被 VecSize 整除且指针满足 16 字节对齐
        bool can_vectorize = (n % VecSize == 0) &&
                             (reinterpret_cast<uintptr_t>(data) % 16 == 0) &&
                             (reinterpret_cast<uintptr_t>(bias) % 16 == 0);

        if (can_vectorize) {
            dim3 block(32, 8);
            dim3 grid((n / VecSize + block.x - 1) / block.x, (m + block.y - 1) / block.y);
            add_bias_silu_fused_vec_kernel<T, VecSize><<<grid, block, 0, stream>>>(
                reinterpret_cast<T*>(data), reinterpret_cast<const T*>(bias), m, n);
        } else {
            dim3 block(32, 8);
            dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
            add_bias_silu_fused_scalar_kernel<T><<<grid, block, 0, stream>>>(
                reinterpret_cast<T*>(data), reinterpret_cast<const T*>(bias), m, n);
        }
    };

    return launch_with_type(dtype, launch, "Fused AddBias+SiLU");
}

}
}