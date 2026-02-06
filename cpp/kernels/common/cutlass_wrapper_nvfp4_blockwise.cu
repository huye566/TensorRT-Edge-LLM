#include "cutlass_wrapper_nvfp4_blockwise.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

// CUTLASS错误检查宏
#define CUTLASS_CHECK(status)                                               \
    do {                                                                    \
        if (status != cutlass::Status::kSuccess) {                          \
            std::cerr << "CUTLASS error at " << __FILE__ << ":" << __LINE__ \
                     << " code: " << static_cast<int>(status) << std::endl; \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while(0)

#define CUDA_CHECK(status)                                                  \
    do {                                                                    \
        if (status != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                     << " code: " << static_cast<int>(status) << " "        \
                     << cudaGetErrorString(status) << std::endl;            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while(0)

namespace trt_edgellm {
namespace kernel {

using namespace cute;

// ==================== NVFP4 Quantization Types ====================
using nvfp4_type = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using scale_type = cutlass::float_ue4m3_t;

// ==================== NVFP4 GEMM ====================
// A: NVFP4, B: NVFP4
using ElementInputA = nvfp4_type;
using LayoutInputA = cutlass::layout::RowMajor;
constexpr int AlignmentInputA = 32;  // For nvfp4_type

using ElementInputB = nvfp4_type;  // NVFP4 type
using LayoutInputB = cutlass::layout::ColumnMajor;
constexpr int AlignmentInputB = 32;  // For nvfp4_type

using ElementOutput = cutlass::half_t;
using LayoutOutput = cutlass::layout::RowMajor;
constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// ==================== Epilogue Fusion Definitions ====================
// cutlass/include/cutlass/epilogue/fusion/operations.hpp
template <typename OutputType, typename ElementAccumulator>
using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<OutputType, ElementAccumulator>;

template <typename OutputType, typename ElementAccumulator>
using BiasOperation = cutlass::epilogue::fusion::LinCombPerColBias<OutputType, ElementAccumulator>;

template <typename OutputType, typename ElementAccumulator>
using SiLUOperation = cutlass::epilogue::fusion::LinCombEltAct<
                    cutlass::epilogue::thread::SiLu, OutputType, ElementAccumulator>;

template <typename OutputType, typename ElementAccumulator>
using BiasSiLUOperation = cutlass::epilogue::fusion::LinCombPerColBiasEltAct<
                         cutlass::epilogue::thread::SiLu, OutputType, ElementAccumulator>;

// ==================== Kernel Configurations ====================
template <typename T>
struct KernelConfigM128 {
    using OutputType = T;
    using MmaTileShape = Shape<_128, _256, _256>;
    using ClusterShape = Shape<int, int, _1>;
    using EpilogueTile = Shape<_128, _64>;  // Avoid register spilling
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
    using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100;
    const static dim3 preferred_cluster;
    const static dim3 fallback_cluster;
};
template <typename T>
const dim3 KernelConfigM128<T>::preferred_cluster(1, 4, 1);
template <typename T>
const dim3 KernelConfigM128<T>::fallback_cluster(1, 2, 1);

template <typename T>
struct KernelConfigM256 {
    using OutputType = T;
    using MmaTileShape = Shape<_256, _256, _256>;
    using ClusterShape = Shape<int, int, _1>;
    using EpilogueTile = Shape<_128, _64>;  // Avoid register spilling
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
    using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100;
    const static dim3 preferred_cluster;
    const static dim3 fallback_cluster;
};
template <typename T>
const dim3 KernelConfigM256<T>::preferred_cluster(2, 4, 1);
template <typename T>
const dim3 KernelConfigM256<T>::fallback_cluster(2, 1, 1);

template <typename T>
struct KernelConfigDefault {
    using OutputType = T;
    using MmaTileShape = Shape<_256, _256, _256>;
    using ClusterShape = Shape<int, int, _1>;
    using EpilogueTile = Shape<_128, _64>;  // Avoid register spilling
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
    using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100;
    const static dim3 preferred_cluster;
    const static dim3 fallback_cluster;
};
template <typename T>
const dim3 KernelConfigDefault<T>::preferred_cluster(4, 4, 1);
template <typename T>
const dim3 KernelConfigDefault<T>::fallback_cluster(2, 1, 1);

template <typename T>
struct KernelConfigSpecial {
    using OutputType = T;
    using MmaTileShape = Shape<_256, _128, _64>;
    using ClusterShape = Shape<int, int,_1>;
    using EpilogueTile = Shape<_128, _64>;;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
    using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100;
    const static dim3 preferred_cluster;
    const static dim3 fallback_cluster;
};
template <typename T>
const dim3 KernelConfigSpecial<T>::preferred_cluster(2, 2, 1);
template <typename T>
const dim3 KernelConfigSpecial<T>::fallback_cluster(2, 1, 1);

struct KernelConfigFp32 {
    using OutputType = float;
    using MmaTileShape = Shape<_128, _128, _256>;
    using ClusterShape = Shape<int, int, _1>;
    using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
    using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100;
    const static dim3 preferred_cluster;
    const static dim3 fallback_cluster;
};
const dim3 KernelConfigFp32::preferred_cluster = dim3(1, 4, 1);
const dim3 KernelConfigFp32::fallback_cluster = dim3(1, 2, 1);

// ==================== NVFP4 GEMM Kernel ====================
template <typename KernelConfig, bool kEnableSilu, bool kEnableBias>
struct NvFp4GemmSm100 {
    using Config = KernelConfig;
    using OutputType = typename KernelConfig::OutputType;

    // A matrix configuration (NVFP4)
    using ElementA = ElementInputA;
    using LayoutATag = LayoutInputA;
    static constexpr int AlignmentA = AlignmentInputA;

    // B matrix configuration (NVFP4)
    using ElementB = ElementInputB;
    using LayoutBTag = LayoutInputB;
    static constexpr int AlignmentB = AlignmentInputB;

    // C/D matrix configuration
    using ElementD = OutputType;
    using ElementC = OutputType;
    using LayoutCTag = LayoutOutput;
    using LayoutDTag = LayoutOutput;
    static constexpr int AlignmentD = AlignmentOutput;
    static constexpr int AlignmentC = AlignmentOutput;

    // Kernel functional config
    using ElementAccumulator = float;
    using ArchTag = cutlass::arch::Sm100;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

    // Kernel Perf config
    using MmaTileShape = typename KernelConfig::MmaTileShape;
    using ClusterShape = typename KernelConfig::ClusterShape;
    using EpilogueTile = typename KernelConfig::EpilogueTile;
    using EpilogueSchedule = typename KernelConfig::EpilogueSchedule;
    using MainloopSchedule = typename KernelConfig::MainloopSchedule;

    using EpilogueOperation = 
        cute::conditional_t<kEnableSilu && kEnableBias,
            BiasSiLUOperation<OutputType, ElementAccumulator>,
            cute::conditional_t<kEnableSilu,
                SiLUOperation<OutputType, ElementAccumulator>,
                cute::conditional_t<kEnableBias,
                    BiasOperation<OutputType, ElementAccumulator>,
                    DefaultOperation<OutputType, ElementAccumulator>
                >
            >
        >;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        MmaTileShape,
        ClusterShape,
        EpilogueTile,
        ElementAccumulator,
        ElementAccumulator,
        void,
        LayoutCTag,
        AlignmentC,
        ElementD,
        LayoutDTag,
        AlignmentD,
        EpilogueSchedule,
        EpilogueOperation>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        ElementA,
        LayoutATag,
        AlignmentA,
        ElementB,
        LayoutBTag,
        AlignmentB,
        ElementAccumulator,
        MmaTileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopSchedule>::CollectiveOp;

    using GemmKernel =
        cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                            CollectiveMainloop,
                                            CollectiveEpilogue,
                                            void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    // using LayoutA = decltype(cute::make_layout(make_shape(0, 0, 0), StrideA{}));
    // using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    // using LayoutB = decltype(cute::make_layout(make_shape(0, 0, 0), StrideB{}));
    // using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    // using LayoutC = decltype(cute::make_layout(make_shape(0, 0, 0), StrideC{}));
    using StrideD = typename Gemm::GemmKernel::StrideD;
    // using LayoutD = decltype(cute::make_layout(make_shape(0, 0, 0), StrideD{}));
};

// ==================== GEMM Argument Builder ====================
template <typename T, bool kEnableSilu, bool kEnableBias>
typename T::Gemm::Arguments args_from_options(
    void* output,
    const void* input_a,
    const void* input_b,
    const void* scales_a,
    const void* scales_b,
    const void* bias,
    float alpha,
    int M,
    int N,
    int K) {

    using ElementA = typename T::Gemm::ElementA;
    using ElementB = typename T::Gemm::ElementB;
    using ElementSFA = cutlass::float_ue4m3_t;
    using ElementSFB = cutlass::float_ue4m3_t;
    using ElementD = typename T::Gemm::ElementD;
    using ElementCompute = float;
    using StrideA = typename T::StrideA;
    using StrideB = typename T::StrideB;
    using StrideD = typename T::StrideD;
    using Sm1xxBlkScaledConfig = typename T::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    int m = static_cast<int>(M);
    int n = static_cast<int>(N);
    int k = static_cast<int>(K);
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

    typename T::Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {// Mainloop arguments
        static_cast<ElementA const*>(input_a),
        stride_A,
        static_cast<ElementB const*>(input_b),
        stride_B,
        static_cast<ElementSFA const*>(scales_a),
        layout_SFA,
        static_cast<ElementSFB const*>(scales_b),
        layout_SFB},
        {     // Epilogue arguments
        {alpha, 0.0f},  // epilogue.thread
        nullptr,
        stride_D,
        static_cast<ElementD*>(output),
        stride_D}};

    if constexpr(kEnableBias) {
        auto &fusion_args = arguments.epilogue.thread;
        fusion_args.bias_ptr = static_cast<ElementD const*>(bias);
        // fusion_args.alpha_ptr = static_cast<ElementCompute const*>(&alpha);
    }

    using KernelConfig = typename T::Config;
    arguments.hw_info.cluster_shape = KernelConfig::preferred_cluster;
    arguments.hw_info.cluster_shape_fallback = KernelConfig::fallback_cluster;
    return arguments;
}

// ==================== GEMM Dispatch Function ====================
template <typename T, bool kEnableSilu, bool kEnableBias>
void runNvfp4Gemm(
    void* output,
    const void* input_a,
    const void* input_b,
    const void* scales_a,
    const void* scales_b,
    const void* bias,
    float alpha,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    typename T::Gemm gemm;
    auto arguments = args_from_options<T, kEnableSilu, kEnableBias>(
        output, input_a, input_b, scales_a, scales_b, bias, alpha, M, N, K);

    size_t workspace_size = T::Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm.can_implement(arguments));
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get(), stream));

    CUTLASS_CHECK(gemm.run(arguments, workspace.get(), stream));
}

template <typename OutType, bool kEnableSilu, bool kEnableBias>
void cutlassFp4GemmDispatch(
    void* output,
    const void* input_a,
    const void* input_b,
    const void* scales_a,
    const void* scales_b,
    const void* bias,
    float alpha,
    int M,
    int N,
    int K,
    cudaStream_t stream) {

    if (M <= 128) {
        // m in [1, 128]
        runNvfp4Gemm<NvFp4GemmSm100<KernelConfigM128<OutType>, kEnableSilu, kEnableBias>, kEnableSilu, kEnableBias>(
            output, input_a, input_b, scales_a, scales_b, bias, alpha, M, N, K, stream);
    } else if (M <= 256) {
        // m in (128, 256]
        runNvfp4Gemm<NvFp4GemmSm100<KernelConfigM256<OutType>, kEnableSilu, kEnableBias>, kEnableSilu, kEnableBias>(
            output, input_a, input_b, scales_a, scales_b, bias, alpha, M, N, K, stream);
    } else {
        // m in (256, inf)
        runNvfp4Gemm<NvFp4GemmSm100<KernelConfigDefault<OutType>, kEnableSilu, kEnableBias>, kEnableSilu, kEnableBias>(
            output, input_a, input_b, scales_a, scales_b, bias, alpha, M, N, K, stream);
    }
}

template <bool kEnableSilu, bool kEnableBias>
void cutlassFp4GemmDispatch<float>(
    void* output,
    const void* input_a,
    const void* input_b,
    const void* scales_a,
    const void* scales_b,
    const void* bias,
    float alpha,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    runNvfp4Gemm<NvFp4GemmSm100<KernelConfigFp32, kEnableSilu, kEnableBias>, kEnableSilu, kEnableBias>(
        output, input_a, input_b, scales_a, scales_b, bias, alpha, M, N, K, stream);
}


template <typename GemmTraits, bool kEnableSilu, bool kEnableBias>
class GemmRunner {
public:
    using Gemm = typename GemmTraits::Gemm;
    using Arguments = typename Gemm::Arguments;

    GemmRunner() = default;
    GemmRunner(const GemmRunner&) = delete;
    GemmRunner& operator=(const GemmRunner&) = delete;

    ~GemmRunner() {
        if (internal_workspace_) {
            cudaFree(internal_workspace_);
        }
    }

    void run(void* output, const void* input_a, const void* input_b,
             const void* scales_a, const void* scales_b, const void* bias,
             float alpha, int M, int N, int K, cudaStream_t stream) {

        initialize(output, input_a, input_b, scales_a, scales_b, bias, alpha, M, N, K, stream);
        CUTLASS_CHECK(gemm_.run(stream));
    }

private:
    bool check_inputs(void* output, const void* input_a, const void* input_b,
             const void* scales_a, const void* scales_b, const void* bias,
             float alpha, int M, int N, int K) const {
        if (M != cached_M_ || N != cached_N_ || K != cached_K_) {
            return true;
        }

        if (output != cached_output_ || 
            input_a != cached_input_a_ || 
            input_b != cached_input_b_ || 
            scales_a != cached_scales_a_ ||
            scales_b != cached_scales_b_ ||
            bias != cached_bias_) {
            return true;
        }
        return false;
    }

    void initialize(void* output, const void* input_a, const void* input_b,
             const void* scales_a, const void* scales_b, const void* bias,
             float alpha, int M, int N, int K, cudaStream_t stream) {
        if (!check_inputs(output, input_a, input_b, scales_a, scales_b, bias, alpha, M, N, K)) {
            return;
        }
        arguments_ = args_from_options<GemmTraits, kEnableSilu, kEnableBias>(
                output, input_a, input_b, scales_a, scales_b, bias, alpha, M, N, K);
        CUTLASS_CHECK(gemm_.can_implement(arguments_));
        size_t required_workspace_size = Gemm::get_workspace_size(arguments_);

        if (required_workspace_size > internal_workspace_capacity_) {
            if (internal_workspace_) CUDA_CHECK(cudaFree(internal_workspace_));
            CUDA_CHECK(cudaMalloc(&internal_workspace_, required_workspace_size));
            internal_workspace_capacity_ = required_workspace_size;
        }
        CUTLASS_CHECK(gemm_.initialize(arguments_, internal_workspace_, stream));
        cached_M_ = M;
        cached_N_ = N;
        cached_K_ = K;
        cached_output_ = output;
        cached_input_a_ = input_a;
        cached_input_b_ = input_b;
        cached_scales_a_ = scales_a;
        cached_scales_b_ = scales_b;
        cached_bias_ = bias;
    }

    Gemm gemm_;
    typename GemmTraits::Gemm::Arguments arguments_;
    void* internal_workspace_ = nullptr;
    size_t internal_workspace_capacity_ = 0;
    int cached_M_ = 0;
    int cached_N_ = 0;
    int cached_K_ = 0;
    void* cached_output_ = nullptr;
    const void* cached_input_a_ = nullptr;
    const void* cached_input_b_ = nullptr;
    const void* cached_scales_a_ = nullptr;
    const void* cached_scales_b_ = nullptr;
    const void* cached_bias_ = nullptr;
};

template <typename OutType, bool kEnableSilu, bool kEnableBias>
class CutlassFp4Context {
    using RunnerM128 = GemmRunner<NvFp4GemmSm100<KernelConfigM128<OutType>, kEnableSilu, kEnableBias>, kEnableSilu, kEnableBias>;
    using RunnerM256 = GemmRunner<NvFp4GemmSm100<KernelConfigM256<OutType>, kEnableSilu, kEnableBias>, kEnableSilu, kEnableBias>;
    using RunnerDefault = GemmRunner<NvFp4GemmSm100<KernelConfigDefault<OutType>, kEnableSilu, kEnableBias>, kEnableSilu, kEnableBias>;

    RunnerM128 runner_128_;
    RunnerM256 runner_256_;
    RunnerDefault runner_def_;

public:
    void dispatch(
        void* output, const void* input_a, const void* input_b,
        const void* scales_a, const void* scales_b, const void* bias,
        float alpha, int M, int N, int K, cudaStream_t stream) {
        
        if (M <= 128) {
            runner_128_.run(output, input_a, input_b, scales_a, scales_b, bias, alpha, M, N, K, stream);
        } else if (M <= 256) {
            runner_256_.run(output, input_a, input_b, scales_a, scales_b, bias, alpha, M, N, K, stream);
        } else {
            runner_def_.run(output, input_a, input_b, scales_a, scales_b, bias, alpha, M, N, K, stream);
        }
    }
};


template <typename OutType, bool kEnableSilu, bool kEnableBias>
void cutlassFp4GemmDispatchCached(
    void* output, const void* input_a, const void* input_b,
    const void* scales_a, const void* scales_b, const void* bias,
    float alpha, int M, int N, int K, cudaStream_t stream) {

    thread_local auto context = std::make_unique<CutlassFp4Context<OutType, kEnableSilu, kEnableBias>>();
    context->dispatch(output, input_a, input_b, scales_a, scales_b, bias, alpha, M, N, K, stream);
}

template <bool kEnableSilu, bool kEnableBias>
void cutlassFp4GemmDispatchCachedFloat(
    void* output, const void* input_a, const void* input_b,
    const void* scales_a, const void* scales_b, const void* bias,
    float alpha, int M, int N, int K, cudaStream_t stream) {
    
    using RunnerFp32 = GemmRunner<NvFp4GemmSm100<KernelConfigFp32, kEnableSilu, kEnableBias>, kEnableSilu, kEnableBias>;
    thread_local auto runner = std::make_unique<RunnerFp32>();
    runner->run(output, input_a, input_b, scales_a, scales_b, bias, alpha, M, N, K, stream);
}

template <bool kEnableSilu, bool kEnableBias>
void cutlass_scaled_nvfp4(
    void* output,
    const void* input_a,
    const void* input_b,
    const void* scales_a,
    const void* scales_b,
    const void* bias,
    float alpha,
    int M,
    int N,
    int K,
    cudaStream_t stream,
    nvinfer1::DataType outputType,
    bool use_cached) {

    if (use_cached) {
        if (outputType == nvinfer1::DataType::kHALF) {
            cutlassFp4GemmDispatchCached<cutlass::half_t, kEnableSilu, kEnableBias>(
                output, input_a, input_b, scales_a, scales_b, bias, alpha, M, N, K, stream);
        } else if (outputType == nvinfer1::DataType::kBF16) {
            cutlassFp4GemmDispatchCached<cutlass::bfloat16_t, kEnableSilu, kEnableBias>(
                output, input_a, input_b, scales_a, scales_b, bias, alpha, M, N, K, stream);
        } else if (outputType == nvinfer1::DataType::kFLOAT) {
            cutlassFp4GemmDispatchCachedFloat<kEnableSilu, kEnableBias>(
                output, input_a, input_b, scales_a, scales_b, bias, alpha, M, N, K, stream);
        } else {
            throw std::runtime_error("Unsupported output type");
        }
    } else {
        if (outputType == nvinfer1::DataType::kHALF) {
            cutlassFp4GemmDispatch<cutlass::half_t, kEnableSilu, kEnableBias>(
                output, input_a, input_b, scales_a, scales_b, bias, alpha, M, N, K, stream);
        } else if (outputType == nvinfer1::DataType::kBF16) {
            cutlassFp4GemmDispatch<cutlass::bfloat16_t, kEnableSilu, kEnableBias>(
                output, input_a, input_b, scales_a, scales_b, bias, alpha, M, N, K, stream);
        } 
        else if (outputType == nvinfer1::DataType::kFLOAT) {
            cutlassFp4GemmDispatch<float, kEnableSilu, kEnableBias>(
                output, input_a, input_b, scales_a, scales_b, bias, alpha, M, N, K, stream);
        } else {
            throw std::runtime_error("Unsupported output type");
        }
    }
}

template void cutlass_scaled_nvfp4<false, false>(
    void* output,
    const void* input_a,
    const void* input_b,
    const void* scales_a,
    const void* scales_b,
    const void* bias,
    float alpha,
    int M,
    int N,
    int K,
    cudaStream_t stream,
    nvinfer1::DataType outputType,
    bool use_cached);

template void cutlass_scaled_nvfp4<true, false>(
    void* output,
    const void* input_a,
    const void* input_b,
    const void* scales_a,
    const void* scales_b,
    const void* bias,
    float alpha,
    int M,
    int N,
    int K,
    cudaStream_t stream,
    nvinfer1::DataType outputType,
    bool use_cached);

template void cutlass_scaled_nvfp4<false, true>(
    void* output,
    const void* input_a,
    const void* input_b,
    const void* scales_a,
    const void* scales_b,
    const void* bias,
    float alpha,
    int M,
    int N,
    int K,
    cudaStream_t stream,
    nvinfer1::DataType outputType,
    bool use_cached);

template void cutlass_scaled_nvfp4<true, true>(
    void* output,
    const void* input_a,
    const void* input_b,
    const void* scales_a,
    const void* scales_b,
    const void* bias,
    float alpha,
    int M,
    int N,
    int K,
    cudaStream_t stream,
    nvinfer1::DataType outputType,
    bool use_cached);

} // namespace kernel
} // namespace trt_edgellm
