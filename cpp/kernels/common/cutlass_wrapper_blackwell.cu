#include "cutlass_wrapper_blackwell.h"

#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>

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


#define CUTLASS_CHECK(status)                                                         \
  {                                                                                   \
    cutlass::Status error = status;                                                   \
    if (error != cutlass::Status::kSuccess) {                                         \
      auto msg = std::string("[") + __FILE__ + "] Got cutlass error: " +              \
          cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);         \
      std::cerr << msg << std::endl;                                                  \
      throw std::runtime_error(msg);                                                  \
    }                                                                                 \
  }


namespace trt_edgellm {
namespace kernel {

// ==================== GEMM kernels ====================
template <bool kEnableSilu, bool kEnableBias>
void cutlass_gemm_blackwell_standard(ElementOutput* output,
                            const ElementInputA* input,
                            const ElementInputB* weights,
                            const ElementOutput* bias,
                            int M, int N, int K,
                            cudaStream_t stream) {
    using         LayoutInputA     = cutlass::layout::RowMajor;
    using         LayoutInputB     = cutlass::layout::RowMajor;
    using         LayoutOutput     = cutlass::layout::RowMajor;

    using MmaTileShape_MNK = Shape<_256,_128,_64>;
    using ClusterShape_MNK = Shape<_2,_2,_1>;

    // using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
    // using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
    using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;

    using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementAccumulator>;
    using BiasOperation = cutlass::epilogue::fusion::LinCombPerColBias<ElementOutput, ElementAccumulator>;
    using FusionOperation = cutlass::epilogue::fusion::LinCombEltAct<
                            cutlass::epilogue::thread::SiLu, ElementOutput, ElementAccumulator>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        MmaTileShape_MNK, ClusterShape_MNK,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementOutput, LayoutOutput, AlignmentOutput,
        ElementOutput, LayoutOutput, AlignmentOutput,
        EpilogueSchedule,
        cute::conditional_t<kEnableSilu, FusionOperation, cute::conditional_t<kEnableBias, BiasOperation, DefaultOperation>>
    >::CollectiveOp;

    // using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
    using MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementInputA, LayoutInputA, AlignmentA,
        ElementInputB, LayoutInputB, AlignmentB,
        ElementAccumulator,
        MmaTileShape_MNK, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopSchedule
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    StrideA stride_A;
    StrideB stride_B;
    StrideC stride_C;
    StrideD stride_D;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    float alpha = 1.f;
    float beta = 0.f;

    typename Gemm::Arguments arguments {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {input, stride_A, weights, stride_B},
        {{alpha, beta}, nullptr, stride_C, output, stride_D}
    };

    if constexpr(kEnableBias) {
        auto &fusion_args = arguments.epilogue.thread;
        fusion_args.bias_ptr = bias;
    }

    Gemm gemm;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm.can_implement(arguments));
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(gemm.run(stream));
}


// ==================== GEMV kernels ====================
template <bool kEnableSilu, bool kEnableBias>
void cutlass_gemv_blackwell_standard(ElementOutput* output,
                            const ElementInputA* input,
                            const ElementInputB* weights,
                            const ElementOutput* bias,
                            int M, int N, int K,
                            cudaStream_t stream) {

    using         LayoutInputA     = cutlass::layout::ColumnMajor;
    using         LayoutInputB     = cutlass::layout::ColumnMajor;
    using         LayoutOutput     = cutlass::layout::ColumnMajor;

    using MmaTileShape_MNK = Shape<_128,_16,_64>;
    using ClusterShape_MNK = Shape<_1,_1,_1>;

    // using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
    // using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
    using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;

    using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementAccumulator>;
    using BiasOperation = cutlass::epilogue::fusion::LinCombPerColBias<ElementOutput, ElementAccumulator>;
    using FusionOperation = cutlass::epilogue::fusion::LinCombEltAct<
                            cutlass::epilogue::thread::SiLu, ElementOutput, ElementAccumulator>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        MmaTileShape_MNK, ClusterShape_MNK,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementOutput, LayoutOutput, AlignmentOutput,
        ElementOutput, LayoutOutput, AlignmentOutput,
        EpilogueSchedule,
        cute::conditional_t<kEnableSilu, FusionOperation, cute::conditional_t<kEnableBias, BiasOperation, DefaultOperation>>
    >::CollectiveOp;

    // using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;
    using MainloopSchedule = cutlass::gemm::KernelMixedTmaCpAsyncWarpSpecialized1SmSm100;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementInputA, LayoutInputA, AlignmentA,
        ElementInputB, LayoutInputB, AlignmentB,
        ElementAccumulator,
        MmaTileShape_MNK, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopSchedule
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    StrideA stride_A;
    StrideB stride_B;
    StrideC stride_C;
    StrideD stride_D;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, {N, K, 1});
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, {M, K, 1});
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, {N, M, 1});
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, {N, M, 1});

    float alpha = 1.f;
    float beta = 0.f;

    typename Gemm::Arguments arguments {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {N, M, K, 1},
        {weights, input},
        {{alpha, beta}, nullptr, stride_C, output, stride_D}
    };

    if constexpr(kEnableBias) {
        auto &fusion_args = arguments.epilogue.thread;
        fusion_args.bias_ptr = bias;
    }

    Gemm gemm;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm.can_implement(arguments));
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(gemm.run(stream));
}


// ==================== Cache-enabled GEMM Runner ====================
template <bool kEnableSilu, bool kEnableBias>
class GemmRunnerBlackwell {
private:
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::RowMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    using MmaTileShape_MNK = Shape<_256,_128,_64>;
    using ClusterShape_MNK = Shape<_2,_2,_1>;

    using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;

    using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementAccumulator>;
    using BiasOperation = cutlass::epilogue::fusion::LinCombPerColBias<ElementOutput, ElementAccumulator>;
    using FusionOperation = cutlass::epilogue::fusion::LinCombEltAct<
                            cutlass::epilogue::thread::SiLu, ElementOutput, ElementAccumulator>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        MmaTileShape_MNK, ClusterShape_MNK,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementOutput, LayoutOutput, AlignmentOutput,
        ElementOutput, LayoutOutput, AlignmentOutput,
        EpilogueSchedule,
        cute::conditional_t<kEnableSilu, FusionOperation, cute::conditional_t<kEnableBias, BiasOperation, DefaultOperation>>
    >::CollectiveOp;

    using MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementInputA, LayoutInputA, AlignmentA,
        ElementInputB, LayoutInputB, AlignmentB,
        ElementAccumulator,
        MmaTileShape_MNK, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopSchedule
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

public:
    GemmRunnerBlackwell() = default;

    ~GemmRunnerBlackwell() {
        if (workspace_) {
            cudaFree(workspace_);
        }
    }

    void run(ElementOutput* output,
             const ElementInputA* input,
             const ElementInputB* weights,
             const ElementOutput* bias,
             int M, int N, int K,
             cudaStream_t stream) {
        initialize(output, input, weights, bias, M, N, K, stream);

        // arguments_.mainloop.ptr_A = input;
        // arguments_.mainloop.ptr_B = weights;
        // if constexpr(kEnableBias) {
        //     auto &fusion_args = arguments_.epilogue.thread;
        //     fusion_args.bias_ptr = bias;
        // }
        // arguments_.epilogue.ptr_D = output;
        // CUTLASS_CHECK(gemm_.run(arguments_, workspace_, stream));
        CUTLASS_CHECK(gemm_.run(stream));
    }

private:
    bool check_inputs(ElementOutput* output,
             const ElementInputA* input,
             const ElementInputB* weights,
             const ElementOutput* bias,
             int M, int N, int K) const {
        if (M != cached_M_ || N != cached_N_ || K != cached_K_) {
            return true;
        }

        if (output != cached_output_ || input != cached_input_ || weights != cached_weights_ || bias != cached_bias_) {
            return true;
        }
        return false;
    }

    void initialize(ElementOutput* output,
             const ElementInputA* input,
             const ElementInputB* weights,
             const ElementOutput* bias,
             int M, int N, int K,
             cudaStream_t stream) {
        if (!check_inputs(output, input, weights, bias, M, N, K)) return;
        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
        StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
        StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

        float alpha = 1.f;
        float beta = 0.f;

        // Create new arguments
        arguments_ = typename Gemm::Arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {input, stride_A, weights, stride_B},
            {{alpha, beta}, nullptr, stride_C, output, stride_D}
        };

        if constexpr(kEnableBias) {
            auto &fusion_args = arguments_.epilogue.thread;
            fusion_args.bias_ptr = bias;
        }

        CUTLASS_CHECK(gemm_.can_implement(arguments_));

        // Allocate workspace
        auto workspace_size = Gemm::get_workspace_size(arguments_);
        if (workspace_size_ < workspace_size) {
            cudaFree(workspace_);
            cudaMalloc(&workspace_, workspace_size_);
            workspace_size_ = workspace_size;
        }

        CUTLASS_CHECK(gemm_.initialize(arguments_, workspace_));

        // Cache shape
        cached_M_ = M;
        cached_N_ = N;
        cached_K_ = K;
        cached_input_ = input;
        cached_output_ = output;
        cached_weights_ = weights;
        cached_bias_ = bias;
    }

    Gemm gemm_;
    typename Gemm::Arguments arguments_;
    void* workspace_ = nullptr;
    size_t workspace_size_ = 0;

    int cached_M_ = 0;
    int cached_N_ = 0;
    int cached_K_ = 0;
    const ElementInputA* cached_input_ = nullptr;
    ElementOutput* cached_output_ = nullptr;
    const ElementInputB* cached_weights_ = nullptr;
    const ElementOutput* cached_bias_ = nullptr;
};

// ==================== Cache-enabled GEMV Runner ====================
template <bool kEnableSilu, bool kEnableBias>
class GemvRunnerBlackwell {
private:
    using LayoutInputA = cutlass::layout::ColumnMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::ColumnMajor;

    using MmaTileShape_MNK = Shape<_128,_16,_64>;
    using ClusterShape_MNK = Shape<_1,_1,_1>;

    using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;

    using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementAccumulator>;
    using BiasOperation = cutlass::epilogue::fusion::LinCombPerColBias<ElementOutput, ElementAccumulator>;
    using FusionOperation = cutlass::epilogue::fusion::LinCombEltAct<
                            cutlass::epilogue::thread::SiLu, ElementOutput, ElementAccumulator>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        MmaTileShape_MNK, ClusterShape_MNK,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementOutput, LayoutOutput, AlignmentOutput,
        ElementOutput, LayoutOutput, AlignmentOutput,
        EpilogueSchedule,
        cute::conditional_t<kEnableSilu, FusionOperation, cute::conditional_t<kEnableBias, BiasOperation, DefaultOperation>>
    >::CollectiveOp;

    using MainloopSchedule = cutlass::gemm::KernelMixedTmaCpAsyncWarpSpecialized1SmSm100;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementInputA, LayoutInputA, AlignmentA,
        ElementInputB, LayoutInputB, AlignmentB,
        ElementAccumulator,
        MmaTileShape_MNK, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopSchedule
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

public:
    GemvRunnerBlackwell() = default;

    ~GemvRunnerBlackwell() {
        if (workspace_) {
            cudaFree(workspace_);
        }
    }

    void run(ElementOutput* output,
             const ElementInputA* input,
             const ElementInputB* weights,
             const ElementOutput* bias,
             int M, int N, int K,
             cudaStream_t stream) {
        initialize(output, input, weights, bias, M, N, K, stream);

        // arguments_.mainloop.ptr_A = weights;
        // arguments_.mainloop.ptr_B = input;
        // if constexpr(kEnableBias) {
        //     auto &fusion_args = arguments_.epilogue.thread;
        //     fusion_args.bias_ptr = bias;
        // }
        // arguments_.epilogue.ptr_D = output;
        // CUTLASS_CHECK(gemm_.run(arguments_, workspace_, stream));
        CUTLASS_CHECK(gemm_.run(stream));
    }

private:
    bool check_inputs(ElementOutput* output,
             const ElementInputA* input,
             const ElementInputB* weights,
             const ElementOutput* bias,
             int M, int N, int K) const {
        if (M != cached_M_ || N != cached_N_ || K != cached_K_) {
            return true;
        }

        if (output != cached_output_ || input != cached_input_ || weights != cached_weights_ || bias != cached_bias_) {
            return true;
        }
        return false;
    }

    void initialize(ElementOutput* output,
             const ElementInputA* input,
             const ElementInputB* weights,
             const ElementOutput* bias,
             int M, int N, int K,
             cudaStream_t stream) {
        if (!check_inputs(output, input, weights, bias, M, N, K)) return;

        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {N, K, 1});
        StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {M, K, 1});
        StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {N, M, 1});
        StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {N, M, 1});

        float alpha = 1.f;
        float beta = 0.f;

        arguments_ = typename Gemm::Arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {N, M, K, 1},
            {weights, input},
            {{alpha, beta}, nullptr, stride_C, output, stride_D}
        };

        if constexpr(kEnableBias) {
            auto &fusion_args = arguments_.epilogue.thread;
            fusion_args.bias_ptr = bias;
        }

        CUTLASS_CHECK(gemm_.can_implement(arguments_));

        auto workspace_size = Gemm::get_workspace_size(arguments_);
        if (workspace_size_ < workspace_size) {
            cudaFree(workspace_);
            cudaMalloc(&workspace_, workspace_size_);
            workspace_size_ = workspace_size;
        }

        CUTLASS_CHECK(gemm_.initialize(arguments_, workspace_));

        cached_M_ = M;
        cached_N_ = N;
        cached_K_ = K;
        cached_input_ = input;
        cached_output_ = output;
        cached_weights_ = weights;
        cached_bias_ = bias;
    }

    Gemm gemm_;
    typename Gemm::Arguments arguments_;
    void* workspace_ = nullptr;
    size_t workspace_size_ = 0;

    int cached_M_ = 0;
    int cached_N_ = 0;
    int cached_K_ = 0;
    const ElementInputA* cached_input_ = nullptr;
    ElementOutput* cached_output_ = nullptr;
    const ElementInputB* cached_weights_ = nullptr;
    const ElementOutput* cached_bias_ = nullptr;
};


// ==================== Cached API Functions ====================
template <bool kEnableSilu, bool kEnableBias>
void cutlass_gemm_blackwell_cached(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    const ElementOutput* bias,
    int M, int N, int K,
    cudaStream_t stream) {

    // Thread-local runner for thread safety
    thread_local static GemmRunnerBlackwell<kEnableSilu, kEnableBias> runner;
    runner.run(output, input, weights, bias, M, N, K, stream);
}

template <bool kEnableSilu, bool kEnableBias>
void cutlass_gemv_blackwell_cached(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    const ElementOutput* bias,
    int M, int N, int K,
    cudaStream_t stream) {

    thread_local static GemvRunnerBlackwell<kEnableSilu, kEnableBias> runner;
    runner.run(output, input, weights, bias, M, N, K, stream);
}


template <bool kEnableSilu, bool kEnableBias>
void cutlass_gemm_blackwell(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    const ElementOutput* bias,
    int M, int N, int K,
    cudaStream_t stream,
    bool use_cached) {
    if (use_cached) {
        cutlass_gemm_blackwell_cached<kEnableSilu, kEnableBias>(
                output, input, weights, bias, M, N, K, stream);

    } else {
        cutlass_gemm_blackwell_standard<kEnableSilu, kEnableBias>(
                output, input, weights, bias, M, N, K, stream);
    }
}


template <bool kEnableSilu, bool kEnableBias>
void cutlass_gemv_blackwell(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    const ElementOutput* bias,
    int M, int N, int K,
    cudaStream_t stream,
    bool use_cached) {
    if (use_cached) {
        cutlass_gemv_blackwell_cached<kEnableSilu, kEnableBias>(
                output, input, weights, bias, M, N, K, stream);

    } else {
        cutlass_gemv_blackwell_standard<kEnableSilu, kEnableBias>(
                output, input, weights, bias, M, N, K, stream);
    }
}


// ==================== Unified Dispatcher ====================
template <bool kEnableSilu, bool kEnableBias>
void cutlass_blackwell_dispatch(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    const ElementOutput* bias,
    int M, int N, int K,
    cudaStream_t stream,
    bool use_cached) {

    if (M >= 128) {  // Use GEMM for larger M
        cutlass_gemm_blackwell<kEnableSilu, kEnableBias>(
            output, input, weights, bias, M, N, K, stream, use_cached);
    } else {  // Use GEMV for smaller M
        cutlass_gemv_blackwell<kEnableSilu, kEnableBias>(
            output, input, weights, bias, M, N, K, stream, use_cached);
    }
}

// ==================== Template Instantiations ====================
template void cutlass_blackwell_dispatch<false, false>(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    const ElementOutput* bias,
    int M, int N, int K,
    cudaStream_t stream,
    bool use_cached);

template void cutlass_blackwell_dispatch<true, false>(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    const ElementOutput* bias,
    int M, int N, int K,
    cudaStream_t stream,
    bool use_cached);

template void cutlass_blackwell_dispatch<false, true>(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    const ElementOutput* bias,
    int M, int N, int K,
    cudaStream_t stream,
    bool use_cached);

} // namespace kernel
} // namespace trt_edgellm
