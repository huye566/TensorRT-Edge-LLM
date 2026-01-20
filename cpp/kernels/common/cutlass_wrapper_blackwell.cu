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


// CUTLASS错误检查宏
#define CUTLASS_CHECK(status)                                               \
    do {                                                                    \
        if (status != cutlass::Status::kSuccess) {                          \
            std::cerr << "CUTLASS error at " << __FILE__ << ":" << __LINE__ \
                     << " code: " << static_cast<int>(status) << std::endl; \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while(0)

namespace trt_edgellm {
namespace kernel {

template <bool kEnableSilu, bool kEnableBias>
void cutlass_gemm_blackwell(ElementOutput* output,
                            const ElementInputA* input,
                            const ElementInputB* weights,
                            const ElementOutput* bias,
                            int M, int N, int K,
                            cudaStream_t stream) {

    // using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
    // using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
    // using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
    // using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
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


    // using MainloopSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;
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
    
    typename Gemm::Arguments arguments{
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

    CUTLASS_CHECK(gemm.run());

}

template void cutlass_gemm_blackwell<false, false>(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    const ElementOutput* bias,
    int M, int N, int K,
    cudaStream_t stream);

template void cutlass_gemm_blackwell<true, false>(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    const ElementOutput* bias,
    int M, int N, int K,
    cudaStream_t stream);

template void cutlass_gemm_blackwell<false, true>(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    const ElementOutput* bias,
    int M, int N, int K,
    cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm

