#include "cutlass_wrapper.h"

#include <algorithm>
#include <iostream>
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

#include "cutlass/gemm/kernel/gemv.h"
#include "cutlass/gemm/device/gemv.h"

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

template <int Ma, int Na, int Ca, int Stages>
void cutlass_gemm(ElementOutput* output,
                  const ElementInputA* input,
                  const ElementInputB* weights,
                  int M, int N, int K,
                  cudaStream_t stream) {
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        Ca,
        // 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementComputeEpilogue>;
    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        MMAOp, SmArch,
        ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        Stages, Ma, Na>;

    cutlass::gemm::GemmCoord problem_size(M, N, K);
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);

    int lda = K;
    int ldb = N;
    int ldd = N;

    cutlass::TensorRef<ElementInputA, LayoutInputA> input_device_ref(
        const_cast<ElementInputA*>(input), LayoutInputA(lda));
    cutlass::TensorRef<ElementInputB, LayoutInputB> weights_device_ref(
        const_cast<ElementInputB*>(weights), LayoutInputB(ldb));
    cutlass::TensorRef<ElementOutput, LayoutOutput> output_device_ref(
        output, LayoutOutput(ldd));

    typename Gemm::Arguments arguments{
        problem_size,
        input_device_ref,
        weights_device_ref,
        {nullptr, 0},
        output_device_ref,
        {alpha}};

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    status = gemm_op(stream);
    CUTLASS_CHECK(status);
}

template <int Ma, int Na, int Ca, int Stages>
void cutlass_gemm_silu(ElementOutput* output,
                       const ElementInputA* input,
                       const ElementInputB* weights,
                       int M, int N, int K,
                       cudaStream_t stream) {
    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationSilu<
        ElementOutput,
        Ca,
        ElementAccumulator,
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        MMAOp, SmArch,
        ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        Stages, Ma, Na>;

    cutlass::gemm::GemmCoord problem_size(M, N, K);
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);

    int split_k_slices = 1;
    int lda = K;
    int ldb = N;
    int ldd = N;

    cutlass::TensorRef<ElementInputA, LayoutInputA> input_device_ref(
        const_cast<ElementInputA*>(input), LayoutInputA(lda));
    cutlass::TensorRef<ElementInputB, LayoutInputB> weights_device_ref(
        const_cast<ElementInputB*>(weights), LayoutInputB(ldb));
    cutlass::TensorRef<ElementOutput, LayoutOutput> output_device_ref(
        output, LayoutOutput(ldd));

    typename Gemm::Arguments arguments{
        problem_size,
        input_device_ref,
        weights_device_ref,
        {nullptr, 0},
        output_device_ref,
        {alpha},
        split_k_slices};

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    status = gemm_op(stream);
    CUTLASS_CHECK(status);
}

template <int Ma, int Na, int Ca, int Stages>
void cutlass_gemm_bias(ElementOutput* output,
                       const ElementInputA* input,
                       const ElementInputB* weights,
                       const ElementOutput* bias,
                       int M, int N, int K,
                       cudaStream_t stream) {
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        Ca,
        ElementAccumulator,
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        MMAOp, SmArch,
        ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        Stages, Ma, Na>;

    cutlass::gemm::GemmCoord problem_size(M, N, K);
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);

    int lda = K;
    int ldb = N;
    int ldd = N;

    cutlass::TensorRef<ElementInputA, LayoutInputA> input_device_ref(
        const_cast<ElementInputA*>(input), LayoutInputA(lda));
    cutlass::TensorRef<ElementInputB, LayoutInputB> weights_device_ref(
        const_cast<ElementInputB*>(weights), LayoutInputB(ldb));
    cutlass::TensorRef<ElementOutput, LayoutOutput> output_device_ref(
        output, LayoutOutput(ldd));

    typename Gemm::Arguments arguments{
        problem_size,
        input_device_ref,
        weights_device_ref,
        {const_cast<ElementOutput*>(bias), 0},
        output_device_ref,
        {alpha}};

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    status = gemm_op(stream);
    CUTLASS_CHECK(status);
}


template <int M1, int N1, int K1, int M2, int N2, int K2, bool kEnableSilu>
void cutlass_gemm_grouped(ElementOutput* output,
                          ElementInputA* input,
                          ElementInputB* weight,
                          std::vector<cutlass::gemm::GemmCoord>& problem_sizes,
                          cudaStream_t stream) {
    int problem_count = static_cast<int>(problem_sizes.size());

    // 计算总元素数和偏移
    int64_t total_elements_A = 0;
    int64_t total_elements_B = 0;
    int64_t total_elements_D = 0;

    std::vector<int64_t> offset_A, offset_B, offset_D;
    std::vector<int64_t> lda_host, ldb_host, ldd_host;

    lda_host.resize(problem_count);
    ldb_host.resize(problem_count);
    ldd_host.resize(problem_count);

    for (int i = 0; i < problem_count; ++i) {
        auto problem = problem_sizes[i];

        lda_host[i] = LayoutInputA::packed({problem.m(), problem.k()}).stride(0);
        ldb_host[i] = LayoutInputB::packed({problem.k(), problem.n()}).stride(0);
        ldd_host[i] = LayoutOutput::packed({problem.m(), problem.n()}).stride(0);

        offset_A.push_back(total_elements_A);
        offset_B.push_back(total_elements_B);
        offset_D.push_back(total_elements_D);

        total_elements_A += problem.m() * problem.k();
        total_elements_B += problem.k() * problem.n();
        total_elements_D += problem.m() * problem.n();
    }

    // 分配设备内存
    cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problem_sizes_device(problem_count);
    cutlass::DeviceAllocation<int64_t> lda(problem_count);
    cutlass::DeviceAllocation<int64_t> ldb(problem_count);
    cutlass::DeviceAllocation<int64_t> ldd(problem_count);

    cutlass::DeviceAllocation<ElementInputA*> ptr_A(problem_count);
    cutlass::DeviceAllocation<ElementInputB*> ptr_B(problem_count);
    cutlass::DeviceAllocation<ElementOutput*> ptr_D(problem_count);

    // 拷贝数据到设备
    problem_sizes_device.copy_from_host(problem_sizes.data());
    lda.copy_from_host(lda_host.data());
    ldb.copy_from_host(ldb_host.data());
    ldd.copy_from_host(ldd_host.data());

    // 准备指针数组
    std::vector<ElementInputA*> ptr_A_host(problem_count);
    std::vector<ElementInputB*> ptr_B_host(problem_count);
    std::vector<ElementOutput*> ptr_D_host(problem_count);

    for (int i = 0; i < problem_count; ++i) {
        ptr_A_host[i] = input + offset_A[i];
        ptr_B_host[i] = weight + offset_B[i];
        ptr_D_host[i] = output + offset_D[i];
    }

    ptr_A.copy_from_host(ptr_A_host.data());
    ptr_B.copy_from_host(ptr_B_host.data());
    ptr_D.copy_from_host(ptr_D_host.data());

    // 定义Epilogue类型
    using EpilogueOpSilu = cutlass::epilogue::thread::LinearCombinationSilu<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementAccumulator>;

    using EpilogueType = typename std::conditional<
        kEnableSilu,
        EpilogueOpSilu,
        EpilogueOp>::type;

    // 定义GEMM核和分组GEMM
    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        ElementInputA, LayoutInputA, cutlass::ComplexTransform::kNone, 8,
        ElementInputB, LayoutInputB, cutlass::ComplexTransform::kNone, 8,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        MMAOp, SmArch,
        cutlass::gemm::GemmShape<M1, N1, K1>,
        cutlass::gemm::GemmShape<M2, N2, K2>,
        ShapeMMAOp,
        EpilogueType,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        3,
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;

    using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    float alpha = 1.0f;
    float beta = 0.0f;
    typename Gemm::EpilogueOutputOp::Params epilogue_op(alpha, beta);

    int threadblock_count = Gemm::sufficient(problem_sizes.data(), problem_count);
    if (!threadblock_count) {
        std::cerr << "Insufficient hardware resources for CUTLASS Grouped GEMM" << std::endl;
        return;
    }

    typename Gemm::Arguments args(
        problem_sizes_device.get(), problem_count, threadblock_count,
        epilogue_op,
        ptr_A.get(), ptr_B.get(), ptr_D.get(), ptr_D.get(),
        lda.get(), ldb.get(), ldd.get(), ldd.get(),
        problem_sizes.data());

    Gemm gemm;
    size_t workspace_size = gemm.get_workspace_size(args);
    cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm.initialize(args, workspace.get());
    CUTLASS_CHECK(status);

    status = gemm.run(stream);
    CUTLASS_CHECK(status);
}

template void cutlass_gemm<8, 1, 1, 2>(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    int M, int N, int K,
    cudaStream_t stream);

template void cutlass_gemm<8, 8, 8, 3>(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    int M, int N, int K,
    cudaStream_t stream);

template void cutlass_gemm_silu<8, 1, 1, 2>(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    int M, int N, int K,
    cudaStream_t stream);

template void cutlass_gemm_silu<8, 8, 8, 3>(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    int M, int N, int K,
    cudaStream_t stream);

template void cutlass_gemm_bias<8, 1, 1, 2>(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    const ElementOutput* bias,
    int M, int N, int K,
    cudaStream_t stream);

template void cutlass_gemm_bias<8, 8, 8, 3>(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    const ElementOutput* bias,
    int M, int N, int K,
    cudaStream_t stream);

template void cutlass_gemm_grouped<128, 128, 32, 64, 64, 32, true>(
    ElementOutput* output,
    ElementInputA* input,
    ElementInputB* weight,
    std::vector<cutlass::gemm::GemmCoord>& problem_sizes,
    cudaStream_t stream);

template void cutlass_gemm_grouped<128, 128, 32, 64, 64, 32, false>(
    ElementOutput* output,
    ElementInputA* input,
    ElementInputB* weight,
    std::vector<cutlass::gemm::GemmCoord>& problem_sizes,
    cudaStream_t stream);


template <bool kEnableSilu, bool kEnableBias>
void cutlass_gemm_standard(ElementOutput* output,
                            const ElementInputA* input,
                            const ElementInputB* weights,
                            const ElementOutput* bias,
                            int M, int N, int K,
                            cudaStream_t stream) {

    // 基元类型别名
    using cutlass_half_t = cutlass::half_t;
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;
    using ElementInputA = cutlass_half_t;
    using ElementInputB = cutlass_half_t;
    using ElementOutput = cutlass_half_t;

    // 内存布局
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::RowMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    // 计算架构配置
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;

    // 线程块和warp配置
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
    using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;


    using DefaultEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementAccumulator>;

    using SiluEpilogueOp = cutlass::epilogue::thread::LinearCombinationSilu<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementAccumulator,
        cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;

    using BiasEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementAccumulator,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

    using EpilogueOp = std::conditional_t<
        kEnableSilu,
        SiluEpilogueOp,
        std::conditional_t<kEnableBias, BiasEpilogueOp, DefaultEpilogueOp>
    >;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        MMAOp, SmArch,
        ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3>;

    cutlass::gemm::GemmCoord problem_size(M, N, K);
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);

    int lda = K;
    int ldb = N;
    int ldd = N;

    cutlass::TensorRef<ElementInputA, LayoutInputA> input_device_ref(
        const_cast<ElementInputA*>(input), LayoutInputA(lda));
    cutlass::TensorRef<ElementInputB, LayoutInputB> weights_device_ref(
        const_cast<ElementInputB*>(weights), LayoutInputB(ldb));
    cutlass::TensorRef<ElementOutput, LayoutOutput> output_device_ref(
        output, LayoutOutput(ldd));

    typename Gemm::Arguments arguments{
        problem_size,
        input_device_ref,
        weights_device_ref,
        {const_cast<ElementOutput*>(bias), 0},
        output_device_ref,
        {alpha}};

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    status = gemm_op(stream);
    CUTLASS_CHECK(status);
}

template void cutlass_gemm_standard<false, false>(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    const ElementOutput* bias,
    int M, int N, int K,
    cudaStream_t stream);

template void cutlass_gemm_standard<true, false>(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    const ElementOutput* bias,
    int M, int N, int K,
    cudaStream_t stream);

template void cutlass_gemm_standard<false, true>(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    const ElementOutput* bias,
    int M, int N, int K,
    cudaStream_t stream);


template <bool kEnableSilu, bool kEnableBias>
void cutlass_gemv(ElementOutput* output,
                  const ElementInputA* input,
                  const ElementInputB* weights,
                  const ElementOutput* bias,
                  int M, int N, int K,
                  cudaStream_t stream)
{
    using ElementInput = cutlass::half_t;
    using ElementOutput = cutlass::half_t;
    // using LayoutA = cutlass::layout::ColumnMajor;
    using LayoutA = cutlass::layout::RowMajor;
    using ElementAccumulator = float;
    int const kElementsPerAccess = 8;

    using DefaultEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        1,
        ElementAccumulator,
        ElementAccumulator>;

    using SiluEpilogueOp = cutlass::epilogue::thread::LinearCombinationSilu<
        ElementOutput,
        1,
        ElementAccumulator,
        ElementAccumulator,
        cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;

    using BiasEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        1,
        ElementAccumulator,
        ElementAccumulator,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

    using EpilogueOp = std::conditional_t<
        kEnableSilu,
        SiluEpilogueOp,
        std::conditional_t<kEnableBias, BiasEpilogueOp, DefaultEpilogueOp>
    >;


    using Gemv = cutlass::gemm::device::Gemv<
        cutlass::gemm::kernel::Gemv<
            ElementInput,           // Element A
            LayoutA,                // Layout A
            ElementInput,           // Element B
            ElementOutput,          // Element C
            ElementAccumulator,     // Element accumulator
            EpilogueOp,             // Output operator
            kElementsPerAccess      // Element access granularity
            >
        >;

    float alpha = 1.f;
    float beta = 0.f;

    // cutlass::TensorRef<ElementInput, LayoutA> input_device_ref(
    //     const_cast<ElementInput*>(weights), LayoutA(N)); //column

    cutlass::TensorRef<ElementInput, LayoutA> input_device_ref(
        const_cast<ElementInput*>(weights), LayoutA(K)); //row

    typename Gemv::Arguments arguments{
      {N,K},
      1,
      {alpha, beta},
      input_device_ref,
      input,
      bias,
      output,
      K*N,
      K,
      N,
      N
    };

    Gemv gemv;
    CUTLASS_CHECK(gemv.can_implement(arguments));

    size_t workspace_size = Gemv::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemv.initialize(arguments, workspace.get()));

    CUTLASS_CHECK(gemv.run(stream));

}


template void cutlass_gemv<false, false>(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    const ElementOutput* bias,
    int M, int N, int K,
    cudaStream_t stream);

template void cutlass_gemv<true, false>(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    const ElementOutput* bias,
    int M, int N, int K,
    cudaStream_t stream);

template void cutlass_gemv<false, true>(
    ElementOutput* output,
    const ElementInputA* input,
    const ElementInputB* weights,
    const ElementOutput* bias,
    int M, int N, int K,
    cudaStream_t stream);


} // namespace kernel
} // namespace trt_edgellm