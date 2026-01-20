#ifndef CUTLASS_WRAPPER_BLACKWELL_H
#define CUTLASS_WRAPPER_BLACKWELL_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

#include <cutlass/half.h>
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/layout.h"


namespace trt_edgellm {
namespace kernel {

using namespace cute;
using         ElementInputA    = cutlass::half_t;
using         LayoutInputA     = cutlass::layout::RowMajor;
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementInputA>::value;

using         ElementInputB    = cutlass::half_t;
using         LayoutInputB     = cutlass::layout::RowMajor;
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementInputB>::value;

using         ElementOutput    = cutlass::half_t;
using         LayoutOutput     = cutlass::layout::RowMajor;
constexpr int AlignmentOutput  = 128 / cutlass::sizeof_bits<ElementOutput>::value;

using ElementAccumulator  = float;
using ArchTag             = cutlass::arch::Sm100;
using OperatorClass       = cutlass::arch::OpClassTensorOp;

using MmaTileShape_MNK = Shape<_256,_128,_64>;
using ClusterShape_MNK = Shape<_2,_2,_1>;

template <bool kEnableSilu = false, bool kEnableBias = false>
void cutlass_gemm_blackwell(ElementOutput* output,
                            const ElementInputA* input,
                            const ElementInputB* weights,
                            const ElementOutput* bias,
                            int M, int N, int K,
                            cudaStream_t stream = 0);


} // namespace kernel
} // namespace trt_edgellm


#endif // CUTLASS_WRAPPER_BLACKWELL_H