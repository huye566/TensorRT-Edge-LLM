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
using         ElementInputB    = cutlass::half_t;
using         ElementOutput    = cutlass::half_t;

constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementInputA>::value;
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementInputB>::value;
constexpr int AlignmentOutput  = 128 / cutlass::sizeof_bits<ElementOutput>::value;

using ElementAccumulator  = float;
using ArchTag             = cutlass::arch::Sm100;
using OperatorClass       = cutlass::arch::OpClassTensorOp;


template <bool kEnableSilu = false, bool kEnableBias = false>
void cutlass_gemm_blackwell(ElementOutput* output,
                            const ElementInputA* input,
                            const ElementInputB* weights,
                            const ElementOutput* bias,
                            int M, int N, int K,
                            cudaStream_t stream = 0,
                            bool use_cached = false);

template <bool kEnableSilu = false, bool kEnableBias = false>
void cutlass_gemv_blackwell(ElementOutput* output,
                            const ElementInputA* input,
                            const ElementInputB* weights,
                            const ElementOutput* bias,
                            int M, int N, int K,
                            cudaStream_t stream = 0,
                            bool use_cached = false);

template <bool kEnableSilu, bool kEnableBias>
void cutlass_blackwell_dispatch(ElementOutput* output,
                            const ElementInputA* input,
                            const ElementInputB* weights,
                            const ElementOutput* bias,
                            int M, int N, int K,
                            cudaStream_t stream = 0,
                            bool use_cached = false);

} // namespace kernel
} // namespace trt_edgellm


#endif // CUTLASS_WRAPPER_BLACKWELL_H