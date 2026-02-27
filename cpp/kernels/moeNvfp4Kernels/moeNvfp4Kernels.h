#ifndef MOE_NVFP4_KERNELS_H__
#define MOE_NVFP4_KERNELS_H__

#include <cutlass/half.h>
#include "moeNvfp4InputsParams.h"

namespace trt_edgellm {
namespace kernel {
namespace moe_nvfp4 {

void moe_nvfp4_forward_cuda(const MoeNvfp4InputsParams& params);

} // moe_nvfp4
} // namespace kernel
} // namespace trt_edgellm

#endif // MOE_NVFP4_KERNELS_H__