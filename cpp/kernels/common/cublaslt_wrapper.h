#ifndef CUBLASLT_WRAPPER_H
#define CUBLASLT_WRAPPER_H

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <memory>
#include <algorithm>

namespace trt_edgellm {
namespace kernel {

enum class MemoryFormat {
    ROW_MAJOR,
    COL_MAJOR
};

enum class ComputeType {
    FLOAT,
    HALF,
    TFLOAT32
};

class CublasLtWrapper {
public:
    static CublasLtWrapper& instance() {
        static CublasLtWrapper instance;
        return instance;
    }

    cublasLtHandle_t handle() const { return handle_; }
    bool initialize();
    void cleanup();

    void set_preference(int max_workspace_size = 32 * 1024 * 1024);

    void* workspace() const { return workspace_; }
    size_t max_workspace_size() const { return max_workspace_size_; }

private:
    CublasLtWrapper();
    ~CublasLtWrapper();

    CublasLtWrapper(const CublasLtWrapper&) = delete;
    CublasLtWrapper& operator=(const CublasLtWrapper&) = delete;

    cublasLtHandle_t handle_;
    bool initialized_;

    size_t max_workspace_size_;
    void* workspace_;
};

// GEMM描述符类
template<typename T>
struct GemmDescriptor {
    cublasLtMatrixLayout_t A_desc = nullptr;
    cublasLtMatrixLayout_t B_desc = nullptr;
    cublasLtMatrixLayout_t C_desc = nullptr;
    cublasLtMatmulDesc_t operation_desc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;

    void create(int m, int n, int k, MemoryFormat format, cublasLtHandle_t handle);
    void destroy();
};

// 基础GEMM接口
template<typename T, bool kEnableSilu = false, bool kEnableBias = false>
bool cublaslt_gemm(cublasLtHandle_t handle,
    int m, int n, int k, const T* A, const T* B, const T* bias,
    T* C, MemoryFormat format, ComputeType compute_type, cudaStream_t stream = 0);

} // namespace kernel
} // namespace trt_edgellm

#endif // CUBLASLT_WRAPPER_H
