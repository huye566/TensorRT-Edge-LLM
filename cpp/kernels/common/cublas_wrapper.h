#ifndef CUBLAS_WRAPPER_H
#define CUBLAS_WRAPPER_H

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

namespace trt_edgellm {
namespace kernel {

enum class MemoryFormat {
    ROW_MAJOR,
    COL_MAJOR
};

class CublasWrapper {
public:
    static CublasWrapper& instance() {
        static CublasWrapper instance;
        return instance;
    }

    cublasHandle_t handle() const { return handle_; }
    bool initialize();
    void cleanup();
    void set_math_mode(bool use_tensor_core);
    bool using_tensor_core() const { return use_tensor_core_; }

private:
    CublasWrapper();
    ~CublasWrapper();
    
    CublasWrapper(const CublasWrapper&) = delete;
    CublasWrapper& operator=(const CublasWrapper&) = delete;
    
    cublasHandle_t handle_;
    bool use_tensor_core_;
    bool initialized_;
};

template<typename T>
void cublas_gemm(cublasHandle_t handle,
                int m, int n, int k,
                const T* A,
                const T* B,
                T* C,
                MemoryFormat format = MemoryFormat::ROW_MAJOR);

template<typename T>
void cublas_gemm_silu(cublasHandle_t handle,
                     int m, int n, int k,
                     const T* A,
                     const T* B,
                     T* C);

template<typename T>
void cublas_gemm_bias(cublasHandle_t handle,
                     int m, int n, int k,
                     const T* A,
                     const T* B,
                     const T* bias,  // bias向量，长度为n
                     T* C);

} // namespace kernel
} // namespace trt_edgellm

#endif // CUBLAS_WRAPPER_H