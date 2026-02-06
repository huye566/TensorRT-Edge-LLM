#ifndef ERR_ANALYSIS_H_
#define ERR_ANALYSIS_H_
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

inline const char* cublasGetStatusString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS:           return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:   return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:      return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:     return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:     return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:     return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:  return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:    return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:     return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:     return "CUBLAS_STATUS_LICENSE_ERROR";
        default:                              return "Unknown cuBLAS error";
    }
}

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = (call); \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cublasGetStatusString(status) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define PRINT_BASE_ELE_NUM 5

template<typename T> void printMatrix(int rowCount, int colCount, const T* matrix) {
   for (int i = 0; i < rowCount; i++) {
      for (int j = 0; j < colCount; j++) {
         std::cout << matrix[j * colCount + i] << "\t";
      }
      std::cout << std::endl;
   }
}

template<typename T>
struct ErrorAnalysisResult {
    bool passed;
    double max_abs_error;
    double max_rel_error;
    int error_count;
    int total_count;
};

template<typename T>
ErrorAnalysisResult<T> analyze_errors(
    const std::vector<T>& computed,
    const std::vector<T>& reference,
    int start_idx = 0,
    int count = -1,
    double abs_tolerance = 1e-3,
    double rel_tolerance = 1e-3) {

    ErrorAnalysisResult<T> result;
    result.passed = true;
    result.max_abs_error = 0.0;
    result.max_rel_error = 0.0;
    result.error_count = 0;

    // 如果count为-1，则比较所有元素
    int total_elements = computed.size();
    if (count == -1) {
        count = total_elements - start_idx;
    }

    // 确保索引范围有效
    int end_idx = std::min(start_idx + count, total_elements);
    result.total_count = end_idx - start_idx;

    for (int i = start_idx; i < end_idx; ++i) {
        // 转换为double进行比较
        double computed_f = 0.0;
        double reference_f = 0.0;

        if constexpr (std::is_same<T, half>::value) {
            computed_f = static_cast<double>(computed[i]);
            reference_f = static_cast<double>(reference[i]);
        } else {
            computed_f = static_cast<double>(computed[i]);
            reference_f = static_cast<double>(reference[i]);
        }

        double abs_error = std::abs(computed_f - reference_f);
        double rel_error = (std::abs(reference_f) > 1e-9) ?
                          abs_error / std::abs(reference_f) : abs_error;

        result.max_abs_error = std::max(result.max_abs_error, abs_error);
        result.max_rel_error = std::max(result.max_rel_error, rel_error);

        // 检查是否超出容差
        if (abs_error > abs_tolerance && rel_error > rel_tolerance) {
            result.error_count++;
            if (result.error_count <= 5) {
                std::cout << "  Err[" << i << "]: GPU=" << computed_f
                          << ", REF=" << reference_f
                          << ", abs_err=" << abs_error
                          << ", rel_err=" << rel_error << std::endl;
            }
        } else {
            if (i < PRINT_BASE_ELE_NUM) {
                std::cout << "  Check[" << i << "]: GPU=" << computed_f
                          << ", REF=" << reference_f
                          << ", abs_err=" << abs_error
                          << ", rel_err=" << rel_error << std::endl;
            }
        }
    }

    result.passed = (result.error_count == 0);
    return result;
}

template<typename T>
T random_value() {
    if constexpr (std::is_same<T, half>::value) {
        float val = static_cast<float>(rand() % 100) / 100.0f;
        return half(val);
    } else if constexpr (std::is_same<T, float>::value) {
        return static_cast<T>(rand() % 100) / 100.0f;
    } else {
        return static_cast<T>(rand() % 100) / 100.0f;
    }
}

#endif // ERR_ANALYSIS_H_