#include <iostream>
#include <vector>
#include <chrono>
#include <map>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/kernel/gemv.h>
#include <cutlass/gemm/device/gemv.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/numeric_types.h>
#include <cmath>
#include <type_traits>
#include "check/err_analysis.h"
#include "utils/cuda_check.h"
#include "gemv_bench.h"


template<typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
void compute_gemv_cpu_reference(
    std::vector<ElementC>& h_ref,
    const std::vector<ElementA>& h_A,
    const std::vector<ElementB>& h_B,
    const std::vector<ElementC>& h_C,
    int M, int K,
    double alpha, double beta,
    int max_elements = MAX_COMPARE_COUNT) {

    // 计算实际需要验证的元素数量
    int total_elements = M;
    int verify_count = std::min(total_elements, max_elements);

    // 初始化参考结果向量
    h_ref.resize(total_elements);

    // 只计算前verify_count个元素
    for (int m = 0; m < verify_count; ++m) {
        ElementAccumulator accum = 0.0;

        // 计算矩阵向量乘法部分
        for (int k = 0; k < K; ++k) {
            ElementAccumulator a_val, b_val;

            if constexpr (std::is_same<ElementA, cutlass::half_t>::value) {
                a_val = static_cast<ElementAccumulator>(h_A[m * K + k]);
                b_val = static_cast<ElementAccumulator>(h_B[k]);
            } else {
                a_val = static_cast<ElementAccumulator>(h_A[m * K + k]);
                b_val = static_cast<ElementAccumulator>(h_B[k]);
            }

            accum += a_val * b_val;
        }

        // 加上alpha和beta的影响
        ElementAccumulator c_val;
        if constexpr (std::is_same<ElementC, cutlass::half_t>::value) {
            c_val = static_cast<ElementAccumulator>(h_C[m]);
        } else {
            c_val = static_cast<ElementAccumulator>(h_C[m]);
        }

        accum = alpha * accum + beta * c_val;

        // 存储结果
        if constexpr (std::is_same<ElementC, cutlass::half_t>::value) {
            h_ref[m] = cutlass::half_t(static_cast<float>(accum));
        } else {
            h_ref[m] = static_cast<ElementC>(accum);
        }
    }

    // 对于未计算的部分，可以设为0或者保持未初始化状态
    // 这里我们不做特殊处理，因为在误差分析时只会检查计算过的部分
}

// 模板化的GEMV测试函数
template<typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
TestResult benchmark_gemv_template(int M, int K, int iterations = 100,
                                   double alpha = 1.0, double beta = 0.0) {

    TestResult result;
    result.M = M;
    result.N = 1;  // GEMV的输出是向量，所以N=1
    result.K = K;
    result.operation = "GEMV";

    if constexpr (std::is_same<ElementA, cutlass::half_t>::value) {
        result.data_type = "FP16";
    } else if constexpr (std::is_same<ElementA, float>::value) {
        result.data_type = "FP32";
    }

    // 定义数据类型别名
    using ElementInputA = ElementA;
    using ElementInputB = ElementB;
    using ElementOutput = ElementC;

    // 定义布局 - A矩阵行主序，B向量列主序，C向量列主序
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::ColumnMajor;

    // 定义计算相关类型
    using ElementComputeEpilogue = ElementAccumulator;

    std::ostringstream oss;
    oss << "(" << M << "," << K << ")*(" << K << ",1)";
    result.test_case = oss.str();

    // 创建主机端数据
    std::vector<ElementInputA> h_A(M * K);
    std::vector<ElementInputB> h_B(K);
    std::vector<ElementOutput> h_C(M);
    std::vector<ElementOutput> h_ref(M);

    // 初始化数据
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = random_value<ElementInputA>();
    }
    for (int i = 0; i < K; ++i) {
        h_B[i] = random_value<ElementInputB>();
    }
    for (int i = 0; i < M; ++i) {
        h_C[i] = random_value<ElementOutput>();
    }

    // 设备端内存分配
    ElementInputA* d_A = nullptr;
    ElementInputB* d_B = nullptr;
    ElementOutput* d_C = nullptr;
    ElementOutput* d_D = nullptr;

    size_t size_A = sizeof(ElementInputA) * M * K;
    size_t size_B = sizeof(ElementInputB) * K;
    size_t size_C = sizeof(ElementOutput) * M;

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMalloc(&d_D, size_C));

    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), size_C, cudaMemcpyHostToDevice));

    // 根据数据类型选择不同的配置
    if constexpr (std::is_same<ElementA, cutlass::half_t>::value) {
        // 对于半精度，使用特定的配置
        const int kElementsPerAccess = 8;  // half_t类型，8个元素对齐

        // 定义Epilogue操作
        using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
            ElementOutput,
            1,
            ElementAccumulator,
            ElementComputeEpilogue>;

        // 定义GEMV内核
        using GemvKernel = cutlass::gemm::kernel::Gemv<
            ElementInputA,           // Element A
            LayoutInputA,            // Layout A
            ElementInputB,           // Element B
            ElementOutput,           // Element C
            ElementAccumulator,      // Element accumulator
            EpilogueOp,              // Output operator
            kElementsPerAccess       // Element access granularity
        >;

        // 定义设备级GEMV操作
        using GemvDevice = cutlass::gemm::device::Gemv<GemvKernel>;

        // 创建GEMV操作符
        GemvDevice gemv_op;

        // 创建参数结构
        typename GemvDevice::Arguments arguments(
            {M, K},                     // 问题尺寸 (m, k)
            1,                          // batch_count
            {ElementAccumulator(alpha), ElementAccumulator(beta)},  // alpha, beta
            {d_A, K},                   // TensorRef for A: 指针和步长
            d_B,                        // B向量指针
            d_C,                        // C向量指针
            d_D,                        // D向量指针
            M * K,                      // batch_stride_A
            K,                          // batch_stride_B
            M,                          // batch_stride_C
            M                           // batch_stride_D
        );

        // 检查是否可以实现
        cutlass::Status status = GemvDevice::can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "Cannot implement GEMV operation: " << cutlass::cutlassGetStatusString(status) << std::endl;
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_C));
            CUDA_CHECK(cudaFree(d_D));
            result.passed = false;
            return result;
        }

        // 获取工作空间大小并分配
        size_t workspace_size = GemvDevice::get_workspace_size(arguments);
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        // 初始化操作符
        status = gemv_op.initialize(arguments, workspace.get());
        CUTLASS_CHECK(status);

        // 预热运行
        status = gemv_op();
        CUTLASS_CHECK(status);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 性能测试
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        double total_time_ms = 0.0;
        double min_time_ms = std::numeric_limits<double>::max();
        double max_time_ms = 0.0;

        for (int i = 0; i < iterations; ++i) {
            CUDA_CHECK(cudaEventRecord(start));

            status = gemv_op();
            CUTLASS_CHECK(status);

            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

            total_time_ms += elapsed_ms;
            min_time_ms = std::min(min_time_ms, static_cast<double>(elapsed_ms));
            max_time_ms = std::max(max_time_ms, static_cast<double>(elapsed_ms));
        }

        result.avg_time_ms = total_time_ms / iterations;
        result.min_time_ms = min_time_ms;
        result.max_time_ms = max_time_ms;

        // 计算性能指标
        double flops = 2.0 * M * K;  // GEMV的FLOPs计数: 2MK
        result.avg_tflops = (flops / result.avg_time_ms) / 1e9;
        result.min_tflops = (flops / result.max_time_ms) / 1e9;  // 最小时间对应最大性能
        result.max_tflops = (flops / result.min_time_ms) / 1e9;  // 最大时间对应最小性能

        // 带宽计算
        size_t bytes_transferred = (M * K + K + 2 * M) * sizeof(ElementInputA);
        result.avg_bandwidth_gbs = (bytes_transferred / result.avg_time_ms) / 1e6;

        // 清理事件
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

    } else if constexpr (std::is_same<ElementA, float>::value) {
        // 对于单精度，使用特定的配置
        const int kElementsPerAccess = 4;  // float类型，4个元素对齐

        // 定义Epilogue操作
        using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
            ElementOutput,
            1,
            ElementAccumulator,
            ElementComputeEpilogue>;

        // 定义GEMV内核
        using GemvKernel = cutlass::gemm::kernel::Gemv<
            ElementInputA,           // Element A
            LayoutInputA,            // Layout A
            ElementInputB,           // Element B
            ElementOutput,           // Element C
            ElementAccumulator,      // Element accumulator
            EpilogueOp,              // Output operator
            kElementsPerAccess       // Element access granularity
        >;

        // 定义设备级GEMV操作
        using GemvDevice = cutlass::gemm::device::Gemv<GemvKernel>;

        // 创建GEMV操作符
        GemvDevice gemv_op;

        // 创建参数结构
        typename GemvDevice::Arguments arguments(
            {M, K},                     // 问题尺寸 (m, k)
            1,                          // batch_count
            {ElementAccumulator(alpha), ElementAccumulator(beta)},  // alpha, beta
            {d_A, K},                   // TensorRef for A: 指针和步长
            d_B,                        // B向量指针
            d_C,                        // C向量指针
            d_D,                        // D向量指针
            M * K,                      // batch_stride_A
            K,                          // batch_stride_B
            M,                          // batch_stride_C
            M                           // batch_stride_D
        );

        // 检查是否可以实现
        cutlass::Status status = GemvDevice::can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "Cannot implement GEMV operation: " << cutlass::cutlassGetStatusString(status) << std::endl;
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_C));
            CUDA_CHECK(cudaFree(d_D));
            result.passed = false;
            return result;
        }

        // 获取工作空间大小并分配
        size_t workspace_size = GemvDevice::get_workspace_size(arguments);
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        // 初始化操作符
        status = gemv_op.initialize(arguments, workspace.get());
        CUTLASS_CHECK(status);

        // 预热运行
        status = gemv_op();
        CUTLASS_CHECK(status);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 性能测试
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        double total_time_ms = 0.0;
        double min_time_ms = std::numeric_limits<double>::max();
        double max_time_ms = 0.0;

        for (int i = 0; i < iterations; ++i) {
            CUDA_CHECK(cudaEventRecord(start));

            status = gemv_op();
            CUTLASS_CHECK(status);

            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

            total_time_ms += elapsed_ms;
            min_time_ms = std::min(min_time_ms, static_cast<double>(elapsed_ms));
            max_time_ms = std::max(max_time_ms, static_cast<double>(elapsed_ms));
        }

        result.avg_time_ms = total_time_ms / iterations;
        result.min_time_ms = min_time_ms;
        result.max_time_ms = max_time_ms;

        // 计算性能指标
        double flops = 2.0 * M * K;  // GEMV的FLOPs计数: 2MK
        result.avg_tflops = (flops / result.avg_time_ms) / 1e9;
        result.min_tflops = (flops / result.max_time_ms) / 1e9;  // 最小时间对应最大性能
        result.max_tflops = (flops / result.min_time_ms) / 1e9;  // 最大时间对应最小性能

        // 带宽计算
        size_t bytes_transferred = (M * K + K + 2 * M) * sizeof(ElementInputA);
        result.avg_bandwidth_gbs = (bytes_transferred / result.avg_time_ms) / 1e6;

        // 清理事件
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    // 使用封装的CPU参考计算函数
    compute_gemv_cpu_reference<ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>(
        h_ref, h_A, h_B, h_C, M, K, alpha, beta, MAX_COMPARE_COUNT);

    // 获取GPU结果
    std::vector<ElementOutput> h_D(M);
    CUDA_CHECK(cudaMemcpy(h_D.data(), d_D, size_C, cudaMemcpyDeviceToHost));

    // 使用公共函数进行误差分析
    // 根据数据类型设置不同的容差
    double abs_tolerance = 1e-3;
    double rel_tolerance = 1e-3;

    if constexpr (std::is_same<ElementA, cutlass::half_t>::value) {
        abs_tolerance = 1e-2;
        rel_tolerance = 1e-2;
    }

    // 计算实际验证的元素数量
    result.verify_count = std::min(M, MAX_COMPARE_COUNT);

    auto error_result = analyze_errors(h_D, h_ref, 0, result.verify_count, abs_tolerance, rel_tolerance);

    result.max_abs_error = error_result.max_abs_error;
    result.max_rel_error = error_result.max_rel_error;
    result.passed = error_result.passed;
    result.error_count = error_result.error_count;
    result.total_count = error_result.total_count;

    // 清理设备内存
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D));

    return result;
}

TestResult benchmark_gemv_half(int M, int K, int iterations) {
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;

    return benchmark_gemv_template<ElementA, ElementB, ElementC, ElementAccumulator>(
        M, K, iterations, 1.0, 0.0);
}

TestResult benchmark_gemv_float(int M, int K, int iterations) {
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = float;

    return benchmark_gemv_template<ElementA, ElementB, ElementC, ElementAccumulator>(
        M, K, iterations, 1.0, 0.0);
}
