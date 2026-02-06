#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/epilogue/thread/linear_combination_silu.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/numeric_types.h>
#include <cmath>
#include <type_traits>
#include <iomanip>

#include "utils/cuda_check.h"
#include "check/err_analysis.h"
#include "gemm_silu_bench.h"

// SiLU激活函数: silu(x) = x * sigmoid(x)
template<typename T>
T silu(T x) {
    if constexpr (std::is_same<T, cutlass::half_t>::value) {
        float x_f = static_cast<float>(x);
        float sigmoid = 1.0f / (1.0f + std::exp(-x_f));
        return cutlass::half_t(x_f * sigmoid);
    } else {
        T sigmoid = T(1.0) / (T(1.0) + std::exp(-x));
        return x * sigmoid;
    }
}

template<typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
void compute_gemm_silu_cpu_reference(
    std::vector<ElementC>& h_ref,
    const std::vector<ElementA>& h_A,
    const std::vector<ElementB>& h_B,
    int M, int N, int K,
    double alpha = 1.0,
    int max_elements = MAX_COMPARE_COUNT) {

    // 计算实际需要验证的元素数量
    int total_elements = M * N;
    int verify_count = std::min(total_elements, max_elements);

    // 初始化参考结果向量
    h_ref.resize(total_elements);

    // 只计算前verify_count个元素
    for (int idx = 0; idx < verify_count; ++idx) {
        int m = idx / N;
        int n = idx % N;
        ElementAccumulator accum = 0.0;

        // 计算矩阵乘法部分: C = A * B
        for (int k = 0; k < K; ++k) {
            // A是行主序，B是列主序
            ElementAccumulator a_val, b_val;

            if constexpr (std::is_same<ElementA, cutlass::half_t>::value) {
                a_val = static_cast<ElementAccumulator>(h_A[m * K + k]);
                b_val = static_cast<ElementAccumulator>(h_B[k + n * K]); // B列主序访问
            } else {
                a_val = static_cast<ElementAccumulator>(h_A[m * K + k]);
                b_val = static_cast<ElementAccumulator>(h_B[k + n * K]); // B列主序访问
            }

            accum += a_val * b_val;
        }

        // 应用alpha缩放
        accum = alpha * accum;

        // 应用SiLU激活函数
        ElementAccumulator silu_result = silu<ElementAccumulator>(accum);

        // 存储结果
        if constexpr (std::is_same<ElementC, cutlass::half_t>::value) {
            h_ref[m * N + n] = cutlass::half_t(static_cast<float>(silu_result));
        } else {
            h_ref[m * N + n] = static_cast<ElementC>(silu_result);
        }
    }
}

template<typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
TestResult benchmark_gemm_silu_template(int M, int N, int K, int iterations = 10,
                                        double alpha = 1.0) {

    TestResult result;
    result.M = M;
    result.N = N;
    result.K = K;
    result.operation = "GEMM-SiLU";

    if constexpr (std::is_same<ElementA, cutlass::half_t>::value) {
        result.data_type = "FP16";
    } else if constexpr (std::is_same<ElementA, float>::value) {
        result.data_type = "FP32";
    }

    // 定义数据类型别名
    using ElementInputA = ElementA;
    using ElementInputB = ElementB;
    using ElementOutput = ElementC;

    // 定义布局
    using LayoutInputA = cutlass::layout::RowMajor;    // A矩阵行主序
    using LayoutInputB = cutlass::layout::ColumnMajor; // B矩阵列主序
    using LayoutOutput = cutlass::layout::RowMajor;    // 输出行主序

    // 定义计算相关类型
    using ElementComputeEpilogue = ElementAccumulator;
    using SmArch = cutlass::arch::Sm80;

    std::ostringstream oss;
    oss << "(" << M << "," << K << "," << N << ")";
    result.test_case = oss.str();

    // 创建主机端数据
    std::vector<ElementInputA> h_A(M * K);
    std::vector<ElementInputB> h_B(K * N);
    std::vector<ElementOutput> h_ref(M * N);

    // 初始化数据
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = random_value<ElementInputA>();
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = random_value<ElementInputB>();
    }

    // 设备端内存分配
    ElementInputA* d_A = nullptr;
    ElementInputB* d_B = nullptr;
    ElementOutput* d_D = nullptr;

    size_t size_A = sizeof(ElementInputA) * M * K;
    size_t size_B = sizeof(ElementInputB) * K * N;
    size_t size_D = sizeof(ElementOutput) * M * N;

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_D, size_D));

    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    // 根据数据类型选择不同的配置
    if constexpr (std::is_same<ElementA, cutlass::half_t>::value) {
        // 对于半精度，使用Tensor Core
        using MMAOp = cutlass::arch::OpClassTensorOp;
        using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
        using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
        using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;

        // 定义SiLU Epilogue操作
        using EpilogueOp = cutlass::epilogue::thread::LinearCombinationSilu<
            ElementOutput,
            128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator,
            ElementComputeEpilogue,
            cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;

        // 定义Gemm类型
        using Gemm = cutlass::gemm::device::Gemm<
            ElementInputA,           // ElementA
            LayoutInputA,            // LayoutA
            ElementInputB,           // ElementB
            LayoutInputB,            // LayoutB
            ElementOutput,           // ElementC
            LayoutOutput,            // LayoutC
            ElementAccumulator,      // ElementAccumulator
            MMAOp,                   // Operation class
            SmArch,                  // Architecture
            ShapeMMAThreadBlock,     // Threadblock shape
            ShapeMMAWarp,            // Warp shape
            ShapeMMAOp,              // Instruction shape
            EpilogueOp,              // Epilogue operator (SiLU)
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
            3>;                      // Stages

        // 构造参数
        cutlass::gemm::GemmCoord problem_size(M, N, K);

        // 计算步长
        int lda = K;  // A行主序，所以lda = K
        int ldb = K;  // B列主序，所以ldb = K
        int ldd = N;  // D行主序，所以ldd = N

        // 创建TensorRef
        cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
            d_A, LayoutInputA(lda));

        cutlass::TensorRef<ElementInputB, LayoutInputB> weights_ref(
            d_B, LayoutInputB(ldb));

        cutlass::TensorRef<ElementOutput, LayoutOutput> output_ref(
            d_D, LayoutOutput(ldd));

        // 创建参数（SiLU不需要C矩阵，所以传nullptr）
        typename Gemm::Arguments arguments{
            problem_size,
            input_ref,
            weights_ref,
            {nullptr, 0},  // C矩阵为空
            output_ref,
            {ElementAccumulator(alpha)},  // 只有alpha参数
            1};  // split_k_slices = 1

        // 检查是否可以实现
        cutlass::Status status = Gemm::can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "Cannot implement GEMM-SiLU operation: "
                      << cutlass::cutlassGetStatusString(status) << std::endl;
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_D));
            result.passed = false;
            return result;
        }

        // 获取工作空间大小并分配
        size_t workspace_size = Gemm::get_workspace_size(arguments);
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        // 创建并初始化Gemm操作符
        Gemm gemm_op;
        status = gemm_op.initialize(arguments, workspace.get());
        CUTLASS_CHECK(status);

        // 预热运行
        status = gemm_op();
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

            status = gemm_op();
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
        // GEMM的FLOPs计数: 2MNK (矩阵乘法)
        // SiLU激活函数: 每个元素需要计算exp、除法、乘法，大约6个FLOPs
        double gemm_flops = 2.0 * M * N * K;
        double silu_flops = 6.0 * M * N;  // SiLU的近似FLOPs
        double total_flops = gemm_flops + silu_flops;

        result.avg_tflops = (total_flops / result.avg_time_ms) / 1e9;
        result.min_tflops = (total_flops / result.max_time_ms) / 1e9;  // 最小时间对应最大性能
        result.max_tflops = (total_flops / result.min_time_ms) / 1e9;  // 最大时间对应最小性能

        // 带宽计算
        size_t bytes_transferred = (M * K + K * N + M * N) * sizeof(ElementInputA);
        result.avg_bandwidth_gbs = (bytes_transferred / result.avg_time_ms) / 1e6;

        // 清理事件
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

    } else if constexpr (std::is_same<ElementA, float>::value) {
        // 对于单精度，使用SIMT
        using MMAOp = cutlass::arch::OpClassSimt;
        using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 8>;
        using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 8>;
        using ShapeMMAOp = cutlass::gemm::GemmShape<1, 1, 1>;

        // 定义SiLU Epilogue操作
        using EpilogueOp = cutlass::epilogue::thread::LinearCombinationSilu<
            ElementOutput,
            1,  // SIMT使用标量操作
            ElementAccumulator,
            ElementComputeEpilogue,
            cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;

        // 定义Gemm类型
        using Gemm = cutlass::gemm::device::Gemm<
            ElementInputA,           // ElementA
            LayoutInputA,            // LayoutA
            ElementInputB,           // ElementB
            LayoutInputB,            // LayoutB
            ElementOutput,           // ElementC
            LayoutOutput,            // LayoutC
            ElementAccumulator,      // ElementAccumulator
            MMAOp,                   // Operation class
            cutlass::arch::Sm80,     // Architecture
            ShapeMMAThreadBlock,     // Threadblock shape
            ShapeMMAWarp,            // Warp shape
            ShapeMMAOp,              // Instruction shape
            EpilogueOp,              // Epilogue operator (SiLU)
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
            2>;                      // Stages

        // 构造参数
        cutlass::gemm::GemmCoord problem_size(M, N, K);

        // 计算步长
        int lda = K;  // A行主序，所以lda = K
        int ldb = K;  // B列主序，所以ldb = K
        int ldd = N;  // D行主序，所以ldd = N

        // 创建TensorRef
        cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
            d_A, LayoutInputA(lda));

        cutlass::TensorRef<ElementInputB, LayoutInputB> weights_ref(
            d_B, LayoutInputB(ldb));

        cutlass::TensorRef<ElementOutput, LayoutOutput> output_ref(
            d_D, LayoutOutput(ldd));

        // 创建参数（SiLU不需要C矩阵，所以传nullptr）
        typename Gemm::Arguments arguments{
            problem_size,
            input_ref,
            weights_ref,
            {nullptr, 0},  // C矩阵为空
            output_ref,
            {ElementAccumulator(alpha)},  // 只有alpha参数
            1};  // split_k_slices = 1

        // 检查是否可以实现
        cutlass::Status status = Gemm::can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "Cannot implement GEMM-SiLU operation: "
                      << cutlass::cutlassGetStatusString(status) << std::endl;
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_D));
            result.passed = false;
            return result;
        }

        // 获取工作空间大小并分配
        size_t workspace_size = Gemm::get_workspace_size(arguments);
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        // 创建并初始化Gemm操作符
        Gemm gemm_op;
        status = gemm_op.initialize(arguments, workspace.get());
        CUTLASS_CHECK(status);

        // 预热运行
        status = gemm_op();
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

            status = gemm_op();
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
        double gemm_flops = 2.0 * M * N * K;
        double silu_flops = 6.0 * M * N;  // SiLU的近似FLOPs
        double total_flops = gemm_flops + silu_flops;

        result.avg_tflops = (total_flops / result.avg_time_ms) / 1e9;
        result.min_tflops = (total_flops / result.max_time_ms) / 1e9;  // 最小时间对应最大性能
        result.max_tflops = (total_flops / result.min_time_ms) / 1e9;  // 最大时间对应最小性能

        // 带宽计算
        size_t bytes_transferred = (M * K + K * N + M * N) * sizeof(ElementInputA);
        result.avg_bandwidth_gbs = (bytes_transferred / result.avg_time_ms) / 1e6;

        // 清理事件
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    result.iterations = iterations;

    // 计算CPU参考结果
    compute_gemm_silu_cpu_reference<ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>(
        h_ref, h_A, h_B, M, N, K, alpha, MAX_COMPARE_COUNT);

    // 获取GPU结果
    std::vector<ElementOutput> h_D(M * N);
    CUDA_CHECK(cudaMemcpy(h_D.data(), d_D, size_D, cudaMemcpyDeviceToHost));

    // 使用公共函数进行误差分析
    // 根据数据类型设置不同的容差
    double abs_tolerance = 1e-3;
    double rel_tolerance = 1e-3;

    if constexpr (std::is_same<ElementA, cutlass::half_t>::value) {
        abs_tolerance = 1e-2;
        rel_tolerance = 1e-2;
    }

    // 计算实际验证的元素数量
    result.verify_count = std::min(M * N, MAX_COMPARE_COUNT);

    auto error_result = analyze_errors(h_D, h_ref, 0, result.verify_count, abs_tolerance, rel_tolerance);

    result.max_abs_error = error_result.max_abs_error;
    result.max_rel_error = error_result.max_rel_error;
    result.passed = error_result.passed;
    result.error_count = error_result.error_count;
    result.total_count = error_result.total_count;

    // 清理设备内存
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_D));

    return result;
}

// 包装函数，用于测试不同类型
TestResult benchmark_gemm_silu_half(int M, int N, int K, int iterations) {
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;

    return benchmark_gemm_silu_template<ElementA, ElementB, ElementC, ElementAccumulator>(
        M, N, K, iterations, 1.0);
}

TestResult benchmark_gemm_silu_float(int M, int N, int K, int iterations) {
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = float;

    return benchmark_gemm_silu_template<ElementA, ElementB, ElementC, ElementAccumulator>(
        M, N, K, iterations, 1.0);
}