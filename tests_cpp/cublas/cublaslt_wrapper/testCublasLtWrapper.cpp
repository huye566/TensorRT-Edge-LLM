#include <iostream>
#include <vector>
#include <map>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include "cublaslt_wrapper_bench.h"

using namespace trt_edgellm::kernel;

// 复用的结果打印函数
void print_cublaslt_results_table(const std::vector<TestResult>& results) {
    const int col1 = 15;  // 测试用例
    const int col2 = 10;  // 数据类型
    const int col3 = 30;  // 操作类型
    const int col4 = 25;  // 矩阵大小
    const int col5 = 8;   // 循环次数
    const int col6 = 15;  // 平均时间(ms)
    const int col7 = 20;  // 平均性能(TFLOPS)
    const int col8 = 15;  // 带宽(GB/s)
    const int col9 = 15;  // 最大相对误差
    const int col10 = 20; // 验证元素数
    const int col11 = 15; // 正确性
    const int offset = 0;

    int total_width = col1 + col2 + col3 + col4 + col5 + col6 + col7 +
                     col8 + col9 + col10 + col11;
    total_width = total_width - offset * 11;

    // 打印表头
    std::cout << "\n";
    std::cout << std::string(total_width, '=') << "\n";
    std::cout << "cuBLASLt Wrapper 性能测试结果汇总\n";
    std::cout << std::string(total_width, '=') << "\n";

    std::cout << std::left
              << std::setw(col1) << "TestCase"
              << std::setw(col2) << "DataType"
              << std::setw(col3) << "Operation"
              << std::setw(col4) << "MatrixSize"
              << std::setw(col5) << "Loop"
              << std::setw(col6) << "MeanTime (ms)"
              << std::setw(col7) << "MeanPerf (TFLOPS)"
              << std::setw(col8) << "BW (GB/s)"
              << std::setw(col9) << "MaxRelError"
              << std::setw(col10) << "VerifiedElements"
              << std::setw(col11) << "Check"
              << "\n";

    std::cout << std::string(total_width, '-') << "\n";

    // 打印每一行数据
    for (const auto& result : results) {
        std::ostringstream size_oss;
        size_oss << "(" << result.M << "," << result.K << ")*(" << result.K << "," << result.N << ")";

        // 格式化验证元素数显示
        std::ostringstream verify_oss;
        int total_elements = result.M * result.N;
        if (result.verify_count < total_elements) {
            verify_oss << result.verify_count << "/" << total_elements;
        } else {
            verify_oss << result.verify_count;
        }

        std::cout << std::left
                  << std::setw(col1 - offset) << result.test_case
                  << std::setw(col2 - offset) << result.data_type
                  << std::setw(col3 - offset) << result.operation
                  << std::setw(col4 - offset) << size_oss.str()
                  << std::setw(col5 - offset) << result.iterations
                  << std::setw(col6 - offset) << std::fixed << std::setprecision(3) << result.avg_time_ms
                  << std::setw(col7 - offset) << std::fixed << std::setprecision(3) << result.avg_tflops
                  << std::setw(col8 - offset) << std::fixed << std::setprecision(2) << result.avg_bandwidth_gbs
                  << std::setw(col9 - offset) << std::scientific << std::setprecision(2) << result.max_rel_error
                  << std::setw(col10 - offset) << verify_oss.str();

        if (result.passed) {
            std::cout << std::setw(col11 - offset) << "✓ PASS";
        } else {
            std::ostringstream fail_oss;
            fail_oss << "✗ FAIL (" << result.error_count << ")";
            std::cout << std::setw(col11 - offset) << fail_oss.str();
        }
        std::cout << "\n";
    }

    std::cout << std::string(total_width, '-') << "\n";

    // 打印统计信息
    int total_tests = results.size();
    int passed_tests = std::count_if(results.begin(), results.end(), [](const TestResult& r) { return r.passed; });

    std::cout << "\n统计信息:\n";
    std::cout << "总测试数: " << total_tests << "\n";
    std::cout << "通过测试: " << passed_tests << "\n";
    std::cout << "失败测试: " << (total_tests - passed_tests) << "\n";
    std::cout << "通过率: " << std::fixed << std::setprecision(1)
              << (static_cast<double>(passed_tests) / total_tests * 100) << "%\n";

    // 按数据类型和操作类型统计平均性能
    std::map<std::pair<std::string, std::string>, std::vector<double>> perf_by_type_op;
    for (const auto& result : results) {
        perf_by_type_op[{result.data_type, result.operation}].push_back(result.avg_tflops);
    }

    std::cout << "\n各数据类型和操作类型平均性能:\n";
    for (const auto& [type_op, perfs] : perf_by_type_op) {
        double avg_perf = std::accumulate(perfs.begin(), perfs.end(), 0.0) / perfs.size();
        std::cout << type_op.first << " - " << type_op.second << ": "
                  << std::fixed << std::setprecision(3) << avg_perf << " TFLOPS\n";
    }
}

int main(int argc, char** argv) {
    print_device_info(USE_CUDA_DEVICE_ID);

    srand(2026);
    struct TestCase {
        int M;
        int N;
        int K;
        std::string description;
    };

    // 定义测试用例
    std::vector<TestCase> test_cases = {
        {1, 6144, 2048, "<1,2048>*<2048,6144>"},
        {1, 2048, 6144, "<1,6144>*<6144,2048>"},
        {530, 6144, 2048, "<530,2048>*<2048,6144>"},
        {530, 2048, 6144, "<530,6144>*<6144,2048>"},
        {1, 3, 2048, "<1,2048>*<2048,3>"},
        {530, 3, 2048, "<530,2048>*<2048,3>"}
    };

    std::vector<TestResult> all_test_results;
    int iterations = 10;

    std::cout << "========================================\n";
    std::cout << "cuBLASLt Wrapper 测试 (FP16, FP32, TF32)\n";
    std::cout << "========================================\n\n";

    // 测试FP16
    std::cout << "\n=== 测试 FP16 精度 ===\n";
    ComputeType fp16_compute_type = ComputeType::HALF;

    for (const auto& test_case : test_cases) {
        std::cout << "Testing cuBLASLt GEMM FP16: " << test_case.description << std::endl;
        TestResult result = benchmark_cublaslt_gemm<__half>(
            test_case.M, test_case.N, test_case.K, fp16_compute_type, iterations);
        all_test_results.push_back(result);
    }

    for (const auto& test_case : test_cases) {
        std::cout << "Testing cuBLASLt GEMM+SiLU FP16: " << test_case.description << std::endl;
        TestResult result = benchmark_cublaslt_gemm_silu<__half>(
            test_case.M, test_case.N, test_case.K, fp16_compute_type, iterations);
        all_test_results.push_back(result);
    }

    for (const auto& test_case : test_cases) {
        std::cout << "Testing cuBLASLt GEMM+Bias FP16: " << test_case.description << std::endl;
        TestResult result = benchmark_cublaslt_gemm_bias<__half>(
            test_case.M, test_case.N, test_case.K, fp16_compute_type, iterations);
        all_test_results.push_back(result);
    }

    // 测试FP32和TF32
    std::cout << "\n=== 测试 FP32/TF32 精度 ===\n";
    std::vector<ComputeType> float_compute_types = {ComputeType::FLOAT};

    for (auto compute_type : float_compute_types) {
        std::string compute_type_str = compute_type_to_string(compute_type);
        std::cout << "\n--- 计算类型: " << compute_type_str << " ---\n";

        for (const auto& test_case : test_cases) {
            std::cout << "Testing cuBLASLt GEMM " << compute_type_str << ": "
                      << test_case.description << std::endl;
            TestResult result = benchmark_cublaslt_gemm<float>(
                test_case.M, test_case.N, test_case.K, compute_type, iterations);
            all_test_results.push_back(result);
        }

        for (const auto& test_case : test_cases) {
            std::cout << "Testing cuBLASLt GEMM+SiLU " << compute_type_str << ": "
                      << test_case.description << std::endl;
            TestResult result = benchmark_cublaslt_gemm_silu<float>(
                test_case.M, test_case.N, test_case.K, compute_type, iterations);
            all_test_results.push_back(result);
        }

        for (const auto& test_case : test_cases) {
            std::cout << "Testing cuBLASLt GEMM+Bias " << compute_type_str << ": "
                      << test_case.description << std::endl;
            TestResult result = benchmark_cublaslt_gemm_bias<float>(
                test_case.M, test_case.N, test_case.K, compute_type, iterations);
            all_test_results.push_back(result);
        }
    }

    // 打印结果
    print_cublaslt_results_table(all_test_results);
    std::cout << "\nAll cuBLASLt wrapper tests completed!\n";

    return 0;
}