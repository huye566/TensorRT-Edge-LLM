#include <iostream>
#include <vector>
#include <map>
#include <cuda_runtime.h>
#include "utils/cuda_check.h"
#include "gemm_silu_bench.h"

void print_gemm_silu_results_table(const std::vector<TestResult>& results) {
    const int col1 = 20;  // 测试用例
    const int col2 = 10;  // 数据类型
    const int col3 = 20;  // 操作类型
    const int col4 = 25;  // 矩阵大小
    const int col5 = 8;   // 循环次数
    const int col6 = 15;  // 平均时间(ms)
    const int col7 = 20;  // 平均性能(GFLOPS)
    const int col8 = 20;  // 峰值性能(GFLOPS)
    const int col9 = 18;  // 带宽(GB/s)
    const int col10 = 15; // 最大绝对误差
    const int col11 = 20; // 验证元素数
    const int col12 = 10; // 正确性
    const int offset = 0;

    int total_width = col1 + col2 + col3 + col4 + col5 + col6 + col7 + col8 + col9 + col10 + col11 + col12;
    total_width = total_width - offset * 12;

    // 打印表头
    std::cout << "\n";
    std::cout << std::string(total_width, '=') << "\n";
    std::cout << "GEMM+SiLU 性能测试结果汇总\n";
    std::cout << std::string(total_width, '=') << "\n";

    std::cout << std::left
              << std::setw(col1) << "TestCase"
              << std::setw(col2) << "DataType"
              << std::setw(col3) << "Operation"
              << std::setw(col4) << "MatrixSize"
              << std::setw(col5) << "Loop"
              << std::setw(col6) << "MeanTime (ms)"
              << std::setw(col7) << "MeanPerf (GFLOPS)"
              << std::setw(col8) << "PeakPerf (GFLOPS)"
              << std::setw(col9) << "Bandwidth (GB/s)"
              << std::setw(col10) << "MaxAbsError"
              << std::setw(col11) << "VerifiedElements"
              << std::setw(col12) << "Check"
              << "\n";

    std::cout << std::string(total_width, '-') << "\n";

    // 分离GEMM+SiLU和融合版本的结果
    std::vector<TestResult> gemm_silu_results;
    std::copy_if(results.begin(), results.end(), std::back_inserter(gemm_silu_results),
                 [](const TestResult& r) {
                     return r.operation.find("GEMM_SiLU") != std::string::npos;
                 });

    if (gemm_silu_results.empty()) {
        std::cout << "没有GEMM+SiLU测试结果可显示。\n";
        std::cout << std::string(total_width, '=') << "\n";
        return;
    }

    // 打印每一行数据
    for (const auto& result : gemm_silu_results) {
        std::ostringstream size_oss;
        size_oss << "(" << result.M << "," << result.K << ")*(" << result.K << "," << result.N << ")";

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
                  << std::setw(col8 - offset) << std::fixed << std::setprecision(3) << result.max_tflops
                  << std::setw(col9 - offset) << std::fixed << std::setprecision(2) << result.avg_bandwidth_gbs
                  << std::setw(col10 - offset) << std::scientific << std::setprecision(2) << result.max_abs_error
                  << std::setw(col11 - offset) << verify_oss.str();

        if (result.passed) {
            std::cout << std::left << std::setw(col12 - offset) << "✓ PASS";
        } else {
            std::ostringstream fail_oss;
            fail_oss << "✗ FAIL (" << result.error_count << ")";
            std::cout << std::left << std::setw(col12 - offset) << fail_oss.str();
        }
        std::cout << "\n";
    }

    std::cout << std::string(total_width, '=') << "\n";

    // 计算统计信息
    int total_tests = gemm_silu_results.size();
    int passed_tests = std::count_if(gemm_silu_results.begin(), gemm_silu_results.end(),
                                     [](const TestResult& r) { return r.passed; });

    // 按操作类型分组统计
    std::map<std::string, std::vector<TestResult>> results_by_op;
    for (const auto& result : gemm_silu_results) {
        results_by_op[result.operation].push_back(result);
    }

    // 打印汇总统计信息
    std::cout << "\n汇总统计:\n";
    std::cout << "总测试数: " << total_tests
              << " | 通过: " << passed_tests
              << " | 失败: " << (total_tests - passed_tests)
              << " | 通过率: " << std::fixed << std::setprecision(1)
              << (static_cast<double>(passed_tests) / total_tests * 100) << "%\n";

    // 打印每个操作类型的平均性能
    for (const auto& [op_type, op_results] : results_by_op) {
        double avg_time = 0.0;
        double avg_perf = 0.0;

        for (const auto& result : op_results) {
            avg_time += result.avg_time_ms;
            avg_perf += result.avg_tflops;
        }

        avg_time /= op_results.size();
        avg_perf /= op_results.size();

        std::cout << op_type << ": " << op_results.size() << "个测试 | "
                  << "平均时间: " << std::fixed << std::setprecision(3) << avg_time << " ms | "
                  << "平均性能: " << std::fixed << std::setprecision(3) << avg_perf << " GFLOPS\n";
    }

    // 计算性能加速比（分离 vs 融合）
    if (results_by_op.count("GEMM_SiLU_SEPARATE") && results_by_op.count("GEMM_SiLU_FUSED")) {
        std::cout << "\n性能对比:\n";

        // 对相同矩阵大小的测试进行比较
        std::map<std::string, std::pair<double, double>> perf_map; // 矩阵大小 -> (分离时间, 融合时间)

        for (const auto& result : results_by_op["GEMM_SiLU_SEPARATE"]) {
            std::string key = result.test_case + "_" + result.data_type;
            perf_map[key] = {result.avg_time_ms, 0.0};
        }

        for (const auto& result : results_by_op["GEMM_SiLU_FUSED"]) {
            std::string key = result.test_case + "_" + result.data_type;
            if (perf_map.count(key)) {
                perf_map[key].second = result.avg_time_ms;
            }
        }

        for (const auto& [key, times] : perf_map) {
            if (times.first > 0 && times.second > 0) {
                double speedup = times.first / times.second;
                std::cout << key << ": 分离版 " << std::fixed << std::setprecision(3) << times.first << " ms vs "
                          << "融合版 " << times.second << " ms | "
                          << "加速比: " << std::fixed << std::setprecision(2) << speedup << "x\n";
            }
        }
    }

    std::cout << "\n验证策略说明:\n";
    std::cout << "1. 当输出矩阵元素 > " << MAX_COMPARE_COUNT << " 时，只计算和验证前" << MAX_COMPARE_COUNT << "个元素\n";
    std::cout << "2. 容差设置: FP16为1e-2，FP32为1e-3\n";
    std::cout << "3. SiLU函数: x * sigmoid(x) = x / (1 + exp(-x))\n";
}

int main(int argc, char** argv) {
    print_device_info(2);

    srand(2026);
    struct TestCase {
        int M;
        int N;
        int K;
        std::string description;
    };

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

    // 测试分离版本
    std::cout << "\n========== 测试GEMM+SiLU分离版本 ==========\n";
    for (const auto& test_case : test_cases) {
        std::cout << "Test Case: " << test_case.description << " (FP16)\n";
        TestResult result = benchmark_gemm_silu_separate_half(
            test_case.M, test_case.N, test_case.K, iterations);
        all_test_results.push_back(result);
    }

    for (const auto& test_case : test_cases) {
        std::cout << "Test Case: " << test_case.description << " (FP32)\n";
        TestResult result = benchmark_gemm_silu_separate_float(
            test_case.M, test_case.N, test_case.K, iterations);
        all_test_results.push_back(result);
    }

    // 测试融合版本
    std::cout << "\n========== 测试GEMM+SiLU融合版本 ==========\n";
    for (const auto& test_case : test_cases) {
        std::cout << "Test Case: " << test_case.description << " (FP16)\n";
        TestResult result = benchmark_gemm_silu_fused_half(
            test_case.M, test_case.N, test_case.K, iterations);
        all_test_results.push_back(result);
    }

    for (const auto& test_case : test_cases) {
        std::cout << "Test Case: " << test_case.description << " (FP32)\n";
        TestResult result = benchmark_gemm_silu_fused_float(
            test_case.M, test_case.N, test_case.K, iterations);
        all_test_results.push_back(result);
    }

    print_gemm_silu_results_table(all_test_results);
    std::cout << "\n\nAll GEMM+SiLU tests completed!\n";
    return 0;
}