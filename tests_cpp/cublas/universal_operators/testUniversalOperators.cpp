#include <iostream>
#include <vector>
#include <map>
#include <numeric>
#include "universal_operators_test.h"

void print_universal_operators_results(const std::vector<TestResult>& results) {
    const int col1 = 25;  // 测试用例
    const int col2 = 15;  // 数据类型
    const int col3 = 30;  // 操作类型
    const int col4 = 15;  // 矩阵大小
    const int col5 = 8;   // 循环次数
    const int col6 = 12;  // 平均时间(us)
    const int col7 = 15;  // 带宽(GB/s)
    const int col8 = 12;  // 最大相对误差
    const int col9 = 15;  // 验证元素数
    const int col10 = 10; // 正确性

    int total_width = col1 + col2 + col3 + col4 + col5 + col6 + col7 + col8 + col9 + col10;

    // 打印表头
    std::cout << "\n";
    std::cout << std::string(total_width, '=') << "\n";
    std::cout << "Universal Operators 性能测试结果汇总\n";
    std::cout << std::string(total_width, '=') << "\n";

    std::cout << std::left
              << std::setw(col1) << "TestCase"
              << std::setw(col2) << "DataType"
              << std::setw(col3) << "Operation"
              << std::setw(col4) << "MatrixSize"
              << std::setw(col5) << "Loop"
              << std::setw(col6) << "MeanTime (µs)"
              << std::setw(col7) << "BW (GB/s)"
              << std::setw(col8) << "MaxRelError"
              << std::setw(col9) << "VerifiedElements"
              << std::setw(col10) << "Check"
              << "\n";

    std::cout << std::string(total_width, '-') << "\n";

    // 打印每一行数据
    for (const auto& result : results) {
        std::ostringstream size_oss;
        size_oss << "(" << result.M << "x" << result.N << ")";

        // 格式化验证元素数显示
        std::ostringstream verify_oss;
        int total_elements = result.M * result.N;
        if (result.verify_count < total_elements) {
            verify_oss << result.verify_count << "/" << total_elements;
        } else {
            verify_oss << result.verify_count;
        }

        std::cout << std::left
                  << std::setw(col1) << result.test_case
                  << std::setw(col2) << result.data_type
                  << std::setw(col3) << result.operation
                  << std::setw(col4) << size_oss.str()
                  << std::setw(col5) << result.iterations
                  << std::setw(col6) << std::fixed << std::setprecision(3) << result.avg_time_ms * 1000
                  << std::setw(col7) << std::fixed << std::setprecision(2) << result.avg_bandwidth_gbs
                  << std::setw(col8) << std::scientific << std::setprecision(2) << result.max_rel_error
                  << std::setw(col9) << verify_oss.str();

        if (result.passed) {
            std::cout << std::setw(col10) << "✓ PASS";
        } else {
            std::ostringstream fail_oss;
            fail_oss << "✗ FAIL (" << result.error_count << ")";
            std::cout << std::setw(col10) << fail_oss.str();
        }
        std::cout << "\n";
    }

    std::cout << std::string(total_width, '-') << "\n";

    // 打印统计信息
    int total_tests = results.size();
    int passed_tests = std::count_if(results.begin(), results.end(),
                                     [](const TestResult& r) { return r.passed; });

    std::cout << "\n统计信息:\n";
    std::cout << "总测试数: " << total_tests << "\n";
    std::cout << "通过测试: " << passed_tests << "\n";
    std::cout << "失败测试: " << (total_tests - passed_tests) << "\n";
    std::cout << "通过率: " << std::fixed << std::setprecision(1)
              << (static_cast<double>(passed_tests) / total_tests * 100) << "%\n";

    // 按操作类型统计平均带宽
    std::map<std::string, std::vector<double>> bw_by_op;
    for (const auto& result : results) {
        bw_by_op[result.operation].push_back(result.avg_bandwidth_gbs);
    }

    std::cout << "\n各操作类型平均带宽:\n";
    for (const auto& [op, bws] : bw_by_op) {
        double avg_bw = std::accumulate(bws.begin(), bws.end(), 0.0) / bws.size();
        std::cout << op << ": " << std::fixed << std::setprecision(2) << avg_bw << " GB/s\n";
    }
}

int main(int argc, char** argv) {
    print_device_info(USE_CUDA_DEVICE_ID);

    srand(2026);

    // 定义测试用例：矩阵大小
    struct TestCase {
        int M;
        int N;
        std::string description;
    };

    std::vector<TestCase> test_cases = {
        {1, 8, "1x8"},
        {1, 2048, "1x2048"},
        {1, 6144, "1x6144"},
        {530, 2048, "530x2048"},
        {530, 6144, "530x6144"}
    };

    std::vector<TestResult> all_test_results;
    int iterations = 100;

    std::cout << "========================================\n";
    std::cout << "Universal Operators 测试\n";
    std::cout << "========================================\n\n";

    // 测试每个接口
    for (const auto& test_case : test_cases) {
        std::cout << "Testing Universal AddBias Scalar (FP16): " << test_case.description << std::endl;
        TestResult result = benchmark_universal_add_bias_scalar_half(test_case.M, test_case.N, iterations);
        all_test_results.push_back(result);
        std::cout << "Testing Universal AddBias Vec (FP16): " << test_case.description << std::endl;
        result = benchmark_universal_add_bias_half(test_case.M, test_case.N, iterations);
        all_test_results.push_back(result);
        std::cout << "Testing Universal AddBias Vec Optimized (FP16): " << test_case.description << std::endl;
        result = benchmark_universal_add_bias_vec_optimized_half(test_case.M, test_case.N, iterations);
        all_test_results.push_back(result);

        std::cout << "Testing Universal SiLU Scalar (FP16): " << test_case.description << std::endl;
        result = benchmark_universal_silu_scalar_half(test_case.M, test_case.N, iterations);
        all_test_results.push_back(result);
        std::cout << "Testing Universal SiLU Vec (FP16): " << test_case.description << std::endl;
        result = benchmark_universal_silu_half(test_case.M, test_case.N, iterations);
        all_test_results.push_back(result);
        std::cout << "Testing Universal SiLU Vec Optimized (FP16): " << test_case.description << std::endl;
        result = benchmark_universal_silu_vec_optimized_half(test_case.M, test_case.N, iterations);
        all_test_results.push_back(result);

        std::cout << "Testing Universal AddBias+SiLU Fused (FP16): " << test_case.description << std::endl;
        result = benchmark_universal_add_bias_silu_fused_half(test_case.M, test_case.N, iterations);
        all_test_results.push_back(result);
    }

    for (const auto& test_case : test_cases) {
        std::cout << "Testing Universal AddBias Scalar (FP32): " << test_case.description << std::endl;
        TestResult result = benchmark_universal_add_bias_scalar_float(test_case.M, test_case.N, iterations);
        all_test_results.push_back(result);
        std::cout << "Testing Universal AddBias Vec (FP32): " << test_case.description << std::endl;
        result = benchmark_universal_add_bias_float(test_case.M, test_case.N, iterations);
        all_test_results.push_back(result);
        std::cout << "Testing Universal AddBias Vec Optimized (FP32): " << test_case.description << std::endl;
        result = benchmark_universal_add_bias_vec_optimized_float(test_case.M, test_case.N, iterations);
        all_test_results.push_back(result);

        std::cout << "Testing Universal SiLU Scalar (FP32): " << test_case.description << std::endl;
        result = benchmark_universal_silu_scalar_float(test_case.M, test_case.N, iterations);
        all_test_results.push_back(result);
        std::cout << "Testing Universal SiLU Vec (FP32): " << test_case.description << std::endl;
        result = benchmark_universal_silu_float(test_case.M, test_case.N, iterations);
        all_test_results.push_back(result);
        std::cout << "Testing Universal SiLU Vec Optimized (FP32): " << test_case.description << std::endl;
        result = benchmark_universal_silu_vec_optimized_float(test_case.M, test_case.N, iterations);
        all_test_results.push_back(result);

        std::cout << "Testing Universal AddBias+SiLU Fused (FP32): " << test_case.description << std::endl;
        result = benchmark_universal_add_bias_silu_fused_float(test_case.M, test_case.N, iterations);
        all_test_results.push_back(result);
    }

    // 打印结果
    print_universal_operators_results(all_test_results);
    std::cout << "\nAll Universal Operators tests completed!\n";

    return 0;
}