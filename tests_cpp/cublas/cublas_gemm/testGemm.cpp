#include <iostream>
#include <vector>
#include <map>
#include <cuda_runtime.h>
#include "utils/cuda_check.h"
#include "gemm_bench.h"


void print_gemm_results_table(const std::vector<TestResult>& results) {
    const int col1 = 20;  // 测试用例
    const int col2 = 10;  // 数据类型
    const int col3 = 12;  // 操作类型
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
    std::cout << "cuBLAS GEMM 性能测试结果汇总\n";
    std::cout << std::string(total_width, '=') << "\n";
    
    std::cout << std::left
              << std::setw(col1) << "TestCase"
              << std::setw(col2) << "DataType"
              << std::setw(col3) << "Operation"
              << std::setw(col4) << "MatrixSize"
              << std::setw(col5) << "Loop"
              << std::setw(col6) << "MeanTime (ms)"
              << std::setw(col7) << "MeanPerf (TFLOPS)"
              << std::setw(col8) << "PeakPerf (TFLOPS)"
              << std::setw(col9) << "Bandwidth (GB/s)"
              << std::setw(col10) << "MaxAbsError"
              << std::setw(col11) << "VerifiedElements"
              << std::setw(col12) << "Check"
              << "\n";
    
    std::cout << std::string(total_width, '-') << "\n";
    
    // 统计GEMM结果数量
    std::vector<TestResult> gemm_results;
    std::copy_if(results.begin(), results.end(), std::back_inserter(gemm_results),
                 [](const TestResult& r) { return r.operation == "GEMM"; });
    
    if (gemm_results.empty()) {
        std::cout << "没有GEMM测试结果可显示。\n";
        std::cout << std::string(total_width, '=') << "\n";
        return;
    }
    
    // 打印每一行数据
    for (const auto& result : gemm_results) {
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
        
        // 性能转换为GFLOPS (乘以1000)
        
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
    int total_tests = gemm_results.size();
    int passed_tests = std::count_if(gemm_results.begin(), gemm_results.end(), 
                                     [](const TestResult& r) { return r.passed; });
    
    // 打印汇总统计信息
    std::cout << "\n汇总统计:\n";
    std::cout << "总测试数: " << total_tests 
              << " | 通过: " << passed_tests 
              << " | 失败: " << (total_tests - passed_tests)
              << " | 通过率: " << std::fixed << std::setprecision(1) 
              << (static_cast<double>(passed_tests) / total_tests * 100) << "%\n";
    
    // 计算不同数据类型的平均性能
    std::map<std::string, std::vector<double>> perf_by_type;
    std::map<std::string, int> count_by_type;
    
    for (const auto& result : gemm_results) {
        perf_by_type[result.data_type].push_back(result.avg_tflops * 1000.0);
        count_by_type[result.data_type]++;
    }
    
    for (const auto& [dtype, perf_list] : perf_by_type) {
        double avg_performance = std::accumulate(perf_list.begin(), perf_list.end(), 0.0) / perf_list.size();
        std::cout << dtype << "平均性能: " << std::fixed << std::setprecision(3) 
                  << avg_performance << " GFLOPS (" << count_by_type[dtype] << "个测试)\n";
    }
    
    // 验证策略说明
    std::cout << "\n验证策略说明:\n";
    std::cout << "1. 当输出矩阵元素 > " << MAX_COMPARE_COUNT << " 时，只计算和验证前" << MAX_COMPARE_COUNT << "个元素\n";
    std::cout << "2. 容差设置: FP16为1e-2，FP32为1e-3，FP64为1e-12\n";
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
    int iterations = 10;  // 减少迭代次数，因为GEMM测试较慢
    
    for (const auto& test_case : test_cases) {
        std::cout << "Test Case: " << test_case.description << "\n";
        TestResult result = benchmark_gemm_half(test_case.M, test_case.N, test_case.K, iterations);
        all_test_results.push_back(result);
    }
    
    for (const auto& test_case : test_cases) {
        std::cout << "Test Case: " << test_case.description << "\n";
        TestResult result = benchmark_gemm_float(test_case.M, test_case.N, test_case.K, iterations);
        all_test_results.push_back(result);
    }
    
    print_gemm_results_table(all_test_results);
    std::cout << "\n\nAll tests completed!\n";
    return 0;
}