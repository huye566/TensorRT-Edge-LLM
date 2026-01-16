#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "utils/cuda_check.h"
#include "gemv_bench.h"

void print_gemv_results_table(const std::vector<TestResult>& results) {
    const int col1 = 15;  // 测试用例
    const int col2 = 10;  // 数据类型
    const int col3 = 12;   // 操作类型
    const int col4 = 25;  // 矩阵大小
    const int col5 = 8;  // 循环次数
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
    std::cout << "Cutlass GEMV 性能测试结果汇总\n";
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
    
    // 统计GEMV结果数量
    std::vector<TestResult> gemv_results;
    std::copy_if(results.begin(), results.end(), std::back_inserter(gemv_results),
                 [](const TestResult& r) { return r.operation == "GEMV"; });
    
    if (gemv_results.empty()) {
        std::cout << "没有GEMV测试结果可显示。\n";
        std::cout << std::string(total_width, '=') << "\n";
        return;
    }
    
    // 打印每一行数据
    for (const auto& result : gemv_results) {
        std::ostringstream size_oss;
        size_oss << "(" << result.M << "," << result.K << ")*(" << result.K << ",1)";
        
        // 格式化验证元素数显示
        std::ostringstream verify_oss;
        int total_elements = result.M;
        if (result.verify_count < total_elements) {
            verify_oss << result.verify_count << "/" << total_elements;
        } else {
            verify_oss << result.verify_count;
        }
        
        // GEMV的性能转换为GFLOPS (乘以1000)
        double avg_gflops = result.avg_tflops * 1000.0;
        double max_gflops = result.max_tflops * 1000.0;
        
        std::cout << std::left
                  << std::setw(col1 - offset) << result.test_case
                  << std::setw(col2 - offset) << result.data_type
                  << std::setw(col3 - offset) << result.operation
                  << std::setw(col4 - offset) << size_oss.str()
                  << std::setw(col5 - offset) << result.iterations
                  << std::setw(col6 - offset) << std::fixed << std::setprecision(3) << result.avg_time_ms
                  << std::setw(col7 - offset) << std::fixed << std::setprecision(3) << avg_gflops
                  << std::setw(col8 - offset) << std::fixed << std::setprecision(3) << max_gflops
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
    int total_tests = gemv_results.size();
    int passed_tests = std::count_if(gemv_results.begin(), gemv_results.end(), 
                                     [](const TestResult& r) { return r.passed; });
    
    // 打印汇总统计信息（放在表格下方）
    std::cout << "\n汇总统计:\n";
    std::cout << "总测试数: " << total_tests 
              << " | 通过: " << passed_tests 
              << " | 失败: " << (total_tests - passed_tests)
              << " | 通过率: " << std::fixed << std::setprecision(1) 
              << (static_cast<double>(passed_tests) / total_tests * 100) << "%\n";
    
    // 计算FP16和FP32的平均性能
    double avg_performance_fp16 = 0.0;
    double avg_performance_fp32 = 0.0;
    int count_fp16 = 0, count_fp32 = 0;
    
    for (const auto& result : gemv_results) {
        if (result.data_type == "FP16") {
            avg_performance_fp16 += result.avg_tflops * 1000.0; // 转换为GFLOPS
            count_fp16++;
        } else if (result.data_type == "FP32") {
            avg_performance_fp32 += result.avg_tflops * 1000.0; // 转换为GFLOPS
            count_fp32++;
        }
    }
    
    if (count_fp16 > 0) {
        avg_performance_fp16 /= count_fp16;
        std::cout << "FP16平均性能: " << std::fixed << std::setprecision(3) 
                  << avg_performance_fp16 << " GFLOPS\n";
    }
    
    if (count_fp32 > 0) {
        avg_performance_fp32 /= count_fp32;
        std::cout << "FP32平均性能: " << std::fixed << std::setprecision(3) 
                  << avg_performance_fp32 << " GFLOPS\n";
    }
    
    // 验证策略说明
    std::cout << "\n验证策略说明:\n";
    std::cout << "1. 当输出向量长度 > 10000 时，只计算和验证前10000个元素\n";
    std::cout << "2. 容差设置: FP16为1e-2，FP32为1e-3\n";
}


int main(int argc, char** argv) {
    print_device_info(2);
    srand(2026);

    std::vector<std::pair<int, int>> test_cases = {
        {1, 2048},
        {530, 2048},
    };
    std::vector<TestResult> all_test_results;
    for (const auto& test_case : test_cases) {
        int M = test_case.first;
        int N = test_case.second;
        std::ostringstream oss;
        oss << "(" << M << "," << N << ")";
        TestResult result = benchmark_gemv_half(M, N, 100);  // 100次迭代以获得稳定结果
        
        result.test_case = oss.str();
        all_test_results.push_back(result);
    }

    for (const auto& test_case : test_cases) {
        int M = test_case.first;
        int N = test_case.second;
        std::ostringstream oss;
        oss << "(" << M << "," << N << ")";
        TestResult result = benchmark_gemv_float(M, N, 100);  // 100次迭代以获得稳定结果
        
        result.test_case = oss.str();
        all_test_results.push_back(result);
    }

    print_gemv_results_table(all_test_results);
    return 0;
}