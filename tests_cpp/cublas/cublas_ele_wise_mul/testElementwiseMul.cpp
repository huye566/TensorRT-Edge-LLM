#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <map>
#include "utils/cuda_check.h"
#include "elementwise_mul_bench.h"

// 辅助结构体，用于存储额外的信息
struct TestCaseInfo {
    int rows;
    int cols;
    std::string impl_type; // "CUDA", "CUDA-2D", "Thrust"
};

void print_elementwise_results_table(const std::vector<TestResult>& results,
                                     const std::vector<std::string>& impl_types,
                                     const std::vector<int>& rows_vec,
                                     const std::vector<int>& cols_vec) {
    const int col1 = 25;  // 测试用例
    const int col2 = 10;  // 数据类型
    const int col3 = 10;  // 实现方式
    const int col4 = 12;  // 形状
    const int col5 = 8;   // 循环次数
    const int col6 = 15;  // 平均时间(us)
    const int col7 = 18;  // 平均性能(GFLOPS)
    const int col8 = 18;  // 峰值性能(GFLOPS)
    const int col9 = 18;  // 带宽(GB/s)
    const int col10 = 15; // 最大绝对误差
    const int col11 = 18; // 验证元素数
    const int col12 = 10; // 正确性

    int total_width = col1 + col2 + col3 + col4 + col5 + col6 + col7 +
                     col8 + col9 + col10 + col11 + col12;

    // 打印表头
    std::cout << "\n";
    std::cout << std::string(total_width, '=') << "\n";
    std::cout << "Element-wise Multiplication 性能测试结果汇总\n";
    std::cout << std::string(total_width, '=') << "\n";

    std::cout << std::left
              << std::setw(col1) << "TestCase"
              << std::setw(col2) << "DataType"
              << std::setw(col3) << "Impl"
              << std::setw(col4) << "Shape"
              << std::setw(col5) << "Loop"
              << std::setw(col6) << "MeanTime(µs)"
              << std::setw(col7) << "MeanPerf(GFLOPS)"
              << std::setw(col8) << "PeakPerf(GFLOPS)"
              << std::setw(col9) << "Bandwidth(GB/s)"
              << std::setw(col10) << "MaxAbsError"
              << std::setw(col11) << "VerifiedElements"
              << std::setw(col12) << "Check"
              << "\n";

    std::cout << std::string(total_width, '-') << "\n";

    // 打印每一行数据
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        const auto& impl_type = impl_types[i];
        const int rows = rows_vec[i];
        const int cols = cols_vec[i];
        int total_elements = rows * cols;

        std::ostringstream shape_oss;
        shape_oss << "(" << rows << "," << cols << ")";

        std::ostringstream verify_oss;
        if (result.verify_count < total_elements) {
            verify_oss << result.verify_count << "/" << total_elements;
        } else {
            verify_oss << result.verify_count;
        }

        // 将TFLOPS转换回GFLOPS
        double avg_gflops = result.avg_tflops * 1000.0;
        double max_gflops = result.max_tflops * 1000.0;

        std::cout << std::left
                  << std::setw(col1) << result.test_case
                  << std::setw(col2) << result.data_type
                  << std::setw(col3) << impl_type
                  << std::setw(col4) << shape_oss.str()
                  << std::setw(col5) << result.iterations
                  << std::setw(col6) << std::fixed << std::setprecision(3) << (result.avg_time_ms * 1000)  // 转换为微秒
                  << std::setw(col7) << std::fixed << std::setprecision(3) << avg_gflops
                  << std::setw(col8) << std::fixed << std::setprecision(3) << max_gflops
                  << std::setw(col9) << std::fixed << std::setprecision(2) << result.avg_bandwidth_gbs
                  << std::setw(col10) << std::scientific << std::setprecision(2) << result.max_abs_error
                  << std::setw(col11) << verify_oss.str();

        if (result.passed) {
            std::cout << std::setw(col12) << "✓ PASS";
        } else {
            std::ostringstream fail_oss;
            fail_oss << "✗ FAIL (" << result.error_count << ")";
            std::cout << std::setw(col12) << fail_oss.str();
        }
        std::cout << "\n";
    }

    std::cout << std::string(total_width, '-') << "\n";

    // 打印统计信息
    int total_tests = results.size();
    int passed_tests = std::count_if(results.begin(), results.end(),
                                     [](const TestResult& r) { return r.passed; });

    // 按数据类型和实现方式分组统计性能
    std::map<std::string, std::vector<double>> perf_by_group;
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        const auto& impl_type = impl_types[i];
        std::string group = result.data_type + "_" + impl_type;
        double avg_gflops = result.avg_tflops * 1000.0;
        perf_by_group[group].push_back(avg_gflops);
    }

    std::cout << "\n统计信息:\n";
    std::cout << "总测试数: " << total_tests << "\n";
    std::cout << "通过测试: " << passed_tests << "\n";
    std::cout << "失败测试: " << (total_tests - passed_tests) << "\n";
    std::cout << "通过率: " << std::fixed << std::setprecision(1)
              << (static_cast<double>(passed_tests) / total_tests * 100) << "%\n";

    std::cout << "\n平均性能统计 (GFLOPS):\n";
    for (const auto& group : perf_by_group) {
        double avg_perf = std::accumulate(group.second.begin(),
                                          group.second.end(), 0.0) / group.second.size();
        std::cout << "  " << group.first << ": "
                  << std::fixed << std::setprecision(3) << avg_perf << " GFLOPS\n";
    }

    // 打印验证策略说明
    std::cout << "\n验证策略说明:\n";
    std::cout << "1. 当输出向量长度 > " << MAX_COMPARE_COUNT << " 时，只计算和验证前" << MAX_COMPARE_COUNT << "个元素\n";
    std::cout << "2. 容差设置: FP16为1e-2，FP32为1e-3\n";
    std::cout << "3. 性能计算: 每个元素1次乘法 = 总元素数 FLOPs\n";
    std::cout << "4. 实现方式说明:\n";
    std::cout << "   - CUDA: 一维线程块实现\n";
    std::cout << "   - CUDA-2D: 二维线程块实现（用于矩阵）\n";
    std::cout << "   - Thrust: 使用Thrust库实现\n";
}

int main(int argc, char** argv) {
    print_device_info(2);

    srand(2026);

    // 定义测试用例
    struct TestCase {
        int rows;
        int cols;
        std::string description;
    };

    std::vector<TestCase> test_cases = {
        {1, 6144, "<1,6144>"},
        {530, 6144, "<530,6144>"},
    };

    std::vector<TestResult> all_test_results;
    std::vector<std::string> all_impl_types;
    std::vector<int> all_rows;
    std::vector<int> all_cols;

    int iterations = 100;  // Element-wise操作较快，可以增加迭代次数

    std::cout << "开始Element-wise乘法性能测试...\n";

    // 运行所有测试用例
    for (const auto& test_case : test_cases) {
        std::cout << "\n测试形状: " << test_case.description << "\n";
        // CUDA实现 - half
        std::cout << "  CUDA FP16...";
        auto result_fp16_cuda = benchmark_elementwise_mul_cuda_half(
            test_case.rows, test_case.cols, iterations);
        all_test_results.push_back(result_fp16_cuda);
        all_impl_types.push_back((test_case.rows > 1) ? "CUDA-2D" : "CUDA");
        all_rows.push_back(test_case.rows);
        all_cols.push_back(test_case.cols);
        std::cout << " 完成\n";

        // Thrust实现 - half
        std::cout << "  Thrust FP16...";
        auto result_fp16_thrust = benchmark_elementwise_mul_thrust_half(
            test_case.rows, test_case.cols, iterations);
        all_test_results.push_back(result_fp16_thrust);
        all_impl_types.push_back("Thrust");
        all_rows.push_back(test_case.rows);
        all_cols.push_back(test_case.cols);
        std::cout << " 完成\n";
    }

    for (const auto& test_case : test_cases) {
        std::cout << "\n测试形状: " << test_case.description << "\n";
        // CUDA实现 - float
        std::cout << "  CUDA FP32...";
        auto result_fp32_cuda = benchmark_elementwise_mul_cuda_float(
            test_case.rows, test_case.cols, iterations);
        all_test_results.push_back(result_fp32_cuda);
        all_impl_types.push_back((test_case.rows > 1) ? "CUDA-2D" : "CUDA");
        all_rows.push_back(test_case.rows);
        all_cols.push_back(test_case.cols);
        std::cout << " 完成\n";

        // Thrust实现 - float
        std::cout << "  Thrust FP32...";
        auto result_fp32_thrust = benchmark_elementwise_mul_thrust_float(
            test_case.rows, test_case.cols, iterations);
        all_test_results.push_back(result_fp32_thrust);
        all_impl_types.push_back("Thrust");
        all_rows.push_back(test_case.rows);
        all_cols.push_back(test_case.cols);
        std::cout << " 完成\n";
    }

    // 打印汇总表格
    print_elementwise_results_table(all_test_results, all_impl_types, all_rows, all_cols);

    std::cout << "\n\n所有Element-wise乘法测试完成!\n";

    // 性能对比总结
    std::cout << "\n性能对比总结:\n";
    std::cout << "1. 对于小尺寸 (<128,128>): CUDA和Thrust性能接近\n";
    std::cout << "2. 对于大尺寸 (>256,256>): CUDA实现通常更快\n";
    std::cout << "3. 2D CUDA核函数对于矩阵形状有更好的性能\n";
    std::cout << "4. FP16比FP32带宽更高，但计算精度较低\n";
    std::cout << "5. Thrust实现代码更简洁，但性能可能略低于优化后的CUDA实现\n";

    return 0;
}