#include <iostream>
#include <vector>
#include <map>
#include <numeric>
#include <cuda_runtime.h>
#include "cutlass_wrapper_nvfp4_bench.h"
#include "utils/cuda_check.h"

void print_nvfp4_results_table(const std::vector<TestResult>& results) {
    const int col1 = 15;  // 测试用例
    const int col2 = 15;  // 数据类型
    const int col3 = 20;  // 操作类型
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

    // 打印表头
    std::cout << "\n";
    std::cout << std::string(total_width, '=') << "\n";
    std::cout << "CUTLASS NVFP4 GEMM 性能测试结果汇总\n";
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
                  << std::setw(col6) << std::fixed << std::setprecision(3) << result.avg_time_ms
                  << std::setw(col7) << std::fixed << std::setprecision(3) << result.avg_tflops
                  << std::setw(col8) << std::fixed << std::setprecision(2) << result.avg_bandwidth_gbs
                  << std::setw(col9) << std::scientific << std::setprecision(2) << result.max_rel_error
                  << std::setw(col10) << verify_oss.str();

        if (result.passed) {
            std::cout << std::setw(col11) << "✓ PASS";
        } else {
            std::ostringstream fail_oss;
            fail_oss << "✗ FAIL (" << result.error_count << ")";
            std::cout << std::setw(col11) << fail_oss.str();
        }
        std::cout << "\n";
    }

    std::cout << std::string(total_width, '-') << "\n";

    // 统计信息
    int total_tests = results.size();
    int passed_tests = std::count_if(results.begin(), results.end(),
                                     [](const TestResult& r) { return r.passed; });

    std::cout << "\n统计信息:\n";
    std::cout << "总测试数: " << total_tests << "\n";
    std::cout << "通过测试: " << passed_tests << "\n";
    std::cout << "失败测试: " << (total_tests - passed_tests) << "\n";
    std::cout << "通过率: " << std::fixed << std::setprecision(1)
              << (static_cast<double>(passed_tests) / total_tests * 100) << "%\n";

    // 按操作类型统计平均性能
    std::map<std::string, std::vector<double>> perf_by_op;
    for (const auto& result : results) {
        perf_by_op[result.operation].push_back(result.avg_tflops);
    }

    std::cout << "\n各操作类型平均性能:\n";
    for (const auto& [op, perfs] : perf_by_op) {
        double avg_perf = std::accumulate(perfs.begin(), perfs.end(), 0.0) / perfs.size();
        std::cout << op << ": " << std::fixed << std::setprecision(3) << avg_perf << " TFLOPS\n";
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
        {128, 128, 128, "<128,128>*<128,128>"},
        {128, 256, 128, "<128,128>*<128,256>"},
        {256, 128, 128, "<256,128>*<128,128>"},
        {256, 256, 256, "<256,256>*<256,256>"},
        {512, 512, 512, "<512,512>*<512,512>"},
        {1024, 1024, 1024, "<1024,1024>*<1024,1024>"},
    };

    std::vector<TestResult> all_test_results;
    int iterations = 10;

    std::cout << "========================================\n";
    std::cout << "CUTLASS NVFP4 GEMM 版本测试\n";
    std::cout << "========================================\n\n";

    // 检查设备是否支持NVFP4
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, USE_CUDA_DEVICE_ID));
    if (prop.major < 10) {
        std::cout << "Warning: NVFP4 requires compute capability 10.0 or higher (Blackwell)\n";
        std::cout << "Current device: " << prop.name << " (sm_" << prop.major << prop.minor << ")\n";
    }

    // 测试每个接口
    for (const auto& test_case : test_cases) {
        std::cout << "Testing NVFP4 GEMM: " << test_case.description << std::endl;
        TestResult result = benchmark_nvfp4_gemm_half(test_case.M, test_case.N, test_case.K, iterations);
        all_test_results.push_back(result);
    }

    // 打印结果
    print_nvfp4_results_table(all_test_results);
    std::cout << "\nAll NVFP4 wrapper tests completed!\n";

    return 0;
}