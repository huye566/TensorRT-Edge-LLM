#include <iostream>
#include <vector>
#include <map>
#include <numeric>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "utils/utils.h"
#include "cutlass_wrapper_nvfp4_blockwise_bench.h"

void print_nvfp4_results_table(const std::vector<TestResult>& results) {
    const int col1 = 20;  // 测试用例
    const int col2 = 15;  // 数据类型
    const int col3 = 25;  // 操作类型
    const int col4 = 30;  // 矩阵大小
    const int col5 = 10;  // 循环次数
    const int col6 = 15;  // 平均时间(ms)
    const int col7 = 20;  // 平均性能(TFLOPS)
    const int col8 = 15;  // 带宽(GB/s)
    const int col9 = 15;  // 最大相对误差
    const int col10 = 20; // 验证元素数
    const int col11 = 15; // 正确性

    int total_width = col1 + col2 + col3 + col4 + col5 + col6 + col7 +
                     col8 + col9 + col10 + col11;

    // 打印表头
    std::cout << "\n";
    std::cout << std::string(total_width, '=') << "\n";
    std::cout << "NVFP4 BLOCKWISE GEMM 性能测试结果汇总\n";
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
        size_oss << "(" << result.M << "," << result.K << ")*(" << result.N << "," << result.K << ")";

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

    // 打印统计信息
    int total_tests = results.size();
    int passed_tests = std::count_if(results.begin(), results.end(), [](const TestResult& r) { return r.passed; });

    std::cout << "\n统计信息:\n";
    std::cout << "总测试数: " << total_tests << "\n";
    std::cout << "通过测试: " << passed_tests << "\n";
    std::cout << "失败测试: " << (total_tests - passed_tests) << "\n";
    std::cout << "通过率: " << std::fixed << std::setprecision(1)
              << (static_cast<double>(passed_tests) / total_tests * 100) << "%\n";

    // 计算平均性能
    double avg_tflops = 0.0;
    for (const auto& result : results) {
        avg_tflops += result.avg_tflops;
    }
    avg_tflops /= total_tests;

    std::cout << "平均性能: " << std::fixed << std::setprecision(3) << avg_tflops << " TFLOPS\n";
    std::cout << std::string(total_width, '=') << "\n";
}


int main(int argc, char** argv) {
    print_device_info(USE_CUDA_DEVICE_ID);

    srand(2026);

    // 定义测试用例 - 使用典型的LLM形状
    struct TestCase {
        int M;    // batch size * sequence length
        int N;    // output dimension
        int K;    // input dimension
        std::string description;
    };

    std::vector<TestCase> test_cases = {
        // 小batch大小测试
        {1, 8, 32, "<1,32>*<32,8>"},
        {1, 6144, 2048, "<1,2048>*<2048,6144>"},
        {1, 2048, 6144, "<1,6144>*<2048,6144>"},

        // 中等batch大小测试
        {530, 6144, 2048, "<530,2048>*<2048,6144>"},
        {530, 2048, 6144, "<530,6144>*<6144,2048>"},

        {1, 8, 2048, "<1,2048>*<2048,8>"},
        {530, 8, 2048, "<530,2048>*<2048,8>"},
    };

    bool use_load_file = false;
    if (argc == 2) {
        if (std::string(argv[1]) == "--lf") {
            use_load_file = true;
        }
    } else if (argc >= 4) {
        test_cases.clear();
        int M = atoi(argv[1]);
        int N = atoi(argv[2]);
        int K = atoi(argv[3]);
        test_cases.push_back({M, N, K, "自定义测试"});
        std::cout << "使用自定义测试: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    }

    std::vector<TestResult> all_test_results;
    int iterations = 100;  // 减少迭代次数以加快测试

    std::cout << "===================================================\n";
    std::cout << "CUTLASS NVFP4 Blockwise GEMM 测试\n";
    std::cout << "===================================================\n\n";

    std::cout << "注意: 4-bit量化测试可能需要较长时间，因为包含量化过程\n";
    std::cout << "测试用例数: " << test_cases.size() << "\n";
    std::cout << "每个用例迭代次数: " << iterations << "\n";
    std::cout << "验证元素上限: " << MAX_COMPARE_COUNT << "\n";
    std::cout << "===================================================\n\n";

    std::vector<std::pair<std::string, std::pair<bool, bool>>> test_types = {
        {"标准GEMM", {false, false}},
        {"带偏置GEMM", {true, false}},
        {"带SiLU激活GEMM", {false, true}},
        {"带偏置和SiLU激活GEMM", {true, true}}
    };

    if (use_load_file) {
        auto input_file = trt_edgellm::rt::getResourcesPath() / "nvfp4" / "nvfp4_testdata_1_8_64.bin";
        auto output_file = trt_edgellm::rt::getResourcesPath() / "nvfp4" / "cpp_nvfp4_1_8_64.bin";
        TestParams params = {
            1,
            1,
            1,
            iterations,
            false,
            false,
            true,
            input_file.string(),
            output_file.string()
        };
        TestResult result = benchmark_nvfp4_gemm_half(params);
        all_test_results.push_back(result);
    } else {
        for (size_t i = 0; i < test_cases.size(); ++i) {
            const auto& test_case = test_cases[i];
            for (const auto& test_type : test_types) {
            std::cout << "\n[" << (i+1) << "/" << test_cases.size() << "] "
                      << "测试 cutlass NVFP4 " << test_type.first
                      << ": " << test_case.description
                      << " (M=" << test_case.M << ", N=" << test_case.N << ", K=" << test_case.K << ")"
                      << std::endl;

                try {
                    TestParams params = {
                        test_case.M,
                        test_case.N,
                        test_case.K,
                        iterations,
                        test_type.second.first,  // use_bias
                        test_type.second.second,  // use_silu
                        true,
                        "",
                        ""
                    };
                    TestResult result = benchmark_nvfp4_gemm_half(params);
                    all_test_results.push_back(result);
                } catch (const std::exception& e) {
                    std::cerr << "测试失败: " << e.what() << std::endl;

                    // 创建失败结果记录
                    TestResult failed_result;
                    failed_result.M = test_case.M;
                    failed_result.N = test_case.N;
                    failed_result.K = test_case.K;
                    failed_result.test_case = "M" + std::to_string(test_case.M) +
                                            "_N" + std::to_string(test_case.N) +
                                            "_K" + std::to_string(test_case.K);
                    failed_result.data_type = "NVFP4->FP16";
                    failed_result.operation = test_type.first;;
                    failed_result.iterations = iterations;
                    failed_result.passed = false;
                    failed_result.error_count = 1;
                    failed_result.verify_count = 0;
                    all_test_results.push_back(failed_result);
                }
            }
        }
    }

    // 打印详细结果表格
    print_nvfp4_results_table(all_test_results);

    // 总结
    int total_tests = all_test_results.size();
    int passed_tests = std::count_if(all_test_results.begin(), all_test_results.end(),
                                    [](const TestResult& r) { return r.passed; });

    std::cout << "\n===================================================\n";
    std::cout << "NVFP4 GEMM 测试完成!\n";
    std::cout << "总测试数: " << total_tests << "\n";
    std::cout << "通过测试: " << passed_tests << "\n";
    std::cout << "失败测试: " << (total_tests - passed_tests) << "\n";
    std::cout << "通过率: " << std::fixed << std::setprecision(1)
              << (static_cast<double>(passed_tests) / total_tests * 100) << "%\n";

    if (passed_tests == total_tests) {
        std::cout << "✅ 所有测试通过!\n";
    } else {
        std::cout << "⚠️  有测试失败，请检查错误信息\n";
    }
    std::cout << "===================================================\n";

    return (passed_tests == total_tests) ? 0 : 1;
}