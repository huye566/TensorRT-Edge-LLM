#ifndef RESULTS_PRINT_H
#define RESULTS_PRINT_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <sstream>

#define MAX_COMPARE_COUNT 10000

struct TestResult {
    std::string test_case;
    std::string data_type;
    std::string operation;
    int M;
    int N;
    int K;
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    double avg_tflops;
    double max_tflops;
    double min_tflops;
    double avg_bandwidth_gbs;
    double max_abs_error;
    double max_rel_error;
    bool passed;
    int error_count;
    int total_count;
    int verify_count;
    int iterations;
};

#endif // RESULTS_PRINT_H