# CUPTI Profiler User Guide

This guide covers how to use the `CuptiProfiler` class from `cpp_base_suite` to collect hardware performance metrics via the CUPTI Range Profiler API.

## Prerequisites

- CUDA toolkit with CUPTI support
- `perf_event_paranoid` set to `0` (or run as root):
  ```bash
  sudo sysctl kernel.perf_event_paranoid=0
  ```

## Logging

`cpp_base_suite` ships with a lightweight, standalone logger (`#include "logger.hpp"`).

**Default behaviour**: all messages go to `stderr` (terminal).

**Redirect to a file or host logger**: install a callback.  While a callback is
installed, `stderr` output is suppressed and the callback receives every message:

```cpp
#include "logger.hpp"
namespace cbs_log = cpp_base_suite::logger;

// Route all cpp_base_suite messages through the host project's logger
// (e.g. to a log file, syslog, or the trt_edgellm logger).
cbs_log::SetLogCallback([](int level, const char* msg) {
    my_host_logger.write(level, msg);
});

// Optionally raise the minimum level (default: kInfo).
cbs_log::SetLogLevel(cbs_log::Level::kWarning);
```

Pass `nullptr` to `SetLogCallback` to restore the default `stderr` output.

**Quick Start**

### Basic Usage — Manual Multi-Pass

For most use cases, use `ProfileMultiPass` which automatically handles multi-pass reconfiguration, range pushing/popping, and result collection.

```cpp
#include "profiler/cupti_profiler.hpp"
#include "profiler/metrics.hpp"

using namespace cpp_base_suite::profiler;

void profile_workload() {
    // Create profiler with the metrics you want to collect
    CuptiProfiler profiler(PerformanceProfileMetrics());

    // Initialize the profiler (sets up internal buffers)
    if (!profiler.Initialize()) {
        fprintf(stderr, "Failed to initialize profiler\n");
        return;
    }

    // ProfileMultiPass handles all passes automatically:
    //  1. setConfigForPass → Start → PushRange → workload → PopRange → Stop
    //  2. If more passes needed, reconfigure and repeat
    //  3. Extract results when all passes are complete
    bool ok = profiler.ProfileMultiPass("my_kernel_range",
        [](int /*pass*/) {
            // Your workload goes here — called once per pass
            launchMyKernel();
            cudaDeviceSynchronize();
        });

    if (!ok) {
        fprintf(stderr, "Profiling failed\n");
        return;
    }

    // Print results to stdout
    profiler.PrintResults();

    // Or save to JSON file
    profiler.SaveToJson("/tmp/profile_output.json", WriteMode::Overwrite);
}
```

### ProfileMultiPass — Complete Examples

#### Example 1: Single Workload, Multiple Passes

```cpp
#include "profiler/cupti_profiler.hpp"
#include "profiler/metrics.hpp"

using namespace cpp_base_suite::profiler;

void example_single_workload() {
    // Default metrics: GPU duration, SM utilization, DRAM read/write, SM frequency
    CuptiProfiler profiler(DefaultMetrics());
    profiler.Initialize();

    // The workload is called profiler.NumPasses() times (once per pass)
    profiler.ProfileMultiPass("matrix_multiply", [](int pass) {
        launchMatrixMultiply(stream);
        cudaStreamSynchronize(stream);
    });

    auto results = profiler.GetResults();
    for (auto& r : results) {
        printf("Range: %s\n", r.range_name.c_str());
        for (auto& [id, value] : r.values) {
            printf("  %s = %.2f %s\n", MetricToLabel(id), value, MetricToUnit(id));
        }
    }
}
```

#### Example 2: Compare Different Configurations

```cpp
void example_block_size_comparison() {
    CuptiProfiler profiler({MetricId::SMUtilization, MetricId::IPCExecutedElapsed});
    profiler.Initialize();

    const std::vector<int> block_sizes = {64, 128, 256, 512};

    for (int bs : block_sizes) {
        std::string range_name = "block_size_" + std::to_string(bs);
        profiler.ProfileMultiPass(range_name.c_str(),
            [bs, stream](int /*pass*/) {
                launchMatrixAdd(d_A, d_B, d_C, N, bs, stream);
                cudaStreamSynchronize(stream);
            });

        auto results = profiler.GetResults();
        double sm_util = results[0].values[MetricId::SMUtilization];
        printf("  Block size %4d: SM Util = %.2f%%, IPC = %.2f\n",
               bs, sm_util, results[0].values[MetricId::IPCExecutedElapsed]);
    }
}
```

#### Example 3: Custom Metric Presets

The profiler provides several built-in metric presets:

| Function | Description |
|---|---|
| `DefaultMetrics()` | GPU duration, SM utilization, DRAM bytes, SM frequency |
| `PerformanceProfileMetrics()` | Comprehensive set for performance profiling |
| `WarpStallMetrics()` | All warp stall reasons for bottleneck analysis |
| `CacheMetrics()` | L1/L2 cache hit rates, sector counts |
| `InstructionMixMetrics()` | FAdd, FMul, FFMA instruction counts |
| `LaunchMetrics()` | Occupancy, wave counts, register usage |
| `AllSupportedMetrics()` | Every metric available on the current architecture |

```cpp
void example_custom_metrics() {
    // Analyze warp stalls to find the bottleneck
    CuptiProfiler profiler(WarpStallMetrics());
    profiler.Initialize();

    profiler.ProfileMultiPass("kernel_with_stalls", [](int /*pass*/) {
        runKernel();
        cudaDeviceSynchronize();
    });

    profiler.PrintResults();
}
```

#### Example 4: Save Results to JSON (Append Mode)

For continuous profiling (e.g., profiling every N steps), use `WriteMode::Append` which writes one NDJSON line per call in a non-blocking manner.

```cpp
void example_continuous_profiling() {
    for (int step = 0; step < 100; ++step) {
        CuptiProfiler profiler(DefaultMetrics());
        profiler.Initialize();

        std::string range = "step_" + std::to_string(step);
        profiler.ProfileMultiPass(range.c_str(), [](int /*pass*/) {
            doWorkload();
            cudaDeviceSynchronize();
        });

        // Overwrite for the first step, then append
        auto mode = (step == 0) ? WriteMode::Overwrite : WriteMode::Append;
        profiler.SaveToJson("/tmp/profile_timeline.ndjson", mode);
    }
    // File contains: one JSON array (step 0) + 99 NDJSON lines
}
```

### Low-Level API — Manual Control

When you need finer control over the profiling lifecycle:

```cpp
CuptiProfiler profiler(DefaultMetrics());
profiler.Initialize();
profiler.Start();

  profiler.PushRange("phase_a");
  launchKernelA();
  cudaDeviceSynchronize();
  profiler.PopRange();

  profiler.PushRange("phase_b");
  launchKernelB();
  cudaDeviceSynchronize();
  profiler.PopRange();

profiler.Stop();

auto results = profiler.GetResults();  // 2 ranges: phase_a, phase_b
```

### AutoRange Mode

In `CUPTI_AutoRange` mode, each kernel launch is automatically profiled without manual PushRange/PopRange:

```cpp
CuptiProfiler profiler(DefaultMetrics());
profiler.SetRangeMode(CUPTI_AutoRange);
profiler.Initialize();
profiler.Start();

  // Each kernel is auto-profiled with its launch config captured
  launchKernelA();
  launchKernelB();
  cudaDeviceSynchronize();

profiler.Stop();

auto results = profiler.GetResults();
// Each result includes kernel_info (grid, block, registers, shared memory, duration)
```

### Architecture Compatibility

Metrics have a minimum architecture requirement. Use `IsMetricAvailable()` to check at runtime:

```cpp
if (IsMetricAvailable(MetricId::SMThroughput)) {
    printf("SM Throughput is supported\n");
} else {
    printf("SM Throughput requires arch >= 80\n");
}
```
