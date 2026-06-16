#ifndef PROFILER_CUPTI_PROFILER_HPP_
#define PROFILER_CUPTI_PROFILER_HPP_

#include <cstdint>
#include <functional>
#include <future>
#include <map>
#include <string>
#include <vector>

#include <cupti_profiler_target.h>

#include "profiler/metrics.hpp"

namespace cpp_base_suite {
namespace profiler {

// Internal: captured kernel record from CUPTI Activity API.
struct ActivityKernelRecord;

// Write mode for SaveToJson.
enum class WriteMode {
  Append,   // Append as NDJSON line (non-blocking, O(1) per call)
  Overwrite // Truncate and write fresh JSON
};

// Single-range profiling result: MetricId -> value.
using MetricValues = std::map<MetricId, double>;

// Kernel launch configuration captured via CUPTI Activity API.
struct KernelLaunchInfo {
  std::string name;            // Mangled kernel name (from range profiler)
  std::string demangled_name;  // Human-readable name (if demangle succeeds)
  int32_t grid_x = 0;
  int32_t grid_y = 0;
  int32_t grid_z = 0;
  int32_t block_x = 0;
  int32_t block_y = 0;
  int32_t block_z = 0;
  uint16_t registers_per_thread = 0;
  uint32_t shared_memory_bytes = 0;
  uint32_t shared_memory_executed = 0;
  int64_t grid_id = 0;
  uint64_t start_ns = 0;  // Execution start timestamp (ns)
  uint64_t end_ns = 0;    // Execution end timestamp (ns)
  bool has_activity_info = false;  // true if matched with Activity record
};

struct ProfilingResult {
  std::string range_name;
  MetricValues values;           // MetricId -> value
  KernelLaunchInfo kernel_info;  // Populated when range_mode == CUPTI_AutoRange
};

// RAII wrapper around the CUPTI Range Profiler API.
//
// Usage (UserRange mode — default):
//   CuptiProfiler profiler(metrics);
//   profiler.Start();
//   profiler.PushRange("my_kernel");
//   // ... run kernels ...
//   profiler.PopRange();
//   profiler.Stop();
//   auto results = profiler.GetResults();
//
// Usage (AutoRange mode — automatic per-kernel profiling):
//   CuptiProfiler profiler(metrics);
//   profiler.SetRangeMode(CUPTI_AutoRange);
//   profiler.Initialize();
//   profiler.Start();
//   // ... run kernels (each kernel is auto-profiled) ...
//   profiler.Stop();
//   auto results = profiler.GetResults();  // kernel_info populated
class CuptiProfiler {
 public:
  // Construct with the set of metrics to collect.
  explicit CuptiProfiler(std::vector<MetricId> metrics = DefaultMetrics());
  ~CuptiProfiler();

  CuptiProfiler(const CuptiProfiler&) = delete;
  CuptiProfiler& operator=(const CuptiProfiler&) = delete;

  // Set range mode before Initialize().
  // CUPTI_UserRange (default): manual PushRange/PopRange.
  // CUPTI_AutoRange: automatically profile each kernel; kernel info captured.
  void SetRangeMode(CUpti_ProfilerRange mode);

  // Initialize internal buffers and configure the profiler.
  // Must be called before Start(). Returns false on error.
  bool Initialize();

  // Start profiling. Must have called Initialize() first.
  bool Start();

  // Stop profiling for the current pass and decode collected data.
  // pass_index: which pass just completed (-1 uses the tracked pass from StartWithPass).
  // Returns false on error. Call IsProfilingComplete() to check if more passes needed.
  bool Stop(int pass_index = -1);

  // After Stop(): true if all passes have been collected and results are ready.
  bool IsProfilingComplete() const;

  // Get the next pass index to use (relevant after Stop() when !IsProfilingComplete()).
  int NextPassIndex() const;

  // Get the total number of passes required for the current metric set.
  int NumPasses() const;

  // Helper: run a named range across all required passes automatically.
  // The workload_fn is called once per pass, inside PushRange/PopRange.
  // Handles Start/Stop + re-config internally for each pass.
  // For single-pass metric sets, workload_fn is called exactly once.
  bool ProfileMultiPass(const std::string& range_name,
                        std::function<void(int pass)> workload_fn);

  // Manual single-pass or per-pass control (no function pointer).
  // Reconfigures the profiler for the next pass, starts profiling, and pushes the range.
  // Caller must run workload, then call PopRange() and Stop().
  // Usage (multi-pass):
  //   for (int p = 0; p < profiler.NumPasses(); ++p) {
  //     profiler.ProfileOnePass("my_range");
  //     // ... run your workload ...
  //     profiler.PopRange();
  //     profiler.Stop();
  //   }
  //   auto results = profiler.GetResults();
  bool ProfileOnePass(const std::string& range_name);

  // Manually configure profiler for a specific pass (alternative to ProfileOnePass).
  bool StartWithPass(int pass_index);

  // Push/pop named range (for user-range profiling).
  bool PushRange(const std::string& name);
  bool PopRange();

  // Retrieve collected results after Stop().
  // After all passes are complete, results are automatically extracted.
  std::vector<ProfilingResult> GetResults() const;

  // Format results as a human-readable string (same content as PrintResults).
  std::string FormatResults() const;

  // Print results to stdout in a human-readable table.
  // In AutoRange mode, prints kernel launch config alongside metrics.
  void PrintResults() const;

  // Save results to a JSON file. Returns false on write error.
  // mode=Append: writes a single NDJSON line with timestamp (non-blocking).
  // mode=Overwrite: writes the entire results set as a traditional JSON array.
  bool SaveToJson(const char* path, WriteMode mode = WriteMode::Overwrite) const;

  // Deinitialize profiler resources. Called automatically in destructor.
  void Shutdown();

 private:
  bool extractResults();
  bool setConfigForPass(int pass_index);
  std::vector<ActivityKernelRecord> GetCapturedKernels() const;
  size_t GetRangeCount() const;
  bool EvaluateRange(size_t range_index, struct GlobalCuptiState*,
                     ProfilingResult* out);
  void MatchKernelToRange(size_t range_index,
                          const std::vector<ActivityKernelRecord>& kernels,
                          ProfilingResult* out);

  // Serialize one result to an NDJSON-compatible JSON string (with timestamp).
  // (implemented as static helper in the .cpp)

  struct Impl;
  Impl* impl_ = nullptr;
  mutable std::mutex write_mtx_;       // protects pending writes
  mutable std::future<void> write_fut_; // tracks async write completion
};

}  // namespace profiler
}  // namespace cpp_base_suite

#endif  // PROFILER_CUPTI_PROFILER_HPP_
