#include "profiler/cupti_profiler.hpp"
#include "logger.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <vector>
#include <cuda.h>
#include <cupti_activity.h>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_range_profiler.h>
#include <cupti_target.h>
#include <nlohmann/json.hpp>
#include <cxxabi.h>
#include <cuda_runtime.h>

namespace cpp_base_suite {
namespace profiler {

static bool CheckCupti(CUptiResult result, const char* api_name) {
  if (result != CUPTI_SUCCESS) {
    CBS_LOG_ERROR("CUPTI error in %s: %d", api_name, result);
    return false;
  }
  return true;
}

// ─── Activity API: kernel launch capture ──────────────────────────────────

struct ActivityKernelRecord {
  std::string name;
  int32_t grid_x, grid_y, grid_z;
  int32_t block_x, block_y, block_z;
  uint16_t registers_per_thread;
  uint32_t shared_memory_bytes;
  uint32_t shared_memory_executed;
  int64_t grid_id;
  uint64_t start_ns, end_ns;
};

struct ActivityState {
  bool enabled = false;
  std::vector<ActivityKernelRecord> records;
  std::mutex mtx;  // Protects 'records' and 'enabled'.
  static ActivityState& instance() {
    static ActivityState state;
    return state;
  }

  static void CUPTIAPI BufferRequest(uint8_t** buffer, size_t* size,
                                     size_t* maxNumRecords) {
    static constexpr size_t kBufferSize = 4 * 1024 * 1024;
    *size = kBufferSize;
    *buffer = new uint8_t[kBufferSize];
    *maxNumRecords = 0;
  }

  static void CUPTIAPI BufferComplete(CUcontext /*context*/,
                                      uint32_t /*streamId*/,
                                      uint8_t* buffer, size_t /*size*/,
                                      size_t validSize) {
    auto& state = instance();
    std::lock_guard<std::mutex> lock(state.mtx);
    if (!state.enabled) {
      delete[] buffer;
      return;
    }

    CUpti_Activity* record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
      if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
          record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
        auto* kernel9 = reinterpret_cast<CUpti_ActivityKernel9*>(record);
        ActivityKernelRecord rec;
        rec.name = kernel9->name ? kernel9->name : "<unknown>";
        rec.grid_x = kernel9->gridX;
        rec.grid_y = kernel9->gridY;
        rec.grid_z = kernel9->gridZ;
        rec.block_x = kernel9->blockX;
        rec.block_y = kernel9->blockY;
        rec.block_z = kernel9->blockZ;
        rec.registers_per_thread = kernel9->registersPerThread;
        rec.shared_memory_bytes = kernel9->sharedMemoryExecuted;
        rec.shared_memory_executed = kernel9->sharedMemoryExecuted;
        rec.grid_id = kernel9->gridId;
        rec.start_ns = kernel9->start;
        rec.end_ns = kernel9->end;

        int64_t grid_sz = (int64_t)rec.grid_x * rec.grid_y * rec.grid_z;
        int64_t block_sz = (int64_t)rec.block_x * rec.block_y * rec.block_z;
        if (grid_sz <= 1 && block_sz <= 1) continue;

        state.records.push_back(std::move(rec));
      }
    }

    delete[] buffer;
  }

  bool Enable() {
    std::lock_guard<std::mutex> lock(mtx);
    if (enabled) return true;
    CUptiResult res = cuptiActivityEnable(
        CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (res != CUPTI_SUCCESS) {
      res = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
      if (res != CUPTI_SUCCESS) {
        CBS_LOG_ERROR("CUPTI: failed to enable activity kernel tracing (%d)", res);
        return false;
      }
    }
    cuptiActivityRegisterCallbacks(BufferRequest, BufferComplete);
    enabled = true;
    return true;
  }

  void Flush() {
    cuptiActivityFlushAll(0);
  }

  // Reset state for reuse across multiple profiling sessions.
  void Reset() {
    std::lock_guard<std::mutex> lock(mtx);
    records.clear();
  }
};

// Demangle a C++ mangled name.
static std::string Demangle(const char* mangled) {
  if (!mangled) return "";
  int status = 0;
  char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
  if (status == 0 && demangled) {
    std::string result(demangled);
    free(demangled);
    return result;
  }
  return mangled;
}

// ─── Nesting guard ─────────────────────────────────────────────────────────
//
// CuptiProfiler relies on process-wide CUPTI state (GlobalCuptiState, the
// Activity API, and the per-metric host object). Creating a second profiler
// instance while another is actively profiling on the same thread — e.g. the
// application wraps run_inference() in ProfileMultiPass(), and the runtime
// internally wraps QwenViTRunner::infer() the same way — would otherwise:
//   * re-run the inner workload for every outer pass (exponential replay),
//   * clobber the outer's GlobalCuptiState when metric sets differ,
//   * clear the Activity records the outer is still collecting on destruct.
//
// A thread-local depth counter detects this situation. Initialize() marks the
// new instance as "nested" when depth > 0; nested instances skip all CUPTI
// resource creation, and their Start/Stop/PushRange/PopRange become no-ops.
// ProfileMultiPass still runs the workload exactly once so the real
// computation is not lost.
static thread_local int g_profiling_depth = 0;

// ─── Global/process-level CUPTI state ─────────────────────────────────────

struct GlobalCuptiState {
  bool initialized = false;
  CUcontext cuda_ctx = nullptr;
  CUdevice cuda_device = 0;
  std::string chip_name;
  std::vector<uint8_t> counter_availability_image;

  bool host_initialized = false;
  CUpti_Profiler_Host_Object* host_object = nullptr;
  std::vector<MetricId> host_metrics;
  std::vector<const char*> host_metric_names;
  std::vector<uint8_t> config_image;
  size_t num_passes = 0;

  static GlobalCuptiState& instance() {
    static GlobalCuptiState state;
    return state;
  }

  bool EnsureInitialized() {
    if (initialized) return true;

    cudaFree(0);

    CUresult cu_result = cuCtxGetCurrent(&cuda_ctx);
    if (cu_result != CUDA_SUCCESS || !cuda_ctx) {
      CBS_LOG_ERROR("CUPTI Profiler: no active CUDA context");
      return false;
    }
    cu_result = cuCtxGetDevice(&cuda_device);
    if (cu_result != CUDA_SUCCESS) {
      CBS_LOG_ERROR("CUPTI Profiler: failed to get device");
      return false;
    }

    {
      CUpti_Profiler_Initialize_Params prof_init = {};
      prof_init.structSize = CUpti_Profiler_Initialize_Params_STRUCT_SIZE;
      prof_init.pPriv = nullptr;
      if (!CheckCupti(cuptiProfilerInitialize(&prof_init),
                      "cuptiProfilerInitialize")) {
        return false;
      }
    }

    {
      int dev_idx = 0;
      int device_count = 0;
      if (cuDeviceGetCount(&device_count) == CUDA_SUCCESS) {
        for (int i = 0; i < device_count; ++i) {
          CUdevice d;
          cuDeviceGet(&d, i);
          if (d == cuda_device) {
            dev_idx = i;
            break;
          }
        }
      }

      CUpti_Device_GetChipName_Params chip_params = {};
      chip_params.structSize = CUpti_Device_GetChipName_Params_STRUCT_SIZE;
      chip_params.pPriv = nullptr;
      chip_params.deviceIndex = dev_idx;
      if (!CheckCupti(cuptiDeviceGetChipName(&chip_params),
                      "cuptiDeviceGetChipName")) {
        return false;
      }
      chip_name = chip_params.pChipName;
      CBS_LOG_INFO("CUPTI Profiler: chip = '%s'", chip_name.c_str());
    }

    {
      CUpti_Profiler_GetCounterAvailability_Params ca_params = {};
      ca_params.structSize =
          CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE;
      ca_params.pPriv = nullptr;
      ca_params.ctx = cuda_ctx;
      ca_params.counterAvailabilityImageSize = 0;
      ca_params.pCounterAvailabilityImage = nullptr;

      CUptiResult res = cuptiProfilerGetCounterAvailability(&ca_params);
      if (res != CUPTI_SUCCESS) {
        CBS_LOG_ERROR("CUPTI Profiler: cuptiProfilerGetCounterAvailability failed "
                      "(error %d). Fix: sudo sysctl kernel.perf_event_paranoid=0 "
                      "or run as root.", res);
        return false;
      }

      counter_availability_image.resize(ca_params.counterAvailabilityImageSize);
      ca_params.pCounterAvailabilityImage = counter_availability_image.data();

      if (!CheckCupti(cuptiProfilerGetCounterAvailability(&ca_params),
                      "cuptiProfilerGetCounterAvailability(retrieve)")) {
        return false;
      }
    }

    initialized = true;
    return true;
  }

  bool EnsureHostInitialized(const std::vector<MetricId>& metrics,
                             const std::vector<const char*>& metric_names) {
    if (host_initialized) return true;

    host_metrics = metrics;
    host_metric_names = metric_names;

    CUpti_Profiler_Host_Initialize_Params host_init = {};
    host_init.structSize = CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE;
    host_init.pPriv = nullptr;
    host_init.profilerType = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
    host_init.pChipName = chip_name.c_str();
    host_init.pCounterAvailabilityImage = counter_availability_image.data();

    if (!CheckCupti(cuptiProfilerHostInitialize(&host_init),
                    "cuptiProfilerHostInitialize")) {
      return false;
    }
    host_object = host_init.pHostObject;

    CUpti_Profiler_Host_ConfigAddMetrics_Params add_params = {};
    add_params.structSize =
        CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE;
    add_params.pPriv = nullptr;
    add_params.pHostObject = host_object;
    add_params.ppMetricNames = host_metric_names.data();
    add_params.numMetrics = host_metric_names.size();

    if (!CheckCupti(cuptiProfilerHostConfigAddMetrics(&add_params),
                    "cuptiProfilerHostConfigAddMetrics")) {
      return false;
    }

    CUpti_Profiler_Host_GetConfigImageSize_Params size_params = {};
    size_params.structSize =
        CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE;
    size_params.pPriv = nullptr;
    size_params.pHostObject = host_object;

    if (!CheckCupti(cuptiProfilerHostGetConfigImageSize(&size_params),
                    "cuptiProfilerHostGetConfigImageSize")) {
      return false;
    }
    config_image.resize(size_params.configImageSize);

    CUpti_Profiler_Host_GetConfigImage_Params img_params = {};
    img_params.structSize =
        CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE;
    img_params.pPriv = nullptr;
    img_params.pHostObject = host_object;
    img_params.configImageSize = config_image.size();
    img_params.pConfigImage = config_image.data();

    if (!CheckCupti(cuptiProfilerHostGetConfigImage(&img_params),
                    "cuptiProfilerHostGetConfigImage")) {
      return false;
    }

    CUpti_Profiler_Host_GetNumOfPasses_Params passes_params = {};
    passes_params.structSize =
        CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE;
    passes_params.pPriv = nullptr;
    passes_params.configImageSize = config_image.size();
    passes_params.pConfigImage = config_image.data();

    if (!CheckCupti(cuptiProfilerHostGetNumOfPasses(&passes_params),
                    "cuptiProfilerHostGetNumOfPasses")) {
      return false;
    }
    num_passes = passes_params.numOfPasses;
    CBS_LOG_INFO("CUPTI Profiler: num passes = %zu", num_passes);

    host_initialized = true;
    return true;
  }

  // Check if the current metrics match the requested set.
  bool MetricsMatch(const std::vector<MetricId>& metrics) const {
    if (!host_initialized) return true;
    if (host_metrics.size() != metrics.size()) return false;
    for (size_t i = 0; i < metrics.size(); ++i) {
      if (host_metrics[i] != metrics[i]) return false;
    }
    return true;
  }

  void Reset() {
    host_metrics.clear();
    host_metric_names.clear();
    config_image.clear();
    num_passes = 0;
    host_initialized = false;
    // Note: do NOT reset 'initialized' — profiler/device init is process-wide
    // and does not need to be redone.
  }
};

struct CuptiProfiler::Impl {
  std::vector<MetricId> metrics;
  std::vector<const char*> metric_names;
  std::vector<double> metric_values;

  // Instance-level CUPTI objects (range profiler only — host object is global singleton).
  CUpti_RangeProfiler_Object* range_profiler = nullptr;

  // Images.
  std::vector<uint8_t> config_image;
  std::vector<uint8_t> counter_data_image;

  // Config.
  size_t max_ranges = 100;
  size_t num_nesting_levels = 1;
  size_t min_nesting_level = 1;
  CUpti_ProfilerReplayMode replay_mode = CUPTI_UserReplay;
  CUpti_ProfilerRange range_mode = CUPTI_UserRange; // CUPTI_UserRange, CUPTI_AutoRange;

  bool initialized = false;
  bool started = false;
  bool is_nested = false;  // true if created while another profiler was active on this thread

  // Multipass tracking.
  bool profiling_complete = false;  // all passes collected
  int next_pass_index = 0;          // next pass to submit
  int current_pass_index = 0;       // pass currently being profiled

  // Collected results.
  std::vector<ProfilingResult> results;

  // Activity API: number of kernel records before Start().
  size_t activity_record_start = 0;
};

CuptiProfiler::CuptiProfiler(std::vector<MetricId> metrics)
    : impl_(new Impl()) {
  impl_->metrics = std::move(metrics);
}

CuptiProfiler::~CuptiProfiler() { Shutdown(); }

void CuptiProfiler::SetRangeMode(CUpti_ProfilerRange mode) {
  impl_->range_mode = mode;
  if (mode == CUPTI_AutoRange) {
    impl_->num_nesting_levels = 1;
    impl_->min_nesting_level = 1;
  }
}

bool CuptiProfiler::Initialize() {
  if (impl_->initialized) return true;

  // Detect nesting: if another CuptiProfiler on this thread is already inside
  // ProfileMultiPass/StartWithPass/ProfileOnePass, this instance is nested.
  // Skip all CUPTI resource creation — Start/Stop/PushRange/PopRange will be
  // no-ops, and ProfileMultiPass will just run the workload once.
  impl_->is_nested = (g_profiling_depth > 0);

  auto& global = GlobalCuptiState::instance();
  if (!global.EnsureInitialized()) {
    return false;
  }

  if (impl_->is_nested) {
    // Nested instance: do NOT reset global host state (would invalidate the
    // outer's config_image/host_object), do NOT create a range_profiler or
    // counter_data_image. Just mark initialized so callers can proceed.
    impl_->initialized = true;
    return true;
  }

  // Reset global host state if metrics changed (allows re-profiling
  // with a different metric set in the same process).
  if (!global.MetricsMatch(impl_->metrics)) {
    global.Reset();
  }

  // Build metric names for host initialization.
  std::vector<const char*> metric_names;
  for (auto id : impl_->metrics) {
    metric_names.push_back(MetricToCuptiName(id));
  }

  // Initialize global host object (once per process, per metric set).
  if (!global.EnsureHostInitialized(impl_->metrics, metric_names)) {
    return false;
  }

  // Enable range profiler (per-instance).
  {
    CUpti_RangeProfiler_Enable_Params enable_params = {};
    enable_params.structSize = CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE;
    enable_params.pPriv = nullptr;
    enable_params.ctx = global.cuda_ctx;

    if (!CheckCupti(cuptiRangeProfilerEnable(&enable_params),
                    "cuptiRangeProfilerEnable")) {
      return false;
    }
    impl_->range_profiler = enable_params.pRangeProfilerObject;
  }

  // Create counter data image (per-instance).
  {
    CUpti_RangeProfiler_GetCounterDataSize_Params cd_size = {};
    cd_size.structSize =
        CUpti_RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE;
    cd_size.pPriv = nullptr;
    cd_size.pRangeProfilerObject = impl_->range_profiler;
    cd_size.pMetricNames = global.host_metric_names.data();
    cd_size.numMetrics = global.host_metric_names.size();
    cd_size.maxNumOfRanges = impl_->max_ranges;
    cd_size.maxNumRangeTreeNodes = impl_->max_ranges;

    if (!CheckCupti(cuptiRangeProfilerGetCounterDataSize(&cd_size),
                    "cuptiRangeProfilerGetCounterDataSize")) {
      return false;
    }
    impl_->counter_data_image.resize(cd_size.counterDataSize, 0);

    CUpti_RangeProfiler_CounterDataImage_Initialize_Params cd_init = {};
    cd_init.structSize =
        CUPTI_PROFILER_STRUCT_SIZE(
            CUpti_RangeProfiler_CounterDataImage_Initialize_Params,
            pCounterData);
    cd_init.pPriv = nullptr;
    cd_init.pRangeProfilerObject = impl_->range_profiler;
    cd_init.counterDataSize = impl_->counter_data_image.size();
    cd_init.pCounterData = impl_->counter_data_image.data();

    if (!CheckCupti(cuptiRangeProfilerCounterDataImageInitialize(&cd_init),
                    "cuptiRangeProfilerCounterDataImageInitialize")) {
      return false;
    }
  }

  // Set range profiler config for pass 0.
  if (!setConfigForPass(0)) return false;

  impl_->initialized = true;
  return true;
}

bool CuptiProfiler::setConfigForPass(int pass_index) {
  if (impl_->is_nested) return true;  // no-op for nested instances
  auto& global = GlobalCuptiState::instance();
  CUpti_RangeProfiler_SetConfig_Params set_config = {};
  set_config.structSize = CUpti_RangeProfiler_SetConfig_Params_STRUCT_SIZE;
  set_config.pPriv = nullptr;
  set_config.pRangeProfilerObject = impl_->range_profiler;
  set_config.pConfig = global.config_image.data();
  set_config.configSize = global.config_image.size();
  set_config.pCounterDataImage = impl_->counter_data_image.data();
  set_config.counterDataImageSize = impl_->counter_data_image.size();
  set_config.maxRangesPerPass = impl_->max_ranges;
  set_config.numNestingLevels = impl_->num_nesting_levels;
  set_config.minNestingLevel = impl_->min_nesting_level;
  set_config.passIndex = pass_index;
  set_config.targetNestingLevel = 1;
  set_config.range = impl_->range_mode;
  set_config.replayMode = impl_->replay_mode;

  return CheckCupti(cuptiRangeProfilerSetConfig(&set_config),
                    "cuptiRangeProfilerSetConfig");
}

bool CuptiProfiler::Start() {
  if (!impl_->initialized) {
    CBS_LOG_ERROR("CuptiProfiler: call Initialize() first");
    return false;
  }
  if (impl_->started) return true;
  if (impl_->is_nested) {
    // Nested instance: do not touch CUPTI. Mark started so the matching
    // Stop() path runs cleanly, but skip Activity/RangeProfiler setup.
    impl_->started = true;
    return true;
  }

  // Reset state for re-profiling support.
  impl_->profiling_complete = false;
  impl_->next_pass_index = 0;
  impl_->results.clear();

  // Enable Activity API and record the starting offset.
  auto& activity = ActivityState::instance();
  if (!activity.enabled) {
    if (!activity.Enable()) {
      CBS_LOG_ERROR("CuptiProfiler: failed to enable Activity API");
      return false;
    }
  }
  impl_->activity_record_start = activity.records.size();

  CUpti_RangeProfiler_Start_Params start_params = {};
  start_params.structSize = CUpti_RangeProfiler_Start_Params_STRUCT_SIZE;
  start_params.pPriv = nullptr;
  start_params.pRangeProfilerObject = impl_->range_profiler;

  if (!CheckCupti(cuptiRangeProfilerStart(&start_params),
                  "cuptiRangeProfilerStart")) {
    return false;
  }
  impl_->started = true;
  return true;
}

bool CuptiProfiler::Stop(int pass_index) {
  if (!impl_->started) return true;

  if (impl_->is_nested) {
    // Nested instance: nothing was started in CUPTI, so nothing to stop.
    // Mark profiling_complete so any outer loop short-circuits cleanly.
    impl_->started = false;
    impl_->profiling_complete = true;
    return true;
  }

  // Use the tracked pass index if caller passed the default sentinel.
  if (pass_index < 0) {
    pass_index = impl_->current_pass_index;
  }

  CUpti_RangeProfiler_Stop_Params stop_params = {};
  stop_params.structSize = CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE;
  stop_params.pPriv = nullptr;
  stop_params.pRangeProfilerObject = impl_->range_profiler;
  stop_params.passIndex = pass_index;
  stop_params.targetNestingLevel = 0;
  stop_params.isAllPassSubmitted = 0;

  if (!CheckCupti(cuptiRangeProfilerStop(&stop_params),
                  "cuptiRangeProfilerStop")) {
    return false;
  }

  bool all_submitted = stop_params.isAllPassSubmitted != 0;
  impl_->profiling_complete = all_submitted;
  impl_->next_pass_index = stop_params.passIndex;

  // Decode collected counter data.
  CUpti_RangeProfiler_DecodeData_Params decode_params = {};
  decode_params.structSize = CUpti_RangeProfiler_DecodeData_Params_STRUCT_SIZE;
  decode_params.pPriv = nullptr;
  decode_params.pRangeProfilerObject = impl_->range_profiler;

  if (!CheckCupti(cuptiRangeProfilerDecodeData(&decode_params),
                  "cuptiRangeProfilerDecodeData")) {
    return false;
  }

  impl_->started = false;

  // If multipass replay, caller needs to re-run and call Start() again.
  if (!all_submitted) {
    CBS_LOG_INFO("CUPTI Profiler: multipass replay — pass %d done, "
                 "need pass %d next. Call StartWithPass() and run workload again.",
                 pass_index, impl_->next_pass_index);
    return true;  // not done yet
  }

  // All passes collected — extract results.
  return extractResults();
}

bool CuptiProfiler::IsProfilingComplete() const {
  return impl_->profiling_complete;
}

int CuptiProfiler::NextPassIndex() const {
  return impl_->next_pass_index;
}

int CuptiProfiler::NumPasses() const {
  return static_cast<int>(GlobalCuptiState::instance().num_passes);
}

bool CuptiProfiler::ProfileMultiPass(
    const std::string& range_name,
    std::function<void(int pass)> workload_fn) {
  // Nested instance: another profiler on this thread is already driving CUPTI.
  // Do NOT re-enter multipass replay (would cause exponential workload reruns
  // and corrupt the outer's CUPTI state). Just execute the workload once so
  // the real computation still happens.
  if (impl_->is_nested) {
    workload_fn(0);
    return true;
  }

  // Track profiling depth so nested CuptiProfiler::Initialize() calls (from
  // inside workload_fn) can detect that they are nested.
  struct DepthGuard {
    DepthGuard() { ++g_profiling_depth; }
    ~DepthGuard() { --g_profiling_depth; }
  } guard;

  int pass = 0;

  for (;;) {
    if (!setConfigForPass(pass)) return false;
    if (!Start()) return false;
    if (!PushRange(range_name)) return false;
    workload_fn(pass);
    if (!PopRange()) return false;
    if (!Stop(pass)) return false;

    if (IsProfilingComplete()) break;
    pass = NextPassIndex();
  }
  return true;
}

// ─── Single-pass / manual-pass API (no function pointer) ─────────────────

bool CuptiProfiler::StartWithPass(int pass_index) {
  if (!impl_->initialized) {
    CBS_LOG_ERROR("CuptiProfiler: call Initialize() first");
    return false;
  }
  if (impl_->started) {
    CBS_LOG_ERROR("CuptiProfiler: already started, call Stop() first");
    return false;
  }

  if (impl_->is_nested) {
    // Nested: mark started with the requested pass so Stop/PushRange/PopRange
    // short-circuit cleanly, but do not touch CUPTI. Note: we do NOT bump
    // g_profiling_depth here — the manual API has no natural scope for a
    // RAII decrement, and nesting detection is driven by ProfileMultiPass.
    impl_->current_pass_index = pass_index;
    impl_->started = true;
    return true;
  }

  impl_->current_pass_index = pass_index;

  if (!setConfigForPass(pass_index)) return false;

  CUpti_RangeProfiler_Start_Params start_params = {};
  start_params.structSize = CUpti_RangeProfiler_Start_Params_STRUCT_SIZE;
  start_params.pPriv = nullptr;
  start_params.pRangeProfilerObject = impl_->range_profiler;

  if (!CheckCupti(cuptiRangeProfilerStart(&start_params),
                  "cuptiRangeProfilerStart")) {
    return false;
  }
  impl_->started = true;
  return true;
}

bool CuptiProfiler::ProfileOnePass(const std::string& range_name) {
  int pass = impl_->next_pass_index;

  if (impl_->is_nested) {
    // Nested: run the "start + push" dance as no-ops; caller still drives
    // its workload, then PopRange/Stop which are also no-ops.
    impl_->current_pass_index = pass;
    impl_->started = true;
    return true;
  }

  if (!StartWithPass(pass)) return false;
  if (!PushRange(range_name)) return false;

  // PopRange and Stop are called by the caller after running their workload.
  return true;
}

bool CuptiProfiler::PushRange(const std::string& name) {
  if (!impl_->started) {
    CBS_LOG_ERROR("CuptiProfiler: call Start() first");
    return false;
  }
  if (impl_->is_nested) return true;  // no-op for nested instances

  CUpti_RangeProfiler_PushRange_Params push_params = {};
  push_params.structSize = CUpti_RangeProfiler_PushRange_Params_STRUCT_SIZE;
  push_params.pPriv = nullptr;
  push_params.pRangeProfilerObject = impl_->range_profiler;
  push_params.pRangeName = name.c_str();

  return CheckCupti(cuptiRangeProfilerPushRange(&push_params),
                    "cuptiRangeProfilerPushRange");
}

bool CuptiProfiler::PopRange() {
  if (!impl_->started) {
    CBS_LOG_ERROR("CuptiProfiler: call Start() first");
    return false;
  }
  if (impl_->is_nested) return true;  // no-op for nested instances

  CUpti_RangeProfiler_PopRange_Params pop_params = {};
  pop_params.structSize = CUpti_RangeProfiler_PopRange_Params_STRUCT_SIZE;
  pop_params.pPriv = nullptr;
  pop_params.pRangeProfilerObject = impl_->range_profiler;

  return CheckCupti(cuptiRangeProfilerPopRange(&pop_params),
                    "cuptiRangeProfilerPopRange");
}

std::vector<ProfilingResult> CuptiProfiler::GetResults() const {
  return impl_->results;
}

std::string CuptiProfiler::FormatResults() const {
  std::ostringstream oss;
  if (impl_->is_nested) return oss.str();  // nested instances have no results

  oss << "\n================== CUPTI Profiling Results ==================\n";
  for (const auto& result : impl_->results) {
    oss << "\n--- " << result.range_name << " ---\n";

    // Print kernel launch config in AutoRange mode.
    if (result.kernel_info.has_activity_info) {
      const auto& ki = result.kernel_info;
      oss << "  Kernel:       " << ki.demangled_name << "\n";
      oss << "  Grid:         (" << ki.grid_x << ", " << ki.grid_y << ", "
          << ki.grid_z << ")\n";
      oss << "  Block:        (" << ki.block_x << ", " << ki.block_y << ", "
          << ki.block_z << ")\n";
      oss << "  Registers:    " << ki.registers_per_thread << "/thread\n";
      oss << "  Shared Mem:   " << ki.shared_memory_executed << " bytes\n";
      oss << "  Duration:     " << ((ki.end_ns - ki.start_ns) / 1000.0)
          << " us\n";
      oss << "  ──────────────────────────────────────────────────\n";
    }

    for (const auto& [id, value] : result.values) {
      std::string formatted = FormatMetricValue(id, value);
      char line[256];
      snprintf(line, sizeof(line), "  %-40s %s",
               MetricToLabel(id), formatted.c_str());
      oss << line << "\n";
    }
  }
  oss << "==============================================================\n\n";
  return oss.str();
}

void CuptiProfiler::PrintResults() const {
  std::cout << FormatResults();
}

// ─── Serialization helpers ────────────────────────────────────────────────

static std::string getTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) % 1000;
  auto tm = std::localtime(&time_t_now);
  std::ostringstream oss;
  oss << std::put_time(tm, "%Y-%m-%dT%H:%M:%S")
      << '.' << std::setfill('0') << std::setw(3) << ms.count();
  return oss.str();
}

static nlohmann::json serializeResults(
    const std::vector<ProfilingResult>& results) {
  nlohmann::json ranges = nlohmann::json::array();
  for (const auto& result : results) {
    nlohmann::json range;
    range["range_name"] = result.range_name;

    if (result.kernel_info.has_activity_info) {
      const auto& ki = result.kernel_info;
      nlohmann::json kernel;
      kernel["name"] = ki.demangled_name.empty() ? ki.name : ki.demangled_name;
      kernel["mangled_name"] = ki.name;
      kernel["grid"] = nlohmann::json::array({ki.grid_x, ki.grid_y, ki.grid_z});
      kernel["block"] = nlohmann::json::array({ki.block_x, ki.block_y, ki.block_z});
      kernel["registers_per_thread"] = ki.registers_per_thread;
      kernel["shared_memory_bytes"] = ki.shared_memory_executed;
      kernel["duration_ns"] = ki.end_ns - ki.start_ns;
      range["kernel"] = kernel;
    }

    nlohmann::json metrics = nlohmann::json::object();
    for (const auto& [id, value] : result.values) {
      metrics[MetricToLabel(id)] = value;
    }
    range["metrics"] = metrics;
    ranges.push_back(range);
  }
  return ranges;
}

// ─── SaveToJson ───────────────────────────────────────────────────────────

bool CuptiProfiler::SaveToJson(const char* path, WriteMode mode) const {
  if (impl_->is_nested) return true;  // nested instances have no results to save
  nlohmann::json ranges = serializeResults(impl_->results);

  if (mode == WriteMode::Overwrite) {
    nlohmann::json root;
    root["timestamp"] = getTimestamp();
    root["ranges"] = std::move(ranges);

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
      CBS_LOG_ERROR("CuptiProfiler: cannot open '%s' for writing", path);
      return false;
    }
    ofs << root.dump(2) << "\n";
    ofs.close();
    return true;
  }

  // Append mode: wrap as single NDJSON line, write asynchronously.
  nlohmann::json line;
  line["timestamp"] = getTimestamp();
  line["ranges"] = std::move(ranges);
  std::string json_line = line.dump();

  // Wait for any previous pending async write.
  {
    std::lock_guard<std::mutex> lock(write_mtx_);
    if (write_fut_.valid()) write_fut_.wait();
  }

  write_fut_ = std::async(std::launch::async,
    [path_str = std::string(path), line = std::move(json_line)]() {
      std::ofstream ofs(path_str, std::ios::app);
      if (!ofs.is_open()) {
        CBS_LOG_ERROR("CuptiProfiler: cannot open '%s' for append", path_str.c_str());
        return;
      }
      ofs << line << "\n";
      ofs.close();
    });
  return true;
}

void CuptiProfiler::Shutdown() {
  if (!impl_) return;

  // Wait for any pending async write to complete.
  {
    std::lock_guard<std::mutex> lock(write_mtx_);
    if (write_fut_.valid()) write_fut_.wait();
  }

  if (impl_->started) {
    Stop();
  }

  // Only the outermost profiler on a thread owns the process-wide Activity
  // state. Nested instances must leave it alone — clearing it here would
  // discard the kernel records the outer instance is still collecting.
  if (!impl_->is_nested) {
    ActivityState::instance().Reset();
  }

  // range_profiler is nullptr for nested instances (we skipped creation in
  // Initialize), so this block is naturally a no-op for them.
  if (impl_->range_profiler) {
    CUpti_RangeProfiler_Disable_Params disable_params = {};
    disable_params.structSize =
        CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE;
    disable_params.pPriv = nullptr;
    disable_params.pRangeProfilerObject = impl_->range_profiler;
    cuptiRangeProfilerDisable(&disable_params);
    impl_->range_profiler = nullptr;
  }

  impl_->initialized = false;
}

// ─── extractResults helpers ───────────────────────────────────────────────

std::vector<ActivityKernelRecord> CuptiProfiler::GetCapturedKernels() const {
  auto& activity = ActivityState::instance();
  activity.Flush();

  if (impl_->activity_record_start < activity.records.size()) {
    return std::vector<ActivityKernelRecord>(
        activity.records.begin() + impl_->activity_record_start,
        activity.records.end());
  }
  return {};
}

size_t CuptiProfiler::GetRangeCount() const {
  CUpti_RangeProfiler_GetCounterDataInfo_Params info_params = {};
  info_params.structSize =
      CUPTI_PROFILER_STRUCT_SIZE(
          CUpti_RangeProfiler_GetCounterDataInfo_Params, numTotalRanges);
  info_params.pPriv = nullptr;
  info_params.pCounterDataImage = impl_->counter_data_image.data();
  info_params.counterDataImageSize = impl_->counter_data_image.size();

  CUptiResult res = cuptiRangeProfilerGetCounterDataInfo(&info_params);
  if (res == CUPTI_SUCCESS && info_params.numTotalRanges > 0) {
    return info_params.numTotalRanges;
  }
  return 0;
}

bool CuptiProfiler::EvaluateRange(size_t range_index,
                                  GlobalCuptiState* global,
                                  ProfilingResult* out) {
  CUpti_Profiler_Host_EvaluateToGpuValues_Params eval_params = {};
  eval_params.structSize =
      CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE;
  eval_params.pPriv = nullptr;
  eval_params.pHostObject = global->host_object;
  eval_params.pCounterDataImage = impl_->counter_data_image.data();
  eval_params.counterDataImageSize = impl_->counter_data_image.size();
  eval_params.rangeIndex = range_index;
  eval_params.ppMetricNames = global->host_metric_names.data();
  eval_params.numMetrics = global->host_metric_names.size();
  eval_params.pMetricValues = impl_->metric_values.data();

  if (!CheckCupti(cuptiProfilerHostEvaluateToGpuValues(&eval_params),
                  "cuptiProfilerHostEvaluateToGpuValues")) {
    return false;
  }

  for (size_t i = 0; i < global->host_metrics.size(); ++i) {
    out->values[global->host_metrics[i]] = impl_->metric_values[i];
  }
  return true;
}

void CuptiProfiler::MatchKernelToRange(
    size_t range_index,
    const std::vector<ActivityKernelRecord>& kernels,
    ProfilingResult* out) {
  if (range_index >= kernels.size()) return;

  const auto& rec = kernels[range_index];
  out->kernel_info.name = rec.name;
  out->kernel_info.demangled_name = Demangle(rec.name.c_str());
  out->kernel_info.grid_x = rec.grid_x;
  out->kernel_info.grid_y = rec.grid_y;
  out->kernel_info.grid_z = rec.grid_z;
  out->kernel_info.block_x = rec.block_x;
  out->kernel_info.block_y = rec.block_y;
  out->kernel_info.block_z = rec.block_z;
  out->kernel_info.registers_per_thread = rec.registers_per_thread;
  out->kernel_info.shared_memory_bytes = rec.shared_memory_bytes;
  out->kernel_info.shared_memory_executed = rec.shared_memory_executed;
  out->kernel_info.grid_id = rec.grid_id;
  out->kernel_info.start_ns = rec.start_ns;
  out->kernel_info.end_ns = rec.end_ns;
  out->kernel_info.has_activity_info = true;
}

bool CuptiProfiler::extractResults() {
  auto& global = GlobalCuptiState::instance();
  impl_->results.clear();

  std::vector<ActivityKernelRecord> captured_kernels = GetCapturedKernels();
  size_t num_ranges = GetRangeCount();

  if (num_ranges == 0) {
    CBS_LOG_WARNING("CUPTI Profiler: no counter data available");
    return true;
  }

  impl_->metric_values.resize(global.host_metric_names.size());

  for (size_t r = 0; r < num_ranges; ++r) {
    ProfilingResult result;

    // Get range name.
    result.range_name = "range_" + std::to_string(r);
    {
      CUpti_RangeProfiler_CounterData_GetRangeInfo_Params range_info = {};
      range_info.structSize =
          CUPTI_PROFILER_STRUCT_SIZE(
              CUpti_RangeProfiler_CounterData_GetRangeInfo_Params, rangeName);
      range_info.pPriv = nullptr;
      range_info.pCounterDataImage = impl_->counter_data_image.data();
      range_info.counterDataImageSize = impl_->counter_data_image.size();
      range_info.rangeIndex = r;
      range_info.rangeDelimiter = "/";

      if (cuptiRangeProfilerCounterDataGetRangeInfo(&range_info) ==
          CUPTI_SUCCESS) {
        result.range_name = range_info.rangeName;
      }
    }

    // Evaluate metrics.
    if (!EvaluateRange(r, &global, &result)) {
      continue;
    }

    // Match kernel info.
    MatchKernelToRange(r, captured_kernels, &result);

    impl_->results.push_back(std::move(result));
  }

  return true;
}

}  // namespace profiler
}  // namespace cpp_base_suite
