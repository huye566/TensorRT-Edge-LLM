#include "profiler/metrics.hpp"

#include <cmath>
#include <cstdio>

namespace cpp_base_suite {
namespace profiler {

// ─── Architecture availability ─────────────────────────────────────────────
#ifdef CUPTI_PROFILER_ARCH
constexpr int kCurrentArch = CUPTI_PROFILER_ARCH;
#else
constexpr int kCurrentArch = 0;  // runtime check if not defined at compile time
#endif

// ─── Table: auto-generated from X-macro (order matches enum 100%) ──────────
const MetricEntry kMetricTable[] = {
#define METRIC(name, cupti, lbl, un, arch) { cupti, lbl, un, arch },
  PROFILER_METRIC_LIST
#undef METRIC
};

static_assert(sizeof(kMetricTable) / sizeof(kMetricTable[0]) ==
                  static_cast<int>(MetricId::Count),
              "kMetricTable size must match MetricId enum count");

// ─── Table-based accessors ─────────────────────────────────────────────────
const char* MetricToCuptiName(MetricId id) {
  return kMetricTable[static_cast<int>(id)].cupti_name;
}

const char* MetricToLabel(MetricId id) {
  return kMetricTable[static_cast<int>(id)].label;
}

const char* MetricToUnit(MetricId id) {
  return kMetricTable[static_cast<int>(id)].unit;
}

// Format metric value for display: auto-scale to human-friendly unit.
// Returns "{value} {unit}" (e.g., "128.5 MB", "1.23 GHz", "50.0 us").
// Only affects printing — raw data unchanged.
std::string FormatMetricValue(MetricId id, double raw_value) {
  const char* unit = MetricToUnit(id);
  const char* label = MetricToLabel(id);
  char buf[128];

  // Check if label contains "* 32" suffix, if so multiply value by 32
  double display_value = raw_value;
  if (std::string(label).find("* 32") != std::string::npos) {
    display_value = raw_value * 32.0;
  }

  if (unit[0] == '%' || std::string(unit).find("/warps_issue") != std::string::npos ||
      std::string(unit).find("inst/cycle") != std::string::npos ||
      std::string(unit).find("cycles") != std::string::npos ||
      std::string(unit).find("warp") != std::string::npos ||
      std::string(unit).find("sectors") != std::string::npos ||
      std::string(unit).find("num") != std::string::npos) {
    snprintf(buf, sizeof(buf), "%.2f %s", display_value, unit);
    return buf;
  }

  if (std::string(unit) == "bytes") {
    const char* scales[] = {"B", "KB", "MB", "GB", "TB"};
    int i = 0;
    double v = display_value;
    while (std::fabs(v) >= 1024.0 && i < 4) { v /= 1024.0; i++; }
    snprintf(buf, sizeof(buf), "%.2f %s", v, scales[i]);
    return buf;
  }

  if (std::string(unit) == "byte/s") {
    const char* scales[] = {"B/s", "KB/s", "MB/s", "GB/s", "TB/s"};
    int i = 0;
    double v = display_value;
    while (std::fabs(v) >= 1024.0 && i < 4) { v /= 1024.0; i++; }
    snprintf(buf, sizeof(buf), "%.2f %s", v, scales[i]);
    return buf;
  }

  if (std::string(unit) == "hz") {
    const char* scales[] = {"Hz", "KHz", "MHz", "GHz"};
    int i = 0;
    double v = display_value;
    while (std::fabs(v) >= 1000.0 && i < 3) { v /= 1000.0; i++; }
    snprintf(buf, sizeof(buf), "%.2f %s", v, scales[i]);
    return buf;
  }

  if (std::string(unit) == "ns") {
    const char* scales[] = {"ns", "us", "ms", "s"};
    int i = 0;
    double v = display_value;
    while (std::fabs(v) >= 1000.0 && i < 3) { v /= 1000.0; i++; }
    snprintf(buf, sizeof(buf), "%.2f %s", v, scales[i]);
    return buf;
  }

  // Fallback: print raw value with original unit.
  snprintf(buf, sizeof(buf), "%.2f %s", display_value, unit);
  return buf;
}

// ─── Architecture availability ─────────────────────────────────────────────
bool IsMetricAvailable(MetricId id) {
  int arch = kCurrentArch;
  int idx = static_cast<int>(id);
  if (idx < 0 || idx >= static_cast<int>(MetricId::Count)) return false;
  if (arch > 0 && arch < kMetricTable[idx].arch_min) return false;
  return true;
}

// ─── Preset metric sets ────────────────────────────────────────────────────
std::vector<MetricId> DefaultMetrics() {
  return {
      MetricId::GPUTimeDuration,
      MetricId::SMUtilization,
#ifndef PLATFORM_THOR
      MetricId::DRAMBytesRead,
      MetricId::DRAMBytesWritten,
#else
      MetricId::SOCBytesRead,
      MetricId::SOCBytesWritten,
#endif
      MetricId::SMFrequency,
//       MetricId::DRAMFrequency,
  };
}

std::vector<MetricId> WarpStallMetrics() {
  return {
      MetricId::WarpStallBarrier,
      MetricId::WarpStallBranchResolving,
      MetricId::WarpStallDispatchStall,
      MetricId::WarpStallDrain,
#ifndef PLATFORM_THOR
      MetricId::WarpStallIMCMiss,
#endif
      MetricId::WarpStallLGThrottle,
      MetricId::WarpStallLongScoreboard,
      MetricId::WarpStallMathPipeThrottle,
      MetricId::WarpStallMembar,
      MetricId::WarpStallMIOThrottle,
      MetricId::WarpStallMisc,
      MetricId::WarpStallNoInstruction,
      MetricId::WarpStallNotSelected,
      MetricId::WarpStallSelected,
      MetricId::WarpStallShortScoreboard,
      MetricId::WarpStallSleeping,
      MetricId::WarpStallTexThrottle,
      MetricId::WarpStallWait,
#ifdef CUPTI_PROFILER_ARCH_GE_80
//       MetricId::MemoryStallRate,
#endif
  };
}

std::vector<MetricId> LaunchMetrics() {
  return {
      MetricId::LaunchWavesPerMultiprocessor,
      MetricId::LaunchTpcCount,
      MetricId::LaunchSmCount,
      MetricId::LaunchThreadCount,
      MetricId::LaunchGridSize,
      MetricId::LaunchBlockSize,
      MetricId::LaunchStackSize,
      MetricId::LaunchRegistersPerThread,
      MetricId::LaunchOccupancyLimitWrap,
      MetricId::LaunchOccupancyLimitReg,
      MetricId::LaunchOccupancyLimitSharedMem,
      MetricId::LaunchOccupancyLimitBlock,
      MetricId::LaunchOccupancyPerSharedMemSize,
      MetricId::LaunchOccupancyPerBlockSize,
      MetricId::LaunchOccupancyPerRegisterCount,
      MetricId::LaunchSharedMemPerBlockStatic,
      MetricId::LaunchSharedMemPerBlockDynamic,
      MetricId::LaunchSharedMemPerBlockDriver,
      MetricId::LaunchSharedMemPerBlock,
      MetricId::LaunchSharedMemConfigSize,
      MetricId::SMWarpsActivePerScheduler,
      MetricId::TheoreticalOccupancy,
      MetricId::TheoreticalActiveWarpsPerSM,
      MetricId::AchievedOccupancy,
      MetricId::AchievedActiveWarpsPerSM
  };
}

std::vector<MetricId> CacheMetrics() {
  return {
      MetricId::L2TSectors,
      MetricId::L2TSectorsRead,
      MetricId::L2TSectorsWrite,
      MetricId::L2TSectorsReadHit,
      MetricId::L2TSectorsReadMiss,
      MetricId::L1TEXBytesGlobalLoad,
      MetricId::L1TEXBytesGlobalStore,
      MetricId::L1TEXSectorsGlobalLoadHit,
      MetricId::L1TEXSectorsGlobalLoadMiss,
#ifndef PLATFORM_THOR
      MetricId::L1FromL2Read,
      MetricId::L1FromL2ReadPerSecond,
      MetricId::L1toL2Write,
      MetricId::L1toL2WritePerSecond,
#else
      MetricId::L1FromLRCRead,
      MetricId::L1FromLRCReadPerSecond,
      MetricId::L1toLRCWrite,
      MetricId::L1toLRCWritePerSecond,
#endif
  };
}

std::vector<MetricId> ComputeMemMetrics() {
  return {
      MetricId::SMemInstExecuted,
      MetricId::TMemRead,
      MetricId::TMemReadWrite,
      MetricId::TMemInstExecutedLd,
      MetricId::TMemInstExecutedSt,
#ifdef PLATFORM_THOR
      MetricId::TMAInstLoad,
      MetricId::TMAInstStore,
      MetricId::TMAInstExecutedRate,
      MetricId::TMACyclesActiveRate,
      MetricId::DSMemInstExecuted,
#endif
  };
}

std::vector<MetricId> InstructionMixMetrics() {
  return {
      MetricId::InstExecutedFAdd,
      MetricId::InstExecutedFMul,
      MetricId::InstExecutedFFMA,
      MetricId::InstructionsExecuted,
  };
}

std::vector<MetricId> PerformanceProfileMetrics() {
  std::vector<MetricId> metrics = {
      MetricId::GPUTimeDuration,
#ifndef PLATFORM_THOR
      MetricId::DRAMBytesRead,
      MetricId::DRAMBytesReadPerSecond,
      MetricId::DRAMBytesWritten,
      MetricId::DRAMBytesWrittenPerSecond,
      MetricId::DRAMBytesPerSecond,
      MetricId::DRAMThroughput,
      MetricId::DRAMFrequency,
      MetricId::L1FromL2Read,
      MetricId::L1FromL2ReadPerSecond,
      MetricId::L1toL2Write,
      MetricId::L1toL2WritePerSecond,
#else
      MetricId::SOCBytesRead,
      MetricId::SOCBytesReadPerSecond,
      MetricId::SOCBytesWritten,
      MetricId::SOCBytesWrittenPerSecond,
      MetricId::L1FromLRCRead,
      MetricId::L1FromLRCReadPerSecond,
      MetricId::L1toLRCWrite,
      MetricId::L1toLRCWritePerSecond,
#endif
      MetricId::SMUtilization,
      MetricId::SMThroughput,
      MetricId::GPUMemoryThroughput,
      MetricId::L1TEXThroughput,
      MetricId::L2Throughput,
      MetricId::GPCCyclesElapsedMax,
      MetricId::SMCyclesActive,
      MetricId::SMFrequency,
      MetricId::L1TEXHitRate,
      MetricId::L2HitRate,
      MetricId::GPUMemBusy,
      MetricId::GPUMemMaxBandwidth,
      MetricId::SMMemoryPipesBusy,
      MetricId::IPCExecutedElapsed,
      MetricId::IPCExecutedActive,
      MetricId::IPCIssuedActive,
      MetricId::SMIssueSlotsBusy,
      MetricId::SMMemBusyByInst,
      MetricId::SMWarpsActivePerScheduler,
      // MetricId::TheoreticalOccupancy,
      // MetricId::TheoreticalActiveWarpsPerSM,
      MetricId::AchievedOccupancy,
      MetricId::AchievedActiveWarpsPerSM,
  };

  return metrics;
}

std::vector<MetricId> MainProfileMetrics() {
  auto metrics = PerformanceProfileMetrics();
  auto warp = WarpStallMetrics();
  metrics.insert(metrics.end(),
                  std::make_move_iterator(warp.begin()),
                  std::make_move_iterator(warp.end()));

  return metrics;
}

// Returns all metrics supported at compile time.
std::vector<MetricId> AllSupportedMetrics() {
  std::vector<MetricId> metrics;
  for (int i = 0; i < static_cast<int>(MetricId::Count); ++i) {
    auto id = static_cast<MetricId>(i);
    if (IsMetricAvailable(id)) {
      metrics.push_back(id);
    }
  }
  return metrics;
}

}  // namespace profiler
}  // namespace cpp_base_suite
