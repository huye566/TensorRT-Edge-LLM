#ifndef PROFILER_METRICS_HPP_
#define PROFILER_METRICS_HPP_

#include <string>
#include <vector>

namespace cpp_base_suite {
namespace profiler {

// ─── X-Macro: single source of truth for all metrics ───────────────────────
// Format: METRIC(name, cupti_name, label, unit, arch_min)
//
// Add/delete metrics here ONLY.  Enum and table are auto-generated below.
//
#define PROFILER_METRIC_LIST \
  /* ── Timing / Throughput ──────────────────────────────────────── */ \
  METRIC(GPUTimeDuration, \
         "gpu__time_duration.sum", \
         "GPU Duration", "ns", 70) \
  METRIC(SMUtilization, \
         "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed", \
         "SM Utilization", "%", 70) \
  METRIC(SMThroughput, \
         "sm__throughput.avg.pct_of_peak_sustained_elapsed", \
         "SM Throughput", "%", 80) \
  METRIC(GPUMemoryThroughput, \
         "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", \
         "Memory Throughput", "%", 80) \
  METRIC(L1TEXThroughput, \
         "l1tex__throughput.avg.pct_of_peak_sustained_active", \
         "L1/TEX Throughput", "%", 70) \
  METRIC(L2Throughput, \
         "lts__throughput.avg.pct_of_peak_sustained_elapsed", \
         "L2 Throughput", "%", 70) \
  METRIC(DRAMThroughput, \
         "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", \
         "DRAM Throughput", "%", 70) \
  \
  /* ── Frequency / Cycles ───────────────────────────────────────── */ \
  METRIC(GPCCyclesElapsedMax, \
         "gpc__cycles_elapsed.max", \
         "Elapsed Cycles", "cycles", 70) \
  METRIC(SMCyclesActive, \
         "sm__cycles_active.avg", \
         "SM Active Cycles", "cycles", 70) \
  METRIC(SMFrequency, \
         "gpc__cycles_elapsed.avg.per_second", \
         "SM Frequency", "hz", 70) \
  METRIC(DRAMFrequency, \
         "dram__cycles_elapsed.avg.per_second", \
         "DRAM Frequency", "hz", 70) \
  METRIC(SMCyclesActivePerCycleActive, \
         "smsp__cycles_active.avg.per_cycle_active", \
         "SM Cycles Active", "cycles", 70) \
  \
  /* ── DRAM / Soc Memory ────────────────────────────────────────────── */ \
  METRIC(DRAMBytesRead, \
         "dram__bytes_read.sum", \
         "DRAM Bytes Read", "bytes", 70) \
  METRIC(DRAMBytesReadPerSecond, \
         "dram__bytes_read.sum.per_second", \
         "DRAM Read Throughput", "byte/s", 70) \
  METRIC(DRAMBytesWritten, \
         "dram__bytes_write.sum", \
         "DRAM Bytes Written", "bytes", 70) \
  METRIC(DRAMBytesWrittenPerSecond, \
         "dram__bytes_write.sum.per_second", \
         "DRAM Written Throughput", "byte/s", 70) \
  METRIC(SOCBytesRead, \
         "lts__d_sectors_fill_sysmem.sum", \
         "Soc Bytes Read * 32", "bytes", 80) \
  METRIC(SOCBytesReadPerSecond, \
         "lts__d_sectors_fill_sysmem.sum.per_second", \
         "Soc Read Throughput * 32", "byte/s", 80) \
  METRIC(SOCBytesWritten, \
         "lts__t_sectors_aperture_sysmem_op_write.sum", \
         "Soc Bytes Written * 32", "bytes", 80) \
  METRIC(SOCBytesWrittenPerSecond, \
         "lts__t_sectors_aperture_sysmem_op_write.sum.per_second", \
         "Soc Written Throughput * 32", "byte/s", 80) \
  METRIC(DRAMBytesTotal, \
         "dram__bytes.sum", \
         "DRAM Bytes Total", "bytes", 70) \
  METRIC(DRAMBytesPerSecond, \
         "dram__bytes.sum.per_second", \
         "DRAM Throughput", "byte/s", 70) \
  METRIC(L1TEXHitRate, \
         "l1tex__t_sector_hit_rate.pct", \
         "L1/TEX Hit Rate", "%", 70) \
  METRIC(L2HitRate, \
         "lts__t_sector_hit_rate.pct", \
         "L2 Hit Rate", "%", 70) \
  METRIC(GPUMemBusy, \
         "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed", \
         "Mem Busy", "%", 80) \
  METRIC(GPUMemMaxBandwidth, \
         "gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed", \
         "Max Bandwidth", "%", 80) \
  \
  /* ── SM Memory Throughput ─────────────────────────────────────── */ \
  METRIC(SMMemoryPipesBusy, \
         "sm__memory_throughput.avg.pct_of_peak_sustained_elapsed", \
         "Mem Pipes Busy", "%", 70) \
  \
  /* ── L2 Cache ─────────────────────────────────────────────────── */ \
  METRIC(L2TSectors, \
         "lts__t_sectors.sum", \
         "L2T Sectors", "sectors", 70) \
  METRIC(L2TSectorsRead, \
         "lts__t_sectors_op_read.sum", \
         "L2T Sectors Read", "sectors", 70) \
  METRIC(L2TSectorsWrite, \
         "lts__t_sectors_op_write.sum", \
         "L2T Sectors Write", "sectors", 70) \
  METRIC(L2TSectorsReadHit, \
         "lts__t_sectors_op_read_lookup_hit.sum", \
         "L2T Sectors Read Hit", "sectors", 70) \
  METRIC(L2TSectorsReadMiss, \
         "lts__t_sectors_op_read_lookup_miss.sum", \
         "L2T Sectors Read Miss", "sectors", 70) \
  \
  /* ── L1/TEX Cache ─────────────────────────────────────────────── */ \
  METRIC(L1TEXBytesGlobalLoad, \
         "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum", \
         "L1TEX Bytes Global Load", "bytes", 70) \
  METRIC(L1TEXBytesGlobalStore, \
         "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum", \
         "L1TEX Bytes Global Store", "bytes", 70) \
  METRIC(L1TEXSectorsGlobalLoadHit, \
         "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum", \
         "L1TEX Sectors Global Load Hit", "sectors", 70) \
  METRIC(L1TEXSectorsGlobalLoadMiss, \
         "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum", \
         "L1TEX Sectors Global Load Miss", "sectors", 70) \
  METRIC(L1FromL2Read, \
         "l1tex__m_xbar2l1tex_read_bytes.sum", \
         "L1 Read From L2 Bytes", "bytes", 70) \
  METRIC(L1FromL2ReadPerSecond, \
         "l1tex__m_xbar2l1tex_read_bytes.sum.per_second", \
         "L1 Read From L2 Throughput", "byte/s", 70) \
  METRIC(L1toL2Write, \
         "l1tex__m_l1tex2xbar_write_bytes.sum", \
         "L1 Write to L2 Bytes", "bytes", 70) \
  METRIC(L1toL2WritePerSecond, \
         "l1tex__m_l1tex2xbar_write_bytes.sum.per_second", \
         "L1 Write to L2 Throughput", "byte/s", 70) \
  METRIC(L1FromLRCRead, \
         "lrc__xbar2gpc_sectors_op_read.sum", \
         "L1 Read From LRC Bytes * 32", "bytes", 70) \
  METRIC(L1FromLRCReadPerSecond, \
         "lrc__xbar2gpc_sectors_op_read.sum.per_second", \
         "L1 Read From LRC Throughput * 32", "byte/s", 70) \
  METRIC(L1toLRCWrite, \
         "lrc__xbar2lrc_requests_op_read.sum", \
         "L1 Write to LRC Bytes * 32", "bytes", 70) \
  METRIC(L1toLRCWritePerSecond, \
         "lrc__xbar2lrc_requests_op_read.sum.per_second", \
         "L1 Write to LRC Throughput * 32", "byte/s", 70) \
  \
  /* ── TMA ──────────────────────────────────────────────────────── */ \
  METRIC(TMAInstLoad, \
         "smsp__inst_executed_op_tma_ld.sum", \
         "TMA Instructions-Load", "inst", 90) \
  METRIC(TMAInstStore, \
         "smsp__inst_executed_op_tma_st.sum", \
         "TMA Instructions-Store", "inst", 90) \
  /* 用于衡量tma被启用的频率， 如果很高，说明你在非常频繁地用小数据块启动TMA */ \
  METRIC(TMAInstExecutedRate, \
         "sm__inst_executed_pipe_tma.avg.pct_of_peak_sustained_active", \
         "TMA Inst Executed Rate", "%", 90) \
  /* 记录 TMA 引擎实际用来完成任务的绝对时长， 如果 inst_executed 低但 cycles_active 高，说明每次 TMA 任务数据量大、持续时间长，这通常代表更高效的硬件利用 */ \
  METRIC(TMACyclesActiveRate, \
         "sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_active", \
         "TMA Cycles Active Rate", "%", 90) \
  \
  /* ── Shared Memory (Shared, DSMEM)/ Tensor Memory───────────────── */ \
  METRIC(SMemInstExecuted, \
         "smsp__sass_inst_executed_op_shared.sum", \
         "SMem Inst Executed", "inst", 100) \
  METRIC(DSMemInstExecuted, \
         "smsp__sass_inst_executed_op_dshared.sum", \
         "Distributed SMem Inst Executed", "inst", 100) \
  METRIC(TMemRead, \
         "smsp__mem_tensor_reads_op_ldt.sum", \
         "Tensor Mem Bytes Read", "bytes", 100) \
  METRIC(TMemReadWrite, \
         "smsp__mem_tensor_writes_op_stt.sum", \
         "Tensor Mem Bytes Write", "bytes", 100) \
  METRIC(TMemInstExecutedLd, \
         "smsp__sass_inst_executed_op_tmem_ldt.sum", \
         "Tensor Mem Inst Load", "inst", 100) \
  METRIC(TMemInstExecutedSt, \
         "smsp__sass_inst_executed_op_tmem_stt.sum", \
         "Tensor Mem Inst Store", "inst", 100) \
  \
  /* ── IPC / Issue ──────────────────────────────────────────────── */ \
  METRIC(IPCExecutedElapsed, \
         "sm__inst_executed.avg.per_cycle_elapsed", \
         "Executed Ipc Elapsed", "inst/cycle", 70) \
  METRIC(IPCExecutedActive, \
         "sm__inst_executed.avg.per_cycle_active", \
         "Executed Ipc Active", "inst/cycle", 70) \
  METRIC(IPCIssuedActive, \
         "sm__inst_issued.avg.per_cycle_active", \
         "Issued Ipc Active", "inst/cycle", 70) \
  METRIC(SMIssueSlotsBusy, \
         "sm__inst_issued.avg.pct_of_peak_sustained_active", \
         "Issue Slots Busy", "%", 70) \
  METRIC(SMMemBusyByInst, \
         "sm__instruction_throughput.avg.pct_of_peak_sustained_active", \
         "SM Busy", "%", 70) \
  \
  /* ── Warp Stalls ──────────────────────────────────────────────── */ \
  /* 1 smsp__warps_issue_stalled_barrier.avg.pct_of_peak_sustained_active */ \
  METRIC(WarpStallBarrier, \
         "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio", \
         "Warp Stall: Barrier", "%", 70) \
  /* 2 smsp__warps_issue_stalled_branch_resolving.avg.pct_of_peak_sustained_active */ \
  METRIC(WarpStallBranchResolving, \
         "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio", \
         "Warp Stall: Branch Resolving", "%", 70) \
  /* 3 smsp__warps_issue_stalled_dispatch_stall.avg.pct_of_peak_sustained_active */ \
  METRIC(WarpStallDispatchStall, \
         "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio", \
         "Warp Stall: Dispatch Stall", "%", 70) \
  /* 4 smsp__warps_issue_stalled_drain.avg.pct_of_peak_sustained_active */ \
  METRIC(WarpStallDrain, \
         "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio", \
         "Warp Stall: Drain", "%", 70) \
  /* 5 smsp__warps_issue_stalled_imc_miss.avg.pct_of_peak_sustained_active */ \
  METRIC(WarpStallIMCMiss, \
         "smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio", \
         "Warp Stall: IMC Miss", "%", 70) \
  /* 6 smsp__warps_issue_stalled_lg_throttle.avg.pct_of_peak_sustained_active */ \
  METRIC(WarpStallLGThrottle, \
         "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio", \
         "Warp Stall: LG Throttle", "%", 70) \
  /* 7 smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active */ \
  METRIC(WarpStallLongScoreboard, \
         "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", \
         "Warp Stall: Long Scoreboard", "%", 70) \
  /* 8 smsp__warps_issue_stalled_math_pipe_throttle.avg.pct_of_peak_sustained_active */ \
  METRIC(WarpStallMathPipeThrottle, \
         "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio", \
         "Warp Stall: Math Pipe Throttle", "%", 70) \
  /* 9 smsp__warps_issue_stalled_membar.avg.pct_of_peak_sustained_active */ \
  METRIC(WarpStallMembar, \
         "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio", \
         "Warp Stall: Membar", "%", 70) \
  /* 10 smsp__warps_issue_stalled_mio_throttle.avg.pct_of_peak_sustained_active */ \
  METRIC(WarpStallMIOThrottle, \
         "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio", \
         "Warp Stall: MIO Throttle", "%", 70) \
  /* 11 smsp__warps_issue_stalled_misc.avg.pct_of_peak_sustained_active */ \
  METRIC(WarpStallMisc, \
         "smsp__average_warps_issue_stalled_misc_per_issue_active.ratio", \
         "Warp Stall: Misc", "%", 70) \
  /* 12 smsp__warps_issue_stalled_no_instruction.avg.pct_of_peak_sustained_active */ \
  METRIC(WarpStallNoInstruction, \
         "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio", \
         "Warp Stall: No Instructions", "%", 70) \
  /* 13 smsp__warps_issue_stalled_not_selected.avg.pct_of_peak_sustained_active */ \
  METRIC(WarpStallNotSelected, \
         "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio", \
         "Warp Stall: Not Selected", "%", 70) \
  /* 14 smsp__warps_issue_stalled_selected.avg.pct_of_peak_sustained_active */ \
  METRIC(WarpStallSelected, \
         "smsp__average_warps_issue_stalled_selected_per_issue_active.ratio", \
         "Warp Stall: Selected", "%", 70) \
  /* 15 smsp__warps_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_active */ \
  METRIC(WarpStallShortScoreboard, \
         "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio", \
         "Warp Stall: Short Scoreboard", "%", 70) \
  /* 16 smsp__warps_issue_stalled_sleeping.avg.pct_of_peak_sustained_active */ \
  METRIC(WarpStallSleeping, \
         "smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio", \
         "Warp Stall: Sleeping", "%", 70) \
  /* 17 smsp__warps_issue_stalled_tex_throttle.avg.pct_of_peak_sustained_active */ \
  METRIC(WarpStallTexThrottle, \
         "smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio", \
         "Warp Stall: TEX Throttle", "%", 70) \
  /* 18 smsp__warps_issue_stalled_wait.avg.pct_of_peak_sustained_active */ \
  METRIC(WarpStallWait, \
         "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio", \
         "Warp Stall: Wait", "%", 70) \
  METRIC(WarpStallMemDependence, \
         "smsp__warps_issue_stalled_mem_dependence.avg.pct_of_peak_sustained_active", \
         "Warp Stall: Memory Dependence", "%", 70) \
  METRIC(MemoryStallRate, \
         "smsp__warps_issue_stalled_mbarrier_wait.mios_per_warps_issue", \
         "Memory Stall Rate", "mios/warps_issue", 80) \
  \
  /* ── Launch ────────────────────────────────────────────────── */ \
  METRIC(LaunchWavesPerMultiprocessor, \
         "launch__waves_per_multiprocessor", \
         "Launch Waves Per Multiprocessor", "wave", 70) \
  METRIC(LaunchTpcCount, \
         "launch__tpc_count", \
         "Launch TPC Count", "tpc", 70) \
  METRIC(LaunchSmCount, \
         "launch__sm_count", \
         "Launch SM Count", "sm", 70) \
  METRIC(LaunchThreadCount, \
         "launch__thread_count", \
         "Launch Thread Count", "thread", 70) \
  METRIC(LaunchGridSize, \
         "launch__grid_size", \
         "Launch Grid Size", "num", 70) \
  METRIC(LaunchBlockSize, \
         "launch__block_size", \
         "Launch Block Size", "num", 70) \
  METRIC(LaunchStackSize, \
         "launch__stack_size", \
         "Launch Stack Size", "num", 70) \
  METRIC(LaunchRegistersPerThread, \
         "launch__registers_per_thread", \
         "Launch Registers Per Thread", "reg/thread", 70) \
  METRIC(LaunchOccupancyLimitWrap, \
         "launch__occupancy_limit_warps", \
         "Launch Occupancy (Limit Warps)", "block", 70) \
  METRIC(LaunchOccupancyLimitReg, \
         "launch__occupancy_limit_registers", \
         "Launch Occupancy (Limit Reg)", "block", 70) \
  METRIC(LaunchOccupancyLimitSharedMem, \
         "launch__occupancy_limit_shared_mem", \
         "Launch Occupancy (Limit Shared Mem)", "block", 70) \
  METRIC(LaunchOccupancyLimitBlock, \
         "launch__occupancy_limit_blocks", \
         "Launch Occupancy (Limit Block)", "block", 70) \
  METRIC(LaunchOccupancyPerSharedMemSize, \
         "launch__occupancy_per_shared_mem_size", \
         "Launch occupancy (Number of active warps for given shared memory size)", "wrap", 70) \
  METRIC(LaunchOccupancyPerBlockSize, \
         "launch__occupancy_per_block_size", \
         "Launch occupancy (Number of active warps for given register count)", "wrap", 70) \
  METRIC(LaunchOccupancyPerRegisterCount, \
         "launch__occupancy_per_register_count", \
         "Launch occupancy (Number of active warps for given block size)", "wrap", 70) \
  METRIC(LaunchSharedMemPerBlockStatic, \
        "launch__shared_mem_per_block_static", \
        "Launch Shared Mem Per Block Static", "byte/block", 70) \
  METRIC(LaunchSharedMemPerBlockDynamic, \
        "launch__shared_mem_per_block_dynamic", \
        "Launch Shared Mem Per Block Dynamic", "byte/block", 70) \
  METRIC(LaunchSharedMemPerBlockDriver, \
        "launch__shared_mem_per_block_driver", \
        "Launch Shared Mem Per Block Driver", "byte/block", 70) \
  METRIC(LaunchSharedMemPerBlock, \
        "launch__shared_mem_per_block", \
        "Launch Shared Mem Per Block", "byte/block", 70) \
  METRIC(LaunchSharedMemConfigSize, \
        "launch__shared_mem_config_size", \
        "Launch Shared Mem Config Size", "bytes", 70) \
  METRIC(SMWarpsActivePerScheduler, \
         "smsp__warps_active.avg.per_cycle_active", \
         "Active Warps Per Scheduler", "warp", 70) \
  METRIC(TheoreticalOccupancy, \
         "sm__maximum_warps_per_active_cycle_pct", \
         "Theoretical Occupancy", "%", 70) \
  METRIC(TheoreticalActiveWarpsPerSM, \
         "sm__maximum_warps_avg_per_active_cycle", \
         "Theoretical Active Warps per SM", "warp", 70) \
  METRIC(AchievedOccupancy, \
         "sm__warps_active.avg.pct_of_peak_sustained_active", \
         "Achieved Occupancy", "%", 70) \
  METRIC(AchievedActiveWarpsPerSM, \
         "sm__warps_active.avg.per_cycle_active", \
         "Achieved Active Warps Per SM", "warp", 70) \
  \
  /* ── Instruction Mix ──────────────────────────────────────────── */ \
  METRIC(InstExecutedFAdd, \
         "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", \
         "Inst Executed (FAdd)", "num", 70) \
  METRIC(InstExecutedFMul, \
         "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", \
         "Inst Executed (FMul)", "n", 70) \
  METRIC(InstExecutedFFMA, \
         "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", \
         "Inst Executed (FFMA)", "num", 70) \
  METRIC(InstructionsExecuted, \
         "smsp__inst_executed.sum", \
         "Instructions Executed", "num", 70) \
  \
  /* ── SM Throughput Misc ───────────────────────────────────────── */ \
  METRIC(SMMFMAThroughput, \
         "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed", \
         "SM FMA Cycles Active", "%", 80) \
  METRIC(SMInstExecutedFMA, \
         "sm__inst_executed_pipe_fma.sum", \
         "SM Inst Executed (FMA)", "num", 80)

// ─── Enum: auto-generated from X-macro ─────────────────────────────────────
enum class MetricId : int {
#define METRIC(name, ...) name,
  PROFILER_METRIC_LIST
  Count
#undef METRIC
};

// ─── Per-metric metadata ───────────────────────────────────────────────────
struct MetricEntry {
  const char* cupti_name;
  const char* label;
  const char* unit;
  int arch_min;
};

// ─── Table: defined in metrics.cpp ─────────────────────────────────────────
extern const MetricEntry kMetricTable[static_cast<int>(MetricId::Count)];

// ─── Table-based accessors ─────────────────────────────────────────────────
const char* MetricToCuptiName(MetricId id);
const char* MetricToLabel(MetricId id);
const char* MetricToUnit(MetricId id);

// Format metric value for display: auto-scale to human-friendly unit.
// Returns "{value} {unit}" (e.g., "128.5 MB", "1.23 GHz", "50.0 us").
// Only affects printing — raw data unchanged.
std::string FormatMetricValue(MetricId id, double raw_value);

bool IsMetricAvailable(MetricId id);

// ─── Preset metric sets ────────────────────────────────────────────────────
std::vector<MetricId> DefaultMetrics();
std::vector<MetricId> WarpStallMetrics();
std::vector<MetricId> LaunchMetrics();
std::vector<MetricId> CacheMetrics();
std::vector<MetricId> ComputeMemMetrics();
std::vector<MetricId> InstructionMixMetrics();
std::vector<MetricId> PerformanceProfileMetrics();
std::vector<MetricId> MainProfileMetrics();
std::vector<MetricId> AllSupportedMetrics();

}  // namespace profiler
}  // namespace cpp_base_suite

#endif  // PROFILER_METRICS_HPP_
