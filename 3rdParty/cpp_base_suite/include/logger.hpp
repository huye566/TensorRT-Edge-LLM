#ifndef CPP_BASE_SUITE_LOGGER_HPP_
#define CPP_BASE_SUITE_LOGGER_HPP_

// ─── Lightweight, standalone logging for cpp_base_suite ─────────────────────
//
// Default behaviour: all messages go to stderr (terminal).
// Consumers that need to redirect output (e.g. to a file) install a callback
// via SetLogCallback().  While a callback is installed, stderr output is
// suppressed and the callback receives every message.
//
// This keeps cpp_base_suite independent of any host project's logger.
//
// Typical usage from a host application:
//
//   #include "logger.hpp"
//   namespace cbs_log = cpp_base_suite::logger;
//
//   cbs_log::SetLogCallback([](int level, const char* msg) {
//       my_host_logger.log(level, msg);   // route to file / host logger / etc.
//   });
//
// Internal usage (within cpp_base_suite):
//
//   CBS_LOG_ERROR("CUPTI error in %s: %d", api_name, result);
//   CBS_LOG_INFO("chip = '%s', passes = %zu", chip.c_str(), passes);

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <ctime>
#include <mutex>

namespace cpp_base_suite {
namespace logger {

enum class Level : int {
  kDebug   = 1,
  kInfo    = 2,
  kWarning = 3,
  kError   = 4,
};

// External callback signature.  `level` matches the Level enum;
// `msg` is a fully formatted, NUL-terminated string valid only for the
// duration of the call.
using LogCallback = void (*)(int level, const char* msg);

// Install a custom log callback.  Pass nullptr to restore stderr output.
void SetLogCallback(LogCallback cb);

// Set the minimum log level (default: kInfo — debug messages suppressed).
void SetLogLevel(Level level);
Level GetLogLevel();

}  // namespace logger
}  // namespace cpp_base_suite

// ─── Macros ─────────────────────────────────────────────────────────────────

#define CBS_LOG_DEBUG(...)   ::cpp_base_suite::logger::LogV(::cpp_base_suite::logger::Level::kDebug,   __FILE__, __LINE__, __func__, __VA_ARGS__)
#define CBS_LOG_INFO(...)    ::cpp_base_suite::logger::LogV(::cpp_base_suite::logger::Level::kInfo,    __FILE__, __LINE__, __func__, __VA_ARGS__)
#define CBS_LOG_WARNING(...) ::cpp_base_suite::logger::LogV(::cpp_base_suite::logger::Level::kWarning, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define CBS_LOG_ERROR(...)   ::cpp_base_suite::logger::LogV(::cpp_base_suite::logger::Level::kError,   __FILE__, __LINE__, __func__, __VA_ARGS__)

// ─── Inline implementation (header-only) ────────────────────────────────────

namespace cpp_base_suite {
namespace logger {
namespace detail {

struct State {
  std::mutex   mtx;
  LogCallback  callback  = nullptr;
  Level        min_level = Level::kInfo;
};

inline State& GetState() {
  static State s;
  return s;
}

}  // namespace detail

inline void SetLogCallback(LogCallback cb) {
  auto& s = detail::GetState();
  std::lock_guard<std::mutex> lock(s.mtx);
  s.callback = cb;
}

inline void SetLogLevel(Level level) {
  auto& s = detail::GetState();
  std::lock_guard<std::mutex> lock(s.mtx);
  s.min_level = level;
}

inline Level GetLogLevel() {
  auto& s = detail::GetState();
  std::lock_guard<std::mutex> lock(s.mtx);
  return s.min_level;
}

inline void LogV(Level level, const char* file, int line, const char* func,
                 const char* fmt, ...) {
  auto& s = detail::GetState();

  // Format the user message.
  char user_buf[2048];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(user_buf, sizeof(user_buf), fmt, ap);
  va_end(ap);
  if (n < 0) return;

  // Timestamp: [HH:MM:SS.mmm]
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) % 1000;
  char time_buf[32];
  std::strftime(time_buf, sizeof(time_buf), "%H:%M:%S", std::localtime(&time_t));

  // Extract filename from full path.
  const char* filename = file;
  if (file) {
    const char* slash = std::strrchr(file, '/');
    if (slash) filename = slash + 1;
#ifdef _WIN32
    slash = std::strrchr(filename, '\\');
    if (slash) filename = slash + 1;
#endif
  }

  // Level tag.
  const char* level_tag = "UNKNOWN";
  switch (level) {
    case Level::kDebug:   level_tag = "DEBUG";   break;
    case Level::kInfo:    level_tag = "INFO";    break;
    case Level::kWarning: level_tag = "WARNING"; break;
    case Level::kError:   level_tag = "ERROR";   break;
  }

  // Build decorated line: [timestamp] [level] [file:line:function] message
  char full_buf[2560];
  snprintf(full_buf, sizeof(full_buf),
           "[%s.%03d] [%s] [%s:%d:%s] %s\n",
           time_buf, static_cast<int>(ms.count()), level_tag,
           filename ? filename : "?", line, func ? func : "?", user_buf);

  std::lock_guard<std::mutex> lock(s.mtx);

  // Priority 1: external callback.  Receives every message — not filtered
  // by min_level.  The callback is fully responsible for any filtering.
  if (s.callback) {
    s.callback(static_cast<int>(level), full_buf);
    return;
  }

  // Priority 2 (default): stderr, filtered by min_level.
  if (static_cast<int>(level) < static_cast<int>(s.min_level)) {
    return;
  }
  fputs(full_buf, stderr);
  fflush(stderr);
}

}  // namespace logger
}  // namespace cpp_base_suite

#endif  // CPP_BASE_SUITE_LOGGER_HPP_
