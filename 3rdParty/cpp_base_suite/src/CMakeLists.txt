# Sub-libraries of cpp_base_suite.

# ─── CUPTI profiler ────────────────────────────────────────────────────────

if(CPP_BASE_SUITE_BUILD_PROFILER AND CUDA_FOUND AND CUPTI_FOUND)
    add_subdirectory(profiler)
else()
    if(CPP_BASE_SUITE_BUILD_PROFILER)
        message(STATUS "cpp_base_suite: Skipping profiler (CUDA/CUPTI not found)")
    endif()
endif()
