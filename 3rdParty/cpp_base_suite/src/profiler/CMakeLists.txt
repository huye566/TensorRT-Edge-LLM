# ─── CUPTI profiler library ────────────────────────────────────────────────

add_library(cpp_base_suite_profiler
    ${CMAKE_CURRENT_SOURCE_DIR}/cupti_profiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metrics.cpp
)

add_library(cpp_base_suite::profiler ALIAS cpp_base_suite_profiler)

target_include_directories(cpp_base_suite_profiler
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(cpp_base_suite_profiler
    PUBLIC
        CUDA::cudart
        CUDA::driver
        CUDA::cupti
        nlohmann_json
)

set_target_properties(cpp_base_suite_profiler PROPERTIES
    OUTPUT_NAME     cpp_base_suite_profiler
    POSITION_INDEPENDENT_CODE ON
)
