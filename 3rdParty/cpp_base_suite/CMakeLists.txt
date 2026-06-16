cmake_minimum_required(VERSION 3.18)

# Only declare a standalone project when built directly.
# When consumed via add_subdirectory(), inherit the parent's project context.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(cpp_base_suite
        VERSION 0.1.0
        DESCRIPTION "cpp base suite"
        LANGUAGES CXX CUDA)
endif()


option(CPP_BASE_SUITE_BUILD_PROFILER "Build the CUPTI profiler library" ON)


include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompilerFlags.cmake)

# Only attempt discovery if not already provided by the parent.
set(_CBS_CUDA_TARGETS_FOUND FALSE)
if(TARGET CUDA::cudart)
    set(_CBS_CUDA_TARGETS_FOUND TRUE)
endif()

if(NOT _CBS_CUDA_TARGETS_FOUND AND CPP_BASE_SUITE_BUILD_PROFILER)
    # enable_language(CUDA) must succeed before FindCUDAEcosystem can probe.
    if(NOT CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    endif()
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindCUDAEcosystem.cmake)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ThirdPartyDeps.cmake)

if(PLATFORM_THOR)
    add_compile_definitions(PLATFORM_THOR=1)
    message(STATUS "PLATFORM_THOR is enabled")
    set(CUDA_ARCHITECTURES "100a" CACHE STRING "Target CUDA architecture(s), e.g. 100;100a")
else()
    set(CUDA_ARCHITECTURES "86" CACHE STRING "Target CUDA architecture(s), e.g. 75;80;86")
endif()

message(STATUS "cpp_base_suite: CUDA architectures: ${CUDA_ARCHITECTURES}")
set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")

add_subdirectory(src)

# ─── Installation (only when built standalone) ──────────────────────────────
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND TARGET cpp_base_suite_profiler)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    if(TARGET cpp_base_suite_profiler)
        install(TARGETS cpp_base_suite_profiler
            EXPORT cpp_base_suite-targets
            LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

        install(FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/include/profiler/cupti_profiler.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/profiler/metrics.hpp
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/profiler)

        install(FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/include/logger.hpp
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    endif()

    install(EXPORT cpp_base_suite-targets
        NAMESPACE cpp_base_suite::
        FILE cpp_base_suite-targets.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cpp_base_suite)

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CppBaseSuiteConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/cpp_base_suite-config.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cpp_base_suite)

    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/cpp_base_suite-config-version.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)

    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/cpp_base_suite-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/cpp_base_suite-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cpp_base_suite)

    install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
