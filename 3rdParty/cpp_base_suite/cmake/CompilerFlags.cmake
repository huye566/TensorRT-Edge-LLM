# Compiler and linker flags for the project.
# Only affects targets in this directory tree (add_compile_options / set
# with LOCAL scope do not leak to a parent project).

# C++ standard — only apply if the parent hasn't locked it in already.
if(NOT CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 17)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
set(CMAKE_CXX_EXTENSIONS OFF)

# Common compiler warnings.
# -Wpedantic is applied to C/CXX only — nvcc stub files use non-standard
# #line directives that would trigger it.
add_compile_options(
    -Wall
    -Wextra
    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
    $<$<COMPILE_LANGUAGE:C>:-Wpedantic>
    # -Wno-gnu-zero-variadic-macro-arguments  # glog compat
)

# Position-independent code (needed for static libs used in shared libs)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build type (standalone only — subdirectory inherits parent's).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        message(STATUS "cpp_base_suite: Setting build type to 'Debug' as none was specified.")
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
    endif()
endif()
