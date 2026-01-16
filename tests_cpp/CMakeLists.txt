FILE(GLOB TEST_UTILS_CODE ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp)
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(kernels)
add_subdirectory(plugins)
add_subdirectory(cutlass)
add_subdirectory(cublas)