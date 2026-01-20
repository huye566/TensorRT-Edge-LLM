include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(cutlass_gemv)
add_subdirectory(cutlass_gemm)
add_subdirectory(cutlass_gemm_silu)
add_subdirectory(cutlass_wrapper)
if(SM_101)
  add_subdirectory(cutlass_wrapper_blackwell)
endif()