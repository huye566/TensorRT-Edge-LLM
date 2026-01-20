include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(cublas_gemv)
add_subdirectory(cublas_gemm)
add_subdirectory(cublas_gemm_silu)
add_subdirectory(cublas_ele_wise_mul)
add_subdirectory(cublas_wrapper)