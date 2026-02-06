get_filename_component(FOLDER_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

FILE(GLOB TEST_SRC_CODE 
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(test_${FOLDER_NAME} ${TEST_SRC_CODE} ${TEST_UTILS_CODE})
set_property(TARGET test_${FOLDER_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(test_${FOLDER_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_link_libraries(test_${FOLDER_NAME} PRIVATE commonLibraryExt edgellmCore ${CUDA_DRIVER_LIB})
target_include_directories(
  test_${FOLDER_NAME} PRIVATE ${COMMON_INCLUDE_DIRS})
add_cross_build_link_options(test_${FOLDER_NAME})