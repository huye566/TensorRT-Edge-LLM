get_filename_component(FOLDER_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

FILE(GLOB TEST_SRC_CODE ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(test_${FOLDER_NAME} ${TEST_SRC_CODE} ${TEST_UTILS_CODE})
target_link_libraries(test_${FOLDER_NAME} PRIVATE 
  gtest gtest_main edgellmCore NvInfer_edgellm_plugin)
target_include_directories(
  test_${FOLDER_NAME} PRIVATE ${COMMON_INCLUDE_DIRS}
                        ${CMAKE_SOURCE_DIR}/examples/utils)
add_cross_build_link_options(test_${FOLDER_NAME})