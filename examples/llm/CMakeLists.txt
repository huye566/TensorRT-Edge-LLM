# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.

# Build executables for chat, benchmark and accuracy mode
add_executable(llm_build llm_build.cpp)
target_link_libraries(llm_build PRIVATE ${NV_ONNX_PARSER_LIB} commonLibraryExt
                                        edgellmBuilder)
target_include_directories(llm_build PRIVATE ${ONNX_PARSER_INCLUDE_DIR}
                                             ${COMMON_INCLUDE_DIRS})
add_cross_build_link_options(llm_build)

add_executable(llm_inference llm_inference.cpp)
target_link_libraries(llm_inference PRIVATE edgellmCore edgellmTokenizer
                                            exampleUtils commonLibraryExt)
target_include_directories(
  llm_inference PRIVATE ${COMMON_INCLUDE_DIRS}
                        ${CMAKE_SOURCE_DIR}/examples/utils)
add_cross_build_link_options(llm_inference)

add_library(vlm_infer SHARED vlm_infer.cpp)
target_link_libraries(vlm_infer PRIVATE edgellmCore edgellmTokenizer
                                        exampleUtils commonLibraryExt)
target_include_directories(vlm_infer PRIVATE ${COMMON_INCLUDE_DIRS}
                                             ${CMAKE_SOURCE_DIR}/examples/utils)
add_cross_build_link_options(vlm_infer)

add_executable(vlm_infer_test vlm_infer_test.cpp)
target_link_libraries(vlm_infer_test PRIVATE ${CUDART_LIB} edgellmCore
                                             exampleUtils)
target_include_directories(
  vlm_infer_test PRIVATE ${COMMON_INCLUDE_DIRS}
                         ${CMAKE_SOURCE_DIR}/examples/utils)
add_cross_build_link_options(vlm_infer_test)
