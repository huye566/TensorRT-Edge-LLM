# CMake 工具链文件 - ARM64 交叉编译
# 用于 NVIDIA Thor 平台
# 使用方法: cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/aarch64-toolchain.cmake ..
#
# 注意：
# - CUDA 架构（sm_XX）不在工具链文件中设置，由项目自己决定
# - 可以在项目的 CMakeLists.txt 中设置：
#   set(CMAKE_CUDA_ARCHITECTURES "80")  # 全局设置
#   或 set_target_properties(target PROPERTIES CUDA_ARCHITECTURES "80")  # 按目标设置
# - 常用架构：
#   70 - Volta (V100)
#   75 - Turing (T4)
#   80 - Ampere (A100, RTX 30xx)
#   86 - Ampere (RTX 30xx)
#   89 - Ada Lovelace (RTX 40xx)
#   90 - Hopper (H100)
#   100 - Blackwell (thor)

# ========================================
# 交叉编译系统配置
# ========================================

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# ========================================
# C/C++ 交叉编译器（ARM 工具链 13.3.0，包含 GLIBC 2.38+）
# ========================================

set(CMAKE_C_COMPILER aarch64-none-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-none-linux-gnu-g++)

# ========================================
# CUDA 交叉编译配置
# ========================================

# 使用主机上的 nvcc 编译器（x86_64），通过 -ccbin 交叉编译到 ARM64
# 注意：必须在 project() 声明之前设置
if(NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)
endif()

# CUDA 环境变量（用于 nvcc）
set(ENV{CUDA_PATH} "/opt/cuda-aarch64/current")

# CUDA 编译标志（交叉编译到 ARM64）
# 注意：不包含 -arch=sm_XX，由项目自己指定目标架构
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=aarch64-none-linux-gnu-g++ --compiler-options=-fPIC -I/opt/cuda-aarch64/current/targets/aarch64-linux/include -L/opt/cuda-aarch64/current/targets/aarch64-linux/lib")

# CUDA 链接器（使用 ARM 交叉编译链接器）
set(CMAKE_CUDA_HOST_LINK_LAUNCHER aarch64-none-linux-gnu-g++)

# 默认 CUDA 架构（可在项目中覆盖）
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING "Default CUDA architectures")
endif()

# ========================================
# CUDA ARM64 路径（使用 current 软链接）
# ========================================

# Thor 平台专用 CUDA 路径（包含 curand_kernel.h 等头文件）
set(CUDA_THOR_DIR /opt/cuda-aarch64/current/thor)
# 标准 aarch64 CUDA 路径（包含 libcuda.so 等库文件）
set(CUDA_TOOLKIT_ROOT_DIR /opt/cuda-aarch64/current/targets/aarch64-linux)

# 头文件：thor 目录包含 cublas/curand 等扩展库头文件，
# targets 目录包含 cuda_runtime_api.h 等标准 CUDA runtime 头文件，
# 两者都需要加入搜索路径。
set(CUDA_INCLUDE_DIRS ${CUDA_THOR_DIR}/include ${CUDA_TOOLKIT_ROOT_DIR}/include)
# 库文件使用标准 aarch64 目录
set(CUDA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib)
set(CUDA_LIB_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib)

# CUDA_TARGET_DIR 用于 CMakeLists.txt 中的 find_library/find_path
# 使用 thor 目录作为主要搜索路径（包含头文件和库）
set(CUDA_TARGET_DIR ${CUDA_THOR_DIR})

# 直接指定 CUDA 驱动库（libcuda.so 只在 targets/aarch64-linux/lib/stubs 中）
set(CUDA_DRIVER_LIB ${CUDA_TOOLKIT_ROOT_DIR}/lib/stubs/libcuda.so CACHE FILEPATH "CUDA driver library")

# ========================================
# TensorRT ARM64 路径（使用 current 软链接）
# ========================================

set(TENSORRT_ROOT /opt/tensorrt-aarch64/current)
set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include/aarch64-linux-gnu)
set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib/aarch64-linux-gnu)

# ========================================
# 目标系统根目录（用于 find_* 搜索）
# ========================================

set(CMAKE_FIND_ROOT_PATH
    /opt/arm-toolchain/current/aarch64-none-linux-gnu/libc
    ${CUDA_THOR_DIR}
    ${CUDA_TOOLKIT_ROOT_DIR}
    ${TENSORRT_ROOT}
)

# 搜索配置
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# ========================================
# 编译标志 - NVIDIA Thor 平台优化
# ========================================

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a" CACHE STRING "C flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a" CACHE STRING "CXX flags")

# CUDA RPATH（运行时库搜索路径）
set(CMAKE_CUDA_RPATH "/opt/cuda-aarch64/current/targets/aarch64-linux/lib" CACHE STRING "CUDA RPATH")
