# Find CUDA ecosystem: CUDA toolkit, TensorRT, CUPTI.
#
# Provides IMPORTED targets so consumers just link them without manual
# include-directory / link-directory wiring.
#
# Targets created:
#   CUDA::cudart      — CUDA runtime (libcudart)
#   CUDA::driver      — CUDA driver   (libcuda)
#   CUDA::cupti       — CUPTI library (libcupti)
#   TensorRT::nvinfer — TensorRT inference library
#
# Cache variables (for diagnostics):
#   CUDA_INCLUDE_DIR, CUDA_LIB, CUDA_DRIVER_LIB
#   TENSORRT_INCLUDE_DIR, NVINFER_LIB
#   CUPTI_INCLUDE_DIR, CUPTI_LIB

# ──────────────────────────── CUDA toolkit ────────────────────────────

find_path(CUDA_INCLUDE_DIR
    NAMES cuda_runtime.h
    PATHS /usr/local/cuda/targets/aarch64-linux/include
          /usr/local/cuda/include
          /usr/local/cuda*/include
)

find_library(CUDA_LIB
    NAMES cudart
    PATHS /usr/local/cuda/targets/aarch64-linux/lib
          /usr/local/cuda/lib64
          /usr/local/cuda*/lib64
)

find_program(NVCC_EXECUTABLE nvcc
    PATHS /usr/local/cuda/bin
          /usr/local/cuda*/bin
    NO_DEFAULT_PATH
)

if(CUDA_INCLUDE_DIR AND CUDA_LIB)
    if(NOT TARGET CUDA::cudart)
        add_library(CUDA::cudart UNKNOWN IMPORTED)
        set_target_properties(CUDA::cudart PROPERTIES
            IMPORTED_LOCATION "${CUDA_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIR}")
    endif()
    set(CUDA_FOUND TRUE)
else()
    set(CUDA_FOUND FALSE)
endif()

# ──────────────────────────── CUDA driver ─────────────────────────────

find_library(CUDA_DRIVER_LIB
    NAMES cuda
    PATHS /usr/local/cuda/targets/aarch64-linux/lib/stubs
          /usr/local/cuda/lib64/stubs
          /usr/local/cuda*/compat
          /usr/lib/x86_64-linux-gnu
    NO_DEFAULT_PATH
)
if(NOT CUDA_DRIVER_LIB)
    find_library(CUDA_DRIVER_LIB NAMES cuda)
endif()

if(CUDA_DRIVER_LIB)
    if(NOT TARGET CUDA::driver)
        add_library(CUDA::driver UNKNOWN IMPORTED)
        set_target_properties(CUDA::driver PROPERTIES
            IMPORTED_LOCATION "${CUDA_DRIVER_LIB}")
    endif()
endif()

# ──────────────────────────── TensorRT ────────────────────────────────

# TENSORRT_HOME may be set externally; use it as a hint.
if(TENSORRT_HOME AND EXISTS "${TENSORRT_HOME}/include")
    set(TENSORRT_INCLUDE_DIR "${TENSORRT_HOME}/include")
elseif(TENSORRT_HOME AND EXISTS "${TENSORRT_HOME}/usr/include")
    set(TENSORRT_INCLUDE_DIR "${TENSORRT_HOME}/usr/include")
else()
    find_path(TENSORRT_INCLUDE_DIR
        NAMES NvInfer.h
        PATHS /usr/include
              /usr/local/include
              /usr/include/aarch64-linux-gnu
              /usr/local/cuda/targets/aarch64-linux/include
        NO_DEFAULT_PATH  # optional, but keep to avoid host includes when cross-compiling
    )
    # Fallback to default search if still not found
    if(NOT TENSORRT_INCLUDE_DIR)
        find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h)
    endif()
endif()

find_library(NVINFER_LIB
    NAMES nvinfer
    PATHS "${TENSORRT_HOME}"
          "${TENSORRT_HOME}/usr/lib"
          /usr/lib/aarch64-linux-gnu
          /usr/lib/x86_64-linux-gnu
    NO_DEFAULT_PATH
)
if(NOT NVINFER_LIB)
    find_library(NVINFER_LIB NAMES nvinfer)
endif()

if(NVINFER_LIB AND TENSORRT_INCLUDE_DIR)
    if(NOT TARGET TensorRT::nvinfer)
        add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
        set_target_properties(TensorRT::nvinfer PROPERTIES
            IMPORTED_LOCATION "${NVINFER_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
        )
        # If TENSORRT_HOME is set, also add link directory (optional)
        if(TENSORRT_HOME AND EXISTS "${TENSORRT_HOME}/usr/lib")
            set_property(TARGET TensorRT::nvinfer APPEND PROPERTY
                INTERFACE_LINK_DIRECTORIES "${TENSORRT_HOME}/usr/lib")
        elseif(TENSORRT_HOME AND EXISTS "${TENSORRT_HOME}")
            set_property(TARGET TensorRT::nvinfer APPEND PROPERTY
                INTERFACE_LINK_DIRECTORIES "${TENSORRT_HOME}")
        endif()

    endif()
    set(TENSORRT_FOUND TRUE)
else()
    set(TENSORRT_FOUND FALSE)
    # Provide more detail for debugging
    if(NOT TENSORRT_INCLUDE_DIR)
        message(DEBUG "TensorRT: NvInfer.h not found")
    endif()
    if(NOT NVINFER_LIB)
        message(DEBUG "TensorRT: libnvinfer.so not found")
    endif()
endif()

# ──────────────────────────── CUPTI ───────────────────────────────────

find_library(CUPTI_LIB
    NAMES cupti
    PATHS /usr/local/cuda/targets/aarch64-linux/lib
          /usr/local/cuda/lib64
          /usr/local/cuda/targets/x86_64-linux/lib
          /usr/local/cuda*/targets/x86_64-linux/lib
    NO_DEFAULT_PATH
)

find_path(CUPTI_INCLUDE_DIR
    NAMES cupti_profiler_host.h
    PATHS /usr/local/cuda/targets/aarch64-linux/include
          /usr/local/cuda/include
          /usr/local/cuda/targets/x86_64-linux/include
          /usr/local/cuda*/targets/x86_64-linux/include
    NO_DEFAULT_PATH
)

if(CUPTI_LIB AND CUPTI_INCLUDE_DIR)
    if(NOT TARGET CUDA::cupti)
        add_library(CUDA::cupti UNKNOWN IMPORTED)
        set_target_properties(CUDA::cupti PROPERTIES
            IMPORTED_LOCATION "${CUPTI_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${CUPTI_INCLUDE_DIR}")
    endif()
    set(CUPTI_FOUND TRUE)
else()
    set(CUPTI_FOUND FALSE)
endif()

# ──────────────────────────── Diagnostics ─────────────────────────────

if(CUDA_FOUND)
    message(STATUS "cpp_base_suite: CUDA: include=${CUDA_INCLUDE_DIR}, lib=${CUDA_LIB}")
else()
    message(WARNING "cpp_base_suite: CUDA not found: include=${CUDA_INCLUDE_DIR}, lib=${CUDA_LIB}")
endif()

if(TARGET CUDA::driver)
    message(STATUS "cpp_base_suite: CUDA driver: ${CUDA_DRIVER_LIB}")
endif()

if(TENSORRT_FOUND)
    message(STATUS "cpp_base_suite: TensorRT: include=${TENSORRT_INCLUDE_DIR}, lib=${NVINFER_LIB}")
else()
    message(WARNING "cpp_base_suite: TensorRT (libnvinfer) not found — TensorRT backend will not build.")
endif()

if(CUPTI_FOUND)
    message(STATUS "cpp_base_suite: CUPTI: include=${CUPTI_INCLUDE_DIR}, lib=${CUPTI_LIB}")
else()
    message(WARNING "cpp_base_suite: CUPTI not found — profiler module will not build.")
endif()

if(NVCC_EXECUTABLE)
    message(STATUS "cpp_base_suite: NVCC: ${NVCC_EXECUTABLE}")
else()
    message(WARNING "cpp_base_suite: nvcc not found — CUDA compilation unavailable.")
endif()
