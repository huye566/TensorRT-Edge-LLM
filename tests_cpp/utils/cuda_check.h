#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cassert>

#define CUDA_CHECK(call)                                                      \
do {                                                                          \
    const cudaError_t error = call;                                           \
    if (error != cudaSuccess) {                                               \
        std::cerr << "CUDA error: " << cudaGetErrorString(error)              \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;      \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

#ifdef ENABLE_PROFILE
    #include <chrono>
    #define PROFILE_START(name) auto start_##name = std::chrono::high_resolution_clock::now()
    #define PROFILE_END(name) do { \
        auto end_##name = std::chrono::high_resolution_clock::now(); \
        auto duration_##name = std::chrono::duration_cast<std::chrono::microseconds>(end_##name - start_##name).count(); \
        MOE_PRINT("[PROFILE] %s: %ld us\n", #name, duration_##name); \
    } while(0)
#else
    #define PROFILE_START(name)
    #define PROFILE_END(name)
#endif

#define USE_CUDA_DEVICE_ID 0

inline void print_device_info(int device_id = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    std::string indent_str(device_id, ' ');
    std::cout << indent_str << "========================================\n";
    std::cout << indent_str << "GPU Device Information:\n";
    std::cout << indent_str << "========================================\n";
    std::cout << indent_str << "Device Name: " << prop.name << "\n";
    std::cout << indent_str << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << indent_str << "Total Global Memory: " << (prop.totalGlobalMem >> 20) << " MB\n";
    std::cout << indent_str << "Shared Memory per Block: " << (prop.sharedMemPerBlock >> 10) << " KB\n";
    std::cout << indent_str << "Registers per Block: " << prop.regsPerBlock << "\n";
    std::cout << indent_str << "Warp Size: " << prop.warpSize << "\n";
    std::cout << indent_str << "Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << indent_str << "Max Threads Dim: (" << prop.maxThreadsDim[0] << ", " 
              << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")\n";
    std::cout << indent_str << "Max Grid Size: (" << prop.maxGridSize[0] << ", " 
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")\n";
    std::cout << indent_str << "Clock Rate: " << prop.clockRate / 1000 << " MHz\n";
    std::cout << indent_str << "Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz\n";
    std::cout << indent_str << "Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
    std::cout << indent_str << "L2 Cache Size: " << prop.l2CacheSize / 1024 << " KB\n";
    if (prop.major >= 7) {
        std::cout << indent_str << "Tensor Core: support (> Volta)\n";
    } else {
        std::cout << indent_str << "Tensor Core: unsupported\n";
    }
    std::cout << indent_str << "========================================\n\n";
}

inline bool check_tensor_core_support() {
    cudaDeviceProp prop;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    return prop.major >= 7;
}

#endif // CUDA_CHECK_H