#!/bin/bash
export HOME_DIR="$(dirname "$(dirname $(readlink -f $0))")"
echo "HOME_DIR: ${HOME_DIR}"
cd ${HOME_DIR}

# rm -rf build
mkdir -p build && cd build

# export LD_LIBRARY_PATH=/opt/update/trt10.13_new/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

cmake .. \
  -DTRT_PACKAGE_DIR=/usr/lib/aarch64-linux-gnu \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=auto-thor \
  -DCMAKE_CUDA_ARCHITECTURES=101a \
  -DBUILD_KERNELS_TESTS=ON \
  -DUSE_MOE_CUTLASS=ON \
  -DSM_101=ON \

make -j$(nproc)
