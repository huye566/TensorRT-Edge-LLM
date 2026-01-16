#!/bin/bash

export HOME_DIR="$(dirname "$(dirname "$(dirname $(readlink -f $0))")")"
echo "HOME_DIR: ${HOME_DIR}"
cd ${HOME_DIR}

. ./scripts/thor/common_config.sh

mkdir -p build && cd build

cmake .. \
  -DTRT_PACKAGE_DIR=/usr/lib/aarch64-linux-gnu \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=auto-thor \
  -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} \
  -DBUILD_KERNELS_TESTS=ON \
  -DUSE_MOE_CUTLASS=ON \
  # -DCMAKE_BUILD_TYPE=Debug \

make -j$(nproc)