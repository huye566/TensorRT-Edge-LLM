#!/bin/bash
export HOME_DIR="$(dirname "$(dirname $(readlink -f $0))")"
echo "HOME_DIR: ${HOME_DIR}"
cd ${HOME_DIR}

# rm -rf build_thor
mkdir -p build_thor && cd build_thor

# export LD_LIBRARY_PATH=/opt/update/trt10.13_new/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

cmake .. \
  -DCUDA_CTK_VERSION=12.8 \
  -DTRT_PACKAGE_DIR=/usr/lib/aarch64-linux-gnu \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=auto-thor \
  -DCMAKE_CUDA_ARCHITECTURES=101a \
  -DENABLE_CUTE_DSL=ALL \
  # -DCMAKE_BUILD_TYPE=Debug \

make -j$(nproc)
