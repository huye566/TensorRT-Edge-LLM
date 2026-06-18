#!/bin/bash
export HOME_DIR="$(dirname "$(dirname $(readlink -f $0))")"
echo "HOME_DIR: ${HOME_DIR}"
cd ${HOME_DIR}

# rm -rf build_thor
mkdir -p build_thor && cd build_thor

# 添加交叉编译器到 PATH
export PATH=/opt/arm-toolchain/current/bin:$PATH

cmake .. \
  -DCUDA_CTK_VERSION=12.8 \
  -DCMAKE_TOOLCHAIN_FILE=${HOME_DIR}/scripts/cmake/aarch64-toolchain.cmake \
  -DEMBEDDED_TARGET=auto-thor \
  -DCMAKE_CUDA_ARCHITECTURES=101a \
  # -DENABLE_CUTE_DSL=ALL \
  # -DCMAKE_BUILD_TYPE=Debug \

make -j$(nproc)
