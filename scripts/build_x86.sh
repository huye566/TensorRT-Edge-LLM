#!/bin/bash

export HOME_DIR="$(dirname "$(dirname $(readlink -f $0))")"
echo "HOME_DIR: ${HOME_DIR}"
cd ${HOME_DIR}

. ./scripts/common_config.sh

# rm -rf build
mkdir -p build && cd build
cmake .. \
  -DCMAKE_CUDA_COMPILER=${OPT_DIR}/cuda-${CUDA_VERSION}/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH_X86} \
  -DCUDA_VERSION=${CUDA_VERSION} \
  -DCUDA_DIR=${OPT_DIR}/cuda-${CUDA_VERSION} \
  -DTRT_PACKAGE_DIR=${OPT_DIR}/TensorRT-${TENSORRT_VERSION} \
  -DBUILD_KERNELS_TESTS=ON \
  -DUSE_MOE_CUTLASS=ON \
  # -DCMAKE_BUILD_TYPE=Debug \


make -j$(nproc)