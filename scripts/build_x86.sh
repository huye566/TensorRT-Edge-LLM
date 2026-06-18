#!/bin/bash
export HOME_DIR="$(dirname "$(dirname $(readlink -f $0))")"
echo "HOME_DIR: ${HOME_DIR}"
cd ${HOME_DIR}

. ./scripts/common_config.sh

# rm -rf build
mkdir -p build && cd build
cmake .. \
  -DCUDA_CTK_VERSION=12.8 \
  -DCMAKE_CUDA_COMPILER=${CUDA_PATH}/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH_X86} \
  -DCUDA_DIR=${CUDA_PATH} \
  -DTRT_PACKAGE_DIR=${OPT_DIR}/TensorRT-${TENSORRT_VERSION} \
  -DENABLE_CUTE_DSL=ALL \
  # -DCMAKE_BUILD_TYPE=Debug \

make -j$(nproc)
