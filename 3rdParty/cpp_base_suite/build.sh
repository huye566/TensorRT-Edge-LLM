#!/usr/bin/env bash
# build.sh — Build cpp_base_suite as a shared library.
#
# Usage:
#   ./build.sh                  # Debug shared build
#   ./build.sh Release          # Release shared build
#   ./build.sh Release thor     # Release shared build
#   ./build.sh Release thor /opt/cpp_base_suite  # with custom install prefix

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_TYPE="${1:-Release}"
PLATFORM="${2:-x86}"
PREFIX="${3:-${SCRIPT_DIR}/build/install}"

BUILD_DIR="${SCRIPT_DIR}/build/${BUILD_TYPE}"
CMAKE_OPTIONS="-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_INSTALL_PREFIX=${PREFIX}"
CMAKE_OPTIONS="$CMAKE_OPTIONS -DBUILD_SHARED_LIBS=ON"

if [ "$PLATFORM" == "thor" ]; then
    CMAKE_OPTIONS="$CMAKE_OPTIONS -DTENSORRT_HOME=/opt/update/trt10.13_new/usr"
    CMAKE_OPTIONS="$CMAKE_OPTIONS -DPLATFORM_THOR=ON"
    export PATH=$PATH:/opt/update/trt10.13_new/usr/bin:/usr/local/cuda/bin
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/update/trt10.13_new/usr/lib/aarch64-linux-gnu
elif [ "$PLATFORM" == "x86" ]; then
    CMAKE_OPTIONS="$CMAKE_OPTIONS -DTENSORRT_HOME=/opt/tensorrt"
    CMAKE_OPTIONS="$CMAKE_OPTIONS -DPLATFORM_THOR=OFF"
    export PATH=$PATH:/opt/tensorrt/bin:/usr/local/cuda/bin
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/tensorrt/lib
else
    echo "Unsupported platform: $PLATFORM. Use 'x86' or 'thor'."
    exit 1
fi

echo "==> cpp_base_suite | type=${BUILD_TYPE} | prefix=${PREFIX}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${SCRIPT_DIR}" $CMAKE_OPTIONS

cmake --build . -j"$(nproc)"
cmake --build . --target install

echo "==> Done → ${PREFIX}"
