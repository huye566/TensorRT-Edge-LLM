#!/bin/bash

export HOME_DIR="$(dirname "$(dirname "$(dirname $(readlink -f $0))")")"
echo "HOME_DIR: ${HOME_DIR}"
cd ${HOME_DIR}

. ./scripts/thor/common_config.sh

LLM_BUILD_PATH=./build/examples/llm/llm_build
VIT_BUILD_PATH=./build/examples/multimodal/visual_build
LLM_SDK_LIB_PATH=${HOME_DIR}/build
if [ ! -f "$LLM_BUILD_PATH" ]; then
    LLM_BUILD_PATH=./bin/llm_build
    VIT_BUILD_PATH=./bin/visual_build
    LLM_SDK_LIB_PATH=${HOME_DIR}/lib
fi
echo "Using LLM_BUILD_PATH: ${LLM_BUILD_PATH}"
echo "Using VIT_BUILD_PATH: ${VIT_BUILD_PATH}"

export EDGELLM_PLUGIN_PATH=${LLM_SDK_LIB_PATH}/libNvInfer_edgellm_plugin.so

skip_vit=0
skip_llm=0

for arg in "$@"; do
    case $arg in
        --skip_vit) skip_vit=1 ;;
        --skip_llm) skip_llm=1 ;;
    esac
done

if [[ $vit_only -eq 0 && $llm_only -eq 0 ]]; then
    vit_only=1
    llm_only=1
fi

if [[ $skip_vit -eq 0 ]]; then
    echo "build vit engine..."
    ${VIT_BUILD_PATH} \
    --onnxDir ${ONNX_DIR}/onnx_${LLM_QUANT_TYPE}/visual \
    --engineDir ${ENGINE_DIR}/visual \
    --maxImageTokens 576 \
    --maxImageTokens 576 \
    --maxImageTokensPerImage 576
fi

if [[ $skip_llm -eq 0 ]]; then
    echo "build llm engine..."
    ${LLM_BUILD_PATH} \
    --onnxDir ${ONNX_DIR}/onnx_${LLM_QUANT_TYPE}/llm \
    --engineDir ${ENGINE_DIR}/llm \
    --maxBatchSize 1 \
    --maxInputLen 1024 \
    --maxKVCacheCapacity 4096
fi
