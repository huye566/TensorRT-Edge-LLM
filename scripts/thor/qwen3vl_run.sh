#!/bin/bash

export HOME_DIR="$(dirname "$(dirname "$(dirname $(readlink -f $0))")")"
echo "HOME_DIR: ${HOME_DIR}"
cd ${HOME_DIR}

. ./scripts/thor/common_config.sh

LLM_INFER_PATH=./build/examples/llm/llm_inference
LLM_SDK_LIB_PATH=${HOME_DIR}/build
if [ ! -f "$LLM_INFER_PATH" ]; then
    LLM_INFER_PATH=./bin/llm_inference
    LLM_SDK_LIB_PATH=${HOME_DIR}/lib
fi
echo "Using LLM_INFER_PATH: ${LLM_INFER_PATH}"

export EDGELLM_PLUGIN_PATH=${LLM_SDK_LIB_PATH}/libNvInfer_edgellm_plugin.so

INPUT_JSON=${HOME_DIR}/scripts/thor/input_with_images.json

${LLM_INFER_PATH} \
  --engineDir ${ENGINE_DIR}/llm \
  --multimodalEngineDir ${ENGINE_DIR}/visual \
  --inputFile ${INPUT_JSON} \
  --outputFile output.json
