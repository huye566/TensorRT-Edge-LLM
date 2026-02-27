#!/bin/bash

export HOME_DIR="$(dirname "$(dirname "$(dirname $(readlink -f $0))")")"
echo "HOME_DIR: ${HOME_DIR}"
cd ${HOME_DIR}

. ./scripts/thor/common_config.sh

VLM_INFER_PATH=./build/examples/llm/vlm_infer_test
LLM_SDK_LIB_PATH=${HOME_DIR}/build
LLM_VLM_INFER_LIB_PATH=${LLM_SDK_LIB_PATH}/examples/llm
if [ ! -f "$VLM_INFER_PATH" ]; then
    VLM_INFER_PATH=./bin/vlm_infer_test
    LLM_SDK_LIB_PATH=${HOME_DIR}/lib
    LLM_VLM_INFER_LIB_PATH=${HOME_DIR}/lib
fi
echo "Using VLM_INFER_PATH: ${VLM_INFER_PATH}"

export EDGELLM_PLUGIN_PATH=${LLM_SDK_LIB_PATH}/libNvInfer_edgellm_plugin.so

INPUT_JSON=${HOME_DIR}/scripts/thor/input_with_images.json

# nsys profile -t cudnn,cublas,cuda,nvtx,osrt -s cpu -o moe_nvfp4_vlm --force-overwrite true \
${VLM_INFER_PATH} \
  --llmEnginePath=${ENGINE_DIR}/llm \
  --visualEnginePath=${ENGINE_DIR}/visual \
  --libInferPath=${LLM_VLM_INFER_LIB_PATH}/libvlm_infer.so \
  --modelType=qwen3_vl \
  --inputFile ${INPUT_JSON} \
  --outputFile output.json \
  --closeCudaGraph \
  --dumpProfile
