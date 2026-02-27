#!/bin/bash

# MODEL_PATH=/opt/other/huye/model/qwen3vl_2b
MODEL_PATH=/opt/other/huye/model/qwen3vl_moe_4b
ONNX_DIR=${MODEL_PATH}/vlm_onnx
ENGINE_DIR=${MODEL_PATH}/engines
VIT_QUANT_TYPE=fp8
LLM_QUANT_TYPE=nvfp4
CMAKE_CUDA_ARCHITECTURES=101a
TENSORRT_DIR=/opt/update/trt10.13_new

export LD_LIBRARY_PATH=${TENSORRT_DIR}/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

printf "\n${CYAN}=================================${NC}\n"
printf "${CYAN}【参数信息】${NC}\n"
printf "%-25s : ${YELLOW}%s${NC}\n" "MODEL_PATH" "${MODEL_PATH}"
printf "%-25s : ${YELLOW}%s${NC}\n" "ONNX_DIR" "${ONNX_DIR}"
printf "%-25s : ${YELLOW}%s${NC}\n" "ENGINE_DIR" "${ENGINE_DIR}"
# printf "%-25s : ${YELLOW}%s${NC}\n" "CUDA_VERSION" "${CUDA_VERSION}"
# printf "%-25s : ${YELLOW}%s${NC}\n" "TENSORRT_VERSION" "${TENSORRT_VERSION}"
printf "%-25s : ${YELLOW}%s${NC}\n" "CMAKE_CUDA_ARCHITECTURES" "${CMAKE_CUDA_ARCHITECTURES}"
printf "%-25s : ${YELLOW}%s${NC}\n" "TENSORRT_DIR" "${TENSORRT_DIR}"
printf "%-25s : ${YELLOW}%s${NC}\n" "LD_LIBRARY_PATH" "${LD_LIBRARY_PATH}"
printf "%-25s : ${YELLOW}%s${NC}\n" "VIT_QUANT_TYPE" "${VIT_QUANT_TYPE}"
printf "%-25s : ${YELLOW}%s${NC}\n" "LLM_QUANT_TYPE" "${LLM_QUANT_TYPE}"
printf "\n"