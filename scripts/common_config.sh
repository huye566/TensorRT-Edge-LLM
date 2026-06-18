#!/bin/bash

# model info
MODEL_PATH=/home/nvidia/.cache/huggingface/hub/Qwen3-VL-2B-Instruct
# MODEL_PATH=/root/.cache/huggingface/hub/Qwen3-VL-30B-A3B-Instruct
# MODEL_PATH=/root/.cache/huggingface/hub/Qwen3-VL-4B-MoE-Init
VERSION=080
ONNX_DIR=${MODEL_PATH}/vlm_onnx_${VERSION}
ENGINE_DIR=${MODEL_PATH}/engines_${VERSION}
DATA_SETS="/workspace/app/datasets/cnn_dailymail"
DATA_SETS_VIT="/workspace/app/datasets/lmms-lab/MMMU"

# 3rds
OPT_DIR=/opt
CUDA_ARCH_X86=86
CUDA_VERSION=12.8
TENSORRT_VERSION=10.13.3.9
CUDA_PATH=/usr/local/cuda-${CUDA_VERSION}
export LD_LIBRARY_PATH=${OPT_DIR}/TensorRT-${TENSORRT_VERSION}/lib:${CUDA_PATH}/lib64:$LD_LIBRARY_PATH

export EDGE_LLM_PATH=${HOME_DIR}
export PYTHONPATH=$EDGE_LLM_PATH:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0
CUDA_DEVICE=cuda:0
VIT_QUANT_TYPE=fp16
# LLM_QUANT_TYPE=fp16
LLM_QUANT_TYPE=int4_awq
# LLM_QUANT_TYPE=nvfp4

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

printf "\n${CYAN}=================================${NC}\n"
printf "${CYAN}【模型配置信息】${NC}\n"
printf "%-25s : ${YELLOW}%s${NC}\n" "MODEL_PATH" "${MODEL_PATH}"
printf "%-25s : ${YELLOW}%s${NC}\n" "ONNX_DIR" "${ONNX_DIR}"
printf "%-25s : ${YELLOW}%s${NC}\n" "ENGINE_DIR" "${ENGINE_DIR}"
printf "%-25s : ${YELLOW}%s${NC}\n" "DATA_SETS" "${DATA_SETS}"
printf "%-25s : ${YELLOW}%s${NC}\n" "DATA_SETS_VIT" "${DATA_SETS_VIT}"
printf "\n"

printf "${CYAN}【第三方库配置】${NC}\n"
printf "%-25s : ${YELLOW}%s${NC}\n" "CUDA_VERSION" "${CUDA_VERSION}"
printf "%-25s : ${YELLOW}%s${NC}\n" "TENSORRT_VERSION" "${TENSORRT_VERSION}"
printf "%-25s : ${YELLOW}%s${NC}\n" "LD_LIBRARY_PATH" "${LD_LIBRARY_PATH}"
printf "%-25s : ${YELLOW}%s${NC}\n" "CUDA_ARCH_X86" "${CUDA_ARCH_X86}"
printf "\n"

printf "${CYAN}【量化配置】${NC}\n"
printf "%-25s : ${YELLOW}%s${NC}\n" "VIT_QUANT_TYPE" "${VIT_QUANT_TYPE}"
printf "%-25s : ${YELLOW}%s${NC}\n" "LLM_QUANT_TYPE" "${LLM_QUANT_TYPE}"
printf "%-25s : ${YELLOW}%s${NC}\n" "CUDA_VISIBLE_DEVICES" "${CUDA_VISIBLE_DEVICES}"
printf "%-25s : ${YELLOW}%s${NC}\n" "CUDA_DEVICE" "${CUDA_DEVICE}"
printf "\n"
