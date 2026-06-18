#!/bin/bash
# pip install -e .

export HOME_DIR="$(dirname "$(dirname $(readlink -f $0))")"
echo "HOME_DIR: ${HOME_DIR}"
cd ${HOME_DIR}

. ./scripts/common_config.sh

quant_close=0
ARGS_OPTION=" "

for arg in "$@"; do
    case $arg in
        --quant_close) quant_close=1 ;;
        --skip_vit) ARGS_OPTION="$ARGS_OPTION --skip-visual" ;;
        --skip_llm) ARGS_OPTION="$ARGS_OPTION --skip-llm" ;;
    esac
done


if [[ ${LLM_QUANT_TYPE} == "fp16" ]]; then
    echo "导出模型..."
    # --dtype fp16
    # --dtype fp16
    # --skip-llm
    # --skip-visual
    # --skip-audio
    # --skip-code2wav
    # --eagle-base
    # --fp8-embedding
    # --nvfp4-moe-backend thor,geforce
    # --max-kv-cache-capacity 4096
    # --mtp
    # --skip-action
    tensorrt-edgellm-export \
        ${MODEL_PATH} \
        ${ONNX_DIR}/onnx_${LLM_QUANT_TYPE} \
        ${ARGS_OPTION}
else
    if [[ $quant_close -eq 0 ]]; then
        echo "量化模型..."
        # --quantization nvfp4
        # --lm_head_quantization fp8/nvfp4
        # --visual_quantization fp8
        # --audio_quantization fp8
        # --kv_cache_quantization fp8
        # --dtype fp16
        # --device cuda:0
        # --num_samples 512
        tensorrt-edgellm-quantize llm \
            --model_dir ${MODEL_PATH} \
            --quantization ${LLM_QUANT_TYPE} \
            --output_dir ${ONNX_DIR}/onnx_${LLM_QUANT_TYPE}_quantized \
            --dataset ${DATA_SETS} \
            --device ${CUDA_DEVICE}
    fi
    echo "导出模型..."
    tensorrt-edgellm-export \
        ${ONNX_DIR}/onnx_${LLM_QUANT_TYPE}_quantized \
        ${ONNX_DIR}/onnx_${LLM_QUANT_TYPE} \
        ${ARGS_OPTION}
fi

echo "完成"
