#!/bin/bash
# pip install -e .

export HOME_DIR="$(dirname "$(dirname $(readlink -f $0))")"
echo "HOME_DIR: ${HOME_DIR}"
cd ${HOME_DIR}

. ./scripts/common_config.sh

vit_only=0
llm_only=0
llm_quant_close=0

for arg in "$@"; do
    case $arg in
        --vit_only) vit_only=1 ;;
        --llm_only) llm_only=1 ;;
        --llm_quant_close) llm_quant_close=1 ;;
    esac
done

if [[ $vit_only -eq 0 && $llm_only -eq 0 ]]; then
    vit_only=1
    llm_only=1
fi

if [[ $vit_only -eq 1 ]]; then
    echo "导出视觉编码器..."
    if [[ ${VIT_QUANT_TYPE} == "fp16" ]]; then
        tensorrt-edgellm-export-visual \
            --model_dir ${MODEL_PATH} \
            --output_dir ${ONNX_DIR}/visual_enc_onnx_${VIT_QUANT_TYPE} \
            --dataset_dir ${DATA_SETS_VIT} \
            --device ${CUDA_DEVICE}
    else
        tensorrt-edgellm-export-visual \
            --model_dir ${MODEL_PATH} \
            --output_dir ${ONNX_DIR}/visual_enc_onnx_${VIT_QUANT_TYPE} \
            --quantization ${VIT_QUANT_TYPE} \
            --dataset_dir ${DATA_SETS_VIT} \
            --device ${CUDA_DEVICE}
    fi
fi

if [[ $llm_only -eq 1 ]]; then
    if [[ ${LLM_QUANT_TYPE} == "fp16" ]]; then
        echo "导出语言模型..."
        tensorrt-edgellm-export-llm \
            --model_dir ${MODEL_PATH} \
            --output_dir ${ONNX_DIR}/llm_onnx_${LLM_QUANT_TYPE} \
            --device ${CUDA_DEVICE}
    else
        if [[ $llm_quant_close -eq 0 ]]; then
            echo "量化语言模型..."
            tensorrt-edgellm-quantize-llm \
                --model_dir ${MODEL_PATH} \
                --quantization ${LLM_QUANT_TYPE} \
                --output_dir ${ONNX_DIR}/llm_onnx_${LLM_QUANT_TYPE}_quantized \
                --dataset_dir ${DATA_SETS} \
                --device ${CUDA_DEVICE}
        fi
        echo "导出语言模型..."
        tensorrt-edgellm-export-llm \
            --model_dir ${ONNX_DIR}/llm_onnx_${LLM_QUANT_TYPE}_quantized \
            --output_dir ${ONNX_DIR}/llm_onnx_${LLM_QUANT_TYPE} \
            --device ${CUDA_DEVICE}
    fi
fi

echo "完成"