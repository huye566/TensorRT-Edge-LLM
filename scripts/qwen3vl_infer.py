from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoProcessor
import os
import sys
import torch
import time
import argparse
import subprocess
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

from tensorrt_edgellm.sd_models.sd_modeling_qwen3_vl_moe import SDQwen3VLMoeForConditionalGeneration

class InferConfig(object):
    def __init__(self, model_path=None, max_new_tokens=512):
        self.model_dir = model_path
        self.torch_dtype = torch.float16
        self.max_new_tokens = max_new_tokens
        self.do_sample = True
        self.temperature = 0.1  # 调整到一个更合理的值
        self.top_p = 0.9       # 新增top_p参数
        self.repetition_penalty = 1.0
        self.device = torch.device("cuda:0")

class Qwen3VLInfer(object):
    def __init__(self, model_path, max_new_tokens=512):
        self.config = InferConfig(model_path, max_new_tokens)

        self.model = SDQwen3VLMoeForConditionalGeneration.from_pretrained(
            self.config.model_dir, 
            dtype=self.config.torch_dtype, 
            low_cpu_mem_usage=True,
            # device_map="auto"
        ).to(self.config.device).eval().cuda()

        self.processor = AutoProcessor.from_pretrained(
            self.config.model_dir, 
            use_fast=True,
            min_pixels=128*32*32, max_pixels=1024*32*32
        )

    def export_model(self, save_path):
        factor_l = [0.99, 1.0, 0.98]
        for i in range(len(self.model.model.language_model.layers)):
            for j in range(len(self.model.model.language_model.layers[0].mlp.experts)):
                self.model.model.language_model.layers[i].mlp.experts[j].gate_proj.weight.data.mul_(factor_l[j])
                self.model.model.language_model.layers[i].mlp.experts[j].up_proj.weight.data.mul_(factor_l[j])
                self.model.model.language_model.layers[i].mlp.experts[j].down_proj.weight.data.mul_(factor_l[j])
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

    def extract_onnx_moe_weights(self, save_path, moe_layer_idx=0):
        from safetensors.torch import save_file
        moe_state_dict = {}
        moe_layer = self.model.model.language_model.layers[moe_layer_idx].mlp
        moe_state_dict[f"model.layers.{moe_layer_idx}.mlp.router_weight"] = moe_layer.router.weight.transpose(1, 0).contiguous().data
        moe_state_dict[f"model.layers.{moe_layer_idx}.mlp.router_bias"] = moe_layer.router.bias.data
        gate_weights = []
        up_weights = []
        down_weights = []
        for i in range(len(moe_layer.experts)):
            expert = moe_layer.experts[i]
            gate_weights.append(expert.gate_proj.weight.transpose(1, 0).contiguous().data)
            up_weights.append(expert.up_proj.weight.transpose(1, 0).contiguous().data)
            down_weights.append(expert.down_proj.weight.transpose(1, 0).contiguous().data)
        moe_state_dict[f"model.layers.{moe_layer_idx}.mlp.experts_gate_proj_weight"] = torch.stack(gate_weights, dim=0)
        moe_state_dict[f"model.layers.{moe_layer_idx}.mlp.experts_up_proj_weight"] = torch.stack(up_weights, dim=0)
        moe_state_dict[f"model.layers.{moe_layer_idx}.mlp.experts_down_proj_weight"] = torch.stack(down_weights, dim=0)
        save_file(moe_state_dict, save_path)

        shapes_info = {key: value.shape for key, value in moe_state_dict.items()}
        print("权重形状:")
        for key, shape in shapes_info.items():
            print(f"  {key}: {shape}")


    def check_model_info(self):
        for name, module in self.model.model.language_model.named_modules():
            print(f"Module: {name}, Type: {type(module)}")

    def prepare_model_inputs(self, image, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image","image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        return inputs

    def infer(self, image, prompt, show_log=True):
        try:
            time_stats = {}
            start_time = time.perf_counter()

            inputs = self.prepare_model_inputs(image, prompt)
            time_stats['process'] = (time.perf_counter() - start_time) * 1000

            generate_start = time.perf_counter()
            w0 = self.model.model.language_model.layers[0].mlp.experts[0].gate_proj.weight.data
            w1 = self.model.model.language_model.layers[0].mlp.experts[1].gate_proj.weight.data
            print(f"w0: {w0.shape}, w1: {w1.shape}")
            print(f"w0: {w0[:2,:2]}, w1: {w1[:2,:2]}")
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                num_beams=1,
                early_stopping=False,
                # no_repeat_ngram_size=2  # 防止重复ngram
            )
            time_stats['generate'] = (time.perf_counter() - generate_start) * 1000

            postprocess_start = time.perf_counter()
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            time_stats['postprocess'] = (time.perf_counter() - postprocess_start) * 1000
            time_stats['total'] = (time.perf_counter() - start_time) * 1000

            if show_log:
                print("\n===== 时间统计 (ms) =====")
                print(f"输入准备: {time_stats['process']:.3f} ms")
                print(f"模型生成: {time_stats['generate']:.3f} ms, token nums: {len(generated_ids_trimmed[0])} ({(len(generated_ids_trimmed[0]) / time_stats['generate'] * 1000):.1f} t/s)")
                print(f"结果后处理: {time_stats['postprocess']:.3f} ms")
                print("-----------------------")
                print(f"总耗时: {time_stats['total']:.3f} ms")
                print("=======================\n")
            return output_text[0] if output_text else ""
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            return f"Inference error: {str(e)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3VL Inference")
    parser.add_argument("--model_path", type=str, 
                        default="/root/.cache/huggingface/hub/Qwen3-VL-4B-MoE-Init",
                        help="Path to the model directory")
    parser.add_argument("--image_path", type=str, 
                        default="/workspace/app/datasets/text_recog.png",
                        help="Path to the input image")
    parser.add_argument("--prompt", type=str, 
                        default="<image>\n请将图片中的文字内容提取出来，并以文本形式返回",
                        help="Text prompt for the model")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate")
    parser.add_argument("--export", action='store_true', default=False,
                        help="Whether to export the model after inference")
    parser.add_argument("--model_print", action='store_true', default=False,
                        help="Whether to show log information")
    parser.add_argument("--moe_layer_idx", type=int, default=0, help="MoE layer index")
    parser.add_argument("--extract_moe_safetensor", action='store_true', default=False,
                        help="Whether to extract MoE weights as SafeTensors")
    args = parser.parse_args()
    torch.cuda.set_device(1)

    infer = Qwen3VLInfer(args.model_path, args.max_new_tokens)
    image = Image.open(args.image_path).convert("RGB")
    output = infer.infer(image, args.prompt)
    print("User: ", args.prompt)
    print("ACK:  ", output)

    if args.model_print:
        infer.check_model_info()

    if args.export:
        infer.export_model(os.path.join(args.model_path, "exported_model"))
    
    if args.extract_moe_safetensor:
        # python scripts/qwen3vl_infer.py --model_path=/root/.cache/huggingface/hub/Qwen3-VL-4B-MoE-Init/exported_model --extract_moe_safetensor
        infer.extract_onnx_moe_weights(os.path.join(args.model_path, "moe_weights.safetensors"), args.moe_layer_idx)