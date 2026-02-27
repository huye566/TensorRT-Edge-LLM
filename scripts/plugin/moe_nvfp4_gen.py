import os
import torch
import functools
import torch.nn as nn
import numpy as np
from safetensors.torch import save_file, load_file
from typing import Tuple

try:
    from modelopt.torch.quantization.qtensor import NVFP4QTensor
    MODELOPT_AVAILABLE = True
except ImportError:
    print("警告: modelopt未安装，将使用模拟的NVFP4量化（仅用于生成测试数据框架）")
    MODELOPT_AVAILABLE = False

# ============ 配置 ============
BLOCK_SIZE = 16  # NVFP4 块大小
# 0000 -> 0
E2M1_TO_FLOAT32 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0, # 0.0?
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]
DATA_ROOT_DIR = '/workspace/app/ml/vlm_fastv/TensorRT-Edge-LLM/tests_cpp/resources/moe'  # 请修改为实际路径

# ============ 辅助函数 ============
def cuda_timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        result = func(*args, **kwargs)
        end_event.record()

        torch.cuda.synchronize()  # 等待所有流完成
        elapsed_ms = start_event.elapsed_time(end_event)
        print(f"{func.__name__} 执行耗时: {elapsed_ms:.2f} ms")
        return result
    return wrapper


def _get_padded_shape(M: int, K: int) -> Tuple[int, int]:
    round_up_multiple = lambda x, m: (x + m - 1) // m * m
    M_padded = round_up_multiple(M, 128)
    K_padded = round_up_multiple(K, 4)
    return M_padded, K_padded

def _swizzle_scales(scales: torch.Tensor, scale_ndim: int = 2) -> torch.Tensor:
    B, M, K = scales.shape
    M_padded, K_padded = _get_padded_shape(M, K)

    padded_scales = torch.zeros((B, M_padded, K_padded), dtype=scales.dtype, device=scales.device)
    padded_scales[:B, :M, :K] = scales

    batches, rows, cols = padded_scales.shape
    assert rows % 128 == 0
    assert cols % 4 == 0

    # Reshape and permute
    padded_scales = padded_scales.reshape(batches, rows // 128, 4, 32, cols // 4, 4)
    padded_scales = padded_scales.permute((0, 1, 4, 3, 2, 5))
    padded_scales = padded_scales.contiguous()

    # Final reshape
    if scale_ndim == 2:
        padded_scales = padded_scales.reshape(M_padded, K_padded)
    else:
        padded_scales = padded_scales.reshape(B, M_padded, K_padded)

    return padded_scales

@cuda_timeit
def _break_fp4_bytes(a):
    assert a.dtype == torch.uint8
    m, n = a.shape
    a = a.flatten()
    # Get upper 4 bits
    highHalfByte = (a & 0xF0) >> 4
    # Get lower 4 bits
    lowHalfByte = a & 0x0F
    fH = torch.tensor([E2M1_TO_FLOAT32[x] for x in highHalfByte]).to(a.device)
    fL = torch.tensor([E2M1_TO_FLOAT32[x] for x in lowHalfByte]).to(a.device)
    # [0xAB, 0xCD] -> [0xB, 0xA, 0xD, 0xC], 0xCDAB
    out = torch.stack((fL, fH), dim=-1).reshape(m, n * 2)
    return out

@cuda_timeit
def _break_fp4_bytes_v2(a):
    assert a.dtype == torch.uint8
    m, n = a.shape

    lut = torch.tensor(E2M1_TO_FLOAT32, dtype=torch.float32, device=a.device)
    
    # 提取高4位和低4位（注意：a >> 4 已得到高4位数值，无需再 & 0x0F）
    low = a & 0x0F               # 低4位，范围 0-15
    high = a >> 4                # 高4位，范围 0-15

    fL = lut[low.long()]         # [m, n]
    fH = lut[high.long()]        # [m, n]
    
    out = torch.stack((fL, fH), dim=-1).reshape(m, n * 2)
    return out


def _quantize_to_nvfp4(
        data: torch.Tensor,
        preset_global_scale=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor_amax = torch.abs(data).max().to(torch.float32)
        global_scale_mopt =  tensor_amax / (448.0 * 6.0)
        if preset_global_scale is not None:
            global_scale_mopt = 1.0 / preset_global_scale
        qscales_linear, scaling_factor_2 = (
            NVFP4QTensor.get_weights_scaling_factor(data, BLOCK_SIZE, global_scale_mopt))
        quantized_weight, _, _ = (
            NVFP4QTensor.quantize(data, BLOCK_SIZE, qscales_linear, scaling_factor_2, try_tensorrt=True))
        qweight = quantized_weight._quantized_data
        print(qweight.shape, qscales_linear.shape)
        qscales = _swizzle_scales(qscales_linear.unsqueeze(0), scale_ndim=3).squeeze(0)
        global_scale = 1.0 / scaling_factor_2
        return qweight, qscales, global_scale, qscales_linear


def _dequantize_linear_to_fp32(
    tensor_fp4, tensor_scales_linear, global_scale, dtype, device, block_size=16
):
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = _break_fp4_bytes_v2(tensor_fp4)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    block_scale = tensor_scales_linear.to(torch.float32) / global_scale

    # scale the tensor
    print(tensor_f32.shape, block_scale.shape)
    out = (tensor_f32 * block_scale.unsqueeze(-1)).reshape(m, k).to(device=device, dtype=dtype)
    return out

def recover_swizzled_scales(scale, m, k):
    rounded_m = ((m + 128 - 1) // 128) * 128
    scale_k = k // BLOCK_SIZE
    rounded_k = ((scale_k + 4 - 1) // 4) * 4
    # Recover the swizzled scaling factor to linear layout
    tmp = torch.reshape(scale, (1, rounded_m // 128, rounded_k // 4, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    result = torch.reshape(tmp, (rounded_m, rounded_k)).to(torch.float32)
    return result[:m, :rounded_k]

@cuda_timeit
def _dequantize_swizzle_to_fp32(
    tensor_fp4, tensor_sf, global_scale, dtype, device, block_size=16
):
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = _break_fp4_bytes_v2(tensor_fp4)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf_linear = recover_swizzled_scales(tensor_sf, m, k)
    block_scale = tensor_sf_linear.to(torch.float32) / global_scale

    # scale the tensor
    print(tensor_f32.shape, block_scale.shape)
    out = (tensor_f32 * block_scale.unsqueeze(-1)).reshape(m, k).to(device=device, dtype=dtype)
    return out

def print_array_bytes(arr, name, limit=5):
    if arr is None:
        print(f"{name} = None")
        return
    arr_cpu = arr.cpu().detach()
    bytes_data = arr_cpu.numpy().tobytes()
    total_bytes = len(bytes_data)
    print(f"{name} [{total_bytes} bytes]: ", end="")
    if total_bytes > 2 * limit:
        for i in range(limit):
            print(f"0x{bytes_data[i]:02x} ", end="")
        print("... ", end="")
        for i in range(total_bytes - limit, total_bytes):
            print(f"0x{bytes_data[i]:02x}" + (" " if i != total_bytes - 1 else ""), end="")
    else:
        for i in range(total_bytes):
            print(f"0x{bytes_data[i]:02x}" + (" " if i != total_bytes - 1 else ""), end="")
    print()

def print_array_values(arr, name, limit=5):
    if arr is None:
        print(f"{name} = None")
        return
    arr_cpu = arr.cpu().detach().flatten()
    total_elems = arr_cpu.numel()
    print(f"{name} [{total_elems}]: ", end="")
    if total_elems > 2 * limit:
        for i in range(limit):
            print(f"{arr_cpu[i].item():.6f} ", end="")
        print("... ", end="")
        for i in range(total_elems - limit, total_elems):
            print(f"{arr_cpu[i].item():.6f}" + (" " if i != total_elems - 1 else ""), end="")
    else:
        for i in range(total_elems):
            print(f"{arr_cpu[i].item():.6f}" + (" " if i != total_elems - 1 else ""), end="")
    print()

def print_nvfp4_matmul_info(a_fp4, a_scales, a_global_scale,
                             b_fp4, b_scales, b_global_scale, out):
    print("\n========== Python NVFP4 Matmul Debug Info ==========")
    print('m = {}, k = {}, n = {}'.format(a_fp4.shape[0], a_fp4.shape[1] * 2, b_fp4.shape[0]))
    a_gs = a_global_scale.item() if torch.is_tensor(a_global_scale) else a_global_scale
    b_gs = b_global_scale.item() if torch.is_tensor(b_global_scale) else b_global_scale
    print(f"a_global_scale: {a_gs}")
    print(f"b_global_scale: {b_gs}")
    print(f'alpha = {1.0 / (a_gs * b_gs)}')

    print_array_bytes(a_fp4, "a_fp4")
    print_array_bytes(b_fp4, "b_fp4")
    print_array_bytes(a_scales.view(torch.uint8), "a_scales")
    print_array_values(a_scales, "a_scales")
    print_array_bytes(b_scales.view(torch.uint8), "b_scales")
    print_array_values(b_scales, "b_scales")

    print_array_values(out, "out")
    print("=====================================================\n")


def nvfp4_matmul(a_fp4, a_scales, a_global_scale,
                 b_fp4, b_scales, b_global_scale,
                 dtype, device, block_size=16):
    _, m_k = a_fp4.shape
    _, n_k = b_fp4.shape
    assert m_k == n_k

    a = _dequantize_swizzle_to_fp32(a_fp4, a_scales, a_global_scale, dtype, device, block_size)
    b = _dequantize_swizzle_to_fp32(b_fp4, b_scales, b_global_scale, dtype, device, block_size)
    out = torch.matmul(a, b.t())

    print_nvfp4_matmul_info(a_fp4, a_scales, a_global_scale, 
                            b_fp4, b_scales, b_global_scale, out)
    return out


class Qwen3VLTextConfig:
    def __init__(self, hidden_size=2048, intermediate_size=6144, hidden_act="silu"):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act


class Qwen3VLTextMLP_NVFP4(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = nn.SiLU()

        self.register_buffer('gate_qweight', torch.empty(0))
        self.register_buffer('gate_qscales', torch.empty(0))
        self.register_buffer('gate_input_global_scale', torch.empty(0))
        self.register_buffer('gate_weight_global_scale', torch.empty(0))
        self.register_buffer('up_qweight', torch.empty(0))
        self.register_buffer('up_qscales', torch.empty(0))
        self.register_buffer('up_input_global_scale', torch.empty(0))
        self.register_buffer('up_weight_global_scale', torch.empty(0))
        self.register_buffer('down_qweight', torch.empty(0))
        self.register_buffer('down_qscales', torch.empty(0))
        self.register_buffer('down_input_global_scale', torch.empty(0))
        self.register_buffer('down_weight_global_scale', torch.empty(0))

    def forward(self, x_fp16):
        # x_fp16: [num_tokens, hidden_size]
        if len(x_fp16.shape) == 2:
            x_flat = x_fp16.unsqueeze(0)
        batch_size, seq, hidden = x_flat.shape
        total_tokens = seq * batch_size
        x_flat = x_flat.reshape(total_tokens, hidden)

        gate_q, gate_s, gate_gs, _ = _quantize_to_nvfp4(
            x_flat, preset_global_scale=self.gate_input_global_scale
        )
        # 2. gate投影 (NVFP4矩阵乘)
        gate_out_fp16 = nvfp4_matmul(
            gate_q, gate_s, gate_gs,
            self.gate_qweight, self.gate_qscales, self.gate_weight_global_scale,
            dtype=torch.float16, device=x_flat.device, block_size=BLOCK_SIZE
        )

        # 3. up投影
        up_q, up_s, up_gs, _ = _quantize_to_nvfp4(
            x_flat, preset_global_scale=self.up_input_global_scale
        )
        up_out_fp16 = nvfp4_matmul(
            up_q, up_s, up_gs,
            self.up_qweight, self.up_qscales, self.up_weight_global_scale,
            dtype=torch.float16, device=x_flat.device, block_size=BLOCK_SIZE
        )

        # 4. SiLU和Hadamard积（FP16）
        silu_out = self.act_fn(gate_out_fp16)
        hadamard_out = silu_out * up_out_fp16

        # 5. down投影（需要量化hadamard_out）
        had_q, had_s, had_gs, _ = _quantize_to_nvfp4(
            hadamard_out, preset_global_scale=self.down_input_global_scale)
        down_out_fp16 = nvfp4_matmul(
            had_q, had_s, had_gs,
            self.down_qweight, self.down_qscales, self.down_weight_global_scale,
            dtype=torch.float16, device=x_flat.device, block_size=BLOCK_SIZE
        )

        return down_out_fp16, gate_out_fp16, up_out_fp16, silu_out, hadamard_out

class Qwen3VLTextMoE_NVFP4(nn.Module):
    def __init__(self, config, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = config.hidden_size
        self.experts = nn.ModuleList([
            Qwen3VLTextMLP_NVFP4(config)
            for _ in range(num_experts)
        ])
        self.router = nn.Linear(self.hidden_size, num_experts, bias=True)

    def set_router_params(self, weight, bias):
        self.router.weight.data = weight
        self.router.bias.data = bias

    def set_expert_weights(self, expert_idx,
                           gate_qw, gate_qs, gate_gs,
                           up_qw, up_qs, up_gs,
                           down_qw, down_qs, down_gs):
        exp = self.experts[expert_idx]
        exp.gate_qweight = gate_qw
        exp.gate_qscales = gate_qs
        exp.gate_weight_global_scale = gate_gs
        exp.up_qweight = up_qw
        exp.up_qscales = up_qs
        exp.up_weight_global_scale = up_gs
        exp.down_qweight = down_qw
        exp.down_qscales = down_qs
        exp.down_weight_global_scale = down_gs

    def set_expert_input_gs(self, expert_idx, gate_input_gs, up_input_gs, down_input_gs):
        exp = self.experts[expert_idx]
        exp.gate_input_global_scale = gate_input_gs
        exp.up_input_global_scale = up_input_gs
        exp.down_input_global_scale = down_input_gs

    def forward(self, x_fp16):
        # router FP16
        logits = self.router(x_fp16)
        expert_idx = logits.argmax(dim=-1)  # [total_tokens]

        out = torch.zeros_like(x_fp16)
        intermediate = {
            'router_logits': logits,
            'expert_idx': expert_idx,
        }

        for i in range(self.num_experts):
            mask = (expert_idx == i)
            if mask.any():
                expert_in = x_fp16[mask]
                expert_out, gate_out, up_out, silu_out, had_out = self.experts[i](expert_in)
                out[mask] = expert_out
                intermediate[f'expert_{i}'] = {
                    'output': expert_out,
                    'gate_out': gate_out,
                    'up_out': up_out,
                    'silu_out': silu_out,
                    'hadamard_out': had_out
                }

        return out, intermediate

def load_moe_nvfp4_weights(model, fp16_weight_file, save_path=None, moe_layer_idx=0):
    if fp16_weight_file is None:
        if save_path is not None and os.path.exists(save_path):
            moe_weights = load_file(save_path)
            model.set_router_params(
                moe_weights[f'model.layers.{moe_layer_idx}.mlp.router_weight'].transpose(1, 0).contiguous(), 
                moe_weights[f'model.layers.{moe_layer_idx}.mlp.router_bias']
            )
            for i in range(model.num_experts):
                model.set_expert_weights(
                    i,
                    moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.gate_qweight'][i],
                    moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.gate_qscales'][i],
                    moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.gate_input_global_scale'][i],
                    moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.gate_weight_global_scale'][i],
                    moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.up_qweight'][i],
                    moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.up_qscales'][i],
                    moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.up_input_global_scale'][i],
                    moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.up_weight_global_scale'][i],
                    moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.down_qweight'][i],
                    moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.down_qscales'][i],
                    moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.down_input_global_scale'][i],
                    moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.down_weight_global_scale'][i]
                )
        else:
            raise FileNotFoundError("No valid weight file found.")
    else:
        fp16_weights = load_file(fp16_weight_file)
        router_weight = fp16_weights[f'model.layers.{moe_layer_idx}.mlp.router_weight']
        router_bias = fp16_weights[f'model.layers.{moe_layer_idx}.mlp.router_bias']
        gate_weights = fp16_weights[f'model.layers.{moe_layer_idx}.mlp.experts_gate_proj_weight']
        up_weights = fp16_weights[f'model.layers.{moe_layer_idx}.mlp.experts_up_proj_weight']
        down_weights = fp16_weights[f'model.layers.{moe_layer_idx}.mlp.experts_down_proj_weight']

        model.set_router_params(router_weight.transpose(1, 0).contiguous(), router_bias)
        moe_weights = {}
        moe_weights[f'model.layers.{moe_layer_idx}.mlp.router_weight'] = router_weight.cpu()
        moe_weights[f'model.layers.{moe_layer_idx}.mlp.router_bias'] = router_bias.cpu()
        gate_qw_list = []
        gate_qs_list = []
        gate_gs_list = []
        up_qw_list = []
        up_qs_list = []
        up_gs_list = []
        down_qw_list = []
        down_qs_list = []
        down_gs_list = []
        for i in range(model.num_experts):
            gate_qw, gate_qs, gate_gs, _ = _quantize_to_nvfp4(gate_weights[i].transpose(1, 0).contiguous())
            up_qw, up_qs, up_gs, _ = _quantize_to_nvfp4(up_weights[i].transpose(1, 0).contiguous())
            down_qw, down_qs, down_gs, _ = _quantize_to_nvfp4(down_weights[i].transpose(1, 0).contiguous())
            model.set_expert_weights(i, gate_qw, gate_qs, gate_gs,
                                        up_qw, up_qs, up_gs,
                                        down_qw, down_qs, down_gs)
            gate_qw_list.append(gate_qw)
            gate_qs_list.append(gate_qs)
            gate_gs_list.append(gate_gs)
            up_qw_list.append(up_qw)
            up_qs_list.append(up_qs)
            up_gs_list.append(up_gs)
            down_qw_list.append(down_qw)
            down_qs_list.append(down_qs)
            down_gs_list.append(down_gs)
        moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.gate_qweight'] = torch.stack(gate_qw_list, dim=0)
        moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.gate_qscales'] = torch.stack(gate_qs_list, dim=0)
        moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.gate_input_global_scale'] = torch.tensor(
            [701.0, 363.0, 363.0], dtype=torch.float32) # 363.0
        moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.gate_weight_global_scale'] = torch.stack(gate_gs_list, dim=0)
        moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.up_qweight'] = torch.stack(up_qw_list, dim=0)
        moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.up_qscales'] = torch.stack(up_qs_list, dim=0)
        moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.up_input_global_scale'] = torch.tensor(
            [701.0, 363.0, 363.0], dtype=torch.float32)
        moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.up_weight_global_scale'] = torch.stack(up_gs_list, dim=0)

        moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.down_qweight'] = torch.stack(down_qw_list, dim=0)
        moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.down_qscales'] = torch.stack(down_qs_list, dim=0)
        moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.down_input_global_scale'] = torch.tensor(
            [146.0, 201.25, 209.875], dtype=torch.float32) # 205.25
        moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.down_weight_global_scale'] = torch.stack(down_gs_list, dim=0)

        moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.gate_alpha'] = torch.ones(model.num_experts, dtype=torch.float32)
        moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.up_alpha'] = torch.ones(model.num_experts, dtype=torch.float32)
        moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.down_alpha'] = torch.ones(model.num_experts, dtype=torch.float32)
        for i in range(model.num_experts):
            moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.gate_alpha'][i] = float(1.0 / (
                    moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.gate_input_global_scale'][i] *
                    moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.gate_weight_global_scale'][i]))
            moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.up_alpha'][i] = float(1.0 / (
                    moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.up_input_global_scale'][i] *
                    moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.up_weight_global_scale'][i]))
            moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.down_alpha'][i] = float(1.0 / (
                    moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.down_input_global_scale'][i] *
                    moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.down_weight_global_scale'][i]))
        for i in range(model.num_experts):
            model.set_expert_input_gs(i, 
                moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.gate_input_global_scale'][i],
                moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.up_input_global_scale'][i],
                moe_weights[f'model.layers.{moe_layer_idx}.mlp.experts.down_input_global_scale'][i]
            )

        if save_path is not None and not os.path.exists(save_path):
            print(f"✅ 量化权重已保存到 {save_path}.")
            save_file(moe_weights, save_path)
    return model

def generate_input_nvfp4(filename, seq_len=1, dtype=torch.float16):
    if not os.path.exists(filename):
        torch.manual_seed(42)
        np.random.seed(42)
        
        batch_size = 1
        hidden_size = 2048
        input = torch.randn(batch_size, seq_len, hidden_size).to(dtype)
        inputs = {}
        inputs['hidden_state'] = input

        save_file(inputs, filename)
        print(f"✅ 测试输入已保存到 {filename}, 输入形状: {input.shape}.")
    else:
        input = load_file(filename)['hidden_state']
        print(f"✅ 测试输入已从 {filename} 加载, 输入形状: {input.shape}.")
    return input


def print_data_info_python(model, x, intermediate_results, output):
    print("=" * 80)
    print("PYTHON IMPLEMENTATION DATA INFO")
    print("=" * 80)
    
    # 基本参数
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    hidden_size = model.hidden_size
    intermediate_size = model.experts[0].intermediate_size
    experts_num = model.num_experts
    experts_topk = 1
    total_tokens = batch_size * seq_len
    
    print(f"MoE Input Parameters:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Hidden Size: {hidden_size}")
    print(f"  Intermediate Size: {intermediate_size}")
    print(f"  Experts Number: {experts_num}")
    print(f"  Experts Top-K: {experts_topk}")
    print(f"  Total Tokens: {total_tokens}")
    
    # 打印路由器权重
    router_weight = model.router.weight.data
    router_bias = model.router.bias.data
    print(f"\nRouter weight shape: {router_weight.shape}, dtype: {router_weight.dtype}")
    print(f"Router weight data (first 5 elements):")
    router_weight_flat = router_weight.flatten()
    for i in range(min(5, len(router_weight_flat))):
        print(f"  [{i}] = {router_weight_flat[i].item():.6f}")
    
    print(f"\nRouter bias data:")
    for i in range(len(router_bias)):
        print(f"  [{i}] = {router_bias[i].item():.6f}")
    
    for expert_idx in range(experts_num):
        print(f"\nExpert {expert_idx}:")
        print(f"\nExpert {expert_idx} gate_proj qweight (first 5 elements):")
        gate_qweight = model.experts[expert_idx].gate_qweight.flatten()
        for i in range(min(5, len(gate_qweight))):
            print(f"  [{i}] = {gate_qweight[i].item()}")
        print(f"\nExpert {expert_idx} gate_proj qscales (first 5 elements):")
        gate_qscales = model.experts[expert_idx].gate_qscales.flatten()
        for i in range(min(5, len(gate_qscales))):
            print(f"  [{i}] = {gate_qscales[i].view(torch.uint8).item()}")

        print(f"\nExpert {expert_idx} up_proj qweight (first 5 elements):")
        up_qweight = model.experts[expert_idx].up_qweight.flatten()
        for i in range(min(5, len(up_qweight))):
            print(f"  [{i}] = {up_qweight[i].item()}")
        print(f"\nExpert {expert_idx} up_proj qscales (first 5 elements):")
        up_qscales = model.experts[expert_idx].up_qscales.flatten()
        for i in range(min(5, len(up_qscales))):
            print(f"  [{i}] = {up_qscales[i].view(torch.uint8).item()}")

        print(f"\nExpert {expert_idx} down_proj qweight (first 5 elements):")
        down_qweight = model.experts[expert_idx].down_qweight.flatten()
        for i in range(min(5, len(down_qweight))):
            print(f"  [{i}] = {down_qweight[i].item()}")
        print(f"\nExpert {expert_idx} down_proj qscales (first 5 elements):")
        down_qscales = model.experts[expert_idx].down_qscales.flatten()
        for i in range(min(5, len(down_qscales))):
            print(f"  [{i}] = {down_qscales[i].view(torch.uint8).item()}")

    print(f"\nRouter logits:")
    logits = intermediate_results['router_logits']
    print(f"  Shape: {logits.shape}, dtype: {logits.dtype}")
    for i in range(logits.shape[-1]):
        print(f"  Expert {i}: {logits[0, 0, i].item():.6f}")
    
    print(f"\nSelected expert indices:")
    expert_idx = intermediate_results['expert_idx']
    for i in range(total_tokens):
        selected_expert = expert_idx[0, i].item()
        print(f"  Token {i} -> Expert {selected_expert}")

        print(f"\nInput data (first 5 elements):")
        input_data = x[0, i]
        for j in range(min(5, len(input_data))):
            print(f"  [{j}] = {input_data[j].item():.6f}")

        # 打印中间结果（如果有）
        if f'expert_{selected_expert}' in intermediate_results:
            expert_result = intermediate_results[f'expert_{selected_expert}']
            
            print(f"\nExpert {selected_expert} intermediate results:")
            
            if 'gate_out' in expert_result:
                gate_out = expert_result['gate_out']
                print(f"  Gate output shape: {gate_out.shape}, dtype: {gate_out.dtype}")
                print(f"  Gate output (first 5 elements):")
                gate_flat = gate_out.flatten()
                for i in range(min(5, len(gate_flat))):
                    print(f"    [{i}] = {gate_flat[i].item():.6f}")
            
            if 'up_out' in expert_result:
                up_out = expert_result['up_out']
                print(f"  Up output shape: {up_out.shape}, dtype: {up_out.dtype}")
                print(f"  Up output (first 5 elements):")
                up_flat = up_out.flatten()
                for i in range(min(5, len(up_flat))):
                    print(f"    [{i}] = {up_flat[i].item():.6f}")
            
            if 'silu_out' in expert_result:
                silu_out = expert_result['silu_out']
                print(f"  SiLU output shape: {silu_out.shape}, dtype: {silu_out.dtype}")
                print(f"  SiLU output (first 5 elements):")
                silu_flat = silu_out.flatten()
                for i in range(min(5, len(silu_flat))):
                    print(f"    [{i}] = {silu_flat[i].item():.6f}")
            
            if 'hadamard_out' in expert_result:
                hadamard_out = expert_result['hadamard_out']
                print(f"  Hadamard output shape: {hadamard_out.shape}, dtype: {hadamard_out.dtype}")
                print(f"  Hadamard output (first 5 elements):")
                hadamard_flat = hadamard_out.flatten()
                for i in range(min(5, len(hadamard_flat))):
                    print(f"    [{i}] = {hadamard_flat[i].item():.6f}")
            
            if 'output' in expert_result:
                expert_output = expert_result['output']
                print(f"  Expert output shape: {expert_output.shape}, dtype: {expert_output.dtype}")
                print(f"  Expert output (first 5 elements):")
                output_flat = expert_output.flatten()
                for i in range(min(5, len(output_flat))):
                    print(f"    [{i}] = {output_flat[i].item():.6f}")

    print(f"\nFinal output shape: {output.shape}, dtype: {output.dtype}")
    print(f"Final output data (first 10 elements):")
    output_flat = output.flatten()
    for i in range(min(10, len(output_flat))):
        print(f"  [{i}] = {output_flat[i].item():.6f}")


def save_intermediate_results(intermediate_results, output_file, moe_layer_idx=0, experts_num=3):
    results = {}
    results['expert_idx'] = intermediate_results['expert_idx']
    for expert_idx in range(experts_num):
        if f'expert_{expert_idx}' in intermediate_results:
            results[f'model.layers.{moe_layer_idx}.mlp.experts.{expert_idx}.gate_out'] = intermediate_results[f'expert_{expert_idx}']['gate_out']
            results[f'model.layers.{moe_layer_idx}.mlp.experts.{expert_idx}.up_out'] = intermediate_results[f'expert_{expert_idx}']['up_out']
            results[f'model.layers.{moe_layer_idx}.mlp.experts.{expert_idx}.silu_out'] = intermediate_results[f'expert_{expert_idx}']['silu_out']
            results[f'model.layers.{moe_layer_idx}.mlp.experts.{expert_idx}.hadamard_out'] = intermediate_results[f'expert_{expert_idx}']['hadamard_out']
            results[f'model.layers.{moe_layer_idx}.mlp.experts.{expert_idx}.output'] = intermediate_results[f'expert_{expert_idx}']['output']
    save_file(results, output_file)
    print(f"✅ 中间结果已保存到 {output_file}")


def generate_nvfp4_test_data(cpu_offload=True):
    FP16_WEIGHT_FILE = os.path.join(DATA_ROOT_DIR, "moe_weights.safetensors")
    FP16_INPUT_FILE = os.path.join(DATA_ROOT_DIR, "seq1/moe_input.safetensors")
    OUTPUT_DIR = os.path.join(DATA_ROOT_DIR, "seq1")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    NVFP4_WEIGHT_FILE = os.path.join(OUTPUT_DIR, "moe_nvfp4_weights.safetensors")
    NVFP4_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "moe_nvfp4_output.safetensors")
    NVFP4_INTERMEDIATE_FILE = os.path.join(OUTPUT_DIR, "moe_nvfp4_intermediate_results.safetensors")

    experts_num = 3
    config = Qwen3VLTextConfig(hidden_size=2048, intermediate_size=6144)
    model_nvfp4 = Qwen3VLTextMoE_NVFP4(config, num_experts=experts_num)
    model = load_moe_nvfp4_weights(model_nvfp4, FP16_WEIGHT_FILE, NVFP4_WEIGHT_FILE)

    if cpu_offload:
        model = model.eval().cpu()
    else:
        model = model.eval().cuda()
    model.eval()

    print("运行NVFP4模拟前向...")
    with torch.no_grad():
        x = generate_input_nvfp4(FP16_INPUT_FILE)
        if not cpu_offload:
            x = x.cuda()
        output, intermediate_results = model_nvfp4(x)
        print_data_info_python(model, x, intermediate_results, output)

    print("保存文件...")
    save_file({'output': output.cpu()}, NVFP4_OUTPUT_FILE)
    save_intermediate_results(intermediate_results, NVFP4_INTERMEDIATE_FILE, experts_num=experts_num)

    print("完成！生成的文件：")
    print(f"  {NVFP4_WEIGHT_FILE}")
    print(f"  {NVFP4_OUTPUT_FILE}")
    print(f"  {NVFP4_INTERMEDIATE_FILE}")


def compare_output():
    output_file = os.path.join(DATA_ROOT_DIR, "seq1/moe_nvfp4_output.safetensors")
    ref_output_file = os.path.join(DATA_ROOT_DIR, "seq1/moe_output_ref.safetensors")
    output = load_file(output_file)['output']
    ref_output = load_file(ref_output_file)['output']

    diff = torch.abs(output - ref_output)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"最大绝对误差: {max_diff:.10f}")
    print(f"平均绝对误差: {mean_diff:.10f}")

    if torch.allclose(output, ref_output, rtol=1e-5, atol=1e-5):
        print("✅ 输出一致!")
        return True
    else:
        print("❌ 输出不一致!")
        print(output)
        print(ref_output)
        return False


generate_nvfp4_test_data(False)
# compare_output()