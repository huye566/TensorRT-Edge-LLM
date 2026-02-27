# python moe check
import os
import torch
import torch.nn as nn
import numpy as np
from safetensors.torch import save_file, load_file

class Qwen3VLTextConfig:
    def __init__(self, hidden_size=2048, intermediate_size=6144, hidden_act="silu"):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

class Qwen3VLTextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward_v1(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    
    def forward_v2(self, x):
        gate_out = self.gate_proj(x)
        up_out = self.up_proj(x)
        silu_out = self.act_fn(gate_out)
        hadamard_out = silu_out * up_out
        down_proj = self.down_proj(hadamard_out)
        return (down_proj, gate_out, up_out, silu_out, hadamard_out)

    def forward(self, x):
        return self.forward_v2(x)

class Qwen3VLTextMoE(nn.Module):
    def __init__(self, config, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = config.hidden_size

        self.experts = nn.ModuleList(
            [Qwen3VLTextMLP(config) for _ in range(num_experts)]
        )

        self.router = nn.Linear(self.hidden_size, num_experts, bias=True)

    def forward(self, x):
        logits = self.router(x)
        expert_idx = logits.argmax(dim=-1)
        out = torch.zeros_like(x)

        intermediate_results = {}
        intermediate_results['router_logits'] = logits
        intermediate_results['expert_idx'] = expert_idx
        for i, expert in enumerate(self.experts):
            mask = (expert_idx == i)
            if mask.any():
                expert_out = expert(x[mask])
                if isinstance(expert_out, tuple):
                    out[mask] = expert_out[0]
                    intermediate_results[f'expert_{i}'] = {
                        'output': expert_out[0],
                        'gate_out': expert_out[1],
                        'up_out': expert_out[2],
                        'silu_out': expert_out[3],
                        'hadamard_out': expert_out[4]
                    }
                else:
                    out[mask] = expert_out
        return out, intermediate_results
    
def gen_input_data(filename, seq_len=1, dtype=torch.float16):
    if not os.path.exists(filename):
        # 固定随机种子确保可重复
        torch.manual_seed(42)
        np.random.seed(42)
        
        batch_size = 1
        hidden_size = 2048
        input = torch.randn(batch_size, seq_len, hidden_size).to(dtype)

        save_file({'hidden_state': input}, filename)
        print(f"✅ 测试输入已保存到 {filename}, 输入形状: {input.shape}.")
    else:
        input = load_file(filename)['hidden_state']
        print(f"✅ 测试输入已从 {filename} 加载, 输入形状: {input.shape}.")
    return input

def load_moe_weights(model, filename, moe_layer_idx=0):
    weights = load_file(filename)

    router_weight = weights[f'model.layers.{moe_layer_idx}.mlp.router_weight']
    router_bias = weights[f'model.layers.{moe_layer_idx}.mlp.router_bias']
    gate_weights = weights[f'model.layers.{moe_layer_idx}.mlp.experts_gate_proj_weight']
    up_weights = weights[f'model.layers.{moe_layer_idx}.mlp.experts_up_proj_weight']
    down_weights = weights[f'model.layers.{moe_layer_idx}.mlp.experts_down_proj_weight']

    model.router.weight.data = router_weight.transpose(1, 0).contiguous()
    model.router.bias.data = router_bias
    
    # 打印数据类型信息
    print(f"路由器权重 dtype: {router_weight.dtype}, shape: {router_weight.shape}")
    print(f"专家门控权重 dtype: {gate_weights.dtype}, shape: {gate_weights.shape}")

    for i in range(model.num_experts):
        model.experts[i].gate_proj.weight.data = gate_weights[i].transpose(1, 0).contiguous()
        model.experts[i].up_proj.weight.data = up_weights[i].transpose(1, 0).contiguous()
        model.experts[i].down_proj.weight.data = down_weights[i].transpose(1, 0).contiguous()
    
    print(f"✅ 权重已从 {filename} 加载")
    return model

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
        print(f"\nExpert {expert_idx} gate_proj weight (first 5 elements):")
        gate_weight = model.experts[expert_idx].gate_proj.weight.data.flatten()
        for i in range(min(5, len(gate_weight))):
            print(f"  [{i}] = {gate_weight[i].item():.6f}")

        print(f"\nExpert {expert_idx} up_proj weight (first 5 elements):")
        up_weight = model.experts[expert_idx].up_proj.weight.data.flatten()
        for i in range(min(5, len(up_weight))):
            print(f"  [{i}] = {up_weight[i].item():.6f}")

        print(f"\nExpert {expert_idx} down_proj weight (first 5 elements):")
        down_weight = model.experts[expert_idx].down_proj.weight.data.flatten()
        for i in range(min(5, len(down_weight))):
            print(f"  [{i}] = {down_weight[i].item():.6f}")

    print(f"\nRouter logits:")
    logits = intermediate_results['router_logits']
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
    print(f"Final output data (first 5 elements):")
    output_flat = output.flatten()
    for i in range(min(10, len(output_flat))):
        print(f"  [{i}] = {output_flat[i].item():.6f}")

SEQ_NUM = 1
DATA_ROOT_DIR = '/workspace/app/ml/vlm_fastv/TensorRT-Edge-LLM/tests_cpp/resources/moe'
GEN_DATA_DIR = os.path.join(DATA_ROOT_DIR, f"seq{SEQ_NUM}")

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

def test_moe(cpu_offload=True):
    input_file = os.path.join(GEN_DATA_DIR, "moe_input.safetensors")
    weight_file = os.path.join(DATA_ROOT_DIR, "moe_weights.safetensors")
    output_file = os.path.join(GEN_DATA_DIR, "moe_output.safetensors")
    im_res_file = os.path.join(GEN_DATA_DIR, "moe_intermediate_results.safetensors")
    config = Qwen3VLTextConfig(hidden_size=2048, intermediate_size=6144)
    model = Qwen3VLTextMoE(config, num_experts=3)
    model = load_moe_weights(model, weight_file)
    if cpu_offload:
        model = model.eval().cpu()
    else:
        model = model.eval().cuda()
    model.eval()
    with torch.no_grad():
        x = gen_input_data(input_file, SEQ_NUM)
        if not cpu_offload:
            x = x.cuda()

        output, intermediate_results = model(x)
        print_data_info_python(model, x, intermediate_results, output)
        output = output.cpu()

    save_file({'output': output}, output_file)
    save_intermediate_results(intermediate_results, im_res_file)
    print(f"✅ 模型输出已保存到 {output_file}")

def compare_output():
    output_file = os.path.join(GEN_DATA_DIR, "moe_output.safetensors")
    ref_output_file = os.path.join(GEN_DATA_DIR, "moe_output_ref.safetensors")
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
        return False

test_moe(False)