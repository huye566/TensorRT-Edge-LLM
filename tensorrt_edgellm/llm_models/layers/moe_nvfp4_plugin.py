from typing import Optional, Tuple, List
import math

import onnx
import onnx_graphsurgeon as gs
import torch
import torch.nn as nn
import numpy as np
from onnx.defs import OpSchema
from torch.onnx import register_custom_op_symbolic, symbolic_helper
from torch.onnx.symbolic_helper import _get_tensor_sizes

from ...common import ONNX_OPSET_VERSION
from modelopt.torch.quantization.qtensor import NVFP4QTensor

BLOCK_SIZE = 16

def _cast_fp8(array: np.ndarray) -> np.ndarray:
    array_f32_t = torch.from_numpy(array)
    if torch.cuda.is_available():
        array_f32_t = array_f32_t.cuda()
    array_f8_t = array_f32_t.clamp(min=-448, max=448).to(torch.float8_e4m3fn).view(QUANT_DATA_TYPE)
    array_f8 = array_f8_t.cpu().numpy().astype(np.uint8)
    return array_f8

def _cast_fp4(array: np.ndarray) -> np.ndarray:
    array_f32_t = torch.from_numpy(array)
    array_f32_t_shape = array_f32_t.shape
    assert array_f32_t_shape[0] % 2 == 0, "array_f32_t_shape[0] must be divisible by 2"
    array_f4_t_shape = (array_f32_t_shape[0] // 2, *array_f32_t_shape[1:])
    if torch.cuda.is_available():
        array_f32_t = array_f32_t.cuda()
    array_f4_t = NVFP4QTensor._cast_fp4(array_f32_t)
    array_f4_t = array_f4_t.flatten()
    array_f4_t_packed = (array_f4_t[::2] | (array_f4_t[1::2] << 4)).reshape(array_f4_t_shape)
    array_f4 = array_f4_t_packed.cpu().numpy().astype(np.uint8)
    return array_f4

def pack_uint8_to_int32(tensor: torch.Tensor) -> torch.Tensor:
    orig_shape = tensor.shape
    last_dim = orig_shape[-1]
    # assert last_dim % 4 == 0, "last_dim must be divisible by 4"
    pad_size = (4 - last_dim % 4) % 4
    if pad_size > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_size), "constant", 0)
    flat = tensor.view(-1, 4)
    int32_vals = flat.view(torch.uint8).view(torch.int32).squeeze(-1)  # shape: (num_rows,)
    new_last_dim = (last_dim + 3) // 4
    new_shape = orig_shape[:-1] + (new_last_dim,)
    return int32_vals.view(new_shape)

def _get_padded_shape(M, K):
    round_up_multiple = lambda x, m: (x + m - 1) // m * m
    M_padded = round_up_multiple(M, 128)
    K_padded = round_up_multiple(K, 4)
    return M_padded, K_padded

def _prepare_scales(scales: torch.Tensor, scale_ndim: int = 2) -> torch.Tensor:
    B, M, K = scales.shape
    M_padded, K_padded = _get_padded_shape(M, K)

    # Create padded tensor
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

GLOBAL_SCALE_TYPE = torch.float32
QUANT_DATA_TYPE = torch.int8
QUANT_DATA_TYPE_BYTES = torch.iinfo(QUANT_DATA_TYPE).bits // 8
TENSOR_TYPE_NAME = {
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.int32: "int32",
    torch.float16: "float16",
    torch.float32: "float",
}

moe_nvfp4_plugin_schema = OpSchema(
    name="MoENvFp4Plugin",
    domain="trt",
    since_version=ONNX_OPSET_VERSION,
    doc="Custom TensorRT Mixture of Experts (MoE) plugin with NVFP4 quantization.",
    inputs=[
        OpSchema.FormalParameter(
            name="hidden_states",
            description="hidden states tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="router_weight",
            description="router quantized weight tensor (FP4, packed as uint8)",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="router_bias",
            description="router bias tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="experts_gate_proj_qweight",
            description="Experts gate projection quantized weight tensor (FP4, packed as uint8)",
            type_str="QT",
        ),
        OpSchema.FormalParameter(
            name="experts_gate_proj_qscales",
            description="Experts gate projection quantization scales (FP8, packed as uint8)",
            type_str="QT",
        ),
        OpSchema.FormalParameter(
            name="experts_gate_proj_input_global_scale",
            description="Experts gate projection input global scale (fp32)",
            type_str="GST",
        ),
        OpSchema.FormalParameter(
            name="experts_gate_proj_weight_global_scale",
            description="Experts gate projection weight global scale (fp32)",
            type_str="GST",
        ),
        OpSchema.FormalParameter(
            name="experts_up_proj_qweight",
            description="Experts up projection quantized weight tensor (FP4, packed as uint8)",
            type_str="QT",
        ),
        OpSchema.FormalParameter(
            name="experts_up_proj_qscales",
            description="Experts up projection quantization scales (FP8, packed as uint8)",
            type_str="QT",
        ),
        OpSchema.FormalParameter(
            name="experts_up_proj_input_global_scale",
            description="Experts up projection input global scale (fp32)",
            type_str="GST",
        ),
        OpSchema.FormalParameter(
            name="experts_up_proj_weight_global_scale",
            description="Experts up projection weight global scale (fp32)",
            type_str="GST",
        ),
        OpSchema.FormalParameter(
            name="experts_down_proj_qweight",
            description="Experts down projection quantized weight tensor (FP4, packed as uint8)",
            type_str="QT",
        ),
        OpSchema.FormalParameter(
            name="experts_down_proj_qscales",
            description="Experts down projection quantization scales (FP8, packed as uint8)",
            type_str="QT",
        ),
        OpSchema.FormalParameter(
            name="experts_down_proj_input_global_scale",
            description="Experts down projection input global scale (fp32)",
            type_str="GST",
        ),
        OpSchema.FormalParameter(
            name="experts_down_proj_weight_global_scale",
            description="Experts down projection weight global scale (fp32)",
            type_str="GST",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="moe_output",
            description="MoE output tensor",
            type_str="T",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"],
            "Input and output data type for hidden_states and bias.",
        ),
        (
            "QT",
            [f"tensor({TENSOR_TYPE_NAME[QUANT_DATA_TYPE]})"],
            "Quantized data type.",
        ),
        (
            "GST",
            [f"tensor({TENSOR_TYPE_NAME[GLOBAL_SCALE_TYPE]})"],
            "Global scale type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="experts_gate_alpha",
            type=OpSchema.AttrType.FLOATS,
            description="Per-expert alpha for gate projection",
            required=True,
        ),
        OpSchema.Attribute(
            name="experts_up_alpha",
            type=OpSchema.AttrType.FLOATS,
            description="Per-expert alpha for up projection",
            required=True,
        ),
        OpSchema.Attribute(
            name="experts_down_alpha",
            type=OpSchema.AttrType.FLOATS,
            description="Per-expert alpha for down projection",
            required=True,
        ),
        OpSchema.Attribute(
            name="experts_num",
            type=OpSchema.AttrType.INT,
            description="Number of experts",
            required=True,
        ),
        OpSchema.Attribute(
            name="experts_topk",
            type=OpSchema.AttrType.INT,
            description="TopK number of experts",
            required=True,
        )
    ],
)
onnx.defs.register_schema(moe_nvfp4_plugin_schema)

@symbolic_helper.parse_args("v", "v", "v", "v", "v", "v", "v", "v", "v", "v",
                            "v", "v", "v", "v", "v", "fs", "fs", "fs", "i", "i")
def symbolic_moe_nvfp4_plugin(
    g: torch.onnx._internal.torchscript_exporter.jit_utils.GraphContext,
    hidden_states: torch._C.Value,
    router_weight: torch._C.Value,
    router_bias: torch._C.Value,
    experts_gate_proj_qweight: torch._C.Value,
    experts_gate_proj_qscales: torch._C.Value,
    experts_gate_proj_input_global_scale: torch._C.Value,
    experts_gate_proj_weight_global_scale: torch._C.Value,
    experts_up_proj_qweight: torch._C.Value,
    experts_up_proj_qscales: torch._C.Value,
    experts_up_proj_input_global_scale: torch._C.Value,
    experts_up_proj_weight_global_scale: torch._C.Value,
    experts_down_proj_qweight: torch._C.Value,
    experts_down_proj_qscales: torch._C.Value,
    experts_down_proj_input_global_scale: torch._C.Value,
    experts_down_proj_weight_global_scale: torch._C.Value,
    experts_gate_alpha: List[float],
    experts_up_alpha: List[float],
    experts_down_alpha: List[float],
    experts_num: int,
    experts_topk: int
):
    output = g.op(
        "trt::MoENvFp4Plugin",
        hidden_states,
        router_weight,
        router_bias,
        experts_gate_proj_qweight,
        experts_gate_proj_qscales,
        experts_gate_proj_input_global_scale,
        experts_gate_proj_weight_global_scale,
        experts_up_proj_qweight,
        experts_up_proj_qscales,
        experts_up_proj_input_global_scale,
        experts_up_proj_weight_global_scale,
        experts_down_proj_qweight,
        experts_down_proj_qscales,
        experts_down_proj_input_global_scale,
        experts_down_proj_weight_global_scale,
        experts_gate_alpha_f = experts_gate_alpha,
        experts_up_alpha_f = experts_up_alpha,
        experts_down_alpha_f = experts_down_alpha,
        experts_num_i = experts_num,
        experts_topk_i = experts_topk
    )

    # Set output type based on input type
    hidden_states_type = hidden_states.type()
    output_sizes = _get_tensor_sizes(hidden_states)
    output.setType(hidden_states_type.with_sizes(output_sizes))

    return output

@torch.library.custom_op("trt::moe_nvfp4_plugin", mutates_args=())
def moe_nvfp4_plugin(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: torch.Tensor,
    experts_gate_proj_qweight: torch.Tensor,
    experts_gate_proj_qscales: torch.Tensor,
    experts_gate_proj_input_global_scale: torch.Tensor,
    experts_gate_proj_weight_global_scale: torch.Tensor,
    experts_up_proj_qweight: torch.Tensor,
    experts_up_proj_qscales: torch.Tensor,
    experts_up_proj_input_global_scale: torch.Tensor,
    experts_up_proj_weight_global_scale: torch.Tensor,
    experts_down_proj_qweight: torch.Tensor,
    experts_down_proj_qscales: torch.Tensor,
    experts_down_proj_input_global_scale: torch.Tensor,
    experts_down_proj_weight_global_scale: torch.Tensor,
    experts_gate_alpha: List[float],
    experts_up_alpha: List[float],
    experts_down_alpha: List[float],
    experts_num: int,
    experts_topk: int
) -> torch.Tensor:
    assert hidden_states.dtype == torch.float16, f"hidden_states {hidden_states.dtype} should be in float16"
    assert experts_gate_proj_qweight.dtype == QUANT_DATA_TYPE, f"experts_gate_proj_qweight {experts_gate_proj_qweight.dtype} should be in uint8"
    assert experts_gate_proj_qscales.dtype == QUANT_DATA_TYPE, f"experts_gate_proj_qscales {experts_gate_proj_qscales.dtype} should be in uint8"
    assert experts_gate_proj_input_global_scale.dtype == GLOBAL_SCALE_TYPE, f"experts_gate_proj_input_global_scale {experts_gate_proj_input_global_scale.dtype} should be in float32"
    assert experts_gate_proj_weight_global_scale.dtype == GLOBAL_SCALE_TYPE, f"experts_gate_proj_weight_global_scale {experts_gate_proj_weight_global_scale.dtype} should be in float32"
    assert router_bias.dtype == torch.float16, f"router_bias {router_bias.dtype} should be in float16"
    
    # Check shapes for router
    hidden_size = hidden_states.shape[-1]
    assert router_weight.shape[0] == hidden_size, f"router_weight shape {router_weight.shape} first dim should be equal to hidden_size {hidden_size}"
    assert router_weight.shape[1] == experts_num, f"router_weight shape {router_weight.shape} second dim should be equal to experts_num {experts_num}"
    assert router_bias.shape[0] == experts_num, f"router_bias shape {router_bias.shape} first dim should be equal to experts_num {experts_num}"
    return hidden_states.clone()

def register_moe_nvfp4_plugin_onnx_symbolic_functions() -> None:
    """Register symbolic functions for ONNX export."""
    register_custom_op_symbolic("trt::moe_nvfp4_plugin",
                                symbolic_moe_nvfp4_plugin, ONNX_OPSET_VERSION)
    print("Registered ONNX symbolic functions for custom MoE NVFP4 plugin")

class MoeNvFp4PluginModule(torch.nn.Module):
    def __init__(
            self,
            experts_num: int,
            experts_topk: int,
            hidden_size: int,
            intermediate_size: int,
            pack_dtype: torch.dtype = torch.float16):
        super().__init__()

        self.experts_num = experts_num
        self.experts_topk = experts_topk
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.pack_dtype = pack_dtype
        self.experts_gate_alpha = [0.0] * experts_num
        self.experts_up_alpha = [0.0] * experts_num
        self.experts_down_alpha = [0.0] * experts_num

        # Register buffers for router
        self.register_buffer(
            "router_weight",
            torch.zeros((hidden_size, experts_num), dtype=pack_dtype),
        )
        self.register_buffer(
            "router_bias",
            torch.zeros((experts_num, ), dtype=pack_dtype),
        )
        
        # Register buffers for experts gate projection
        self.register_buffer(
            "experts_gate_proj_qweight",
            torch.zeros((experts_num, intermediate_size, hidden_size // 2 // QUANT_DATA_TYPE_BYTES), dtype=QUANT_DATA_TYPE),
        )
        padd_shape = _get_padded_shape(intermediate_size, hidden_size // BLOCK_SIZE)
        self.register_buffer(
            "experts_gate_proj_qscales",
            torch.zeros((experts_num, padd_shape[0], padd_shape[1] // QUANT_DATA_TYPE_BYTES), dtype=QUANT_DATA_TYPE),
        )
        self.register_buffer(
            "experts_gate_proj_input_global_scale",
            torch.zeros((experts_num, ), dtype=GLOBAL_SCALE_TYPE),
        )
        self.register_buffer(
            "experts_gate_proj_weight_global_scale",
            torch.zeros((experts_num, ), dtype=GLOBAL_SCALE_TYPE),
        )
        
        # Register buffers for experts up projection
        self.register_buffer(
            "experts_up_proj_qweight",
            torch.zeros((experts_num, intermediate_size, hidden_size // 2 // QUANT_DATA_TYPE_BYTES), dtype=QUANT_DATA_TYPE),
        )
        padd_shape = _get_padded_shape(intermediate_size, hidden_size // BLOCK_SIZE)
        self.register_buffer(
            "experts_up_proj_qscales",
            torch.zeros((experts_num, padd_shape[0], padd_shape[1] // QUANT_DATA_TYPE_BYTES), dtype=QUANT_DATA_TYPE),
        )
        self.register_buffer(
            "experts_up_proj_input_global_scale",
            torch.zeros((experts_num, ), dtype=GLOBAL_SCALE_TYPE),
        )
        self.register_buffer(
            "experts_up_proj_weight_global_scale",
            torch.zeros((experts_num, ), dtype=GLOBAL_SCALE_TYPE),
        )

        # Register buffers for experts down projection
        self.register_buffer(
            "experts_down_proj_qweight",
            torch.zeros((experts_num, hidden_size, intermediate_size // 2 // QUANT_DATA_TYPE_BYTES), dtype=QUANT_DATA_TYPE),
        )
        padd_shape = _get_padded_shape(hidden_size, intermediate_size // BLOCK_SIZE)
        self.register_buffer(
            "experts_down_proj_qscales",
            torch.zeros((experts_num, padd_shape[0], padd_shape[1] // QUANT_DATA_TYPE_BYTES), dtype=QUANT_DATA_TYPE),
        )
        self.register_buffer(
            "experts_down_proj_input_global_scale",
            torch.zeros((experts_num, ), dtype=GLOBAL_SCALE_TYPE),
        )
        self.register_buffer(
            "experts_down_proj_weight_global_scale",
            torch.zeros((experts_num, ), dtype=GLOBAL_SCALE_TYPE),
        )

    def _quantize_weight_fp4(
        self, 
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from modelopt.torch.quantization.qtensor import NVFP4QTensor
        tensor_amax = torch.abs(weight).max().to(torch.float32)
        global_scale_mopt =  tensor_amax / (448.0 * 6.0)
        weight_scaling_factor, weight_scaling_factor_2 = (
            NVFP4QTensor.get_weights_scaling_factor(weight, BLOCK_SIZE, global_scale_mopt))
        quantized_weight, _, _ = (
            NVFP4QTensor.quantize(weight, BLOCK_SIZE, weight_scaling_factor, weight_scaling_factor_2, try_tensorrt=True))
        qweight = quantized_weight._quantized_data
        qscales = _prepare_scales(weight_scaling_factor.unsqueeze(0), scale_ndim=3).squeeze(0)
        global_scale = 1.0 / weight_scaling_factor_2
        return qweight, qscales, global_scale

    def load_state_dict_from_torch(
        self, 
        mlp_module: torch.nn.Module
    ) -> None:
        # print(mlp_module)
        # print(mlp_module.router.weight)
        # exit()
        amax_factor = 448.0 * 6.0
        self.router_weight.copy_(mlp_module.router.weight.transpose(1, 0).contiguous())
        self.router_bias.copy_(mlp_module.router.bias)

        # Quantize experts weights
        for i in range(self.experts_num):
            # Gate projection
            gate_proj_weight = mlp_module.experts[i].gate_proj.weight.contiguous()
            gate_proj_qweight, gate_proj_qscales, gate_proj_weight_global_scale = self._quantize_weight_fp4(gate_proj_weight)
            if QUANT_DATA_TYPE in [torch.int8, torch.uint8]:
                self.experts_gate_proj_qweight[i].copy_(gate_proj_qweight)
                self.experts_gate_proj_qscales[i].copy_(gate_proj_qscales.view(torch.uint8))
            elif QUANT_DATA_TYPE == torch.int32:
                self.experts_gate_proj_qweight[i].copy_(pack_uint8_to_int32(gate_proj_qweight))
                self.experts_gate_proj_qscales[i].copy_(pack_uint8_to_int32(gate_proj_qscales))
            else:
                raise NotImplementedError("Only support pack to BITS-8 or INT32 now.")
            self.experts_gate_proj_weight_global_scale[i].copy_(gate_proj_weight_global_scale)
            self.experts_gate_proj_input_global_scale[i].copy_(amax_factor / mlp_module.experts[i].gate_proj.input_quantizer.amax)
            self.experts_gate_alpha[i] = float(1.0 / (
                self.experts_gate_proj_weight_global_scale[i] * self.experts_gate_proj_input_global_scale[i]
            ))
            # Up projection
            up_proj_weight = mlp_module.experts[i].up_proj.weight.contiguous()
            up_proj_qweight, up_proj_qscales, up_proj_weight_global_scale = self._quantize_weight_fp4(up_proj_weight)
            if QUANT_DATA_TYPE in [torch.int8, torch.uint8]:
                self.experts_up_proj_qweight[i].copy_(up_proj_qweight)
                self.experts_up_proj_qscales[i].copy_(up_proj_qscales.view(torch.uint8))
            elif QUANT_DATA_TYPE == torch.int32:
                self.experts_up_proj_qweight[i].copy_(pack_uint8_to_int32(up_proj_qweight))
                self.experts_up_proj_qscales[i].copy_(pack_uint8_to_int32(up_proj_qscales))
            else:
                raise NotImplementedError("Only support pack to BITS-8 or INT32 now.")
            self.experts_up_proj_weight_global_scale[i].copy_(up_proj_weight_global_scale)
            self.experts_up_proj_input_global_scale[i].copy_(amax_factor / mlp_module.experts[i].up_proj.input_quantizer.amax)
            self.experts_up_alpha[i] = float(1.0 / (
                self.experts_up_proj_weight_global_scale[i] * self.experts_up_proj_input_global_scale[i]
            ))
            # Down projection
            down_proj_weight = mlp_module.experts[i].down_proj.weight.contiguous()
            down_proj_qweight, down_proj_qscales, down_proj_weight_global_scale = self._quantize_weight_fp4(down_proj_weight)
            if QUANT_DATA_TYPE in [torch.int8, torch.uint8]:
                self.experts_down_proj_qweight[i].copy_(down_proj_qweight)
                self.experts_down_proj_qscales[i].copy_(down_proj_qscales.view(torch.uint8))
            elif QUANT_DATA_TYPE == torch.int32:
                self.experts_down_proj_qweight[i].copy_(pack_uint8_to_int32(down_proj_qweight))
                self.experts_down_proj_qscales[i].copy_(pack_uint8_to_int32(down_proj_qscales))
            else:
                raise NotImplementedError("Only support pack to BITS-8 or INT32 now.")
            self.experts_down_proj_weight_global_scale[i].copy_(down_proj_weight_global_scale)
            self.experts_down_proj_input_global_scale[i].copy_(amax_factor / mlp_module.experts[i].down_proj.input_quantizer.amax)
            self.experts_down_alpha[i] = float(1.0 / (
                self.experts_down_proj_weight_global_scale[i] * self.experts_down_proj_input_global_scale[i]
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = moe_nvfp4_plugin(
            hidden_states=x,
            router_weight=self.router_weight,
            router_bias=self.router_bias,
            experts_gate_proj_qweight=self.experts_gate_proj_qweight,
            experts_gate_proj_qscales=self.experts_gate_proj_qscales,
            experts_gate_proj_input_global_scale=self.experts_gate_proj_input_global_scale,
            experts_gate_proj_weight_global_scale=self.experts_gate_proj_weight_global_scale,
            experts_up_proj_qweight=self.experts_up_proj_qweight,
            experts_up_proj_qscales=self.experts_up_proj_qscales,
            experts_up_proj_input_global_scale=self.experts_up_proj_input_global_scale,
            experts_up_proj_weight_global_scale=self.experts_up_proj_weight_global_scale,
            experts_down_proj_qweight=self.experts_down_proj_qweight,
            experts_down_proj_qscales=self.experts_down_proj_qscales,
            experts_down_proj_input_global_scale=self.experts_down_proj_input_global_scale,
            experts_down_proj_weight_global_scale=self.experts_down_proj_weight_global_scale,
            experts_gate_alpha=self.experts_gate_alpha,
            experts_up_alpha=self.experts_up_alpha,
            experts_down_alpha=self.experts_down_alpha,
            experts_num=self.experts_num,
            experts_topk=self.experts_topk
        )
        return output

def replace_moe_nvfp4_module_with_plugin(model: nn.Module) -> nn.Module:
    from tensorrt_edgellm.sd_models.sd_modeling_qwen3_vl_moe import Qwen3VLTextMoE

    # You need to check model structure first
    for name, module in model.named_modules():
        if isinstance(module, Qwen3VLTextMoE):
            print(f"Replacing MoE module {name} with MoeNvFp4PluginModule")
            experts_num = module.num_experts
            hidden_size = module.hidden_size
            intermediate_size = module.experts[0].intermediate_size

            moe_plugin_module = MoeNvFp4PluginModule(
                experts_num=experts_num,
                experts_topk=1,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                pack_dtype=torch.float16
            )

            moe_plugin_module.load_state_dict_from_torch(module)
            parent = model
            if '.' in name:
                parent_name, module_name = name.rsplit('.', 1)
                parent = dict(model.named_modules())[parent_name]
            else:
                module_name = name

            setattr(parent, module_name, moe_plugin_module)
    return model