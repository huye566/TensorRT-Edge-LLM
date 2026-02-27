from typing import Optional, Tuple

import onnx
import onnx_graphsurgeon as gs
import torch
import torch.nn as nn
from onnx.defs import OpSchema
from torch.onnx import register_custom_op_symbolic, symbolic_helper
from torch.onnx.symbolic_helper import _get_tensor_sizes

from ...common import ONNX_OPSET_VERSION

moe_fp16_plugin_schema = OpSchema(
    name="MoEFp16Plugin",
    domain="trt",
    since_version=ONNX_OPSET_VERSION,
    doc=
    "Custom TensorRT Mixture of Experts (MoE) plugin.",
    inputs=[
        OpSchema.FormalParameter(
            name="hidden_states",
            description="hidden states tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="router_weight",
            description="router weights tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="router_bias",
            description="router bias tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name=f"experts_gate_proj_weight",
            description=f"Experts gate projection weight tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name=f"experts_up_proj_weight",
            description=f"Experts up projection weight tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name=f"experts_down_proj_weight",
            description=f"Experts down projection weight tensor",
            type_str="T",
        )
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
            "Input and output data type.",
        ),
    ],
    attributes=[
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
onnx.defs.register_schema(moe_fp16_plugin_schema)


@symbolic_helper.parse_args("v", "v", "v", "v", "v", "v", "i", "i")
def symbolic_moe_fp16_plugin(
    g: torch.onnx._internal.torchscript_exporter.jit_utils.GraphContext,
    hidden_states: torch._C.Value,
    router_weight: torch._C.Value,
    router_bias: torch._C.Value,
    experts_gate_proj_weight: torch._C.Value,
    experts_up_proj_weight: torch._C.Value,
    experts_down_proj_weight: torch._C.Value,
    experts_num: int,
    experts_topk: int
):
    output = g.op(
        "trt::MoEFp16Plugin",
        hidden_states,
        router_weight,
        router_bias,
        experts_gate_proj_weight,
        experts_up_proj_weight,
        experts_down_proj_weight,
        experts_num_i=experts_num,
        experts_topk_i=experts_topk
    )

    # Set output type based on input type
    hidden_states_type = hidden_states.type()
    output_sizes = _get_tensor_sizes(hidden_states)
    output.setType(hidden_states_type.with_sizes(output_sizes))

    return output

@torch.library.custom_op("trt::moe_fp16_plugin", mutates_args=())
def moe_fp16_plugin(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: torch.Tensor,
    experts_gate_proj_weight: torch.Tensor,
    experts_up_proj_weight: torch.Tensor,
    experts_down_proj_weight: torch.Tensor,
    experts_num: int,
    experts_topk: int
) -> torch.Tensor:
    assert hidden_states.dtype == torch.float16, f"hidden_states {hidden_states.dtype} should be in float16"
    assert router_weight.shape[1] == experts_num, f"router_weight shape {router_weight.shape} second dim should be equal to experts_num {experts_num}"
    assert router_bias.shape[0] == experts_num, f"router_bias shape {router_bias.shape} first dim should be equal to experts_num {experts_num}"
    assert experts_gate_proj_weight.shape[0] == experts_num, f"experts_gate_proj_weight {experts_gate_proj_weight.shape[0]} should be equal to experts_num {experts_num}"
    assert experts_up_proj_weight.shape[0] == experts_num, f"experts_up_proj_weight {experts_up_proj_weight.shape[0]} should be equal to experts_num {experts_num}"
    assert experts_down_proj_weight.shape[0] == experts_num, f"experts_down_proj_weight {experts_down_proj_weight.shape[0]} should be equal to experts_num {experts_num}"
    return hidden_states.clone()

def register_moe_fp16_plugin_onnx_symbolic_functions() -> None:
    """Register symbolic functions for ONNX export."""

    # Register our custom symbolic functions
    register_custom_op_symbolic("trt::moe_fp16_plugin",
                                symbolic_moe_fp16_plugin, ONNX_OPSET_VERSION)

    print("Registered ONNX symbolic functions for custom MoE fp16 plugin")

class MoeFp16PluginModule(torch.nn.Module):
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

        self.register_buffer(
            "router_weight",
            torch.zeros((hidden_size, experts_num),
                        dtype=pack_dtype),
        )
        self.register_buffer(
            "router_bias",
            torch.zeros((experts_num, ),
                        dtype=pack_dtype),
        )
        self.register_buffer(
            "experts_gate_proj_weight",
            torch.zeros((experts_num, hidden_size, intermediate_size),
                        dtype=pack_dtype),
        )
        self.register_buffer(
            "experts_up_proj_weight",
            torch.zeros((experts_num, hidden_size, intermediate_size),
                        dtype=pack_dtype),
        )
        self.register_buffer(
            "experts_down_proj_weight",
            torch.zeros((experts_num, intermediate_size, hidden_size),
                        dtype=pack_dtype),
        )

    def load_state_dict_from_torch(self, mlp_module: torch.nn.Module) -> None:
        # cpy weights from the original MoE module to the plugin module
        self.router_weight.copy_(mlp_module.router.weight.transpose(1, 0).contiguous())
        self.router_bias.copy_(mlp_module.router.bias)
        for i in range(self.experts_num):
            self.experts_gate_proj_weight[i].copy_(mlp_module.experts[i].gate_proj.weight.transpose(1, 0).contiguous())
            self.experts_up_proj_weight[i].copy_(mlp_module.experts[i].up_proj.weight.transpose(1, 0).contiguous())
            self.experts_down_proj_weight[i].copy_(mlp_module.experts[i].down_proj.weight.transpose(1, 0).contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = moe_fp16_plugin(
            hidden_states=x,
            router_weight=self.router_weight,
            router_bias=self.router_bias,
            experts_gate_proj_weight=self.experts_gate_proj_weight,
            experts_up_proj_weight=self.experts_up_proj_weight,
            experts_down_proj_weight=self.experts_down_proj_weight,
            experts_num=self.experts_num,
            experts_topk=self.experts_topk
        )
        return output

def replace_moe_fp16_module_with_plugin(model: nn.Module) -> nn.Module:
    from tensorrt_edgellm.sd_models.sd_modeling_qwen3_vl_moe import Qwen3VLTextMoE

    # u need check model structure first
    for name, module in model.named_modules():
        if isinstance(module, Qwen3VLTextMoE):
            print(f"Replacing MoE module {name} with MoeFp16PluginModule")
            experts_num = module.num_experts
            hidden_size = module.hidden_size
            intermediate_size = module.experts[0].intermediate_size

            moe_plugin_module = MoeFp16PluginModule(
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


def inserted_moe_fp16_plugin(graph: gs.Graph) -> gs.Graph:
    import numpy as np
    from modelopt.onnx.quantization.gs_patching import patch_gs_modules
    patch_gs_modules()

    def get_layer_num(graph, layer_key):
        layer_numbers = set()
        for node in graph.nodes:
            if layer_key in node.name:
                try:
                    layer_str = int(node.name.split(f"{layer_key}.")[1].split("/")[0])
                    layer_numbers.add(layer_str)
                except:
                    continue
        return max(layer_numbers) + 1

    def get_op_input_constant(node, input_idx):
        if node and len(node.inputs) > input_idx:
            inp = node.inputs[input_idx]
            if isinstance(inp, gs.Constant):
                return inp.values
        return None
    
    def format_name(name):
        return name.replace('/', '.')
    
    node_map = {node.name: node for node in graph.nodes}
    layer_num = get_layer_num(graph, "layers")
    print(f"Found {layer_num} layers.")

    for i in reversed(range(layer_num)):
        # 1. start node and end node
        layer_prefix = f"/model/layers.{i}"
        start_node_name = f"{layer_prefix}/post_attention_layernorm/Mul_1"
        end_node_name = f"{layer_prefix}/Add_1"
        
        start_node = node_map.get(start_node_name)
        end_node = node_map.get(end_node_name)
        
        if not start_node or not end_node:
            print(f"Skipping layer {i}: Start or End node not found.")
            continue
        print(f"Processing Layer {i}...")

        # 2. Router Weights and Bias
        router_matmul_name = f"{layer_prefix}/mlp/router/MatMul"
        router_matmul = node_map.get(router_matmul_name)
        router_weight = get_op_input_constant(router_matmul, 1)
        
        router_add_name = f"{layer_prefix}/mlp/router/Add"
        router_add = node_map.get(router_add_name)
        router_bias = None
        if router_add:
            inp0 = router_add.inputs[0]
            if isinstance(inp0, gs.Constant) and 'bias' in inp0.name:
                router_bias = inp0.values
            else:
                inp1 = router_add.inputs[1]
                if isinstance(inp1, gs.Constant) and 'bias' in inp1.name:
                    router_bias = inp1.values
        
        if router_weight is None:
            print(f"Error: Router weights not found for layer {i}")
            continue

        # 3. Experts Weights
        experts_up_list = []
        experts_down_list = []
        experts_gate_list = []
        
        expert_idx = 0
        while True:
            expert_base = f"{layer_prefix}/mlp/experts.{expert_idx}"
            up_node = node_map.get(f"{expert_base}/up_proj/MatMul")
            if not up_node:
                break
            
            down_node = node_map.get(f"{expert_base}/down_proj/MatMul")
            gate_node = node_map.get(f"{expert_base}/gate_proj/MatMul")
            
            if not (down_node and gate_node):
                print(f"Warning: Incomplete expert {expert_idx} in layer {i}")
                break
            
            w_gate = get_op_input_constant(gate_node, 1)
            w_up = get_op_input_constant(up_node, 1)
            w_down = get_op_input_constant(down_node, 1)
            
            experts_gate_list.append(w_gate)
            experts_up_list.append(w_up)
            experts_down_list.append(w_down)
            expert_idx += 1
        
        experts_num = len(experts_up_list)
        if experts_num == 0:
            print(f"Error: No experts found for layer {i}")
            continue

        experts_gate_weight = np.stack(experts_gate_list)
        experts_up_weight = np.stack(experts_up_list)
        experts_down_weight = np.stack(experts_down_list)

        c_router_w = gs.Constant(name=format_name(f"{layer_prefix}/mlp/router_weight"), values=router_weight)
        if router_bias is None:
             print(f"Warning: Router bias not found for layer {i}, creating zeros.")
             router_bias = np.zeros((router_weight.shape[-1],), dtype=router_weight.dtype)
        c_router_b = gs.Constant(name=format_name(f"{layer_prefix}/mlp/router_bias"), values=router_bias)

        c_exp_gate = gs.Constant(name=format_name(f"{layer_prefix}/mlp/experts_gate_proj_weight"), values=experts_gate_weight)
        c_exp_up = gs.Constant(name=format_name(f"{layer_prefix}/mlp/experts_up_proj_weight"), values=experts_up_weight)
        c_exp_down = gs.Constant(name=format_name(f"{layer_prefix}/mlp/experts_down_proj_weight"), values=experts_down_weight)

        # 4. Plugin node
        hidden_states_tensor = start_node.outputs[0]
        plugin_inputs = [
            hidden_states_tensor,
            c_router_w,
            c_router_b,
            c_exp_gate,
            c_exp_up,
            c_exp_down
        ]
        plugin_output_tensor = gs.Variable(name=f"{layer_prefix}/mlp/MoEFp16Plugin_output_0", dtype=hidden_states_tensor.dtype)

        moe_plugin_node = gs.Node(
            op="MoEFp16Plugin",
            name=f"{layer_prefix}/mlp/MoEFp16Plugin",
            inputs=plugin_inputs,
            outputs=[plugin_output_tensor],
            attrs={"experts_num": experts_num, "experts_topk": 1}
        )
        graph.nodes.append(moe_plugin_node)

        # 5. Replace MLP inputs
        for idx, inp in enumerate(end_node.inputs):
            if "mlp" in inp.name:
                end_node.inputs[idx] = plugin_output_tensor
                break

    # cleanup and toposort
    graph.cleanup().toposort()
    
    return graph