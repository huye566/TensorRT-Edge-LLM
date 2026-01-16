import argparse
import os
import onnx
import onnx_graphsurgeon as gs

def get_layer_num(graph, layer_key):
    layer_numbers = set()
    for node in graph.nodes:
        if "Add_1" in node.name and f"{layer_key}." in node.name:
            try:
                layer_str = int(node.name.split(f"{layer_key}.")[1].split("/")[0])
                layer_numbers.add(layer_str)
            except:
                continue
    
    return max(layer_numbers) + 1

def save_model(graph, path, external_data=False):
    new_model = gs.export_onnx(graph)
    if external_data and os.path.exists(os.path.join(os.path.dirname(path), "onnx_model.data")):
        os.remove(path)

    onnx.save_model(new_model,
                    path,
                    save_as_external_data=external_data,
                    location="onnx_model.data",
                    all_tensors_to_one_file=True,
                    convert_attribute=True)
    try:
        onnx.checker.check_model(new_model)
        print("Model verification passed!")
    except onnx.checker.ValidationError as e:
        print(f"Model verification failed: {e}")

def prune_vlm_onnx(input_onnx_path, output_onnx_path, n_layers_to_keep, layer_key, save_external_data):
    def get_keep_nodes(g, layer_key, n_layers_to_keep):
        nodes_to_keep = []
        for node in g.nodes:
            keep = True
            if f"{layer_key}." in node.name:
                try:
                    parts = node.name.split(f"{layer_key}.")
                    layer_id = int(parts[1].split("/")[0])
                    if layer_id >= n_layers_to_keep:
                        keep = False  # 删除这一层
                except:
                    pass
            
            if keep:
                nodes_to_keep.append(node)
        return nodes_to_keep

    def get_keep_outputs(g, n_layers_to_keep, layer_n):
        new_outputs = []
        trailing_names = [f"present_key_values.{i}" for i in range(n_layers_to_keep, layer_n)]
        trailing_names += [f"deepstack_features.{i}" for i in range(3)]
        for output in g.outputs:
            output_name = output.name
            if output_name in trailing_names:
                print(f"移除: {output_name}")
            else:
                new_outputs.append(output)
        return new_outputs
    
    def get_keep_inputs(g, n_layers_to_keep, layer_n):
        new_inputs = []
        trailing_names = [f"past_key_values.{i}" for i in range(n_layers_to_keep, layer_n)]
        trailing_names += [f"deepstack_features.{i}" for i in range(n_layers_to_keep, 3)]
        for inp in g.inputs:
            input_name = inp.name
            if input_name in trailing_names:
                print(f"移除: {input_name}")
            else:
                new_inputs.append(inp)
        return new_inputs

    print(f"正在处理模型: {input_onnx_path}")
    g = gs.import_onnx(onnx.load(input_onnx_path))

    layer_n = get_layer_num(g, layer_key)
    if n_layers_to_keep > layer_n:
        n_layers_to_keep = layer_n

    last_node = None
    keep_node = None
    keep_node_name = f"{layer_key}.{n_layers_to_keep-1}/Add_1"
    if 'layers' in layer_key and n_layers_to_keep < 4:
        keep_node_name = f"/model/Where_{n_layers_to_keep+2}"
    # /model/Where_3, 4, 5
    for node in g.nodes:
        if f"{layer_key}.{layer_n-1}/Add_1" == node.name:
            last_node = node
        if keep_node_name == node.name:
            keep_node = node

    last_output = last_node.outputs[0] if last_node.outputs else None
    keep_output = keep_node.outputs[0] if keep_node.outputs else None
    if not last_output or not keep_output:
        print("错误：节点没有输出")
        return

    for node in g.nodes:
        for i, inp in enumerate(node.inputs):
            if inp is not None and inp.name == last_output.name:
                node.inputs[i] = keep_output

    nodes_to_keep = get_keep_nodes(g, layer_key, n_layers_to_keep)
    new_outputs = get_keep_outputs(g, n_layers_to_keep, layer_n)
    new_inputs = get_keep_inputs(g, n_layers_to_keep, layer_n)

    g.nodes = nodes_to_keep
    g.outputs = new_outputs
    g.inputs = new_inputs

    g.fold_constants().cleanup().toposort()
    save_model(g, output_onnx_path, external_data=save_external_data)
    print(f"完成！保留前{n_layers_to_keep}层，保存到{output_onnx_path}")


def prune_vit_onnx(args):
    layer_key = "/blocks"
    n_layers_to_keep = args.n_layers_to_keep
    input_onnx_path = args.vit_onnx_path
    out_dir = os.path.join(args.output_dir, "vit_pruned")
    os.makedirs(out_dir, exist_ok=True)
    output_onnx_path = os.path.join(out_dir, "model.onnx")
    prune_vlm_onnx(input_onnx_path, output_onnx_path, n_layers_to_keep, layer_key, args.save_external_data)

def prune_llm_onnx(args):
    layer_key = "/model/layers"
    n_layers_to_keep = args.n_layers_to_keep
    input_onnx_path = args.llm_onnx_path
    out_dir = os.path.join(args.output_dir, "llm_pruned")
    os.makedirs(out_dir, exist_ok=True)
    output_onnx_path = os.path.join(out_dir, "model.onnx")
    prune_vlm_onnx(input_onnx_path, output_onnx_path, n_layers_to_keep, layer_key, args.save_external_data)


def inserted_moe_fp16_plugin_cplx(graph: gs.Graph) -> gs.Graph:
    import numpy as np
    from modelopt.onnx.quantization.gs_patching import patch_gs_modules
    patch_gs_modules()
    '''
    plugin感觉可以直接替换layers.?.mlp
    不知道会不会和plugin实现冲突，先设计好
    按照我的思路处理如下：
      1. 遍历整个图，找到/model/layers.?/post_attention_layernorm/Mul_1 作为start_node
      2. 同时找到/model/layers.0/Add_1 作为end_node
      3. 然后将start_node和end_node之间的节点替换为MoEFp16Plugin节点
      4. 需要同时找到
      /model/layers.?/mlp/router/MatMul 的权重数据(第二个输入， 可以用MatMul是否在名字中判断)
      /model/layers.?/mlp/router/Add 的bias数据(第一个输入， 可以用'bias'是否在名字中判断)
      /model/layers.?/mlp/experts.?/up_proj/MatMul 的权重数据
      /model/layers.?/mlp/experts.?/down_proj/MatMul 的权重数据
      /model/layers.?/mlp/experts.?/gate_proj/MatMul 的权重数据
      然后将up_proj的所有专家权重拼接成一个带专家num维度的权重张量experts_up_proj_weight,作为MoEFp16Plugin常量输入
      然后将down_proj的所有专家权重拼接成一个带专家num维度的权重张量experts_down_proj_weight,作为MoEFp16Plugin常量输入
      然后将gate_proj的所有专家权重拼接成一个带专家num维度的权重张量experts_gate_proj_weight,作为MoEFp16Plugin常量输入
      同时需要添加attributes num_expert
      上述步骤可以通过MoeFp16PluginModule来完成
      5. start_node的输出作为MoEFp16Plugin的输入hidden_states
      MoEFp16Plugin的输出连接到end_node的某个输入(可以根据名字中是否有'mlp'来判断)
    '''

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
                print(f"{node.name}, input {input_idx} is {inp}")
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
            print(f"{router_add.name}, input {0} is {inp0}")
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

def surgon_onnx_model(args):
    out_dir = os.path.join(args.output_dir, "llm_pruned_inserted")
    output_onnx_path = os.path.join(out_dir, "model.onnx")
    graph = gs.import_onnx(onnx.load(args.surgon_onnx_path))
    new_graph = inserted_moe_fp16_plugin_cplx(graph)
    os.makedirs(out_dir, exist_ok=True)
    save_model(new_graph, output_onnx_path, external_data=args.save_external_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3VL Inference")
    model_root = "/root/.cache/huggingface/hub/Qwen3-VL-4B-MoE-Init/exported_model/vlm_onnx"
    # model_root = "/home/hy/Downloads/Moe/sd"
    parser.add_argument("--vit_onnx_path", type=str, 
                        default=f"{model_root}/visual_enc_onnx_fp16/model.onnx",
                        # default=f"{model_root}/vit/model.onnx",
                        help="Path to the ONNX model file.")
    parser.add_argument("--llm_onnx_path", type=str, 
                        default=f"{model_root}/llm_onnx_int4_awq/model.onnx",
                        # default=f"{model_root}/llm/model.onnx",
                        help="Path to the ONNX model file.")
    parser.add_argument("--surgon_onnx_path", type=str, 
                        default=f"{model_root}/llm_onnx_fp16/model.onnx",
                        # default=f"{model_root}/llm_pruned/model.onnx",
                        help="Path to the ONNX model file.")
    parser.add_argument("--output_dir", type=str, 
                        default=model_root,
                        help="Path to the output ONNX model file.")
    parser.add_argument("--n_layers_to_keep", type=int, default=1,
                        help="Number of layers to keep.")
    parser.add_argument("--prune_vit", action='store_true', default=False,
                        help="Whether to prune only the ViT model.")
    parser.add_argument("--prune_llm", action='store_true', default=False,
                        help="Whether to prune only the LLM model.")
    parser.add_argument("--surgon_llm", action='store_true', default=False,
                        help="Whether to insert MoE plugin into the LLM model.")
    parser.add_argument("--save_external_data", action='store_true', default=False,
                        help="Whether to save external data.")
    args = parser.parse_args()

    if args.prune_vit:
        prune_vit_onnx(args)
    if args.prune_llm:
        prune_llm_onnx(args)
    if args.surgon_llm:
        surgon_onnx_model(args)
