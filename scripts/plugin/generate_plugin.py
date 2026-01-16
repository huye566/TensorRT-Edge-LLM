#!/usr/bin/env python3

import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def lowercase_first_char(s: str) -> str:
    if not s:
        return s
    return s[0].lower() + s[1:]

class PluginConfig:
    """插件配置类"""
    def __init__(self, config: Dict[str, Any]):
        self.name = config.get("plugin_name", "")
        self.namespace = config.get("namespace", "trt_edgellm::plugins")
        self.kernel_namespace = config.get("kernel_namespace", "trt_edgellm::kernel")
        self.description = config.get("description", "TensorRT Plugin")
        self.version = config.get("version", "1")
        self.supports_dynamic = config.get("supports_dynamic", True)
        self.workspace_needed = config.get("workspace_needed", False)
        
        # 输入输出配置
        self.inputs = config.get("inputs", [])
        self.outputs = config.get("outputs", [])
        
        # 属性配置
        self.attributes = config.get("attributes", [])
        
        # CUDA内核配置
        self.kernels = config.get("kernels", [])
        
        # 序列化成员
        self.serialize_members = config.get("serialize_members", [])
        
        # 继承的基类
        self.base_class = "nvinfer1::IPluginV2DynamicExt" if self.supports_dynamic else "nvinfer1::IPluginV2Ext"
        
        # 验证配置
        self._validate_config()
    
    def _validate_config(self):
        """验证配置的完整性"""
        if not self.name:
            raise ValueError("plugin_name is required")
        
        if not self.inputs:
            raise ValueError("inputs configuration is required")
        
        if not self.outputs:
            raise ValueError("outputs configuration is required")

class CodeGenerator:
    """代码生成器"""
    
    def __init__(self, config: PluginConfig):
        self.config = config
        
    def generate_header(self) -> str:
        """生成头文件内容"""
        # 生成成员变量
        member_vars = []
        for attr in self.config.attributes:
            var_name = f"m{self._camel_case(attr['name'])}"
            cpp_type = self._get_cpp_type(attr['type'])
            member_vars.append(f"    {cpp_type} {var_name}{{}}; //!< {attr.get('description', '')}")
        
        for member in self.config.serialize_members:
            var_name = f"m{self._camel_case(member['name'])}"
            cpp_type = self._get_cpp_type(member['type'])
            member_vars.append(f"    {cpp_type} {var_name}{{}}; //!< {member.get('description', '')}")
        
        member_vars_str = "\n".join(member_vars) if member_vars else "    // 成员变量占位"
        
        # 生成构造函数参数
        ctor_params = []
        ctor_param_decl = []
        for attr in self.config.attributes:
            cpp_type = self._get_cpp_type(attr['type'])
            param_name = attr['name']
            ctor_params.append(f"{cpp_type} {param_name}")
            ctor_param_decl.append(f"{cpp_type} {param_name}")
        
        ctor_param_decl_str = ", " + ", ".join(ctor_param_decl) if ctor_param_decl else ""
        
        # 生成构造函数注释
        ctor_comments = "\n     ".join([f"* @param {attr['name']} {attr.get('description', '')}" 
                                      for attr in self.config.attributes])
        # 处理命名空间
        namespace_begin, namespace_end = self._generate_namespace(self.config.namespace)

        # 构建头文件内容
        header_template = '''#pragma once

// Auto-generated on {generation_date}
// {description}

#include <NvInferRuntime.h>
#include <string>
#include <vector>

{namespace_begin}

/*!
 * @brief {description}
 */
class {plugin_name} : public {base_class}
{{
public:
    /*!
     * @brief Construct {plugin_name}
     {constructor_comments}
     */
    {plugin_name}(std::string const& name{constructor_param_decl});

    /*!
     * @brief Construct from serialized data
     * @param name Layer name
     * @param data Serialized plugin data
     * @param length Size of serialized data
     */
    {plugin_name}(std::string const& name, void const* data, size_t length);

    //! @brief Deleted default constructor
    {plugin_name}() = delete;

    //! @brief Deleted copy constructor
    {plugin_name}({plugin_name} const&) = delete;

    //! @brief Destructor
    ~{plugin_name}() override;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    char const* getPluginType() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    char const* getPluginVersion() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;

protected:
    std::string mLayerName; //!< Layer name
    std::string mNamespace; //!< Plugin namespace

{member_variables}
}};

/*!
 * @brief Factory for creating {plugin_name} instances
 */
class {plugin_name}Creator : public nvinfer1::IPluginCreator
{{
public:
    //! @brief Constructor
    {plugin_name}Creator();

    //! @brief Destructor
    ~{plugin_name}Creator() override = default;

    char const* getPluginName() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    char const* getPluginNamespace() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;     //!< Field collection
    static std::vector<nvinfer1::PluginField> mPluginAttributes; //!< Plugin attributes
    std::string mNamespace;                                      //!< Plugin namespace
}};

{namespace_end}
'''
        
        header_content = header_template.format(
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            description=self.config.description,
            namespace_begin=namespace_begin,
            namespace_end=namespace_end,
            plugin_name=self.config.name,
            base_class=self.config.base_class,
            constructor_comments=ctor_comments,
            constructor_param_decl=ctor_param_decl_str,
            member_variables=member_vars_str,
        )
        
        return header_content
    
    def generate_cpp(self) -> str:
        """生成实现文件内容"""
        plugin_name = self.config.name
        plugin_name_lower = plugin_name.lower()
        plugin_name_upper = plugin_name.upper()
        
        # 生成属性字段
        attribute_fields = []
        for attr in self.config.attributes:
            field_type = self._get_plugin_field_type(attr['type'])
            attribute_fields.append(
                f'    mPluginAttributes.emplace_back(PluginField("{attr["name"]}", '
                f'nullptr, PluginFieldType::{field_type}, 1));'
            )
        
        attribute_fields_str = "\n".join(attribute_fields)
        
        # 生成构造函数初始化列表
        init_list = []
        for attr in self.config.attributes:
            var_name = f"m{self._camel_case(attr['name'])}"
            init_list.append(f"    , {var_name}({attr['name']})")
        
        init_list_str = "\n".join(init_list) if init_list else ""
        
        # 生成反序列化代码
        deserialize_code = []
        for member in self.config.serialize_members:
            var_name = f"m{self._camel_case(member['name'])}"
            deserialize_code.append(f"    deserializeValue(&data, &length, &{var_name});")
        
        for attr in self.config.attributes:
            var_name = f"m{self._camel_case(attr['name'])}"
            deserialize_code.append(f"    deserializeValue(&data, &length, &{var_name});")
        
        deserialize_code_str = "\n".join(deserialize_code)
        
        # 生成序列化大小和代码
        serialize_size = len(self.config.serialize_members) + len(self.config.attributes)
        serialize_code = []
        
        for member in self.config.serialize_members:
            var_name = f"m{self._camel_case(member['name'])}"
            serialize_code.append(f"    serializeValue(&buffer, {var_name});")
        
        for attr in self.config.attributes:
            var_name = f"m{self._camel_case(attr['name'])}"
            serialize_code.append(f"    serializeValue(&buffer, {var_name});")
        
        serialize_code_str = "\n".join(serialize_code)
        
        # 生成enqueue函数中的输入提取
        input_extraction = []
        for i, inp in enumerate(self.config.inputs):
            dtype = self._get_cuda_type(inp.get('dtype', 'half'))
            input_name = inp.get('name', f'input{i}')
            input_extraction.append(
                f"        {dtype}* {input_name} = "
                f"reinterpret_cast<{dtype}*>(const_cast<void*>(inputs[{i}]));"
            )
        
        input_extraction_str = "\n".join(input_extraction)
        
        # 生成enqueue函数中的输出提取
        output_extraction = []
        for i, out in enumerate(self.config.outputs):
            dtype = self._get_cuda_type(out.get('dtype', 'half'))
            output_name = out.get('name', f'output{i}')
            output_extraction.append(
                f"        {dtype}* {output_name} = "
                f"reinterpret_cast<{dtype}*>(outputs[{i}]);"
            )
        
        output_extraction_str = "\n".join(output_extraction)
        
        # 生成属性解析代码
        attribute_parsing = []
        for attr in self.config.attributes:
            cpp_type = self._get_cpp_type(attr['type'])
            attr_name = attr['name']
            attribute_parsing.append(
                f"        std::optional<{cpp_type}> {attr_name} = "
                f'parsePluginScalarField<{cpp_type}>("{attr_name}", fc);'
            )
        
        attribute_parsing_str = "\n".join(attribute_parsing)
        
        # 生成属性检查
        attribute_check = " && ".join([f"{attr['name']}.has_value()" for attr in self.config.attributes])
        
        # 生成插件创建参数
        create_params = ", ".join([f"{attr['name']}.value()" for attr in self.config.attributes])
        
        # 生成克隆参数
        clone_member_params = ", " + ", ".join([f"m{self._camel_case(attr['name'])}" 
                                               for attr in self.config.attributes]) if self.config.attributes else ""
        
        # 处理命名空间
        namespace_begin, namespace_end = self._generate_namespace(self.config.namespace)
        
        # 生成构造函数参数声明
        constructor_param_decl = ""
        if self.config.attributes:
            params = []
            for attr in self.config.attributes:
                cpp_type = self._get_cpp_type(attr['type'])
                params.append(f"{cpp_type} {attr['name']}")
            constructor_param_decl = ", " + ", ".join(params)
        
        # 生成格式检查case语句
        format_cases = []
        for i, inp in enumerate(self.config.inputs):
            format_cases.append(self._generate_format_case(i, inp, "input"))
        
        for i, out in enumerate(self.config.outputs):
            idx = len(self.config.inputs) + i
            format_cases.append(self._generate_format_case(idx, out, "output"))
        
        format_cases_str = "\n    ".join(format_cases)
        format_cases_str = "    " + format_cases_str
        
        # 构建实现文件内容
        cpp_template = f'''#include "{plugin_name_lower}.h"
#include "kernels/{plugin_name_lower}Kernels/{plugin_name_lower}Kernels.h"
#include "plugins/utils/pluginUtils.h"

#include <cassert>
#include <cuda_fp16.h>
#include <mutex>
#include <optional>

using namespace nvinfer1;

{namespace_begin}

namespace
{{
constexpr char const* k{plugin_name_upper}_PLUGIN_VERSION{{"{self.config.version}"}};
constexpr char const* k{plugin_name_upper}_PLUGIN_NAME{{"{plugin_name}"}};
}} // namespace

// Static class fields initialization
PluginFieldCollection {plugin_name}Creator::mFieldCollection{{{{}}}};
std::vector<PluginField> {plugin_name}Creator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN({plugin_name}Creator);

{plugin_name}::{plugin_name}(std::string const& name{constructor_param_decl})
    : mLayerName(name){init_list_str}
{{
}}

{plugin_name}::{plugin_name}(std::string const& name, void const* data, size_t length)
    : mLayerName(name)
{{
{deserialize_code_str}
}}

{plugin_name}::~{plugin_name}() {{}}

IPluginV2DynamicExt* {plugin_name}::clone() const noexcept
{{
    {plugin_name}* plugin = new {plugin_name}(mLayerName{clone_member_params});
    return plugin;
}}

char const* {plugin_name}::getPluginType() const noexcept
{{
    return k{plugin_name_upper}_PLUGIN_NAME;
}}

char const* {plugin_name}::getPluginNamespace() const noexcept
{{
    return mNamespace.c_str();
}}

void {plugin_name}::setPluginNamespace(char const* pluginNamespace) noexcept
{{
    mNamespace = std::string(pluginNamespace);
}}

char const* {plugin_name}::getPluginVersion() const noexcept
{{
    return k{plugin_name_upper}_PLUGIN_VERSION;
}}

int32_t {plugin_name}::getNbOutputs() const noexcept
{{
    return {len(self.config.outputs)};
}}

bool {plugin_name}::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{{
    try
    {{
        assert(nbInputs == {len(self.config.inputs)} && nbOutputs == {len(self.config.outputs)});
        assert(pos < (nbInputs + nbOutputs));
        auto const& tensorDesc = inOut[pos];
        bool status = true;

        switch (pos)
        {{
{format_cases_str}
            default: break;
        }}
        return status;
    }}
    catch (std::exception const& e)
    {{
        // Log error if needed
    }}
    return false;
}}

DataType {plugin_name}::getOutputDataType([[maybe_unused]] int32_t index,
    [[maybe_unused]] nvinfer1::DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept
{{
    // TODO: Update based on your plugin's output data type
    return DataType::kHALF;
}}

DimsExprs {plugin_name}::getOutputDimensions([[maybe_unused]] int32_t outputIndex,
    nvinfer1::DimsExprs const* inputs, [[maybe_unused]] int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{{
    // TODO: Implement output dimensions calculation
    DimsExprs output;
    output.nbDims = 3;
    output.d[0] = inputs[0].d[0];
    output.d[1] = inputs[0].d[1];
    output.d[2] = inputs[0].d[2];
    return output;
}}

void {plugin_name}::configurePlugin([[maybe_unused]] nvinfer1::DynamicPluginTensorDesc const* in,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] nvinfer1::DynamicPluginTensorDesc const* out,
    [[maybe_unused]] int32_t nbOutputs) noexcept
{{
    // TODO: Configure plugin based on input descriptors
}}

size_t {plugin_name}::getWorkspaceSize([[maybe_unused]] nvinfer1::PluginTensorDesc const* inputs,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] nvinfer1::PluginTensorDesc const* outputs,
    [[maybe_unused]] int32_t nbOutputs) const noexcept
{{
    // TODO: Calculate workspace size if needed
    return 0;
}}

int32_t {plugin_name}::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    [[maybe_unused]] nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs,
    [[maybe_unused]] void* workspace, cudaStream_t stream) noexcept
{{
    try
    {{
        // Extract inputs
{input_extraction_str}

        // Extract outputs
{output_extraction_str}

        // TODO: Implement plugin logic here
        // Call CUDA kernels if needed
        // {self.config.kernel_namespace}::kernel_cuda(...);

        return 0;
    }}
    catch (std::exception const& e)
    {{
        // Log error
        return -1;
    }}
}}

size_t {plugin_name}::getSerializationSize() const noexcept
{{
    return {serialize_size} * sizeof(int32_t); // TODO: Update based on actual data types
}}

void {plugin_name}::serialize(void* buffer) const noexcept
{{
{serialize_code_str}
}}

int32_t {plugin_name}::initialize() noexcept
{{
    return 0;
}}

void {plugin_name}::terminate() noexcept {{}}

void {plugin_name}::destroy() noexcept
{{
    delete this;
}}

{plugin_name}Creator::{plugin_name}Creator()
{{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
{attribute_fields_str}

    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}}

char const* {plugin_name}Creator::getPluginName() const noexcept
{{
    return k{plugin_name_upper}_PLUGIN_NAME;
}}

nvinfer1::PluginFieldCollection const* {plugin_name}Creator::getFieldNames() noexcept
{{
    return &mFieldCollection;
}}

void {plugin_name}Creator::setPluginNamespace(char const* libNamespace) noexcept
{{
    mNamespace = libNamespace;
}}

char const* {plugin_name}Creator::getPluginNamespace() const noexcept
{{
    return mNamespace.c_str();
}}

char const* {plugin_name}Creator::getPluginVersion() const noexcept
{{
    return k{plugin_name_upper}_PLUGIN_VERSION;
}}

nvinfer1::IPluginV2* {plugin_name}Creator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept
{{
    try
    {{
{attribute_parsing_str}

        bool checkRequiredFields = {attribute_check};
        if (!checkRequiredFields)
        {{
            return nullptr;
        }}

        {plugin_name}* plugin = new {plugin_name}(std::string(name), {create_params});
        return plugin;
    }}
    catch (std::exception const& e)
    {{
        // Log error
    }}
    return nullptr;
}}

nvinfer1::IPluginV2* {plugin_name}Creator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{{
    try
    {{
        return new {plugin_name}(name, serialData, serialLength);
    }}
    catch (std::exception const& e)
    {{
        // Log error
    }}
    return nullptr;
}}

{namespace_end}'''
        
        return cpp_template
    
    def generate_kernel_header(self) -> Optional[str]:
        """生成CUDA内核头文件"""
        if not self.config.kernels:
            return None
        
        # 生成内核函数声明
        kernel_decls = []
        for kernel in self.config.kernels:
            params = []
            for param in kernel.get("parameters", []):
                param_type = self._get_cuda_type(param.get("type", "half"))
                if param.get("is_pointer", False):
                    param_type += "*"
                if param.get("is_const", False):
                    param_type = f"const {param_type}"
                params.append(f"{param_type} {param['name']}")
            
            params_str = ", ".join(params)
            kernel_decls.append(f"void {kernel['name']}_cuda({params_str}, cudaStream_t stream);")
        
        kernel_decls_str = "\n".join(kernel_decls)
        
        # 处理命名空间
        kernel_namespace_begin, kernel_namespace_end = self._generate_namespace(self.config.kernel_namespace)
        
        # 构建内核头文件内容
        kernel_content = f'''#pragma once

// Auto-generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
// CUDA kernels for {self.config.name}

#include <cuda_fp16.h>
#include <stdint.h>

{kernel_namespace_begin}

/*!
 * @brief Main CUDA kernel for {self.config.name}
 */
{kernel_decls_str}

{kernel_namespace_end}
'''
        
        return kernel_content
    
    def _generate_namespace(self, namespace: str) -> tuple:
        """生成命名空间开始和结束部分"""
        parts = namespace.split("::")
        if len(parts) == 1:
            begin = f"namespace {parts[0]} {{\n"
            end = f"}} // namespace {parts[0]}"
        else:
            begin = "\n".join([f"namespace {part} {{" for part in parts]) + "\n"
            end = "\n".join([f"}} // namespace {part}" for part in reversed(parts)])
        return begin, end
    
    def _camel_case(self, name: str) -> str:
        """将snake_case转换为CamelCase"""
        parts = name.split('_')
        return ''.join(part.capitalize() for part in parts)
    
    def _generate_format_case(self, index: int, io_config: Dict, io_type: str) -> str:
        """生成格式检查的case语句"""
        indent_first = "        "
        indent = indent_first + "    "
        name = io_config.get('name', f'{io_type}_{index}')
        
        lines = []
        lines.append(f'{indent_first}case {index}: // {name}')
        lines.append(f'{indent}{{')
        
        # 数据类型检查
        dtype = io_config.get('dtype', 'half')
        trt_dtype = self._get_trt_data_type(dtype)
        lines.append(f'{indent}    status &= tensorDesc.type == DataType::{trt_dtype};')
        
        # 格式检查
        lines.append(f'{indent}    status &= tensorDesc.format == TensorFormat::kLINEAR;')
        
        # 维度检查
        dims = io_config.get('dims', [])
        if dims:
            lines.append(f'{indent}    status &= tensorDesc.dims.nbDims == {len(dims)};')
            for i, dim in enumerate(dims):
                if dim != -1:  # -1表示动态维度
                    lines.append(f'{indent}    status &= tensorDesc.dims.d[{i}] == {dim};')
        
        lines.append(f'{indent}    break;')
        lines.append(f'{indent}}}')
        
        return '\n'.join(lines)
    
    def _get_cpp_type(self, type_str: str) -> str:
        type_map = {
            'int32': 'int32_t',
            'int64': 'int64_t', 
            'float32': 'float',
            'float64': 'double',
            'bool': 'bool',
            'string': 'std::string',
        }
        return type_map.get(type_str, type_str)
    
    def _get_cuda_type(self, type_str: str) -> str:
        type_map = {
            'half': 'half',
            'float': 'float',
            'float16': 'half',
            'float32': 'float',
            'int8': 'int8_t',
            'int16': 'int16_t',
            'int32': 'int32_t',
            'int64': 'int64_t',
        }
        return type_map.get(type_str, type_str)
    
    def _get_trt_data_type(self, type_str: str) -> str:
        type_map = {
            'half': 'kHALF',
            'float16': 'kHALF',
            'float': 'kFLOAT',
            'float32': 'kFLOAT',
            'int8': 'kINT8',
            'int32': 'kINT32',
            'int64': 'kINT64',
            'bool': 'kBOOL',
        }
        return type_map.get(type_str, 'kHALF')
    
    def _get_plugin_field_type(self, type_str: str) -> str:
        type_map = {
            'int32': 'kINT32',
            'int64': 'kINT64',
            'float32': 'kFLOAT32',
            'float64': 'kFLOAT64',
            'string': 'kCHAR',
            'bool': 'kBOOL',
        }
        return type_map.get(type_str, 'kINT32')

def main(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    plugin_config = PluginConfig(config_data)
    generator = CodeGenerator(plugin_config)
    os.makedirs(args.output_dir, exist_ok=True)
    
    header_content = generator.generate_header()
    header_file = os.path.join(args.output_dir, f"{lowercase_first_char(plugin_config.name)}.h")
    with open(header_file, 'w', encoding='utf-8') as f:
        f.write(header_content)
    print(f"Generated header file: {header_file}")
    
    cpp_content = generator.generate_cpp()
    cpp_file = os.path.join(args.output_dir, f"{lowercase_first_char(plugin_config.name)}.cpp")
    with open(cpp_file, 'w', encoding='utf-8') as f:
        f.write(cpp_content)
    print(f"Generated implementation file: {cpp_file}")
    
    if args.generate_kernels and plugin_config.kernels:
        kernel_content = generator.generate_kernel_header()
        if kernel_content:
            kernel_dir = os.path.join(args.output_dir, "kernels", 
                                     f"{lowercase_first_char(plugin_config.name)}Kernels")
            os.makedirs(kernel_dir, exist_ok=True)
            
            kernel_file = os.path.join(kernel_dir, 
                                      f"{lowercase_first_char(plugin_config.name)}Kernels.h")
            with open(kernel_file, 'w', encoding='utf-8') as f:
                f.write(kernel_content)
            print(f"Generated kernel header file: {kernel_file}")
    
    print(f"\nPlugin '{plugin_config.name}' generated successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TensorRT plugin template")
    parser.add_argument("--config", 
                        default=os.path.join(CURRENT_DIR, "moe_fp16_config.json"),
                        help="JSON configuration file")
    parser.add_argument("--output_dir", "-o", 
                        default=os.path.join(CURRENT_DIR, "generated"),
                        help="Output directory")
    parser.add_argument("--generate_kernels", "-k", action="store_true", 
                       help="Generate kernel header files")
    
    args = parser.parse_args()
    main(args)