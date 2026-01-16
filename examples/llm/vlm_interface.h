/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef VLM_INTERFACE_H
#define VLM_INTERFACE_H

#include <stddef.h>
#include <stdint.h>

struct VlmExtraConfig
{
    VlmExtraConfig() = default;
    VlmExtraConfig(bool staticImageSize, bool staticPrompt, int visCompressMode, float keepRate)
        : staticImageSize(staticImageSize)
        , staticPrompt(staticPrompt)
        , visCompressMode(visCompressMode)
        , keepRate(keepRate)
    {
    }
    bool staticImageSize = false; // 固定图像尺寸优化
    bool staticPrompt = false;    // 固定提示词优化
    int visCompressMode = 0;      // 0=不压缩, 1=vis pruner, 2=ali compress method
    float keepRate = 1.0f;
};

#ifdef __cplusplus
extern "C"
{
#endif

// 符号可见性定义
#if defined(_WIN32) || defined(__CYGWIN__)
#define API_EXPORT __declspec(dllexport)
#define API_IMPORT __declspec(dllimport)
#else
#define API_EXPORT __attribute__((visibility("default")))
#define API_IMPORT
#endif

#ifdef BUILDING_SHARED_LIB
#define API API_EXPORT
#else
#define API API_IMPORT
#endif

    // 模型类型枚举
    typedef enum
    {
        MODEL_TYPE_QWEN_2_VL_2B = 0,
        MODEL_TYPE_QWEN_2_5_VL_3B = 1,
        MODEL_TYPE_INTERNVL3_1B = 2,
        MODEL_TYPE_QWEN_3_VL_2B = 3,
    } ModelType;

    typedef struct
    {
        char* first;
        char* second;
    } LoraWeights;

    // GPU内存句柄结构体
    typedef struct
    {
        unsigned char* ptr; // 图像内存指针
        size_t size;        // 内存大小
        int64_t width;      // 图像宽度
        int64_t height;     // 图像高度
        int64_t channels;   // 图像通道数
    } ImageBufferHandle;

    // 推理结果结构体
    typedef struct
    {
        char* text;          // 生成的文本
        float confidence;    // 置信度分数
        int64_t token_count; // 文本长度
    } InferenceResult;

    /**
     * 一次可以推理多个batch
     * 一个batch包含任意个图片、一个提示词和一个输出结果
     * 所有内存均有外部调用者自行申请、释放
     */
    typedef struct
    {
        ImageBufferHandle* images;
        int64_t image_size;
        char* inputString;
        InferenceResult* result;
    } InferenceBatchHandle;

    // 模型类型枚举
    typedef enum
    {
        TASK_STATUS_WAITING = 0,
        TASK_STATUS_PROCESSING = 1,
        TASK_STATUS_DONE = 2,
        TASK_STATUS_ERROR = -1
    } TaskStatus;

    // 推理任务句柄
    typedef struct
    {
        int64_t task_id; // 任务ID
        int64_t status;  // 状态码: 0=等待, 1=处理中, 2=完成, -1=错误
    } InferenceTaskHandle;

    // 日志等级枚举
    typedef enum
    {
        VLM_LOG_LEVEL_NONE = 0,
        VLM_LOG_LEVEL_DEBUG = 1,
        VLM_LOG_LEVEL_INFO = 2,
        VLM_LOG_LEVEL_WARN = 3,
        VLM_LOG_LEVEL_ERROR = 4
    } VLMLogLevel;

    typedef void (*VLMLogCallback)(int level, char const* text);

    // 回调函数类型定义
    // 有两种方法异步等待推理结果，这是第一种，回调函数。
    typedef void (*InferenceCompletionCallback)(InferenceBatchHandle* handle, int64_t batch_size);

    typedef struct
    {
        bool is_eagle3;                       // 是否为投机采样模式，如果是，默认为eagle3
        char const* eagle3_draft_engine_path; // 草稿模型路径
        int32_t max_path_len;                 // eagle3 层数
        int32_t topk;                         // eagle3 topk
        int32_t max_decoding_tokens;          // = max_path_len * topk
    } Eagle3Info;

    typedef struct
    {
        int64_t max_length;             // 最大上下文长度
        int64_t batch_size;             // batch size
        ModelType model_type;           // 模型类型
        char const* tokenizer_path;     // 模型 tokenizer 路径
        char const* visual_engine_path; // vit 路径
        char const* llm_engine_path;    // llm 路径
        Eagle3Info eagle3_info;         // ealge3 信息
        LoraWeights loraWeights;        // lora 权重
        void* cudaStream;               // 外部传入的 CudaStream
        VlmExtraConfig extra_config;    // 额外配置项
        bool dump_profile;              // 是否统计性能信息
        bool close_cuda_graph;          // 是否关闭cuda graph
    } VLMModelInfo;

    /**
     * 初始化和清理
     *
     * 执行 vlm_initialize 等操作前，首先调用 vlm_get_version 确认版本信息
     */
    API int vlm_initialize(VLMModelInfo* model_info);
    API void vlm_cleanup();

    // 推理API
    API InferenceTaskHandle* vlm_submit_query(
        InferenceBatchHandle* batch, int64_t batch_size, InferenceCompletionCallback callback = nullptr);

    // 任务管理
    API int vlm_check_task_status(InferenceTaskHandle* handle);
    // 有两种方法异步等待推理结果，这是第二种，异步等待。
    API bool vlm_wait_for_result(InferenceTaskHandle* handle);
    API void vlm_release_task(InferenceTaskHandle* handle);

/**
 * 该版本号用于对齐 workflow 和 sdk 的接口信息
 *
 * 版本号说明：
 * VLMInfer         确定这是libvlm_infer.so
 * v0.1.0.2         sdk 的主要版本
 * 250818           接口更新日期
 * AdaptToEagle3    接口更新原因
 *
 * 工作模式：
 * 头文件一式两份，分别位于workflow和sdk中，
 * workflow调用libvlm_infer.so后，首先通过vlm_get_version函数获取libvlm_infer.so的版本号
 * 然后与自己头文件中的版本号进行字符串对比，不同则表示接口不能对齐，拒绝调用推理
 *
 */
#define VLM_INFER_VERSION_INFO "VLMInfer_v1.01_251222_SoWrapper"

    // 实用函数
    API const char* vlm_get_error_message(); // deprecated
    API void vlm_set_log_callback(VLMLogCallback);
    API const char* vlm_get_version();

#ifdef __cplusplus
}
#endif

#endif // VLM_INTERFACE_H