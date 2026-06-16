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

#define BUILDING_SHARED_LIB
#include "vlm_interface.h"

#include "common/trtUtils.h"
#include "memoryMonitor.h"
#include "profileFormatter.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmInferenceSpecDecodeRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include "profiler/cupti_profiler.hpp"
#include "logger.hpp"

using namespace trt_edgellm;
using Json = nlohmann::json;

// 模型封装类
class VLModel
{
public:
    VLMModelInfo model_info;

private:
    std::unique_ptr<void, DlDeleter> pluginHandles;
    cudaStream_t stream;
    std::unique_ptr<rt::LLMInferenceRuntime> llmInferenceRuntime{nullptr};
    std::unique_ptr<rt::LLMInferenceSpecDecodeRuntime> eagleInferenceRuntime{nullptr};

    MemoryMonitor memoryMonitor;

public:
    VLModel(VLMModelInfo* info)
        : model_info(*info)
    {
    }

    ~VLModel()
    {
        memoryMonitor.stop();
    }

    bool init()
    {
        void* cudaStream = model_info.cudaStream;
        if (!cudaStream)
        {
            LOG_ERROR("cuda stream is nullptr!");
            return false;
        }
        // cuda stream 是一个句柄指针，可以拷贝，指向同一个流
        stream = *(static_cast<cudaStream_t*>(cudaStream));

        pluginHandles = loadEdgellmPluginLib();
        std::unordered_map<std::string, std::string> loraWeightsMap;

        bool enable_eagle = model_info.eagle3_info.is_eagle3;
        if (enable_eagle)
        {
            rt::EagleDraftingConfig draftingConfig{model_info.eagle3_info.topk, model_info.eagle3_info.max_path_len,
                model_info.eagle3_info.max_decoding_tokens};
            try
            {
                eagleInferenceRuntime = std::make_unique<rt::LLMInferenceSpecDecodeRuntime>(
                    model_info.llm_engine_path, model_info.visual_engine_path, draftingConfig, stream);
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Failed to initialize LLMInferenceSpecDecodeRuntime: %s", e.what());
                return false;
            }

            bool const draftProposalCaptureStatus = eagleInferenceRuntime->captureDraftProposalCudaGraph(stream);
            if (!draftProposalCaptureStatus)
            {
                LOG_WARNING(
                    "Failed to capture CUDA graph for draft proposal usage, proceeding with normal engine execution.");
            }

            bool const draftAcceptCaptureStatus = eagleInferenceRuntime->captureDraftAcceptDecodeTokenCudaGraph(stream);
            if (!draftAcceptCaptureStatus)
            {
                LOG_WARNING(
                    "Failed to capture CUDA graph for draft accept decode token usage, proceeding with normal engine "
                    "execution.");
            }

            bool const baseCaptureStatus = eagleInferenceRuntime->captureBaseVerificationCudaGraph(stream);
            if (!baseCaptureStatus)
            {
                LOG_WARNING(
                    "Failed to capture CUDA graph for base model verification usage, proceeding with normal engine "
                    "execution.");
            }
        }
        else
        {
            // Standard mode
            try
            {
                llmInferenceRuntime = std::make_unique<rt::LLMInferenceRuntime>(
                    model_info.llm_engine_path, model_info.visual_engine_path, loraWeightsMap, stream);
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Failed to initialize LLMInferenceRuntime: %s", e.what());
                return false;
            }
            if (!model_info.close_cuda_graph && !llmInferenceRuntime->captureDecodingCUDAGraph(stream))
            {
                LOG_WARNING(
                    "Failed to capture CUDA graph for decoding usage, proceeding with normal engine execution.");
            }
        }

        if (model_info.dump_profile)
        {
            setProfilingEnabled(true);
            memoryMonitor.start();
        }

        return true;
    }

    bool run_inference(rt::LLMGenerationRequest const& request, rt::LLMGenerationResponse& response)
    {
        bool enable_eagle = model_info.eagle3_info.is_eagle3;
        bool requestStatus = false;
        if (enable_eagle)
        {
            requestStatus = eagleInferenceRuntime->handleRequest(request, response, stream);
        }
        else
        {
            requestStatus = llmInferenceRuntime->handleRequest(request, response, stream);
        }
        if (!requestStatus)
        {
            LOG_ERROR("Failed to inference!");
        }
        if (model_info.dump_profile)
        {
            std::ostringstream profileOutput;
            profileOutput << std::endl;
            profileOutput << "=== Performance Summary ===" << std::endl;
            if (model_info.eagle3_info.is_eagle3)
            {
                // Eagle runtime with detailed metrics
                auto prefillMetrics = eagleInferenceRuntime->getPrefillMetrics();
                auto eagleGenerationMetrics = eagleInferenceRuntime->getEagleGenerationMetrics();
                auto multimodalMetrics = eagleInferenceRuntime->getMultimodalMetrics();
                outputPrefillProfile(profileOutput, prefillMetrics);
                outputEagleGenerationProfile(profileOutput, eagleGenerationMetrics);
                outputMultimodalProfile(profileOutput, multimodalMetrics);
                outputMemoryProfile(profileOutput, memoryMonitor);
            }
            else
            {
                auto multimodalMetrics = llmInferenceRuntime->getMultimodalMetrics();
                outputPrefillProfile(profileOutput, llmInferenceRuntime->getPrefillMetrics());
                outputGenerationProfile(profileOutput, llmInferenceRuntime->getGenerationMetrics());
                outputMultimodalProfile(profileOutput, multimodalMetrics);
                outputMemoryProfile(profileOutput, memoryMonitor);
            }
            profileOutput << "=====================================" << std::endl;
            LOG_INFO("%s", profileOutput.str().c_str());
        }
        return requestStatus;
    }
};

// 全局变量
static std::unique_ptr<VLModel> g_model = nullptr;
static std::mutex g_model_mutex;
static bool g_initialized = false;

// 任务管理
struct InferenceTask
{
    InferenceTaskHandle handle;
    InferenceBatchHandle* batch;
    int64_t batch_size;
    InferenceCompletionCallback callback;
    std::thread worker;
    bool completed;
    std::condition_variable cv;
    std::mutex mutex;
};

static std::unordered_map<int, std::unique_ptr<InferenceTask>> g_tasks;
static std::mutex g_tasks_mutex;
static int g_next_task_id = 1;

// API实现
extern "C" API int vlm_initialize(VLMModelInfo* model_info)
{
    std::lock_guard<std::mutex> lock(g_model_mutex);
    gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);

    if (g_initialized)
    {
        LOG_ERROR("Model has been initialized.");
        return 0;
    }

    if (!model_info)
    {
        LOG_ERROR("Model info is nullptr!");
        return 0;
    }

    // 创建并初始化模型
    try
    {
        g_model = std::make_unique<VLModel>(model_info);
        if (!g_model->init())
        {
            LOG_ERROR("Model initialization failed.");
            g_model.reset();
            return 0;
        }
    }
    catch (std::exception const& e)
    {
        std::string msg = std::string("Model initialization exception: ") + std::string(e.what());
        LOG_ERROR(msg.c_str());
        g_model.reset();
        return 0;
    }

    g_initialized = true;
    return 1;
}

extern "C" API void vlm_cleanup()
{
    std::lock_guard<std::mutex> model_lock(g_model_mutex);

    if (!g_initialized)
        return;

    // 等待所有任务完成
    {
        std::lock_guard<std::mutex> tasks_lock(g_tasks_mutex);

        for (auto& pair : g_tasks)
        {
            auto& task = pair.second;
            if (task->worker.joinable())
            {
                task->worker.join();
            }
        }
        g_tasks.clear();
    }

    // 清理模型
    g_model.reset();
    g_initialized = false;
}

// 工作线程函数
void inference_thread_func(InferenceTask* task)
{
    {
        std::lock_guard<std::mutex> lock(task->mutex);
        task->handle.status = TASK_STATUS_PROCESSING; // 处理中
    }

    rt::LLMGenerationRequest batchedRequest;
    rt::LLMGenerationResponse response;

    batchedRequest.temperature = 0.0;
    batchedRequest.topP = 1.0;
    batchedRequest.topK = 1;
    batchedRequest.saveSystemPromptKVCache = false;
    batchedRequest.enableThinking = false;
    batchedRequest.maxGenerateLength = 200;
    batchedRequest.staticPrompt = g_model->model_info.extra_config.staticPrompt;

    std::vector<void*> temp_device_image;
    for (int64_t batch_idx = 0; batch_idx < task->batch_size; ++batch_idx)
    {
        InferenceBatchHandle* batch = task->batch + batch_idx;
        rt::LLMGenerationRequest::Request request;
        std::vector<rt::Message> chatMessage;
        std::vector<rt::imageUtils::ImageData> imageBuffers;

        {
            rt::Message sysMsg;
            sysMsg.role = "system";
            rt::Message::MessageContent msgContent;
            msgContent.type = "text";
            msgContent.content = "You are a helpful assistant.";
            sysMsg.contents.push_back(std::move(msgContent));
            chatMessage.emplace_back(sysMsg);
        }

        rt::Message userMsg;
        userMsg.role = "user";
        rt::Message::MessageContent textContent;
        textContent.type = "text";
        textContent.content = batch->inputString;
        userMsg.contents.push_back(std::move(textContent));
        for (int64_t image_idx = 0; image_idx < batch->image_size; ++image_idx)
        {
            auto* image = batch->images + image_idx;

            void* d_image = nullptr;
            cudaPointerAttributes attributes;
            cudaPointerGetAttributes(&attributes, image->ptr);
            if (attributes.type == cudaMemoryTypeHost)
            {
                CUDA_CHECK(cudaHostGetDevicePointer((void**) &d_image, image->ptr, 0));
                LOG_DEBUG("Image memory device type is cudaMemoryTypeHost");
            }
            else if (attributes.type == cudaMemoryTypeDevice)
            {
                d_image = image->ptr;
                LOG_DEBUG("Image memory device type is cudaMemoryTypeDevice");
            }
            else
            {
                size_t image_size = image->height * image->width * image->channels * sizeof(unsigned char);
                CUDA_CHECK(cudaMalloc((void**) &d_image, image_size));
                CUDA_CHECK(cudaMemcpy(d_image, image->ptr, image_size, cudaMemcpyHostToDevice));
                temp_device_image.emplace_back(d_image);
                LOG_WARNING("Image memory device type is CPU, copy to GPU!");
            }

            rt::Coords shape{image->height, image->width, image->channels};
            // ONLY support GPU addr.
            auto imgTensor
                = rt::Tensor(static_cast<void*>(d_image), shape, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8);
            auto imageData = rt::imageUtils::ImageData(std::move(imgTensor));
            imageBuffers.push_back(imageData);

            rt::Message::MessageContent imageContent;
            imageContent.type = "image";
            imageContent.content = "GPU data"; // 此处content不重要，可以填图像文件路径、图像类型等
            userMsg.contents.push_back(std::move(imageContent));
        }
        chatMessage.emplace_back(userMsg);
        // 可以继续向 chatMessage 中添加消息，这样就是连续对话了。

        request.messages = std::move(chatMessage);
        request.imageBuffers = std::move(imageBuffers);
        batchedRequest.requests.push_back(std::move(request));
    }

    // 运行推理
    auto infer_start = std::chrono::high_resolution_clock::now();

    bool run_res = false;
    if (g_model->model_info.extra_config.cuptiProfileLevel > 0) {
        std::vector<cpp_base_suite::profiler::MetricId> metrics;
        int const level = g_model->model_info.extra_config.cuptiProfileLevel;
        if (level == 1) {
            metrics = cpp_base_suite::profiler::DefaultMetrics();
        } else if (level == 2) {
            metrics = cpp_base_suite::profiler::PerformanceProfileMetrics();
        } else {
            metrics = cpp_base_suite::profiler::MainProfileMetrics();
        }

        cpp_base_suite::profiler::CuptiProfiler profiler(metrics);
        if (!profiler.Initialize())
        {
            LOG_WARNING("inference_thread_func(): Failed to initialize CUPTI profiler — running unprofiled.");
            run_res = g_model->run_inference(batchedRequest, response);
        } else {
            run_res = profiler.ProfileMultiPass("qwen infer",
                [&batchedRequest, &response](int /*pass*/) {
                if (!g_model->run_inference(batchedRequest, response))
                {
                    LOG_ERROR("inference_thread_func: run_inference failed during profiling.");
                }
            });

            LOG_INFO(profiler.FormatResults().c_str());
            profiler.SaveToJson("vlm_profile.njson", cpp_base_suite::profiler::WriteMode::Append);
        }
    } else {
        run_res = g_model->run_inference(batchedRequest, response);
    }

    auto infer_end = std::chrono::high_resolution_clock::now();
    auto infer_dur = std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_start).count();
    LOG_INFO("g_model->run_inference cost %d ms", infer_dur);

    for (auto* d_image : temp_device_image)
    {
        CUDA_CHECK(cudaFree(d_image));
    }
    temp_device_image.clear();

    for (size_t batch_idx = 0; batch_idx < response.outputTexts.size(); ++batch_idx)
    {
        auto& resultString = response.outputTexts[batch_idx];
        InferenceBatchHandle* batch = task->batch + batch_idx;
        InferenceResult* result = batch->result;
        resultString.copy(result->text, resultString.size());
        result->text[resultString.size()] = '\0';
        result->confidence = 1.0f;
        result->token_count = response.outputIds[batch_idx].size();
    }

    // 标记完成并通知
    {
        std::lock_guard<std::mutex> lock(task->mutex);
        task->completed = true;
        task->handle.status = run_res ? TASK_STATUS_DONE : TASK_STATUS_ERROR; // 完成
    }
    task->cv.notify_all();

    // 调用回调函数
    if (task->callback)
    {
        task->callback(task->batch, task->batch_size);
    }
}

extern "C" API InferenceTaskHandle* vlm_submit_query(
    InferenceBatchHandle* batch, int64_t batch_size, InferenceCompletionCallback callback)
{
    if (!g_initialized)
    {
        LOG_ERROR("Library is not initialized, please call the vlm_initialize() first.");
        return nullptr;
    }

    if (!batch || batch_size <= 0)
    {
        LOG_ERROR("Invalid parameters.");
        return nullptr;
    }

    // 创建任务
    auto task = std::make_unique<InferenceTask>();
    task->batch = batch;
    task->batch_size = batch_size;
    task->callback = callback;
    task->completed = false;
    task->handle.status = TASK_STATUS_WAITING; // 等待状态

    // 分配任务ID
    int task_id;
    {
        std::lock_guard<std::mutex> lock(g_tasks_mutex);
        task_id = g_next_task_id++;
        task->handle.task_id = task_id;
    }

    // 保存任务副本指针，用于返回
    InferenceTaskHandle* handle_ptr = &task->handle;

    // 启动工作线程
    task->worker = std::thread(inference_thread_func, task.get());

    // 将任务转移到全局管理
    {
        std::lock_guard<std::mutex> lock(g_tasks_mutex);
        g_tasks[task_id] = std::move(task);
    }

    return handle_ptr;
}

extern "C" API int vlm_check_task_status(InferenceTaskHandle* handle)
{
    if (!handle)
    {
        LOG_ERROR("Invalid task handle.");
        return -1;
    }

    std::lock_guard<std::mutex> lock(g_tasks_mutex);
    auto it = g_tasks.find(handle->task_id);
    if (it == g_tasks.end())
    {
        LOG_ERROR("Task does not exist.");
        return -1;
    }

    return it->second->handle.status;
}

extern "C" API bool vlm_wait_for_result(InferenceTaskHandle* handle)
{
    if (!handle)
    {
        LOG_ERROR("Invalid task handle.");
        return false;
    }

    // 查找任务
    std::unique_ptr<InferenceTask>* task_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_tasks_mutex);
        auto it = g_tasks.find(handle->task_id);
        if (it == g_tasks.end())
        {
            LOG_ERROR("Task does not exist.");
            return false;
        }
        task_ptr = &it->second;
    }

    InferenceTask* task = task_ptr->get();

    // 等待任务完成
    {
        std::unique_lock<std::mutex> lock(task->mutex);
        if (!task->completed)
        {
            task->cv.wait(lock, [task] { return task->completed; });
        }
    }

    return true;
}

extern "C" API void vlm_release_task(InferenceTaskHandle* handle)
{
    if (!handle)
        return;

    std::lock_guard<std::mutex> lock(g_tasks_mutex);
    auto it = g_tasks.find(handle->task_id);
    if (it == g_tasks.end())
        return;

    // 确保线程已完成
    auto& task = it->second;
    if (task->worker.joinable())
    {
        task->worker.join();
    }

    // 任务会在这里自动销毁
    g_tasks.erase(it);
}

// deprecated
extern "C" API const char* vlm_get_error_message()
{
    return "";
}

extern "C" API void vlm_set_log_callback(VLMLogCallback cb)
{
    gLogger.set_log_callback(cb);
    cpp_base_suite::logger::SetLogCallback(cb);
}

extern "C" API const char* vlm_get_version()
{
    return VLM_INFER_VERSION_INFO;
}
