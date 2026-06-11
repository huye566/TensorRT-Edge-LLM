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

#include "profileFormatter.h"
#include "vlm_interface.h"
#include <chrono>
#include <common/logger.h>
#include <cstring>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_resize2.h>

using namespace trt_edgellm;
using Json = nlohmann::json;

// 动态库函数指针类型定义
typedef int (*InitFunc)(VLMModelInfo* model_info);
typedef void (*CleanupFunc)();
typedef InferenceTaskHandle* (*SubmitFunc)(InferenceBatchHandle*, int64_t, InferenceCompletionCallback);
typedef bool (*WaitResultFunc)(InferenceTaskHandle*);
typedef void (*ReleaseTaskFunc)(InferenceTaskHandle*);
typedef char const* (*GetErrorFunc)();
typedef char const* (*GetVersionFunc)();

// 函数指针
struct VLMFunctions
{
    InitFunc initialize;
    CleanupFunc cleanup;
    SubmitFunc submit_query;
    WaitResultFunc wait_for_result;
    ReleaseTaskFunc release_task;
    GetErrorFunc get_error;
    GetVersionFunc get_version;
};

static std::tuple<int64_t, int64_t> getResizedImageSize(
    int64_t const height, int64_t const width, int64_t const maxRatio, ModelType type)
{
    int64_t patchSize = 14;
    int64_t mergeSize = 2;
    switch (type)
    {
    case MODEL_TYPE_QWEN_2_VL_2B:
        patchSize = 14;
        mergeSize = 2;
        break;
    case MODEL_TYPE_QWEN_3_VL_2B:
        patchSize = 16;
        mergeSize = 2;
        break;
    default: LOG_WARNING("Unsupported model type."); break;
    }

    int64_t const factor = patchSize * mergeSize;
    int64_t const minPixels = 128 * factor * factor;
    int64_t const maxPixels = 1024 * factor * factor;

    auto roundByFactor = [](int64_t value, int64_t factor) -> int64_t {
        return std::round(static_cast<double>(value) / factor) * factor;
    };
    auto floorByFactor = [](int64_t value, int64_t factor) -> int64_t {
        return std::floor(static_cast<double>(value) / factor) * factor;
    };
    auto ceilByFactor = [](int64_t value, int64_t factor) -> int64_t {
        return std::ceil(static_cast<double>(value) / factor) * factor;
    };

    if (std::max(height, width) / std::min(height, width) > maxRatio)
    {
        throw std::runtime_error("absolute aspect ratio must be smaller than " + std::to_string(maxRatio) + ", got "
            + std::to_string(std::max(height, width) / std::min(height, width)));
    }

    int64_t hBar = std::max(factor, roundByFactor(height, factor));
    int64_t wBar = std::max(factor, roundByFactor(width, factor));

    if (hBar * wBar > maxPixels)
    {
        double beta = std::sqrt(static_cast<double>(height * width) / maxPixels);
        hBar = floorByFactor(static_cast<int64_t>(height / beta), factor);
        wBar = floorByFactor(static_cast<int64_t>(width / beta), factor);
    }
    else if (hBar * wBar < minPixels)
    {
        double beta = std::sqrt(static_cast<double>(minPixels) / (height * width));
        hBar = ceilByFactor(static_cast<int64_t>(height * beta), factor);
        wBar = ceilByFactor(static_cast<int64_t>(width * beta), factor);
    }

    return {hBar, wBar};
}

void init_batch(InferenceBatchHandle* handle, std::string const& input, std::vector<std::string> const& image_path,
    int max_length, ModelType type)
{
    handle->inputString = new char[input.size() + 1];
    input.copy(handle->inputString, input.size());
    handle->inputString[input.size()] = '\0';
    handle->image_size = image_path.size();
    handle->images = new ImageBufferHandle[handle->image_size];
    for (int i = 0; i < handle->image_size; ++i)
    {
        int width{0}, height{0}, channels{0};
        int desiredChannels = 3;
        // Loaded pixels in hwc, rgb order
        unsigned char* image = stbi_load(image_path[i].c_str(), &width, &height, &channels, desiredChannels);
        // check resize
        auto [resizedHeight, resizedWidth] = getResizedImageSize(height, width, 200, type);

        ImageBufferHandle* imageBuf = handle->images + i;
        imageBuf->size = resizedHeight * resizedWidth * desiredChannels * sizeof(unsigned char);
        imageBuf->width = resizedWidth;
        imageBuf->height = resizedHeight;
        imageBuf->channels = desiredChannels;

        unsigned char* resizedImage = nullptr;
        if (width != resizedWidth || height != resizedHeight)
        {
            // need resize
            LOG_WARNING("Resize image from (%d, %d) to (%d, %d).", width, height, resizedWidth, resizedHeight);
            resizedImage = (unsigned char*) malloc(imageBuf->size);
            constexpr int32_t kINPUT_STRIDE_BYTES{0};
            constexpr int32_t kOUTPUT_STRIDE_BYTES{0};
            stbir_resize_uint8_linear(image, width, height, kINPUT_STRIDE_BYTES, resizedImage, resizedWidth,
                resizedHeight, kOUTPUT_STRIDE_BYTES, stbir_pixel_layout::STBIR_RGB);
            stbi_image_free(image);
        }
        else
        {
            resizedImage = image;
        }

        unsigned char* cuda_image;
        cudaMalloc(&cuda_image, imageBuf->size);
        cudaMemcpy(cuda_image, resizedImage, imageBuf->size, cudaMemcpyHostToDevice);
        imageBuf->ptr = cuda_image;
        stbi_image_free(resizedImage);
    }
    handle->result = new InferenceResult;
    handle->result->text = new char[max_length];
}

void deinit_batch(InferenceBatchHandle* handle)
{
    if (!handle)
        return;
    delete[] handle->inputString;
    for (int j = 0; j < handle->image_size; ++j)
    {
        ImageBufferHandle* imageBuf = handle->images + j;
        cudaFree(imageBuf->ptr);
    }
    delete[] handle->images;
    delete[] handle->result->text;
    delete handle->result;
}

struct RuntimeArgs
{
    bool help{false};
    std::string inputFile;
    std::string outputFile;
    std::string llmEnginePath;
    std::string visualEnginePath;
    std::string libInferPath;
    std::string modelType{"qwen2_vl"};
    bool dumpProfile{false};
    bool staticImageSize{false};
    bool staticPrompt{false};
    int visCompressMode{0};
    float keepRate{1.0};
    bool isEagle3{false};
    bool closeCudaGraph{false};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [-h] [-e or --llmEnginePath=<path to LLM engine>] [-v or --visualEnginePath=<path to visual engine>]"
                 " [-s or --maxLength=<int>] [-t or --tokenizerPath=<path to HF tokenizer>]"
                 " [--inputString=<input string for one batch>] [--imagePaths=<image paths for one batch>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -h                  Display this help message" << std::endl;
    std::cerr << "  --llmEnginePath     Provide the Qwen TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --visualEnginePath  Provide the visual TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --libInferPath      Provide the path to libvlm_infer.so. default=./libvlm_infer.so. " << std::endl;
    std::cerr << "  --inputFile         Provide the input json file path. " << std::endl;
    std::cerr << "  --outputFile        Provide the output json file path. " << std::endl;
    std::cerr << "  --maxLength         Provide the maximum output length for the generation session (including the "
                 "input). Default = 1024."
              << std::endl;
    std::cerr << "  --modelType         Provide the model type. Default = qwen2_vl." << std::endl;
    std::cerr << "  --dumpProfile             Use debug mode, which outputs tensors." << std::endl;
    std::cerr << "  --staticImageSize   Image size is static or not(dynamic), Default = false." << std::endl;
    std::cerr << "  --staticPrompt      Prompt is static or not(dynamic), Default = false." << std::endl;
    std::cerr << "  --visCompressMode   0: no compress, 1 - vis pruner, 2 - ali compress method" << std::endl;
    std::cerr << "  --keepRate          Provide the pruner keep rate. Default = 1.0." << std::endl;
    std::cerr << "  --eagle3            Use eagle3, Default = false." << std::endl;
    std::cerr << "  --closeCudaGraph    Close CUDA Graph optimization, Default = false." << std::endl;
};

bool parseRuntimeArgs(RuntimeArgs& args, int argc, char* argv[])
{
    static struct option long_options[] = {{"help", no_argument, 0, 'h'}, {"inputFile", required_argument, 0, 'i'},
        {"outputFile", required_argument, 0, 'o'}, {"llmEnginePath", required_argument, 0, 'e'},
        {"visualEnginePath", required_argument, 0, 'v'}, {"libInferPath", required_argument, 0, 'l'},
        {"modelType", required_argument, 0, 0}, {"staticImageSize", no_argument, 0, 'z'},
        {"staticPrompt", no_argument, 0, 'm'}, {"draftEnginePath", required_argument, 0, 'D'},
        {"eagle3", no_argument, 0, 'E'}, {"visCompressMode", required_argument, 0, 'P'},
        {"keepRate", required_argument, 0, 'K'}, {"dumpProfile", no_argument, 0, 'd'},
        {"closeCudaGraph", no_argument, 0, 'C'}, {0, 0, 0, 0}};

    int opt;

    // Loop to process each option
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "hi:o:e:v:l:D:P:K:zmEdC", long_options, &option_index)) != -1)
    {
        switch (opt)
        {
        case 'h': args.help = true; return true;
        case 'i':
            if (optarg)
            {
                args.inputFile = optarg;
            }
            else
            {
                std::cerr << "ERROR: --inputString requires option argument" << std::endl;
                return false;
            }
            break;
        case 'o':
            if (optarg)
            {
                args.outputFile = optarg;
            }
            else
            {
                std::cerr << "ERROR: --imagePaths requires option argument" << std::endl;
                return false;
            }
            break;
        case 'e':
            if (optarg)
            {
                args.llmEnginePath = optarg;
            }
            else
            {
                std::cerr << "ERROR: --llmEnginePath requires option argument" << std::endl;
                return false;
            }
            break;
        case 'E': args.isEagle3 = true; break;
        case 'v':
            if (optarg)
            {
                args.visualEnginePath = optarg;
            }
            else
            {
                std::cerr << "ERROR: --visualEnginePath requires option argument" << std::endl;
                return false;
            }
            break;
        case 'l':
            if (optarg)
            {
                args.libInferPath = optarg;
            }
            else
            {
                args.libInferPath = "./libvlm_infer.so";
                std::cerr << "WARNING: --libInferPath default set to: ./libvlm_infer.so " << std::endl;
            }
            break;
        case 'd': args.dumpProfile = true; break;
        case 'z': args.staticImageSize = true; break;
        case 'm': args.staticPrompt = true; break;
        case 'P':
            if (optarg)
            {
                args.visCompressMode = std::stoi(optarg);
            }
            else
            {
                std::cerr << "ERROR: --visCompressMode requires option argument" << std::endl;
                return false;
            }
            break;
        case 'K':
            if (optarg)
            {
                args.keepRate = std::stof(optarg);
            }
            else
            {
                std::cerr << "ERROR: --keepRate requires option argument" << std::endl;
                return false;
            }
            break;
        case 'C': args.closeCudaGraph = true; break;
        case 0:
            if (strcmp(long_options[option_index].name, "modelType") == 0)
            {
                {
                    if (optarg)
                    {
                        args.modelType = optarg;
                    }
                    else
                    {
                        std::cerr << "ERROR: model type requires option argument,support only qwen2_vl currently"
                                  << std::endl;
                        return false;
                    }
                }
            }
            break;
        default: return false;
        }
    }
    return true;
}

#include <iostream>

class Metrics
{
public:
    Metrics(std::string const& n)
        : name(n)
    {
    }

    void add(bool pred, bool gt)
    {
        if (pred && gt)
            TP++;
        else if (!pred && !gt)
            TN++;
        else if (pred && !gt)
            FP++;
        else if (!pred && gt)
            FN++;
    }
    void add(std::pair<bool, bool> const& r)
    {
        add(r.first, r.second);
    }

    long tp() const
    {
        return TP;
    }
    long tn() const
    {
        return TN;
    }
    long fp() const
    {
        return FP;
    }
    long fn() const
    {
        return FN;
    }

    double precision() const
    {
        return TP + FP == 0 ? 0.0 : double(TP) / (TP + FP);
    }

    double recall() const
    {
        return TP + FN == 0 ? 0.0 : double(TP) / (TP + FN);
    }

    double accuracy() const
    {
        long total = TP + TN + FP + FN;
        return total == 0 ? 0.0 : double(TP + TN) / total;
    }

    double f1() const
    {
        double p = precision();
        double r = recall();
        return (p + r) == 0 ? 0.0 : 2 * p * r / (p + r);
    }

    std::string fmtString()
    {
        std::ostringstream result;
        result << std::fixed << std::setprecision(2);
        result << std::endl;
        result << "=== " << name << " Summary ===" << std::endl;
        result << "TP: " << TP << ", TN: " << TN << ", FP: " << FP << ", FN: " << FN << std::endl;
        result << "precision: " << precision() << ", recall: " << recall() << ", accuracy: " << accuracy()
               << ", f1: " << f1();
        return result.str();
    }

private:
    long TP = 0, TN = 0, FP = 0, FN = 0;
    std::string name;
};

std::pair<bool, bool> checkConstruct(std::string const& pred_str, std::string const& gt_str)
{
    bool pred = pred_str.find("施工") != std::string::npos;
    bool gt = gt_str.find("施工") != std::string::npos;
    return {pred, gt};
}

std::pair<bool, bool> checkLight(std::string const& pred_str, std::string const& gt_str)
{
    bool pred = true;
    bool gt = true;
    return {pred, gt};
}

std::pair<bool, bool> checkRoad(std::string const& pred_str, std::string const& gt_str)
{
    bool pred = true;
    bool gt = true;
    return {pred, gt};
}

struct ParsedInput
{
    std::vector<std::string> inputStrings;
    std::vector<std::vector<std::string>> imagePaths;
    std::vector<std::string> groundTruth;
    int batchSize{1};
    int maxLength{256};
    int cuptiProfileLevel{0};
};

void parseInputFile(std::string const& inputFilePath, ParsedInput& output)
{
    Json inputData;
    std::ifstream inputFileStream(inputFilePath);
    if (!inputFileStream.is_open())
    {
        LOG_ERROR("Failed to open input file: %s", inputFilePath.c_str());
        throw std::runtime_error("Failed to open input file: " + inputFilePath);
    }
    try
    {
        inputData = Json::parse(inputFileStream);
        inputFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse input file with error: %s", e.what());
        throw std::runtime_error("Failed to parse input file: " + inputFilePath);
    }

    // Extract global parameters
    output.batchSize = inputData.value("batch_size", 1);
    output.maxLength = inputData.value("max_generate_length", 256);
    output.cuptiProfileLevel = inputData.value("cupti_profile_level", 0);

    // Parse requests
    if (inputData.contains("requests") && inputData["requests"].is_array())
    {
        auto& requests = inputData["requests"];

        // Process requests in batches according to batchSize
        for (auto const& request : requests)
        {
            if (request.contains("messages") && request["messages"].is_array())
            {
                auto& messages = request["messages"];
                for (auto const& message : messages)
                {
                    if (!message.contains("role") || !message.contains("content"))
                    {
                        LOG_ERROR("Each message must have 'role' and 'content' fields");
                        throw std::runtime_error("Each message must have 'role' and 'content' fields");
                    }
                    if (message.contains("role") && message["role"] == "user")
                    {
                        auto const& content = message["content"];
                        if (content.is_string())
                        {
                            output.inputStrings.push_back(content.get<std::string>());
                        }
                        else if (content.is_array())
                        {
                            std::vector<std::string> images;
                            for (auto const& contentItem : content)
                            {
                                if (!contentItem.contains("type"))
                                {
                                    LOG_ERROR("Each content item must have a 'type' field");
                                    throw std::runtime_error("Each content item must have a 'type' field");
                                }

                                auto type = contentItem["type"].get<std::string>();
                                if (type == "text")
                                {
                                    output.inputStrings.push_back(contentItem["text"].get<std::string>());
                                }
                                else if (type == "image")
                                {
                                    images.push_back(contentItem["image"].get<std::string>());
                                }
                                else
                                {
                                    LOG_ERROR("Content type must be 'text', 'image', but got: %s", type.c_str());
                                    throw std::runtime_error(format::fmtstr(
                                        "Content type must be 'text', 'image', but got: %s", type.c_str()));
                                }
                            }
                            output.imagePaths.push_back(std::move(images));
                        }
                        if (message.contains("truth"))
                        {
                            output.groundTruth.push_back(message["truth"].get<std::string>());
                        }
                    }
                }
            }
            else
            {
                LOG_ERROR("messages is not an array");
                throw std::runtime_error("messages is not an array");
            }
        }
    }
    else
    {
        LOG_ERROR("requests is not an array");
        throw std::runtime_error("requests is not an array");
    }
}

int main(int argc, char* argv[])
{
    RuntimeArgs args;
    if ((argc < 2) || (!parseRuntimeArgs(args, argc, argv)))
    {
        printUsage(argv[0]);
        return 0;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return 0;
    }

    if (args.inputFile == "")
    {
        std::cerr << "Error: --inputFile is required" << std::endl;
        return 1;
    }

    // 加载共享库
    void* lib_handle = dlopen(args.libInferPath.c_str(), RTLD_LAZY);
    if (!lib_handle)
    {
        std::cerr << "无法加载库: " << dlerror() << std::endl;
        return 1;
    }

    // 解析函数
    VLMFunctions funcs;
    funcs.initialize = (InitFunc) dlsym(lib_handle, "vlm_initialize");
    funcs.cleanup = (CleanupFunc) dlsym(lib_handle, "vlm_cleanup");
    funcs.submit_query = (SubmitFunc) dlsym(lib_handle, "vlm_submit_query");
    funcs.wait_for_result = (WaitResultFunc) dlsym(lib_handle, "vlm_wait_for_result");
    funcs.release_task = (ReleaseTaskFunc) dlsym(lib_handle, "vlm_release_task");
    funcs.get_error = (GetErrorFunc) dlsym(lib_handle, "vlm_get_error_message");
    funcs.get_version = (GetVersionFunc) dlsym(lib_handle, "vlm_get_version");

    // 检查函数加载是否成功
    bool functions_loaded = funcs.initialize && funcs.cleanup && funcs.submit_query && funcs.wait_for_result
        && funcs.release_task && funcs.get_error && funcs.get_version;

    if (!functions_loaded)
    {
        std::cerr << "无法解析函数符号: " << dlerror() << std::endl;
        dlclose(lib_handle);
        return 1;
    }
    std::string version = VLM_INFER_VERSION_INFO;
    std::string so_version = funcs.get_version();
    if (version != so_version)
    {
        std::cerr << "版本号不同，version：" << version << ", so version: " << so_version << std::endl;
        dlclose(lib_handle);
        return 1;
    }

    ParsedInput input;
    parseInputFile(args.inputFile, input);

    std::string visual_engine_path = args.visualEnginePath;
    std::string llm_engine_path = args.llmEnginePath;

    LoraWeights loraWeights;
    loraWeights.first = nullptr;
    loraWeights.second = nullptr;

    ModelType model_type;

    if (args.modelType == "qwen2_vl")
    {
        model_type = ModelType::MODEL_TYPE_QWEN_2_VL_2B;
    }
    else if (args.modelType == "qwen2_5_vl")
    {
        model_type = ModelType::MODEL_TYPE_QWEN_2_5_VL_3B;
    }
    else if (args.modelType == "qwen3_vl")
    {
        model_type = ModelType::MODEL_TYPE_QWEN_3_VL_2B;
    }
    else if (args.modelType == "internvl_3")
    {
        model_type = ModelType::MODEL_TYPE_INTERNVL3_1B;
    }
    else
    {
        std::cerr << "Only support 'qwen2_vl'/'qwen2_5_vl' model and  'internvl' model !! please check model_type !!"
                  << std::endl;
    }

    // std::cerr << "vlm_infer_test is NOT runnable! If you want to run, please modify image size to (952, 504) and put
    // image data into gpu device memory!" << std::endl; return -1;

    cudaStream_t stream;
    auto ret = cudaStreamCreate(&stream);
    if (ret != cudaSuccess)
    {
        std::cerr << "cudaStreamCreate failed!" << std::endl;
        return -1;
    }
    VLMModelInfo input_info{input.maxLength, input.batchSize, model_type, "", visual_engine_path.c_str(), llm_engine_path.c_str(),
        {args.isEagle3, "", 6, 10, 60}, loraWeights, (void*) (&stream),
        {args.staticImageSize, args.staticPrompt, input.cuptiProfileLevel, args.visCompressMode, args.keepRate}, args.dumpProfile, args.closeCudaGraph};
    // 初始化库
    if (!funcs.initialize(&input_info))
    {
        std::cerr << "库初始化失败: " << funcs.get_error() << std::endl;
        dlclose(lib_handle);
        return 1;
    }
    std::cout << "库初始化完成" << std::endl;

    Json outputData;
    outputData["input_file"] = args.inputFile;
    outputData["responses"] = Json::array();

    Metrics construct("construct");

    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    for (size_t m = 0; m < input.inputStrings.size(); m += input.batchSize)
    {
        LOG_INFO("------------------%d/%d--------------------", m+1, input.inputStrings.size());
        InferenceBatchHandle* batch_handlers;
        batch_handlers = new InferenceBatchHandle[input.batchSize];
        for (int i = 0; i < input.batchSize; ++i)
        {
            init_batch(batch_handlers + i, input.inputStrings[m + i], input.imagePaths[m + i], input.maxLength, model_type);
        }

        // 提交推理任务
        auto infer_start = clock::now();
        InferenceTaskHandle* task = funcs.submit_query(batch_handlers, input.batchSize, nullptr);
        if (!task)
        {
            LOG_ERROR("任务提交失败: %s.", funcs.get_error());
            funcs.cleanup();
            dlclose(lib_handle);
            return 1;
        }

        // 同步等待结果
        bool result = funcs.wait_for_result(task);
        auto infer_end = clock::now();
        double infer_time_milli_sec = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
        if (result)
        {
            for (int i = 0; i < input.batchSize; ++i)
            {
                InferenceBatchHandle* handle = batch_handlers + i;
                LOG_INFO("推理Token数：%d, 结果：%s.", handle->result->token_count, handle->result->text);

                Json responseJson;
                std::string outputText = handle->result->text;
                // Validate UTF-8 for output text (inputs are always valid)
                // If invalid UTF-8 detected, error message is returned and original text is logged
                responseJson["output_text"] = sanitizeUtf8ForJson(outputText);
                responseJson["request_idx"] = m + i;
                responseJson["image"] = input.imagePaths[m][i];
                responseJson["latency"] = infer_time_milli_sec;
                responseJson["failed_reason"] = "";
                outputData["responses"].push_back(responseJson);

                if (!input.groundTruth.empty())
                {
                    construct.add(checkConstruct(outputText, input.groundTruth[m + i]));
                }
            }
        }
        else
        {
            LOG_ERROR("获取结果失败: %s.", funcs.get_error());
            for (int i = 0; i < input.batchSize; ++i)
            {
                Json responseJson;
                std::string outputText = "推理失败！";
                // Validate UTF-8 for output text (inputs are always valid)
                // If invalid UTF-8 detected, error message is returned and original text is logged
                responseJson["output_text"] = sanitizeUtf8ForJson(outputText);
                responseJson["request_idx"] = m + i;
                responseJson["image"] = input.imagePaths[m][i];
                responseJson["latency"] = infer_time_milli_sec;
                responseJson["failed_reason"] = outputText;
                outputData["responses"].push_back(responseJson);

                if (!input.groundTruth.empty())
                {
                    construct.add(checkConstruct(outputText, input.groundTruth[m + i]));
                }
            }
        }

        // 清理资源
        funcs.release_task(task);
        for (int i = 0; i < input.batchSize; ++i)
        {
            deinit_batch(batch_handlers + i);
        }
        delete[] batch_handlers;
        auto end = clock::now();
        double cost_time_sec = std::chrono::duration<double>(end - start).count();
        double fps = (m + input.batchSize) / cost_time_sec;
        if (!input.groundTruth.empty())
        {
            LOG_INFO(construct.fmtString().c_str());
        }
        LOG_INFO("------------------FPS:%.2f--------------------", fps);
    }

    LOG_INFO("Ready to write %s.", args.outputFile.c_str());
    // Export to JSON file
    try
    {
        std::ofstream outputFile(args.outputFile);
        if (outputFile.is_open())
        {
            outputFile << outputData.dump(4); // Pretty print with 4 spaces indentation
            outputFile.close();
            LOG_INFO("All responses exported to: %s", args.outputFile.c_str());
        }
        else
        {
            LOG_ERROR("Failed to open output file: %s", args.outputFile.c_str());
            return EXIT_FAILURE;
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to write output file: %s", e.what());
        return EXIT_FAILURE;
    }

    funcs.cleanup();
    dlclose(lib_handle);

    return 0;
}
