#ifndef TRT_EDGELLM_MOENVFP4_PLUGIN_H__
#define TRT_EDGELLM_MOENVFP4_PLUGIN_H__

#include <NvInferRuntime.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace trt_edgellm {
namespace plugins {

class MoeNvfp4Plugin : public nvinfer1::IPluginV2DynamicExt {
public:
    MoeNvfp4Plugin(std::string const& name, 
                   int32_t expertsNum, 
                   int32_t expertsTopK,
                   std::vector<float> gateAlpha,
                   std::vector<float> upAlpha,
                   std::vector<float> downAlpha);
    
    MoeNvfp4Plugin(std::string const& name, void const* data, size_t length);
    MoeNvfp4Plugin() = delete;
    MoeNvfp4Plugin(MoeNvfp4Plugin const&) = delete;
    ~MoeNvfp4Plugin() override;

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
    void allocateWorkspace();
    void loadWorkspace();
    void freeWorkspace();

    // Helper function for scale padding size
    int getScalePadSize(int M, int N) const;

private:
    std::string mLayerName;
    std::string mNamespace;

    int32_t mHiddenSize{};
    int32_t mIntermediateSize{};
    int32_t mExpertsNum{};
    int32_t mExpertsTopK{};
    
    // Alpha parameters from attributes
    std::vector<float> mGateAlpha;
    std::vector<float> mUpAlpha;
    std::vector<float> mDownAlpha;
    
    bool mIsWorkspaceAllocated{false};
    bool mIsDataInitialized{false};
    std::unordered_map<std::string, void*> mWorkspace;
};

class MoeNvfp4PluginCreator : public nvinfer1::IPluginCreator
{
public:
    MoeNvfp4PluginCreator();
    ~MoeNvfp4PluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    char const* getPluginNamespace() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm

#endif // TRT_EDGELLM_MOENVFP4_PLUGIN_H__