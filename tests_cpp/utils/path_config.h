#ifndef TESTS_UTILS_PATH_CONFIG_H
#define TESTS_UTILS_PATH_CONFIG_H

#include <string>
#include <filesystem>
#include <unordered_map>
#include <memory>

namespace trt_edgellm {
namespace tests {

class PathConfig {
public:
    static PathConfig& getInstance();
    std::filesystem::path getResourcesPath() const;
    std::filesystem::path getTestDataPath() const;
    std::filesystem::path getSafetensorPath(const std::string& filename) const;
    void setCustomPath(const std::string& key, const std::filesystem::path& path);

private:
    PathConfig();
    void initDefaultPaths();

private:
    std::filesystem::path projectRoot_;
    std::filesystem::path resourcesRoot_;
    std::filesystem::path testDataRoot_;

    // 自定义路径映射
    std::unordered_map<std::string, std::filesystem::path> customPaths_;
};

} // namespace tests
} // namespace trt_edgellm

#endif // TESTS_UTILS_PATH_CONFIG_H