#include "utils/path_config.h"
#include <iostream>

namespace trt_edgellm {
namespace tests {

PathConfig& PathConfig::getInstance() {
    static PathConfig instance;
    return instance;
}

PathConfig::PathConfig() {
    std::filesystem::path currentPath = std::filesystem::current_path();

    while (!currentPath.empty() && 
           !std::filesystem::exists(currentPath / "tests_cpp")) {
        currentPath = currentPath.parent_path();
    }
    
    projectRoot_ = currentPath;
    initDefaultPaths();
}

void PathConfig::initDefaultPaths() {
    if (projectRoot_.empty()) {
        std::cerr << "Warning: Cannot determine project root. Using current directory." << std::endl;
        projectRoot_ = std::filesystem::current_path();
    }

    resourcesRoot_ = projectRoot_ / "tests_cpp" / "resources" / "moe";
    testDataRoot_ = resourcesRoot_ / "seq1";
}

std::filesystem::path PathConfig::getResourcesPath() const {
    auto it = customPaths_.find("resources");
    if (it != customPaths_.end()) {
        return it->second;
    }
    return resourcesRoot_;
}

std::filesystem::path PathConfig::getTestDataPath() const {
    auto it = customPaths_.find("testdata");
    if (it != customPaths_.end()) {
        return it->second;
    }

    return testDataRoot_;
}

std::filesystem::path PathConfig::getSafetensorPath(const std::string& filename) const {
    for (const auto& [key, path] : customPaths_) {
        std::filesystem::path fullPath = path / filename;
        if (std::filesystem::exists(fullPath)) {
            return fullPath;
        }
    }

    std::filesystem::path testDataPath = getTestDataPath();
    std::filesystem::path fullPath = testDataPath / filename;
    if (std::filesystem::exists(fullPath)) {
        return fullPath;
    }

    std::filesystem::path resourcesPath = getResourcesPath();
    fullPath = resourcesPath / filename;
    if (std::filesystem::exists(fullPath)) {
        return fullPath;
    }
    
    return fullPath;
}

void PathConfig::setCustomPath(const std::string& key, const std::filesystem::path& path) {
    customPaths_[key] = path;
    std::cout << "Set custom path [" << key << "]: " << path.string() << std::endl;
}

} // namespace tests
} // namespace trt_edgellm