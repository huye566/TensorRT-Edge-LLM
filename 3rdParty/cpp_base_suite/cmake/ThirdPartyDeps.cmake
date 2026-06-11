set(THIRDPARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty CACHE PATH "Third-party libraries directory")

if(NOT TARGET nlohmann_json)
    if(EXISTS "${THIRDPARTY_DIR}/nlohmann_json/include/nlohmann/json.hpp")
        add_library(nlohmann_json INTERFACE IMPORTED)
        set_target_properties(nlohmann_json PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${THIRDPARTY_DIR}/nlohmann_json/include")
        message(STATUS "cpp_base_suite: Using pre-built nlohmann_json headers")
    else()
        # Best-effort: try the standard CMake module.
        find_package(nlohmann_json QUIET)
        if(TARGET nlohmann_json)
            message(STATUS "cpp_base_suite: Using system nlohmann_json")
        else()
            message(WARNING "cpp_base_suite: nlohmann_json not found — JSON features will not build")
            add_library(nlohmann_json INTERFACE IMPORTED)
        endif()
    endif()
endif()
