cmake_minimum_required(VERSION 3.50...3.28)
project(l515_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Ensure third-party submodules (like librealsense) inherit the policy minimum
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/librealsense/CMakeLists.txt")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Minimum policy version" FORCE)
endif()

option(L515_BUILD_WITH_BUNDLED_LIBREALSENSE
       "Build against a checked-out librealsense source tree"
       ON)
set(LIBREALSENSE_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/librealsense"
    CACHE PATH "Path to librealsense source (v2.54.2)")

if(L515_BUILD_WITH_BUNDLED_LIBREALSENSE AND EXISTS "${LIBREALSENSE_SOURCE_DIR}/CMakeLists.txt")
    # Critical: do NOT build the Python bindings. This project talks to
    # librealsense through its C++ API and exports a tiny C ABI of its own.
    set(BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_GRAPHICAL_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory(
        "${LIBREALSENSE_SOURCE_DIR}"
        "${CMAKE_BINARY_DIR}/librealsense-build"
        EXCLUDE_FROM_ALL
    )
    set(REAL_SENSE_TARGET realsense2)
else()
    find_package(realsense2 2.54.2 REQUIRED CONFIG)
    if(TARGET realsense2::realsense2)
        set(REAL_SENSE_TARGET realsense2::realsense2)
    elseif(TARGET realsense2)
        set(REAL_SENSE_TARGET realsense2)
    else()
        message(FATAL_ERROR "Could not find a usable realsense2 CMake target.")
    endif()
endif()

add_library(l515_native SHARED native/l515_native.cpp)
target_link_libraries(l515_native PRIVATE ${REAL_SENSE_TARGET})
target_include_directories(l515_native PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/native"
)

target_compile_definitions(l515_native PRIVATE L515_NATIVE_BUILD=1)

if(NOT WIN32)
    set_target_properties(l515_native PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()

# The setup script places l515_native and librealsense in the same directory.
# Use a relative runtime path so the installed Python module does not depend on
# the temporary CMake build directory.
if(APPLE)
    set_target_properties(l515_native PROPERTIES
        BUILD_RPATH "@loader_path"
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
elseif(UNIX)
    set_target_properties(l515_native PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Put all native artifacts in one predictable folder so the Python loader can
# locate them without depending on the active build configuration.
set(L515_NATIVE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib")

# Keep the librealsense runtime next to l515_native so the Python ctypes
# loader can find both libraries after setup.py copies them into python/.
if(TARGET realsense2 AND L515_BUILD_WITH_BUNDLED_LIBREALSENSE)
    set_target_properties(realsense2 PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${L515_NATIVE_OUTPUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${L515_NATIVE_OUTPUT_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${L515_NATIVE_OUTPUT_DIR}"
    )
    foreach(CONFIG Debug Release RelWithDebInfo MinSizeRel)
        set_target_properties(realsense2 PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${CONFIG} "${L515_NATIVE_OUTPUT_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_${CONFIG} "${L515_NATIVE_OUTPUT_DIR}"
            ARCHIVE_OUTPUT_DIRECTORY_${CONFIG} "${L515_NATIVE_OUTPUT_DIR}"
        )
    endforeach()
endif()

set_target_properties(l515_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${L515_NATIVE_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${L515_NATIVE_OUTPUT_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${L515_NATIVE_OUTPUT_DIR}"
)

foreach(CONFIG Debug Release RelWithDebInfo MinSizeRel)
    set_target_properties(l515_native PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${CONFIG} "${L515_NATIVE_OUTPUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_${CONFIG} "${L515_NATIVE_OUTPUT_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY_${CONFIG} "${L515_NATIVE_OUTPUT_DIR}"
    )
endforeach()

message(STATUS "L515 native backend will link against librealsense target: ${REAL_SENSE_TARGET}")
