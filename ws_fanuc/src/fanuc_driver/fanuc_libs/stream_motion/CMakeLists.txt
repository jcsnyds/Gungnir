# SPDX-FileCopyrightText: 2025, FANUC America Corporation
# SPDX-FileCopyrightText: 2025, FANUC CORPORATION
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22)
project(stream_motion)

option(BUILD_EXAMPLES "Build examples" OFF)

add_library(stream_motion SHARED src/stream.cpp src/byte_ops.cpp)
set_target_properties(stream_motion PROPERTIES CXX_STANDARD 17)
target_link_libraries(stream_motion PRIVATE Sockpp::sockpp)
target_include_directories(stream_motion PUBLIC
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if (BUILD_TESTING)
  add_subdirectory(test)
endif ()
