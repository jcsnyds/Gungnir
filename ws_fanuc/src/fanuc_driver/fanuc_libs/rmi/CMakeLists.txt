# SPDX-FileCopyrightText: 2025, FANUC America Corporation
# SPDX-FileCopyrightText: 2025, FANUC CORPORATION
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22)
project(rmi)

add_library(rmi SHARED src/rmi.cpp src/serialization.cpp src/deserialization.cpp)
target_link_libraries(rmi PRIVATE reflectcpp Sockpp::sockpp)
target_include_directories(rmi PUBLIC
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(rmi PROPERTIES CXX_STANDARD 20)

if (BUILD_TESTING)
  add_subdirectory(test)
endif ()
