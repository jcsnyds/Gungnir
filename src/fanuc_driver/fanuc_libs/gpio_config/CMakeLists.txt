# SPDX-FileCopyrightText: 2025, FANUC America Corporation
# SPDX-FileCopyrightText: 2025, FANUC CORPORATION
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22)
project(gpio_config)

add_library(gpio_config SHARED src/gpio_config.cpp )
target_link_libraries(gpio_config PRIVATE reflectcpp)
target_include_directories(gpio_config PUBLIC
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(gpio_config PROPERTIES CXX_STANDARD 20)

if (BUILD_TESTING)
  add_subdirectory(test)
endif ()
