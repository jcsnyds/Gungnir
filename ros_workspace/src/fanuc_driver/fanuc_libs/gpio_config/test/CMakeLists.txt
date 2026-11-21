# SPDX-FileCopyrightText: 2025, FANUC America Corporation
# SPDX-FileCopyrightText: 2025, FANUC CORPORATION
#
# SPDX-License-Identifier: Apache-2.0

add_executable(gpio_config_test gpio_config_test.cpp)
target_link_libraries(gpio_config_test PRIVATE gpio_config GTest::gtest_main GTest::gmock_main)
target_compile_definitions(gpio_config_test PRIVATE YAML_TEST_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}/data")
gtest_discover_tests(gpio_config_test)
