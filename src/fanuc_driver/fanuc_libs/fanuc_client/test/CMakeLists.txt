# SPDX-FileCopyrightText: 2025, FANUC America Corporation
# SPDX-FileCopyrightText: 2025, FANUC CORPORATION
#
# SPDX-License-Identifier: Apache-2.0

add_executable(fanuc_client_test fanuc_client_test.cpp)
target_link_libraries(fanuc_client_test PRIVATE fanuc_client GTest::gtest_main GTest::gmock_main)
gtest_discover_tests(fanuc_client_test)

add_executable(gpio_buffer_test gpio_buffer_test.cpp)
target_link_libraries(gpio_buffer_test PRIVATE fanuc_client GTest::gtest_main GTest::gmock_main)
gtest_discover_tests(gpio_buffer_test)
