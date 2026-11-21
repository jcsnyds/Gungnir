# SPDX-FileCopyrightText: 2025, FANUC America Corporation
# SPDX-FileCopyrightText: 2025, FANUC CORPORATION
#
# SPDX-License-Identifier: Apache-2.0

add_executable(rmi_test rmi_test.cpp)
set_target_properties(rmi_test PROPERTIES CXX_STANDARD 20)
target_link_libraries(rmi_test PRIVATE rmi reflectcpp GTest::gtest_main)
gtest_discover_tests(rmi_test)
