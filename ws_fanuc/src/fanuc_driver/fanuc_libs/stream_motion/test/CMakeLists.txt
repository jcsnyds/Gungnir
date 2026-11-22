# SPDX-FileCopyrightText: 2025, FANUC America Corporation
# SPDX-FileCopyrightText: 2025, FANUC CORPORATION
#
# SPDX-License-Identifier: Apache-2.0

add_executable(stream_motion_test stream_test.cpp)
target_link_libraries(stream_motion_test PRIVATE stream_motion GTest::gtest_main Sockpp::sockpp)
gtest_discover_tests(stream_motion_test)
