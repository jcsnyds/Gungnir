# SPDX-FileCopyrightText: 2025, FANUC America Corporation
# SPDX-FileCopyrightText: 2025, FANUC CORPORATION
#
# SPDX-License-Identifier: Apache-2.0

add_executable(stream_motion_example stream_motion_example.cpp)
target_link_libraries(stream_motion_example PRIVATE stream_motion)

add_executable(rmi_connection_example rmi_connection_example.cpp)
target_link_libraries(rmi_connection_example PRIVATE rmi)

add_executable(fanuc_client_example fanuc_client_example.cpp)
target_link_libraries(fanuc_client_example PRIVATE fanuc_client)
