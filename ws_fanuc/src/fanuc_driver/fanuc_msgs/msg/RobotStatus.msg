# SPDX-FileCopyrightText: 2025, FANUC America Corporation
# SPDX-FileCopyrightText: 2025, FANUC CORPORATION
#
# SPDX-License-Identifier: Apache-2.0

uint8 CONTACT_STOP_MODE_NONE = 0
uint8 CONTACT_STOP_MODE_SAFE = 1
uint8 CONTACT_STOP_MODE_STOP = 2
uint8 CONTACT_STOP_MODE_DSBL = 3
uint8 CONTACT_STOP_MODE_ESCP = 4

bool in_error
bool tp_enabled
bool e_stopped
bool motion_possible
uint8 contact_stop_mode
