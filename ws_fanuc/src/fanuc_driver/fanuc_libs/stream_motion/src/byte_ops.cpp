// SPDX-FileCopyrightText: 2025, FANUC America Corporation
// SPDX-FileCopyrightText: 2025, FANUC CORPORATION
//
// SPDX-License-Identifier: Apache-2.0

#include "stream_motion/byte_ops.hpp"
#include <fmt/format.h>

#include <cstdint>

namespace stream_motion
{

bool isLittleEndian()
{
  static const uint16_t test = 0x1;
  static bool is_little = *reinterpret_cast<const uint8_t*>(&test) == 0x1;
  return is_little;
}

}  // namespace stream_motion
