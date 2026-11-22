// SPDX-FileCopyrightText: 2025, FANUC America Corporation
// SPDX-FileCopyrightText: 2025, FANUC CORPORATION
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

namespace stream_motion
{

bool isLittleEndian();

/**
 * @brief Swaps the byte order of the given value.
 *
 * @tparam T The arithmetic type of the value to swap (must be 2, 4, or 8 bytes).
 * @param value The value whose bytes will be swapped.
 * @param should_swap A boolean indicating if the bytes should be swapped. This defaults to true if the host system is
 * little endian because most stream motion data is big endian on the wire.
 * @return The value with its bytes swapped or the original value.
 */
template <typename T>
T swapBytesIfNeeded(T value, const bool should_swap = isLittleEndian())
{
  static_assert(std::is_arithmetic_v<T>, "swapBytesIfNeeded can only be used with arithmetic types");
  static_assert(sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, "swapBytes is not implemented for this type size");
  if (!should_swap)
  {
    return value;
  }

  auto buf = reinterpret_cast<uint8_t*>(&value);
  std::reverse(buf, buf + sizeof(T));
  return *reinterpret_cast<T*>(buf);
}
}  // namespace stream_motion
