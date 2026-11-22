// SPDX-FileCopyrightText: 2025, FANUC America Corporation
// SPDX-FileCopyrightText: 2025, FANUC CORPORATION
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace gpio_config
{

enum class BoolIOCmdType
{
  DO,
  RO,
  F
};
enum class BoolIOStateType
{
  DI,
  DO,
  RI,
  RO,
  F
};

struct BoolIOStateConfig
{
  BoolIOStateType type;
  uint32_t start;
  uint32_t length;  // Number of consecutive points.
};
struct BoolIOCmdConfig
{
  BoolIOCmdType type;
  uint32_t start;
  uint32_t length;  // Number of consecutive points.
};

enum class AnalogIOCmdType
{
  AO
};
enum class AnalogIOStateType
{
  AI,
  AO
};

struct AnalogIOCmdConfig
{
  AnalogIOCmdType type;
  uint32_t start;
  uint32_t length;  // Number of consecutive points.
};
struct AnalogIOStateConfig
{
  AnalogIOStateType type;
  uint32_t start;
  uint32_t length;  // Number of consecutive points.
};

struct NumRegConfig
{
  uint32_t start;
  uint32_t length;  // Number of consecutive points.
};

struct GPIOTopicConfig
{
  std::optional<std::vector<BoolIOCmdConfig>> io_cmd;
  std::optional<std::vector<BoolIOStateConfig>> io_state;
  std::optional<std::vector<AnalogIOCmdConfig>> analog_io_cmd;
  std::optional<std::vector<AnalogIOStateConfig>> analog_io_state;
  std::optional<std::vector<NumRegConfig>> num_reg_cmd;
  std::optional<std::vector<NumRegConfig>> num_reg_state;
};

struct GPIOConfig
{
  GPIOTopicConfig gpio_topic_config;
};

GPIOConfig ParseGPIOConfig(const std::filesystem::path& file_name);

}  // namespace gpio_config
