// SPDX-FileCopyrightText: 2025, FANUC America Corporation
// SPDX-FileCopyrightText: 2025, FANUC CORPORATION
//
// SPDX-License-Identifier: Apache-2.0

#include "gpio_config/gpio_config.hpp"

#include <filesystem>

#include "rfl/yaml.hpp"

namespace gpio_config
{

GPIOConfig ParseGPIOConfig(const std::filesystem::path& file_name)
{
  std::string file_name_str = std::filesystem::absolute(file_name);
  rfl::Result<GPIOConfig> result = rfl::yaml::load<GPIOConfig>(file_name_str);

  if (!result.has_value())
  {
    throw std::runtime_error("Failed to parse GPIO configuration yaml file with error: " + result.error().what());
  }
  return result.value();
}

}  // namespace gpio_config
