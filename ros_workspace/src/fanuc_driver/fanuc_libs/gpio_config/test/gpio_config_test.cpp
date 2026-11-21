// SPDX-FileCopyrightText: 2025, FANUC America Corporation
// SPDX-FileCopyrightText: 2025, FANUC CORPORATION
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <filesystem>

#include "gpio_config/gpio_config.hpp"

namespace gpio_config
{
namespace
{
using ::testing::ElementsAre;
using ::testing::FieldsAre;
using ::testing::Optional;

std::filesystem::path GetGoodYaml()
{
  return std::filesystem::path(YAML_TEST_FOLDER) / "good_config.yaml";
}

std::filesystem::path GetBadYaml()
{
  return std::filesystem::path(YAML_TEST_FOLDER) / "bad_config.yaml";
}

TEST(ParseGPIOConfigTest, ParsesSuccessfully)
{
  GPIOTopicConfig config = ParseGPIOConfig(GetGoodYaml()).gpio_topic_config;

  EXPECT_THAT(config.io_state,
              Optional(ElementsAre(FieldsAre(BoolIOStateType::DI, 1, 2), FieldsAre(BoolIOStateType::DO, 3, 4),
                                   FieldsAre(BoolIOStateType::RI, 5, 6), FieldsAre(BoolIOStateType::RO, 7, 8),
                                   FieldsAre(BoolIOStateType::F, 9, 10))));
  EXPECT_THAT(config.io_cmd,
              Optional(ElementsAre(FieldsAre(BoolIOCmdType::DO, 11, 12), FieldsAre(BoolIOCmdType::RO, 13, 14),
                                   FieldsAre(BoolIOCmdType::F, 15, 16))));
  EXPECT_THAT(config.analog_io_state, Optional(ElementsAre(FieldsAre(AnalogIOStateType::AI, 21, 22),
                                                           FieldsAre(AnalogIOStateType::AO, 23, 24))));
  EXPECT_THAT(config.analog_io_cmd,
              Optional(ElementsAre(FieldsAre(AnalogIOCmdType::AO, 31, 32), FieldsAre(AnalogIOCmdType::AO, 33, 34))));
  EXPECT_THAT(config.num_reg_state, Optional(ElementsAre(FieldsAre(101, 102))));
  EXPECT_THAT(config.num_reg_cmd, Optional(ElementsAre(FieldsAre(201, 202))));
}

TEST(ParseGPIOConfigTest, ParsingFails)
{
  auto parse_config = []() { ParseGPIOConfig(GetBadYaml()); };

  EXPECT_THAT(parse_config, ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Failed to parse GPIO")));
}

}  // namespace
}  // namespace gpio_config
