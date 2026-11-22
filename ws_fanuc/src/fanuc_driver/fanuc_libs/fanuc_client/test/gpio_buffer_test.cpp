// SPDX-FileCopyrightText: 2025, FANUC America Corporation
// SPDX-FileCopyrightText: 2025, FANUC CORPORATION
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "stream_motion/packets.hpp"

#include "fanuc_client/gpio_buffer.hpp"

namespace fanuc_client
{
namespace
{

using ::testing::Each;
using ::testing::ElementsAreArray;
using ::testing::FieldsAre;
using CommandGPIOTypes = ::fanuc_client::GPIOBuffer::CommandGPIOTypes;
using StatusGPIOTypes = ::fanuc_client::GPIOBuffer::StatusGPIOTypes;

TEST(GPIOBufferTest, EmptyBufferZeroInitialized)
{
  GPIOBuffer buffer = GPIOBuffer::Builder().build();

  EXPECT_THAT(buffer.command_buffer(), Each(0));
  EXPECT_THAT(buffer.status_buffer(), Each(0));
}

TEST(GPIOBufferTest, WithCommandBuffer)
{
  GPIOBuffer::Builder builder = GPIOBuffer::Builder();

  // Starts at byte index 0.
  constexpr int kNumFlags = 9;
  auto flags = builder.addCommandConfig<CommandGPIOTypes::F, bool>(1, kNumFlags);
  // Starts at byte index 4.
  constexpr int kDOStart = 10;
  constexpr int kNumDO = 33;
  auto d_o = builder.addCommandConfig<CommandGPIOTypes::DO, bool>(kDOStart, kNumDO);
  // Starts at byte index 12.
  auto r_o = builder.addCommandConfig<CommandGPIOTypes::RO, bool>(300, 34);
  // Starts at byte index 20.
  constexpr int kNumAO = 5;
  auto a_o = builder.addCommandConfig<CommandGPIOTypes::AO, uint16_t>(1, kNumAO);

  GPIOBuffer buffer = builder.build();

  EXPECT_THAT(buffer.command_buffer(), Each(0));
  EXPECT_THAT(buffer.status_buffer(), Each(0));

  // Change up the flags.
  ASSERT_EQ(flags.start_index(), 1);
  ASSERT_EQ(flags.size(), kNumFlags);
  EXPECT_EQ(flags.type(), "F");
  for (int i = 1; i <= kNumFlags; ++i)
  {
    flags.set(i, i % 2);
  }

  std::array<uint8_t, 256> expected_array{};
  expected_array[0] = 0b01010101;
  expected_array[1] = 0x01;
  EXPECT_THAT(buffer.command_buffer(), ElementsAreArray(expected_array));
  EXPECT_THAT(buffer.status_buffer(), Each(0));

  // Change up DO.
  ASSERT_EQ(d_o.start_index(), kDOStart);
  ASSERT_EQ(d_o.size(), kNumDO);
  EXPECT_EQ(d_o.type(), "DO");
  for (int i = kDOStart; i < kDOStart + kNumDO; ++i)
  {
    d_o.set(i, i % 3);
  }

  expected_array[4] = 0b11011011;
  expected_array[5] = 0b10110110;
  expected_array[6] = 0b01101101;
  expected_array[7] = 0b11011011;
  expected_array[8] = 0x00;
  EXPECT_THAT(buffer.command_buffer(), ElementsAreArray(expected_array));
  EXPECT_THAT(buffer.status_buffer(), Each(0));

  // Edit analog flags.
  ASSERT_EQ(a_o.start_index(), 1);
  ASSERT_EQ(a_o.size(), kNumAO);
  EXPECT_EQ(a_o.type(), "AO");
  for (int i = 1; i <= kNumAO; ++i)
  {
    a_o.set(i, i * 256 + /* lower byte */ i + 1);
  }

  // Stored little endian.
  expected_array[20] = 0x2;  // Index 1
  expected_array[21] = 0x1;
  expected_array[22] = 0x3;  // Index 2
  expected_array[23] = 0x2;
  expected_array[24] = 0x4;  // Index 3
  expected_array[25] = 0x3;
  expected_array[26] = 0x5;  // Index 4
  expected_array[27] = 0x4;
  expected_array[28] = 0x6;  // Index 5
  expected_array[29] = 0x5;
  EXPECT_THAT(buffer.command_buffer(), ElementsAreArray(expected_array));
  EXPECT_THAT(buffer.status_buffer(), Each(0));
}

TEST(GPIOBufferTest, WithStatusBuffer)
{
  GPIOBuffer::Builder builder = GPIOBuffer::Builder();

  // Starts at byte index 0.
  constexpr int kNumFlags = 9;
  auto flags = builder.addStatusConfig<StatusGPIOTypes::F, bool>(1, kNumFlags);
  // Starts at byte index 4.
  constexpr int kDOStart = 10;
  constexpr int kNumDO = 33;
  auto d_o = builder.addStatusConfig<StatusGPIOTypes::DO, bool>(kDOStart, kNumDO);
  // Starts at byte index 12.
  auto r_o = builder.addStatusConfig<StatusGPIOTypes::RO, bool>(300, 34);
  // Starts at byte index 20.
  constexpr int kNumAO = 5;
  auto a_o = builder.addStatusConfig<StatusGPIOTypes::AO, uint16_t>(1, kNumAO);

  GPIOBuffer buffer = builder.build();

  EXPECT_THAT(buffer.command_buffer(), Each(0));
  EXPECT_THAT(buffer.status_buffer(), Each(0));

  // Change the flags
  buffer.status_buffer()[0] = 0b01010101;
  buffer.status_buffer()[1] = 0x01;

  ASSERT_EQ(flags.start_index(), 1);
  ASSERT_EQ(flags.size(), kNumFlags);
  EXPECT_EQ(flags.type(), "F");
  for (int i = 1; i <= kNumFlags; ++i)
  {
    EXPECT_EQ(flags.get(i), i % 2 != 0);
  }

  // Change up DO.
  buffer.status_buffer()[4] = 0b11011011;
  buffer.status_buffer()[5] = 0b10110110;
  buffer.status_buffer()[6] = 0b01101101;
  buffer.status_buffer()[7] = 0b11011011;
  buffer.status_buffer()[8] = 0x00;

  ASSERT_EQ(d_o.start_index(), kDOStart);
  ASSERT_EQ(d_o.size(), kNumDO);
  EXPECT_EQ(d_o.type(), "DO");
  for (int i = kDOStart; i < kDOStart + kNumDO; ++i)
  {
    EXPECT_EQ(d_o.get(i), i % 3 != 0);
  }

  // Edit analog flags. Stored little endian.
  buffer.status_buffer()[20] = 0x2;  // Index 1
  buffer.status_buffer()[21] = 0x1;
  buffer.status_buffer()[22] = 0x3;  // Index 2
  buffer.status_buffer()[23] = 0x2;
  buffer.status_buffer()[24] = 0x4;  // Index 3
  buffer.status_buffer()[25] = 0x3;
  buffer.status_buffer()[26] = 0x5;  // Index 4
  buffer.status_buffer()[27] = 0x4;
  buffer.status_buffer()[28] = 0x6;  // Index 5
  buffer.status_buffer()[29] = 0x5;
  ASSERT_EQ(a_o.start_index(), 1);
  ASSERT_EQ(a_o.size(), kNumAO);
  EXPECT_EQ(a_o.type(), "AO");
  for (int i = 1; i <= kNumAO; ++i)
  {
    EXPECT_EQ(a_o.get(i), i * 256 + /* lower byte */ i + 1);
  }
}

template <CommandGPIOTypes t>
struct CmdTypeWrapper
{
  static constexpr CommandGPIOTypes value = t;
};

template <StatusGPIOTypes t>
struct StatusTypeWrapper
{
  static constexpr StatusGPIOTypes value = t;
};

template <stream_motion::IOType t>
struct IOTypeWrapper
{
  static constexpr stream_motion::IOType value = t;
};

template <typename T>
class ToGPIOBoolCommandConfigTest : public ::testing::Test
{
protected:
  static constexpr CommandGPIOTypes cmd_type = std::tuple_element<0, T>::type::value;
  using ValueType = typename std::tuple_element<1, T>::type;
  static constexpr auto io_type = static_cast<uint32_t>(std::tuple_element<2, T>::type::value);

  static constexpr int start_index = 1;
  static constexpr int size = 2;

  ToGPIOBoolCommandConfigTest()
    : builder_{}, cmd_block_(builder_.addCommandConfig<cmd_type, ValueType>(start_index, size))
  {
  }

  GPIOBuffer::Builder builder_{};
  const GPIOBuffer::CommandBlock<cmd_type, ValueType>& cmd_block_;
};

TYPED_TEST_SUITE_P(ToGPIOBoolCommandConfigTest);

TYPED_TEST_P(ToGPIOBoolCommandConfigTest, GeneratesGPIOControlConfig)
{
  stream_motion::GPIOControlConfig sm_config = this->cmd_block_.toGPIOControlConfig();

  EXPECT_THAT(sm_config, FieldsAre(stream_motion::GPIOCommandType::IOCmd, this->io_type, this->start_index, this->size));
}

REGISTER_TYPED_TEST_CASE_P(ToGPIOBoolCommandConfigTest, GeneratesGPIOControlConfig);

using BoolIOCmdTestTypes =
    ::testing::Types<std::tuple<CmdTypeWrapper<CommandGPIOTypes::DO>, bool, IOTypeWrapper<stream_motion::IOType::DO>>,
                     std::tuple<CmdTypeWrapper<CommandGPIOTypes::RO>, bool, IOTypeWrapper<stream_motion::IOType::RO>>,
                     std::tuple<CmdTypeWrapper<CommandGPIOTypes::F>, bool, IOTypeWrapper<stream_motion::IOType::F>>>;

INSTANTIATE_TYPED_TEST_SUITE_P(ToStreamMotionConfigTestSuite, ToGPIOBoolCommandConfigTest, BoolIOCmdTestTypes);

template <typename T>
class ToGPIOBoolStatusConfigTest : public ::testing::Test
{
protected:
  static constexpr StatusGPIOTypes cmd_type = std::tuple_element<0, T>::type::value;
  using ValueType = typename std::tuple_element<1, T>::type;
  static constexpr auto io_type = static_cast<uint32_t>(std::tuple_element<2, T>::type::value);

  static constexpr int start_index = 1;
  static constexpr int size = 2;

  ToGPIOBoolStatusConfigTest()
    : builder_{}, status_block_(builder_.addStatusConfig<cmd_type, ValueType>(start_index, size))
  {
  }

  GPIOBuffer::Builder builder_{};
  const GPIOBuffer::StatusBlock<cmd_type, ValueType>& status_block_;
};

TYPED_TEST_SUITE_P(ToGPIOBoolStatusConfigTest);

TYPED_TEST_P(ToGPIOBoolStatusConfigTest, GeneratesGPIOControlConfig)
{
  stream_motion::GPIOControlConfig sm_config = this->status_block_.toGPIOControlConfig();

  EXPECT_THAT(sm_config,
              FieldsAre(stream_motion::GPIOCommandType::IOState, this->io_type, this->start_index, this->size));
}

REGISTER_TYPED_TEST_CASE_P(ToGPIOBoolStatusConfigTest, GeneratesGPIOControlConfig);

using BoolIOStatusTestTypes =
    ::testing::Types<std::tuple<StatusTypeWrapper<StatusGPIOTypes::DO>, bool, IOTypeWrapper<stream_motion::IOType::DO>>,
                     std::tuple<StatusTypeWrapper<StatusGPIOTypes::DI>, bool, IOTypeWrapper<stream_motion::IOType::DI>>,
                     std::tuple<StatusTypeWrapper<StatusGPIOTypes::RO>, bool, IOTypeWrapper<stream_motion::IOType::RO>>,
                     std::tuple<StatusTypeWrapper<StatusGPIOTypes::RI>, bool, IOTypeWrapper<stream_motion::IOType::RI>>,
                     std::tuple<StatusTypeWrapper<StatusGPIOTypes::F>, bool, IOTypeWrapper<stream_motion::IOType::F>>>;

INSTANTIATE_TYPED_TEST_SUITE_P(ToStreamMotionConfigTestSuite, ToGPIOBoolStatusConfigTest, BoolIOStatusTestTypes);

TEST(ToGPIOControlConfigConversionTest, ConvertsTheRestOfCommandTypes)
{
  GPIOBuffer::Builder builder{};
  builder.addCommandConfig<CommandGPIOTypes::AO, uint16_t>(1, 2);
  builder.addCommandConfig<CommandGPIOTypes::FloatReg, float>(3, 4);
  GPIOBuffer buffer = builder.build();

  stream_motion::GPIOConfiguration configs = buffer.toStreamMotionConfig();

  using ::stream_motion::GPIOCommandType;
  using ::stream_motion::IOType;
  using ::stream_motion::NumRegType;
  EXPECT_THAT(configs, ::testing::IsSupersetOf(
                           { FieldsAre(GPIOCommandType::IOCmd, static_cast<uint32_t>(IOType::AO), 1, 2),
                             FieldsAre(GPIOCommandType::NumRegCmd, static_cast<uint32_t>(NumRegType::Float), 3, 4) }));
}

TEST(ToGPIOControlConfigConversionTest, ConvertsTheRestOfStatusTypes)
{
  GPIOBuffer::Builder builder{};
  builder.addStatusConfig<StatusGPIOTypes::AO, uint16_t>(1, 2);
  builder.addStatusConfig<StatusGPIOTypes::AI, uint16_t>(3, 4);
  builder.addStatusConfig<StatusGPIOTypes::FloatReg, float>(5, 6);
  GPIOBuffer buffer = builder.build();

  stream_motion::GPIOConfiguration configs = buffer.toStreamMotionConfig();

  using ::stream_motion::GPIOCommandType;
  using ::stream_motion::IOType;
  using ::stream_motion::NumRegType;
  EXPECT_THAT(configs, ::testing::IsSupersetOf(
                           { FieldsAre(GPIOCommandType::IOState, static_cast<uint32_t>(IOType::AO), 1, 2),
                             FieldsAre(GPIOCommandType::IOState, static_cast<uint32_t>(IOType::AI), 3, 4),
                             FieldsAre(GPIOCommandType::NumRegState, static_cast<uint32_t>(NumRegType::Float), 5, 6) }));
}

}  // namespace
}  // namespace fanuc_client
