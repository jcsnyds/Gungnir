// SPDX-FileCopyrightText: 2025, FANUC America Corporation
// SPDX-FileCopyrightText: 2025, FANUC CORPORATION
//
// SPDX-License-Identifier: Apache-2.0

#include "fanuc_client/gpio_buffer.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "stream_motion/byte_ops.hpp"
#include "stream_motion/packets.hpp"

namespace fanuc_client
{
namespace
{
using ::stream_motion::isLittleEndian;
using ::stream_motion::swapBytesIfNeeded;
using CommandGPIOTypes = ::fanuc_client::GPIOBuffer::CommandGPIOTypes;
using StatusGPIOTypes = ::fanuc_client::GPIOBuffer::StatusGPIOTypes;

constexpr auto kIOTypeConversionErrorMsg = "INTERNAL: Attempted to convert register type to stream motion IO type.";

void ThrowNoMoreConfigs()
{
  throw std::invalid_argument("Cannot support more than " + std::to_string(GPIOBuffer::kMaxConfigs) + " GPIO configs.");
}

void ThrowOutOfBufferSpace()
{
  throw std::invalid_argument("Cannot support any more GPIO configs. The provided configs would take more "
                              "than the current limit of " +
                              std::to_string(GPIOBuffer::kBufferLength) + " bytes.");
}

constexpr std::string_view CommandTypeToString(CommandGPIOTypes type)
{
  switch (type)
  {
    case CommandGPIOTypes::DO:
      return "DO";
    case CommandGPIOTypes::RO:
      return "RO";
    case CommandGPIOTypes::AO:
      return "AO";
    case CommandGPIOTypes::F:
      return "F";
    case CommandGPIOTypes::FloatReg:
      return "FloatReg";
    default:
      return "Unknown";
  }
  // Should never be hit
  return "";
}

constexpr std::string_view StatusTypeToString(StatusGPIOTypes type)
{
  switch (type)
  {
    case StatusGPIOTypes::DI:
      return "DI";
    case StatusGPIOTypes::DO:
      return "DO";
    case StatusGPIOTypes::RI:
      return "RI";
    case StatusGPIOTypes::RO:
      return "RO";
    case StatusGPIOTypes::AI:
      return "AI";
    case StatusGPIOTypes::AO:
      return "AO";
    case StatusGPIOTypes::F:
      return "F";
    case StatusGPIOTypes::FloatReg:
      return "FloatReg";
    default:
      return "Unknown";
  }
  // Should never be hit
  return "";
}

// Only converts IO types (throws on registers).
stream_motion::IOType ToStreamMotionIOType(GPIOBuffer::CommandGPIOTypes type)
{
  using ::stream_motion::IOType;
  switch (type)
  {
    case CommandGPIOTypes::DO:
      return IOType::DO;
    case CommandGPIOTypes::RO:
      return IOType::RO;
    case CommandGPIOTypes::AO:
      return IOType::AO;
    case CommandGPIOTypes::F:
      return IOType::F;
    case CommandGPIOTypes::FloatReg:
      break;
  }
  throw std::runtime_error(kIOTypeConversionErrorMsg);
}

stream_motion::IOType toStreamMotionIOType(GPIOBuffer::StatusGPIOTypes type)
{
  using ::stream_motion::IOType;
  switch (type)
  {
    case StatusGPIOTypes::DI:
      return IOType::DI;
    case StatusGPIOTypes::DO:
      return IOType::DO;
    case StatusGPIOTypes::RI:
      return IOType::RI;
    case StatusGPIOTypes::RO:
      return IOType::RO;
    case StatusGPIOTypes::AI:
      return IOType::AI;
    case StatusGPIOTypes::AO:
      return IOType::AO;
    case StatusGPIOTypes::F:
      return IOType::F;
    case StatusGPIOTypes::FloatReg:
      break;
  }
  throw std::runtime_error(kIOTypeConversionErrorMsg);
}

int Align(int value, int alignment)
{
  const int remainder = value % alignment;
  return remainder == 0 ? value : value - remainder + alignment;
}

}  // namespace

template <CommandGPIOTypes gpio_type, typename T>
stream_motion::GPIOControlConfig GPIOBuffer::CommandBlock<gpio_type, T>::toGPIOControlConfig() const
{
  stream_motion::GPIOCommandType command_type = gpio_type == CommandGPIOTypes::FloatReg ?
                                                    stream_motion::GPIOCommandType::NumRegCmd :
                                                    stream_motion::GPIOCommandType::IOCmd;

  uint32_t sm_gpio_type;
  if (command_type == stream_motion::GPIOCommandType::NumRegCmd)
  {
    sm_gpio_type = static_cast<uint32_t>(stream_motion::NumRegType::Float);
  }
  else
  {
    sm_gpio_type = static_cast<uint32_t>(ToStreamMotionIOType(gpio_type));
  }
  return stream_motion::GPIOControlConfig{ .command_type = command_type,
                                           .gpio_type = sm_gpio_type,
                                           .start = start_index_,
                                           .length = static_cast<uint32_t>(size_) };
}

template <StatusGPIOTypes gpio_type, typename T>
stream_motion::GPIOControlConfig GPIOBuffer::StatusBlock<gpio_type, T>::toGPIOControlConfig() const
{
  stream_motion::GPIOCommandType command_type = gpio_type == StatusGPIOTypes::FloatReg ?
                                                    stream_motion::GPIOCommandType::NumRegState :
                                                    stream_motion::GPIOCommandType::IOState;

  uint32_t sm_gpio_type;
  if (command_type == stream_motion::GPIOCommandType::NumRegState)
  {
    sm_gpio_type = static_cast<uint32_t>(stream_motion::NumRegType::Float);
  }
  else
  {
    sm_gpio_type = static_cast<uint32_t>(toStreamMotionIOType(gpio_type));
  }
  return stream_motion::GPIOControlConfig{ .command_type = command_type,
                                           .gpio_type = sm_gpio_type,
                                           .start = start_index_,
                                           .length = static_cast<uint32_t>(size_) };
}

template <CommandGPIOTypes gpio_type, typename T>
void GPIOBuffer::CommandBlock<gpio_type, T>::set(int index, T value)
{
  if (index >= start_index_ + size_ || index < start_index_)
  {
    throw std::out_of_range("Unable to set index " + std::to_string(index) + " in GPIO block starting at " +
                            std::to_string(start_index_) + " with size " + std::to_string(size_));
  }

  const int buf_index = (index - start_index_) * sizeof(T);

  if constexpr (std::is_same_v<T, bool>)
  {
    const int byte_index = buf_index / 8;
    const int bit_index = buf_index % 8;

    const uint8_t mask = 0x1 << bit_index;
    if (value)
    {
      ptr_[byte_index] |= mask;
    }
    else
    {
      ptr_[byte_index] &= ~mask;
    }
  }
  else
  {
    // Keep bytes stored little endian.
    *reinterpret_cast<T*>(ptr_ + buf_index) = swapBytesIfNeeded(value, !isLittleEndian());
  }
}

template <StatusGPIOTypes gpio_type, typename T>
T GPIOBuffer::StatusBlock<gpio_type, T>::get(int index)
{
  if (index >= start_index_ + size_ || index < start_index_)
  {
    throw std::out_of_range("Unable to get index " + std::to_string(index) + " in GPIO block starting at " +
                            std::to_string(start_index_) + " with size " + std::to_string(size_));
  }

  const int buf_index = (index - start_index_) * sizeof(T);

  if constexpr (std::is_same_v<T, bool>)
  {
    const int byte_index = buf_index / 8;
    const int bit_index = buf_index % 8;

    return (ptr_[byte_index] & (0x1 << bit_index)) != 0;
  }
  else
  {
    // Bytes stored little endian.
    return swapBytesIfNeeded(*reinterpret_cast<T*>(ptr_ + buf_index), !isLittleEndian());
  }
};

template <CommandGPIOTypes gpio_type, typename T>
std::string_view GPIOBuffer::CommandBlock<gpio_type, T>::type() const
{
  return CommandTypeToString(gpio_type);
}

template <StatusGPIOTypes gpio_type, typename T>
std::string_view GPIOBuffer::StatusBlock<gpio_type, T>::type() const
{
  return StatusTypeToString(gpio_type);
}

template class GPIOBuffer::CommandBlock<CommandGPIOTypes::DO, bool>;
template class GPIOBuffer::CommandBlock<CommandGPIOTypes::RO, bool>;
template class GPIOBuffer::CommandBlock<CommandGPIOTypes::AO, uint16_t>;
template class GPIOBuffer::CommandBlock<CommandGPIOTypes::F, bool>;
template class GPIOBuffer::CommandBlock<CommandGPIOTypes::FloatReg, float>;

template class GPIOBuffer::StatusBlock<StatusGPIOTypes::DI, bool>;
template class GPIOBuffer::StatusBlock<StatusGPIOTypes::DO, bool>;
template class GPIOBuffer::StatusBlock<StatusGPIOTypes::RI, bool>;
template class GPIOBuffer::StatusBlock<StatusGPIOTypes::RO, bool>;
template class GPIOBuffer::StatusBlock<StatusGPIOTypes::AI, uint16_t>;
template class GPIOBuffer::StatusBlock<StatusGPIOTypes::AO, uint16_t>;
template class GPIOBuffer::StatusBlock<StatusGPIOTypes::F, bool>;
template class GPIOBuffer::StatusBlock<StatusGPIOTypes::FloatReg, float>;

template <CommandGPIOTypes gpio_type, typename T>
GPIOBuffer::CommandBlock<gpio_type, T>& GPIOBuffer::Builder::addCommandConfig(uint32_t start_index, size_t size)
{
  if (blocks_.size() == kMaxConfigs)
  {
    ThrowNoMoreConfigs();
  }

  // If bool type, we pack 8 values to a byte.
  const int num_bytes = std::is_same_v<T, bool> ? Align(size, 8) / 8 : sizeof(T) * size;
  if (next_command_buf_index_ + num_bytes > kBufferLength)
  {
    ThrowOutOfBufferSpace();
  }

  auto block = std::make_unique<CommandBlock<gpio_type, T>>(command_buffer_->data() + next_command_buf_index_,
                                                            start_index, size);
  auto& block_ref = *block;
  blocks_.emplace_back(std::move(block));

  // 4 byte block alignment.
  next_command_buf_index_ += Align(num_bytes, 4);
  return block_ref;
}

template <StatusGPIOTypes gpio_type, typename T>
GPIOBuffer::StatusBlock<gpio_type, T>& GPIOBuffer::Builder::addStatusConfig(uint32_t start_index, size_t size)
{
  if (blocks_.size() == kMaxConfigs)
  {
    ThrowNoMoreConfigs();
  }

  // If bool type, we pack 8 values to a byte.
  const int num_bytes = std::is_same_v<T, bool> ? Align(size, 8) / 8 : sizeof(T) * size;
  if (next_status_buf_index_ + num_bytes > kBufferLength)
  {
    ThrowOutOfBufferSpace();
  }

  auto block =
      std::make_unique<StatusBlock<gpio_type, T>>(status_buffer_->data() + next_status_buf_index_, start_index, size);
  auto& block_ref = *block;
  blocks_.emplace_back(std::move(block));

  // 4 byte block alignment.
  next_status_buf_index_ += Align(num_bytes, 4);
  return block_ref;
}

// Explicit instantiations for supported config types.
template GPIOBuffer::CommandBlock<CommandGPIOTypes::DO, bool>& GPIOBuffer::Builder::addCommandConfig(uint32_t, size_t);
template GPIOBuffer::CommandBlock<CommandGPIOTypes::RO, bool>& GPIOBuffer::Builder::addCommandConfig(uint32_t, size_t);
template GPIOBuffer::CommandBlock<CommandGPIOTypes::AO, uint16_t>& GPIOBuffer::Builder::addCommandConfig(uint32_t,
                                                                                                         size_t);
template GPIOBuffer::CommandBlock<CommandGPIOTypes::F, bool>& GPIOBuffer::Builder::addCommandConfig(uint32_t, size_t);
template GPIOBuffer::CommandBlock<CommandGPIOTypes::FloatReg, float>& GPIOBuffer::Builder::addCommandConfig(uint32_t,
                                                                                                            size_t);
template GPIOBuffer::StatusBlock<StatusGPIOTypes::DI, bool>& GPIOBuffer::Builder::addStatusConfig(uint32_t, size_t);
template GPIOBuffer::StatusBlock<StatusGPIOTypes::DO, bool>& GPIOBuffer::Builder::addStatusConfig(uint32_t, size_t);
template GPIOBuffer::StatusBlock<StatusGPIOTypes::RI, bool>& GPIOBuffer::Builder::addStatusConfig(uint32_t, size_t);
template GPIOBuffer::StatusBlock<StatusGPIOTypes::RO, bool>& GPIOBuffer::Builder::addStatusConfig(uint32_t, size_t);
template GPIOBuffer::StatusBlock<StatusGPIOTypes::AI, uint16_t>& GPIOBuffer::Builder::addStatusConfig(uint32_t, size_t);
template GPIOBuffer::StatusBlock<StatusGPIOTypes::AO, uint16_t>& GPIOBuffer::Builder::addStatusConfig(uint32_t, size_t);
template GPIOBuffer::StatusBlock<StatusGPIOTypes::F, bool>& GPIOBuffer::Builder::addStatusConfig(uint32_t, size_t);
template GPIOBuffer::StatusBlock<StatusGPIOTypes::FloatReg, float>& GPIOBuffer::Builder::addStatusConfig(uint32_t,
                                                                                                         size_t);

GPIOBuffer GPIOBuffer::Builder::build()
{
  return { std::move(command_buffer_), std::move(status_buffer_), std::move(blocks_) };
}

stream_motion::GPIOConfiguration GPIOBuffer::toStreamMotionConfig() const
{
  stream_motion::GPIOConfiguration sm_config{};
  int index = 0;
  for (const std::unique_ptr<BlockInterface>& block : blocks_)
  {
    sm_config[index++] = block->toGPIOControlConfig();
  }
  return sm_config;
}

}  // namespace fanuc_client
