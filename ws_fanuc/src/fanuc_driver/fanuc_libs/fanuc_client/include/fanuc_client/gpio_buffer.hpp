// SPDX-FileCopyrightText: 2025, FANUC America Corporation
// SPDX-FileCopyrightText: 2025, FANUC CORPORATION
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>
#include "stream_motion/packets.hpp"

namespace fanuc_client
{

class GPIOBuffer
{
public:
  static constexpr int kMaxConfigs = stream_motion::kMaxGPIOConfigs;
  static constexpr int kBufferLength = 256;
  using RawGPIOBuffer = std::array<uint8_t, kBufferLength>;

  enum class CommandGPIOTypes
  {
    DO,
    RO,
    AO,
    F,
    FloatReg,
  };

  enum class StatusGPIOTypes
  {
    DI,
    DO,
    RI,
    RO,
    AI,
    AO,
    F,
    FloatReg,
  };

  // Represents a view into the command or status buffer.
  class BlockInterface
  {
  public:
    virtual ~BlockInterface() = default;
    virtual stream_motion::GPIOControlConfig toGPIOControlConfig() const = 0;
    virtual std::string_view type() const = 0;
    virtual uint32_t start_index() const = 0;
    virtual size_t size() const = 0;
  };

  template <CommandGPIOTypes gpio_type, typename T>
  class CommandBlock final : public BlockInterface
  {
  public:
    CommandBlock() = delete;
    CommandBlock(uint8_t* ptr, uint32_t start_index, size_t size)
      : ptr_{ ptr }, start_index_{ start_index }, size_{ size }
    {
      if (ptr == nullptr)
      {
        throw std::invalid_argument("Pointer to command block cannot be null.");
      }
    }

    /** Converts this block into a stream motion config. */
    stream_motion::GPIOControlConfig toGPIOControlConfig() const override;

    void set(int index, T value);

    std::string_view type() const override;

    uint32_t start_index() const override
    {
      return start_index_;
    }
    size_t size() const override
    {
      return size_;
    }

  private:
    uint8_t* ptr_;
    const uint32_t start_index_;
    // Number of values of type T that can fit in this block.
    const size_t size_;
  };

  template <StatusGPIOTypes gpio_type, typename T>
  class StatusBlock final : public BlockInterface
  {
  public:
    StatusBlock() = delete;
    StatusBlock(uint8_t* ptr, uint32_t start_index, size_t size)
      : ptr_{ ptr }, start_index_{ start_index }, size_{ size }
    {
      if (ptr == nullptr)
      {
        throw std::invalid_argument("Pointer to command block cannot be null.");
      }
    }

    /** Converts this block into a stream motion config. */
    stream_motion::GPIOControlConfig toGPIOControlConfig() const override;

    T get(int index);

    std::string_view type() const override;

    uint32_t start_index() const override
    {
      return start_index_;
    }
    size_t size() const override
    {
      return size_;
    }

  private:
    uint8_t* ptr_;
    const uint32_t start_index_;
    // Number of values of type T that can fit in this block.
    const size_t size_;
  };

  class Builder
  {
  public:
    Builder()
      : command_buffer_{ std::make_unique<RawGPIOBuffer>() }, status_buffer_{ std::make_unique<RawGPIOBuffer>() }
    {
      blocks_.reserve(stream_motion::kMaxGPIOConfigs);
    }

    Builder(const Builder&) = delete;
    Builder& operator=(const Builder&) = delete;

    template <CommandGPIOTypes gpio_type, typename T>
    CommandBlock<gpio_type, T>& addCommandConfig(uint32_t start_index, size_t size);

    template <StatusGPIOTypes gpio_type, typename T>
    StatusBlock<gpio_type, T>& addStatusConfig(uint32_t start_index, size_t size);

    GPIOBuffer build();

  private:
    size_t next_command_buf_index_ = 0;
    std::unique_ptr<RawGPIOBuffer> command_buffer_;
    size_t next_status_buf_index_ = 0;
    std::unique_ptr<RawGPIOBuffer> status_buffer_;
    std::vector<std::unique_ptr<BlockInterface>> blocks_;
  };

  GPIOBuffer() = delete;
  GPIOBuffer(const GPIOBuffer&) = delete;
  GPIOBuffer& operator=(const GPIOBuffer&) = delete;

  GPIOBuffer(GPIOBuffer&& other) noexcept
    : command_buffer_{ std::move(other.command_buffer_) }
    , status_buffer_{ std::move(other.status_buffer_) }
    , blocks_{ std::move(other.blocks_) }
  {
  }

  GPIOBuffer& operator=(GPIOBuffer&& other) noexcept
  {
    command_buffer_ = std::move(other.command_buffer_);
    status_buffer_ = std::move(other.status_buffer_);
    blocks_ = std::move(other.blocks_);
    return *this;
  }

  /** Converts the format of the command buffer (as defined by the blocks it holds) into a config that stream motion can
   * use to configure commanding gpio. */
  stream_motion::GPIOConfiguration toStreamMotionConfig() const;

  const RawGPIOBuffer& command_buffer() const
  {
    return *command_buffer_;
  }

  RawGPIOBuffer& status_buffer() const
  {
    return *status_buffer_;
  }

private:
  GPIOBuffer(std::unique_ptr<RawGPIOBuffer> command_buffer, std::unique_ptr<RawGPIOBuffer> status_buffer,
             std::vector<std::unique_ptr<BlockInterface>> blocks)
    : command_buffer_{ std::move(command_buffer) }
    , status_buffer_{ std::move(status_buffer) }
    , blocks_{ std::move(blocks) }
  {
  }

  std::unique_ptr<RawGPIOBuffer> command_buffer_;
  std::unique_ptr<RawGPIOBuffer> status_buffer_;
  std::vector<std::unique_ptr<BlockInterface>> blocks_;
};

}  // namespace fanuc_client
