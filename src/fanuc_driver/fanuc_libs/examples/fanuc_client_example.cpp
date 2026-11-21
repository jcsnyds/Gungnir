// SPDX-FileCopyrightText: 2025, FANUC America Corporation
// SPDX-FileCopyrightText: 2025, FANUC CORPORATION
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cmath>
#include <iostream>

#include <Eigen/Core>

#include "fanuc_client/fanuc_client.hpp"
#include "fanuc_client/gpio_buffer.hpp"

int main()
{
  try
  {
    fanuc_client::FanucClient fanuc_client("192.168.1.100");
    const double v_peak = 1000.0;
    const double payload = 0.0;
    std::vector<double> vel_limit;
    std::vector<double> acc_limit;
    std::vector<double> jerk_limit;
    fanuc_client.getLimits(v_peak, payload, vel_limit, acc_limit, jerk_limit);

    // Configure IO
    fanuc_client::GPIOBuffer::Builder gpio_builder{};
    auto regs = gpio_builder.addCommandConfig<fanuc_client::GPIOBuffer::CommandGPIOTypes::FloatReg, float>(
        /*start_index=*/1, /*size=*/10);
    auto regs_state = gpio_builder.addStatusConfig<fanuc_client::GPIOBuffer::StatusGPIOTypes::FloatReg, float>(
        /*start_index=*/1, /*size=*/10);

    auto gpio_buffer = std::make_shared<fanuc_client::GPIOBuffer>(gpio_builder.build());

    fanuc_client.startRMI();
    Eigen::VectorXd initial_command = Eigen::VectorXd::Zero(9);
    initial_command[4] = -90;
    fanuc_client.writeJointTargetRMI(initial_command);

    fanuc_client.startRealtimeStream(gpio_buffer);
    Eigen::VectorXd current_command = initial_command;
    fanuc_client.writeJointTarget(initial_command);

    // Generate a sine wave motion for 3 seconds
    const int period = 2800;
    const auto start_time = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start_time < std::chrono::milliseconds(1 * period) &&
           fanuc_client.isStreaming())
    {
      std::cout << "j1: " << fanuc_client.readJointAngles()[0] << " " << std::endl;
      const double time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
      const double omega = 2 * M_PI / (static_cast<double>(period) / 1000);
      current_command.array() = initial_command.array() + 20.0 * (1 + -cos(time * omega));
      fanuc_client.writeJointTarget(current_command);

      const auto value = static_cast<float>(time);
      for (int i = 0; i < regs.size(); ++i)
      {
        regs.set(regs.start_index() + i, value + i);
      }
      fanuc_client.sendIOCommand();

      for (int i = regs_state.start_index(); i < regs_state.start_index() + regs_state.size(); ++i)
      {
        std::cout << "reg value " << i << ": " << regs_state.get(i) << std::endl;
      }

      std::this_thread::sleep_for(std::chrono::microseconds(10 * fanuc_client.getControlPeriod()));
    }
  }
  catch (const std::runtime_error& e)
  {
    std::cout << e.what() << std::endl;
  }
}
