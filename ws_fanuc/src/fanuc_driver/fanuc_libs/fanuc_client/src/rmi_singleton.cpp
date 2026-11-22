// SPDX-FileCopyrightText: 2025, FANUC America Corporation
// SPDX-FileCopyrightText: 2025, FANUC CORPORATION
//
// SPDX-License-Identifier: Apache-2.0

#include "fanuc_client/fanuc_client.hpp"

namespace fanuc_client
{

RMISingleton::RMISingleton() = default;

RMISingleton::~RMISingleton() = default;

RMISingleton& RMISingleton::getInstance()
{
  static RMISingleton instance;
  return instance;
}

std::shared_ptr<rmi::RMIConnectionInterface> RMISingleton::getRMIInstance()
{
  std::scoped_lock lock(getInstance().mtx_);
  if (getInstance().rmi_connection_interface_ == nullptr)
  {
    throw std::logic_error(
        "Cannot get the RMI instance before creating it. Use creatNewRMIInstance() before calling this method.");
  }
  return getInstance().rmi_connection_interface_;
}

std::shared_ptr<rmi::RMIConnectionInterface> RMISingleton::creatNewRMIInstance(const std::string& robot_ip_address,
                                                                               const uint16_t rmi_port)
{
  std::scoped_lock lock(getInstance().mtx_);
  getInstance().rmi_connection_interface_ = std::make_unique<rmi::RMIConnection>(robot_ip_address, rmi_port);
  return getInstance().rmi_connection_interface_;
}

std::shared_ptr<rmi::RMIConnectionInterface>
RMISingleton::setRMIInstance(std::unique_ptr<rmi::RMIConnectionInterface> rmi_connection)
{
  std::scoped_lock lock(getInstance().mtx_);
  if (rmi_connection == nullptr)
  {
    throw std::logic_error("Cannot set RMI instance to a non-nullptr value. Use creatNewRMIInstance() instead.");
  }
  getInstance().rmi_connection_interface_ = std::move(rmi_connection);
  return getInstance().rmi_connection_interface_;
}

}  // namespace fanuc_client
