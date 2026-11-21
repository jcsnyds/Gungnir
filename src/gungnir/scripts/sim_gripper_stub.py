#!/usr/bin/env python3
"""Stand-in for FANUC's /fanuc_gpio_controller/set_bool_io so gripper code is identical in sim.
The gripper is a static mesh in sim, so this only logs the request."""
import rclpy
from rclpy.node import Node
from fanuc_msgs.srv import SetBoolIO


class SimGripperStub(Node):
    def __init__(self):
        super().__init__("fanuc_gpio_controller")  # makes the service /fanuc_gpio_controller/set_bool_io
        self.create_service(SetBoolIO, "~/set_bool_io", self.on_request)
        self.get_logger().info("Simulated GPIO ready (gripper is a static mesh)")

    def on_request(self, request, response):
        self.get_logger().info(f"[sim] {request.io_type.type}[{request.index}] = {request.value}")
        fields = response.get_fields_and_field_types()
        if "success" in fields:
            response.success = True
        if "message" in fields:
            response.message = "simulated"
        return response


def main():
    rclpy.init()
    node = SimGripperStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
