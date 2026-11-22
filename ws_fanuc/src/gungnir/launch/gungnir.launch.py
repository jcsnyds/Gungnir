"""
AI use: this launch file was written by the project's author (with earlier AI assistance that was not recorded in
this file). Claude (Anthropic; Claude Sonnet 5.5 via claude.ai), November 2026, changed ONLY the handler that
starts MoveIt and RViz in simulation mode (see "later" below, and the new imports EmitEvent, LogInfo and
Shutdown): it now starts them only if the joint_trajectory_controller spawner finished successfully, and
otherwise prints an explanation and stops the whole launch. Claude checked that the edited file compiles
(py_compile) and that the rest of the file is unchanged (a text diff). Claude has NOT run it with ROS 2,
Gazebo or MoveIt, because those are not installed where Claude works.
Reviewed and tested by: not yet reviewed or tested by the author.
"""
import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess, LogInfo, OpaqueFunction,
                            RegisterEventHandler, TimerAction)
from launch.events import Shutdown  # lets the launch stop itself when a start-up step failed
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder

PKG = "gungnir"


def _spawner(name):
    return Node(package="controller_manager", executable="spawner", output="screen",
                arguments=[name, "--controller-manager", "/controller_manager",
                           "--controller-manager-timeout", "180"])


def _join(*parts):
    return os.pathsep.join(p for p in parts if p)


def launch_setup(context, *args, **kwargs):
    arg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    mode = arg("mode")
    sim = mode == "sim"

    camera = arg("camera")
    if camera == "auto":
        camera = {"sim": "sim", "real": "real"}.get(mode, "none")
    if camera == "sim" and not sim:
        raise RuntimeError(f"camera:=sim needs mode:=sim (the simulated camera lives in Gazebo). "
                           f"With mode:={mode} use camera:=real or camera:=none.")
    if camera == "real" and sim:
        raise RuntimeError("mode:=sim cannot be combined with camera:=real (sim time vs wall-clock time). "
                           "Use mode:=mock camera:=real to test the physical camera without the robot.")
    show_world = sim if arg("show_world") == "auto" else arg("show_world") == "true"

    sim_time = [{"use_sim_time": True}] if sim else []
    pkg_share = get_package_share_directory(PKG)

    description_args = {
        "mode": mode,
        "robot_ip": arg("robot_ip"),
        "motion_control": arg("motion_control"),
        "encoding": arg("encoding"),
        "sim_camera": "true" if (sim and camera == "sim") else "false",
        "robot_model": arg("robot_model"),
    }
    moveit_config = (
        MoveItConfigsBuilder("gungnir", package_name=PKG)
        .robot_description(file_path="config/gungnir.urdf.xacro", mappings=description_args)
        .robot_description_semantic(file_path="config/gungnir.srdf")
        .trajectory_execution(file_path="config/gungnir_moveit_controllers.yaml")
        .planning_scene_monitor(publish_robot_description=True, publish_robot_description_semantic=True)
        .to_moveit_configs()
    )

    rviz_name = "gungnir_camera.rviz" if (camera != "none" or show_world) else "moveit.rviz"
    if not os.path.exists(os.path.join(pkg_share, "config", rviz_name)):
        print(f"[gungnir] {rviz_name} not found (run scripts/make_rviz_config.py and rebuild); using moveit.rviz")
        rviz_name = "moveit.rviz"

    rsp = Node(package="robot_state_publisher", executable="robot_state_publisher", output="both",
               parameters=[moveit_config.robot_description] + sim_time)
    move_group = Node(package="moveit_ros_move_group", executable="move_group", output="screen",
                      parameters=[moveit_config.to_dict(), {"monitor_dynamics": False}] + sim_time)
    rviz = Node(package="rviz2", executable="rviz2", name="rviz2", output="log",
                arguments=["-d", os.path.join(pkg_share, "config", rviz_name)],
                parameters=[moveit_config.robot_description, moveit_config.robot_description_semantic,
                            moveit_config.planning_pipelines, moveit_config.robot_description_kinematics,
                            moveit_config.joint_limits] + sim_time,
                condition=IfCondition(arg("launch_rviz")))
    jsb = _spawner("joint_state_broadcaster")
    jtc = _spawner("joint_trajectory_controller")
    # Same service name as FANUC's gpio controller: /fanuc_gpio_controller/set_bool_io
    gripper_stub = Node(package=PKG, executable="sim_gripper_stub.py", output="screen",
                        parameters=[{"use_sim_time": sim}])

    # ---- camera + scene helpers (any mode) ----
    extras = []
    if camera == "real":
        extras.append(Node(package=PKG, executable="l515_node.py", name="l515_node", output="screen",
                           parameters=[{"frame_id": "camera_depth_optical_frame",
                                        "max_rate": float(arg("camera_rate")),
                                        "swap_rb": arg("camera_swap_rb") == "true",
                                        "l515_path": arg("l515_path"),
                                        "fake_dir": arg("fake_camera_dir")}]))
    if camera != "none" and arg("camera_viz") == "true":
        extras.append(Node(package=PKG, executable="camera_viz.py", name="camera_viz", output="screen",
                           parameters=[{"use_sim_time": sim}]))
    if show_world:
        extras.append(Node(package=PKG, executable="world_markers.py", name="world_markers", output="screen",
                           parameters=[{"world_file": os.path.join(pkg_share, "worlds", "gungnir_world.sdf"),
                                        "use_sim_time": sim}]))

    if not sim:  # mock or real: FANUC's controller stack
        control_node = Node(
            package="controller_manager", executable="ros2_control_node", output="both",
            parameters=[moveit_config.robot_description,
                        ParameterFile(arg("ros2_control_config"), allow_substs=True)])
        nodes = [rsp, control_node, jsb, jtc]
        # FANUC's mock macro lacks an interface their gpio controller needs, so mock uses the stub.
        nodes.append(_spawner("fanuc_gpio_controller") if mode == "real" else gripper_stub)
        return nodes + extras + [move_group, rviz]

    # ---- sim ----
    # Gazebo server only; software rendering is needed for the depth camera on this machine and
    # applies to the Gazebo processes only.
    plugin_path = _join(os.environ.get("IGN_GAZEBO_SYSTEM_PLUGIN_PATH", ""),
                        os.path.join(get_package_prefix("gz_ros2_control"), "lib"), "/opt/ros/humble/lib")
    resource_path = _join(os.environ.get("IGN_GAZEBO_RESOURCE_PATH", ""), os.path.dirname(pkg_share))
    gazebo = ExecuteProcess(
        cmd=["ign", "gazebo", "-s", "-r", "-v", "3", os.path.join(pkg_share, "worlds", "gungnir_world.sdf")],
        additional_env={"LIBGL_ALWAYS_SOFTWARE": "1", "IGN_GAZEBO_RESOURCE_PATH": resource_path,
                        "IGN_GAZEBO_SYSTEM_PLUGIN_PATH": plugin_path},
        output="screen")
    # Optional Gazebo GUI as a separate client, so a GUI crash cannot take the server down.
    gui = TimerAction(period=8.0, actions=[ExecuteProcess(
        cmd=["ign", "gazebo", "-g", "--render-engine-gui", "ogre"],
        additional_env={"LIBGL_ALWAYS_SOFTWARE": "1", "IGN_GAZEBO_RESOURCE_PATH": resource_path},
        output="screen", condition=IfCondition(arg("gazebo_gui")))])
    spawn = Node(package="ros_gz_sim", executable="create", output="screen",
                 arguments=["-topic", "robot_description", "-name", arg("robot_model")])
    bridge_args = ["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"]
    if camera == "sim":
        bridge_args += ["/sim_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
                        "/sim_camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
                        "/sim_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"]
    bridge = Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen", arguments=bridge_args)
    # Start MoveIt/RViz only once the controllers exist (Gazebo takes a while to come up).
    # The spawner process ends when it has loaded, configured and started the controller. Its exit code says
    # whether that worked: 0 = yes. The old code started MoveIt on ANY exit, so after a failed spawner MoveIt
    # ran with no controllers and every move ended in "Unable to identify any set of controllers" (error -4).
    def start_moveit_if_controllers_ok(event, context):
        # event.returncode is the exit code of the joint_trajectory_controller spawner process.
        if event.returncode == 0:
            return [move_group, rviz]
        # Failure: say what probably happened and stop everything instead of running a robot that cannot move.
        return [LogInfo(msg=(
                    "[gungnir] The joint_trajectory_controller spawner FAILED (exit code "
                    f"{event.returncode}), so the robot has no working controller. The usual cause is a "
                    "Gazebo (ign gazebo) process left over from an earlier launch that still holds the "
                    "controllers. Stop this launch, run: pkill -9 -f gazebo ; ros2 daemon stop ; and launch again. "
                    "Look for lines mentioning 'Controller already loaded' above.")),
                EmitEvent(event=Shutdown(reason="controller spawner failed"))]

    later = RegisterEventHandler(OnProcessExit(target_action=jtc, on_exit=start_moveit_if_controllers_ok))
    nodes = [gazebo, gui, rsp, bridge, spawn, gripper_stub, jsb, jtc, later]
    if camera == "sim":
        nodes.append(Node(package=PKG, executable="sim_camera_adapter.py", output="screen",
                          parameters=[os.path.join(pkg_share, "config", "sim_camera_noise.yaml"),
                                      {"use_sim_time": True, "noise_enabled": arg("depth_noise") == "true"}]))
    return nodes + extras


def generate_launch_description():
    args = [
        DeclareLaunchArgument("mode", default_value="mock", choices=["mock", "sim", "real"],
                              description="Arm backend."),
        DeclareLaunchArgument("camera", default_value="auto", choices=["auto", "sim", "real", "none"],
                              description="Camera source. auto: sim for mode:=sim, real for mode:=real, none for mock."),
        DeclareLaunchArgument("camera_viz", default_value="true", choices=["true", "false"],
                              description="Publish the point cloud and colour-coded depth image."),
        DeclareLaunchArgument("show_world", default_value="auto", choices=["auto", "true", "false"],
                              description="Draw the sim world's objects in RViz (auto: only in mode:=sim)."),
        DeclareLaunchArgument("gazebo_gui", default_value="false", choices=["true", "false"],
                              description="Also start the Gazebo GUI (software rendered; may be slow or crash)."),
        DeclareLaunchArgument("depth_noise", default_value="true", choices=["true", "false"],
                              description="Add a depth-noise model to the simulated camera."),
        DeclareLaunchArgument("camera_rate", default_value="10.0", description="Max publish rate of the real camera (Hz)."),
        DeclareLaunchArgument("camera_swap_rb", default_value="false", choices=["true", "false"],
                              description="Swap red/blue of the real camera (as swap_rb in CameraFeed)."),
        DeclareLaunchArgument("l515_path", default_value="",
                              description="Folder containing l515.py, if it is not importable by python3."),
        DeclareLaunchArgument("fake_camera_dir", default_value="",
                              description="With camera:=real: replay a saved capture instead of opening the camera."),
        DeclareLaunchArgument("robot_model", default_value="crx30ia",
                              description="CRX model; needs config/robots/<model>_urdf_macro.xacro (see scripts/adopt_robot.py)."),
        DeclareLaunchArgument("robot_ip", default_value="192.168.1.100"),
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        DeclareLaunchArgument("motion_control", default_value="1"),
        DeclareLaunchArgument("encoding", default_value="UTF-8"),
        # Substituted into FANUC's controller yaml via $(var ...)
        DeclareLaunchArgument("prefix", default_value=""),
        DeclareLaunchArgument("gpio_config_package", default_value="fanuc_hardware_interface"),
        DeclareLaunchArgument("gpio_config_path", default_value="config/example_gpio_config.yaml"),
        DeclareLaunchArgument(
            "ros2_control_config",
            default_value=PathJoinSubstitution(
                [FindPackageShare("fanuc_hardware_interface"), "config", "ros2_controllers.yaml"])),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=launch_setup)])