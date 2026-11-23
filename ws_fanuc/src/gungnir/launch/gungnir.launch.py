"""
AI use: this launch file was written by the project's author (with earlier AI assistance that was not recorded in
this file). Claude (Anthropic; Claude Sonnet 5.5 via claude.ai), November 2026, changed ONLY the handler that
starts MoveIt and RViz in simulation mode (see "later" below, and the new imports EmitEvent, LogInfo and
Shutdown): it now starts them only if the joint_trajectory_controller spawner finished successfully, and
otherwise prints an explanation and stops the whole launch. Claude checked that the edited file compiles
(py_compile) and that the rest of the file is unchanged (a text diff). Claude has NOT run it with ROS 2,
Gazebo or MoveIt, because those are not installed where Claude works.
Claude (Anthropic; Claude Opus 5.5 via claude.ai), November 2026, made two more changes at the author's request:
  1. camera:=real now finds the L515 wrapper by itself. When l515_path:= is empty, the launch looks for
     IntelRealsense_Driver/python/l515.py in the project folder (found from this file's real location and from
     the package's install prefix), and passes that folder to l515_node. If the wrapper cannot be found at all,
     the launch prints what to do and does not start l515_node, instead of the node dying with exit code 1.
     New: L515_RELATIVE_DIR, _project_dirs(), _resolve_l515_path().
  2. mode:=sim no longer needs `pkill -9 -f gazebo` between runs. Before Gazebo starts, the launch stops any
     Gazebo (ign gazebo / gz sim) processes of this user that are still running from an earlier launch (they
     hold the joint controllers, so the spawner failed and MoveIt gave error -4), and when this launch shuts
     down it stops the Gazebo processes again so none are left behind. New launch argument clean_gazebo
     (default true) and the functions _gazebo_pids(), _stop_pids(), _clean_gazebo().
Claude checked the file with py_compile and ran the new helper functions on a copy of the repository folder tree
and against stand-in "ign gazebo" processes (found, stopped with SIGTERM, SIGKILL fallback, own process and other
programs left alone). The file has NOT been run with ROS 2, Gazebo or the camera.
Reviewed and tested by: not yet reviewed or tested by the author.
"""
import importlib.util  # checks whether python3 can already import the l515 wrapper (no camera is opened)
import os
import signal  # SIGTERM / SIGKILL for stopping left-over Gazebo processes
import time  # short waits while left-over Gazebo processes exit

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess, LogInfo, OpaqueFunction,
                            RegisterEventHandler, TimerAction)
from launch.events import Shutdown  # lets the launch stop itself when a start-up step failed
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit, OnShutdown  # OnShutdown: clean up Gazebo when the launch stops
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


# ======================================================================================================
# Finding the L515 wrapper (camera:=real)
# ======================================================================================================
# Where the wrapper lives inside the project (the repository root, e.g. ~/Gungnir). setup.py builds the native
# library into IntelRealsense_Driver/build/lib, and l515.py finds it there relative to its own location, so only
# this folder has to be put on python3's import path.
L515_RELATIVE_DIR = os.path.join("IntelRealsense_Driver", "python")


def _project_dirs():
    """Candidate project (repository root) folders, nearest first.

    The launch file can be installed in two ways, so the project folder is found in two ways (the same idea as
    _workspace_script_dirs() in gungnir.full.launch.py):
      * colcon build --symlink-install: this file is a link into <project>/ws_fanuc/src/gungnir/launch, and
        os.path.realpath() follows the link back into the source tree.
      * plain colcon build: this file is a copy in <project>/ws_fanuc/install/..., and the package's install
        prefix leads back to the workspace folder.
    From each starting point every parent folder is a candidate, up to and including the repository root (the
    first folder that contains .git), and never more than eight levels.
    """
    starts = [os.path.dirname(os.path.realpath(__file__))]
    try:
        # <workspace>/install/gungnir (isolated layout) or <workspace>/install (merged layout).
        starts.append(os.path.normpath(get_package_prefix(PKG)))
    except Exception:  # noqa: BLE001  (package not found: only the first route is used)
        pass
    folders = []
    for start in starts:
        folder = start
        for _ in range(8):
            folders.append(folder)
            # A .git entry marks the repository root: nothing above it belongs to the project.
            if os.path.exists(os.path.join(folder, ".git")):
                break
            parent = os.path.dirname(folder)
            if parent == folder:  # top of the file system
                break
            folder = parent
    # Keep each folder once, in order (both routes usually reach the same folders).
    return list(dict.fromkeys(folders))


def _resolve_l515_path(explicit):
    """Decide which folder l515_node should add to its import path.

    Returns (path, problem):
      * path: the folder to pass as the node's l515_path parameter ("" = python3 can import l515 by itself).
      * problem: None if the camera node can start, otherwise a message saying what is wrong and how to fix it.
    """
    # 1. l515_path:= given on the command line: use it, but say so clearly if it does not hold l515.py.
    if explicit:
        folder = os.path.abspath(os.path.expanduser(explicit))
        if os.path.isfile(os.path.join(folder, "l515.py")):
            return folder, None
        return folder, (f"l515_path:={explicit} does not contain l515.py (looked for {folder}/l515.py). "
                        f"Pass the folder that holds l515.py, normally <project>/{L515_RELATIVE_DIR}.")
    # 2. The project's own copy of the wrapper (built by setup.py).
    for project in _project_dirs():
        folder = os.path.join(project, L515_RELATIVE_DIR)
        if os.path.isfile(os.path.join(folder, "l515.py")):
            return folder, None
    # 3. Installed for python3 some other way (pip install for the system python, PYTHONPATH, ...). The launch
    #    runs on the same system python3 as l515_node, so this check is valid for the node. find_spec() only
    #    locates the module: nothing is imported and the camera is not opened.
    if importlib.util.find_spec("l515") is not None:
        return "", None
    searched = "\n    ".join(os.path.join(p, L515_RELATIVE_DIR, "l515.py") for p in _project_dirs())
    return "", ("Cannot find the L515 wrapper (l515.py), so l515_node is NOT started and there will be no camera "
                "images. Run setup.py in the project folder to build IntelRealsense_Driver, or pass "
                "l515_path:=/path/to/IntelRealsense_Driver/python. Looked for:\n    " + searched)


# ======================================================================================================
# Left-over Gazebo processes (mode:=sim)
# ======================================================================================================
# In simulation the joint controllers are loaded INSIDE the Gazebo server (gz_ros2_control). Stopping a launch
# with Ctrl-C does not always stop that server: `ign gazebo` is a wrapper script, and a server process it started
# can outlive it. The left-over server keeps its controller_manager, so the next launch's spawner fails
# ("Controller already loaded") and MoveIt then reports error -4. The functions below find and stop such
# processes, so a new launch always starts with a clean Gazebo.
GAZEBO_LAUNCHERS = ("ign", "gz")        # command-line tools that start Gazebo (Fortress: ign, newer: gz)
GAZEBO_SUBCOMMANDS = ("gazebo", "sim")  # their subcommands that run a simulation (ign gazebo / gz sim)


def _gazebo_pids():
    """Process ids of this user's Gazebo processes (ign gazebo ... / gz sim ...), excluding this process.

    Reads /proc directly (Linux only; no extra Python package needed). A process counts as Gazebo when one of its
    command-line words is the program ign or gz (possibly as the script run by ruby, e.g. "ruby /usr/bin/ign
    gazebo ...") and a LATER word is gazebo or sim. Processes of other users are never touched.
    """
    pids = []
    own_pid, own_uid = os.getpid(), os.getuid()
    try:
        entries = os.listdir("/proc")
    except OSError:  # not Linux: nothing to clean up
        return pids
    for entry in entries:
        if not entry.isdigit() or int(entry) == own_pid:
            continue
        try:
            # Only this user's processes (the owner of /proc/<pid> is the process owner).
            if os.stat(f"/proc/{entry}").st_uid != own_uid:
                continue
            with open(f"/proc/{entry}/cmdline", "rb") as handle:
                # The command line is stored as NUL-separated words.
                words = [w.decode(errors="ignore") for w in handle.read().split(b"\0") if w]
        except OSError:  # the process exited while we looked, or is not readable
            continue
        for i, word in enumerate(words):
            if os.path.basename(word) in GAZEBO_LAUNCHERS and any(w in GAZEBO_SUBCOMMANDS for w in words[i + 1:]):
                pids.append(int(entry))
                break
    return pids


def _stop_pids(pids, grace=3.0):
    """Ask the processes to stop (SIGTERM); after `grace` seconds force the remaining ones (SIGKILL).

    Returns the list of pids that were still alive after SIGKILL (normally empty).
    """
    def alive(pid):
        try:
            os.kill(pid, 0)  # signal 0 only checks that the process exists
        except ProcessLookupError:
            return False
        except PermissionError:  # exists but is not ours (should not happen: _gazebo_pids() checks the owner)
            return True
        # A finished child that nobody has collected yet ("zombie") counts as gone: it holds no resources.
        try:
            with open(f"/proc/{pid}/stat", "r") as handle:
                return handle.read().split(")")[-1].split()[0] != "Z"
        except OSError:
            return False

    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 2.0)):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and any(alive(p) for p in pids):
            time.sleep(0.1)
        pids = [p for p in pids if alive(p)]
        if not pids:
            break
    return pids


def _clean_gazebo(when):
    """Stop all left-over Gazebo processes of this user and say so. `when` is shown in the message."""
    pids = _gazebo_pids()
    if not pids:
        return
    print(f"[gungnir] {when}: stopping {len(pids)} Gazebo process(es) {pids} so that no old Gazebo keeps the "
          "joint controllers (use clean_gazebo:=false to keep them).", flush=True)
    left = _stop_pids(pids)
    if left:
        print(f"[gungnir] WARNING: Gazebo process(es) {left} did not stop; run: pkill -9 -f gazebo", flush=True)


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
        # Work out where the L515 wrapper is. A replay (fake_camera_dir:=...) does not need the wrapper at all.
        if arg("fake_camera_dir"):
            l515_path, l515_problem = arg("l515_path"), None
        else:
            l515_path, l515_problem = _resolve_l515_path(arg("l515_path"))
        if l515_problem:
            # Do not start a node that can only die with "Cannot import l515"; explain the fix instead. The rest of
            # the stack (robot, MoveIt, RViz, client) still starts.
            extras.append(LogInfo(msg="[gungnir] ERROR: " + l515_problem))
        else:
            if l515_path and not arg("l515_path"):
                # Found automatically: show where, so a wrong copy is easy to spot.
                extras.append(LogInfo(msg=f"[gungnir] L515 wrapper: {l515_path}"))
            extras.append(Node(package=PKG, executable="l515_node.py", name="l515_node", output="screen",
                               parameters=[{"frame_id": "camera_depth_optical_frame",
                                            "max_rate": float(arg("camera_rate")),
                                            "swap_rb": arg("camera_swap_rb") == "true",
                                            "l515_path": l515_path,
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
    # Stop any Gazebo left over from an earlier launch BEFORE starting a new one: the old server still holds the
    # joint controllers (see _clean_gazebo above). This runs now, while the launch is being set up, so it is
    # finished before the new Gazebo and the controller spawners start.
    clean_gazebo = arg("clean_gazebo") == "true"
    if clean_gazebo:
        _clean_gazebo("Before starting Gazebo")
    # When this launch shuts down (Ctrl-C, or the spawner-failure handler below), stop the Gazebo processes again,
    # so nothing is left behind for the next launch. The handler runs once at the start of the shutdown.
    cleanup_on_exit = RegisterEventHandler(OnShutdown(
        on_shutdown=lambda event, context: _clean_gazebo("Launch shutting down")),
        condition=IfCondition(arg("clean_gazebo")))
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
                    "controllers. This launch now stops left-over Gazebo processes itself (unless "
                    "clean_gazebo:=false), so first just launch again; if it fails again, run: "
                    "pkill -9 -f gazebo ; ros2 daemon stop ; and launch again. "
                    "Look for lines mentioning 'Controller already loaded' above.")),
                EmitEvent(event=Shutdown(reason="controller spawner failed"))]

    later = RegisterEventHandler(OnProcessExit(target_action=jtc, on_exit=start_moveit_if_controllers_ok))
    nodes = [cleanup_on_exit, gazebo, gui, rsp, bridge, spawn, gripper_stub, jsb, jtc, later]
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
                              description="Folder containing l515.py. Empty: <project>/IntelRealsense_Driver/python "
                                          "(found automatically), else an l515 that python3 can already import."),
        DeclareLaunchArgument("clean_gazebo", default_value="true", choices=["true", "false"],
                              description="mode:=sim: stop this user's left-over Gazebo processes before starting "
                                          "and when the launch shuts down (they hold the joint controllers)."),
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