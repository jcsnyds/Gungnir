"""gungnir.full.launch.py: the complete Gungnir stack = gungnir.launch.py + the Gungnir client web UI.

    ros2 launch gungnir gungnir.full.launch.py mode:=sim camera:=sim grasp_server:=192.168.1.50
    ros2 launch gungnir gungnir.full.launch.py mode:=real grasp_server:=192.168.1.50 launch_rviz:=false
    ros2 launch gungnir gungnir.full.launch.py mode:=sim client_args:="--min-score 0.3 --move-speed 0.1"
    ros2 launch gungnir gungnir.full.launch.py mode:=sim launch_client:=false     # same as gungnir.launch.py

What this file does:
  1. Includes gungnir.launch.py unchanged (arm backend, MoveIt, RViz, camera, Gazebo, ...). Every argument of
     gungnir.launch.py (mode, camera, robot_model, launch_rviz, ...) keeps working: arguments given on the
     command line are visible to included launch files, so nothing has to be forwarded here.
  2. Starts Gungnir_Client.py (browser UI: live camera, Capture & run, best grasp, approve-and-move) as a
     separate operating-system process, then prints the web address. Open http://localhost:8000.

Why the client is an ExecuteProcess and not a launch_ros Node: the client needs the project's Python virtual
environment (torch, ultralytics, flask, plotly ...), which the system python3 that a Node would use does not
have. ExecuteProcess lets us choose the interpreter explicitly. The client is still a ROS 2 node (it
subscribes to /camera/depth/image_rect_raw, /camera/color/image_raw and /camera/color/camera_info), and the
planner it launches after the user approves a move talks to the move_group started by gungnir.launch.py.
Because the interpreter is chosen explicitly (the .venv/bin/python path), the virtual environment does NOT have
to be activated in the terminal that runs `ros2 launch`.

Things that must be true for the client to be useful (the launch file cannot check them):
  * A camera is running: camera:=sim (with mode:=sim) or camera:=real. With the default mode:=mock there is
    NO camera, so the client's Capture button stays disabled; this file prints a warning in that case.
  * The camera publishes on the three topic names above (they are hard-coded in the client).
  * ContactGraspnet_Server is already running on the GPU computer (grasp_server:=<its address>).

AI use: This file was originally written by Claude (Anthropic; Claude Sonnet 5.5 via claude.ai) in November 2026, based on
the author's gungnir.launch.py and Gungnir_Client.py and the author's requirement to start the client together
with the simulation. Claude checked only that the file compiles (py_compile); it has NOT been run with ROS 2,
Gazebo, the real robot or the camera. The author is responsible for reviewing and testing it.
After the author's first test run showed the client rejecting --move-frame (the client on disk was an older
version without the grasp-approval options), Claude changed the move options to be passed only when set, and
added a check that reports an older client clearly. That edit was again only compile-checked.
ChatGPT (OpenAI) update, November 2026: configured the WSL source-tree path "/home/rpoli/Gungnir/ws_fanuc/src/gungnir/scripts/Gungnir_Client_ROS2.py" as the first automatic client-script discovery candidate. Existing client_script:= and GUNGNIR_CLIENT_SCRIPT overrides remain available, and the normal package/current-directory fallbacks are retained. This update has not been runtime-tested with ROS 2.
Claude (Anthropic; Claude Sonnet 5.5 via claude.ai) update, November 2026, after the author's run printed
"Gungnir client script not found". Three changes: (1) the WSL path above starts with "~", which os.path.isfile()
does not expand, so it could never match; it now goes through os.path.expanduser(). (2) New function
_workspace_script_dirs() finds the client in the source tree (src/gungnir/scripts) from the launch file's real
path (symlink install) or from the package's install prefix (plain colcon build); those candidates now come
BEFORE the WSL path and before the installed copies, so the order described in the ChatGPT paragraph above no
longer holds. (3) New function _find_venv_python() looks for .venv/ or venv/ in the client script's folder and
its parent folders up to the repository root, because setup.py creates .venv/ in the repository root while the
script is four folders deeper; without this the client would start with the system python3 and fail to import
torch. The launch now also logs which script and interpreter it chose. Claude checked this file with py_compile
and by running it against stand-in launch modules and a temporary folder tree (symlink install, copied install,
"~" path, explicit script, missing script, missing venv). It has NOT been run with ROS 2, Gazebo, the robot or
the camera.
Reviewed and tested by: not yet reviewed or tested by the author.
"""

import os
import shlex  # splits the client_args string into separate command-line arguments (respects quotes)
import sys

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, LogInfo,
                            OpaqueFunction, RegisterEventHandler, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

# Name of the ROS 2 package that contains gungnir.launch.py (the same constant as in gungnir.launch.py).
PKG = "gungnir"
# File names under which the client may be saved. The second one is the name used in the author's project.
CLIENT_FILE_NAMES = ("Gungnir_Client_ROS2.py", "Gungnir_Client.py")
# Preferred source-tree location on the user's WSL Ubuntu installation. It starts with "~", so it MUST go through
# os.path.expanduser() before it is tested: os.path.isfile("~/x") is False because no shell expands the "~" here.
# If this path is absent on another machine, normal discovery continues below.
DEFAULT_CLIENT_SCRIPT = "~/Gungnir/ws_fanuc/src/gungnir/scripts/Gungnir_Client_ROS2.py"


def _first_existing_file(paths):
    """Return the first path in `paths` that is an existing file, or None if none of them is."""
    for path in paths:
        if path and os.path.isfile(path):
            return path
    return None


def _client_has_option(script, option):
    """True if the client script's text mentions the command-line option (e.g. "--move-frame").

    A plain text search is enough here: the client defines its options with argparse, so an option it
    supports appears in its source as a quoted string. This lets the launch file tell an older client
    (before the grasp-approval / robot-move feature) from a current one without importing it.
    """
    try:
        with open(script, "r", encoding="utf-8", errors="ignore") as handle:
            return option in handle.read()
    except OSError:
        return False


def _workspace_script_dirs():
    """Folders in the SOURCE tree (<workspace>/src/gungnir/scripts) where the client script may live.

    The client script is not necessarily installed into the package by colcon (it is not, in the author's
    workspace), so looking only in install/gungnir/lib and install/gungnir/share finds nothing. The source
    tree is found in two ways, because colcon can install this launch file in two ways:

      * `colcon build --symlink-install`: the installed launch file is a symbolic link into the source tree.
        os.path.realpath() follows the link, so the REAL location of this file is
        <workspace>/src/gungnir/launch/ and its sibling folder is <workspace>/src/gungnir/scripts/.
      * plain `colcon build`: the installed launch file is a COPY that links back to nothing. But the package
        is installed at <workspace>/install/gungnir, so the workspace folder can be worked out from the
        package's install prefix, and the scripts folder is <workspace>/src/gungnir/scripts/.
    """
    folders = []

    # Route 1: the real (symlink-resolved) folder of this launch file, then its sibling "scripts" folder.
    launch_dir = os.path.dirname(os.path.realpath(__file__))
    folders.append(os.path.join(os.path.dirname(launch_dir), "scripts"))

    # Route 2: derive the workspace folder from where colcon installed the package.
    prefix = os.path.normpath(get_package_prefix(PKG))
    if os.path.basename(prefix) == PKG:
        # Default "isolated" layout: the prefix is <workspace>/install/gungnir.
        workspace = os.path.dirname(os.path.dirname(prefix))
    else:
        # "Merged" layout (colcon build --merge-install): the prefix is <workspace>/install.
        workspace = os.path.dirname(prefix)
    folders.append(os.path.join(workspace, "src", PKG, "scripts"))

    # Both routes usually give the same folder; keep each once, in order.
    return list(dict.fromkeys(folders))


def _find_venv_python(start_dir):
    """Find the project's virtual-environment interpreter (.venv/bin/python or venv/bin/python).

    Starts in `start_dir` (the client script's folder) and walks UP through its parent folders. This is needed
    because setup.py creates .venv/ in the repository root (~/Gungnir) while the client script lives four
    folders deeper (ws_fanuc/src/gungnir/scripts), so a search limited to the script's own folder never finds it.

    The walk stops after the repository root (the first folder that contains .git) so that an unrelated
    venv further up (for example ~/venv) is not picked up by accident, and never climbs more than six levels.
    Returns the path as a string, or None if there is no virtual environment.

    The returned path is deliberately NOT resolved with realpath: a venv's python is a symlink to the system
    python, and only the unresolved path lets Python find the venv's pyvenv.cfg and therefore its packages.
    """
    folder = os.path.abspath(start_dir)
    for _ in range(7):  # the starting folder plus up to six parents
        for name in (".venv", "venv"):
            candidate = os.path.join(folder, name, "bin", "python")
            # isfile() follows the symlink, so a working venv python counts as a file.
            if os.path.isfile(candidate):
                return candidate
        # A .git entry marks the repository root: this was the last folder that is part of the project.
        if os.path.exists(os.path.join(folder, ".git")):
            break
        parent = os.path.dirname(folder)
        if parent == folder:  # reached the top of the file system
            break
        folder = parent
    return None


def _base_launch_file():
    """Locate gungnir.launch.py.

    The sibling of this file is preferred: when both launch files sit in the same folder (installed, or
    symlink-installed from the source tree), the base file next to this one is always the right one. The
    package's installed share/launch folder is the fallback.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(here, "gungnir.launch.py"),
                  os.path.join(get_package_share_directory(PKG), "launch", "gungnir.launch.py")]
    found = _first_existing_file(candidates)
    if found is None:
        raise RuntimeError("gungnir.launch.py not found. Looked in:\n  " + "\n  ".join(candidates))
    return found


def client_setup(context, *args, **kwargs):
    """Work out how to start the client and return the launch actions for it.

    This is an OpaqueFunction (like launch_setup in gungnir.launch.py) because the script and interpreter
    paths depend on launch argument values and on files on disk, which are only known at launch time.
    """
    # Shortcut: the current value of a launch argument as a plain string.
    arg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731

    # launch_client:=false turns this file back into plain gungnir.launch.py (nothing is searched or started).
    if arg("launch_client") != "true":
        return []

    messages = []  # LogInfo actions that tell the user about the choices made below

    # ---- 1. Which client script? -------------------------------------------------------------------------
    pkg_share = get_package_share_directory(PKG)
    pkg_lib = os.path.join(get_package_prefix(PKG), "lib", PKG)  # where ament installs a package's executables
    # An explicit choice (launch argument, then environment variable) must exist: never silently fall back to a
    # different file than the one the user asked for.
    explicit_script = arg("client_script") or os.environ.get("GUNGNIR_CLIENT_SCRIPT", "")
    if explicit_script:
        script = os.path.abspath(os.path.expanduser(explicit_script))
        if not os.path.isfile(script):
            raise RuntimeError(f"Client script not found: {script} (from client_script:= or $GUNGNIR_CLIENT_SCRIPT)")
    else:
        # No explicit choice: look in the usual places, in this order.
        install_dirs = (pkg_lib, pkg_share, os.path.join(pkg_share, "scripts"), os.getcwd())
        candidates = []
        # 1. The source tree the package was built from. Preferred over installed copies because the client
        #    writes its output/ folder and looks for Cartesian_Path_Planner.py next to ITSELF, which should be
        #    the source folder, not a build artefact.
        candidates += [os.path.join(folder, name) for folder in _workspace_script_dirs() for name in CLIENT_FILE_NAMES]
        # 2. The author's WSL path. expanduser() turns "~" into the home folder; without it this never matched.
        candidates.append(os.path.expanduser(DEFAULT_CLIENT_SCRIPT))
        # 3. Where colcon may have installed the script, then the current directory.
        candidates += [os.path.join(folder, name) for folder in install_dirs for name in CLIENT_FILE_NAMES]
        # Keep each path once (the routes above often overlap); this also keeps the error message short.
        candidates = list(dict.fromkeys(candidates))
        script = _first_existing_file(candidates)
        if script is None:
            raise RuntimeError("Gungnir client script not found. Pass client_script:=/path/to/Gungnir_Client.py "
                               "(or set $GUNGNIR_CLIENT_SCRIPT). Looked in:\n  " + "\n  ".join(candidates))

    # The folder holding the REAL file (realpath follows symlinks, so a symlink-installed script resolves back
    # to the source tree). The client runs from here: it resolves its own output folder and its default planner
    # script relative to this folder, and the FastSAM weights are downloaded to / looked up in it.
    project_dir = os.path.dirname(os.path.realpath(script))

    # ---- 2. Which Python interpreter? --------------------------------------------------------------------
    explicit_python = arg("client_python") or os.environ.get("GUNGNIR_PYTHON", "")
    if explicit_python:
        python = os.path.expanduser(explicit_python)
    else:
        # The project's virtual environment created by setup.py. It is in the REPOSITORY ROOT, several folders
        # above the client script, so the search walks upward (see _find_venv_python). Nothing needs to be
        # "activated": starting the interpreter by its path inside the venv is enough for Python to use the
        # venv's packages.
        python = _find_venv_python(project_dir)
        if python is None:
            # Last resort: the interpreter running this launch file (the sourced system python3). It probably
            # lacks torch/ultralytics/flask, so tell the user instead of failing mysteriously later.
            python = sys.executable
            messages.append(LogInfo(msg=(
                f"[gungnir] WARNING: no .venv/ or venv/ found in {project_dir} or the folders above it (up to "
                f"the repository root); starting the client with {python}. If it fails to import "
                "torch/ultralytics/flask, run setup.py or pass client_python:=/path/to/venv/bin/python")))
    # Always say which script and interpreter were chosen: the first thing to check when something is off.
    messages.append(LogInfo(msg=f"[gungnir] Client script: {script}\n[gungnir] Client python: {python}"))

    # ---- 3. The client's command line --------------------------------------------------------------------
    cmd = [python, script]
    # ContactGraspnet_Server address. If empty, the positional argument is omitted and the client falls back to
    # $GUNGNIR_SERVER, then to 127.0.0.1 (its own defaults). The token is never put on the command line (it would
    # be visible in `ps`); the client reads $GUNGNIR_TOKEN, which this process passes on through the environment.
    if arg("grasp_server"):
        cmd.append(arg("grasp_server"))
    cmd += ["--port", arg("client_port")]
    # Robot-move options. Only clients with the grasp-approval feature understand them; an older client exits
    # with "unrecognized arguments" if it is given one. So they are passed ONLY when the user set them.
    has_move_options = _client_has_option(script, "--move-frame")
    if (arg("move_frame") or arg("planner_script")) and not has_move_options:
        # The user asked for a move option but this client cannot take it: stop here with a clear reason
        # instead of letting the client die with an argparse usage message.
        raise RuntimeError(f"move_frame:= / planner_script:= were given, but {script} has no --move-frame option: "
                           "it is an older Gungnir client without the grasp-approval / robot-move feature. "
                           "Update the client, or remove those two arguments.")
    if not has_move_options:
        messages.append(LogInfo(msg="[gungnir] NOTE: this Gungnir client has no robot-move feature (no --move-frame "
                                    "option), so there will be no 'Proposed robot move' panel. Update "
                                    f"{script} to get it."))
    # The TF frame the grasp poses are in. It must match the frame_id the camera publishes and must exist in
    # the robot's TF tree. Empty: the client's own default, which is "camera_depth_optical_frame".
    if arg("move_frame"):
        cmd += ["--move-frame", arg("move_frame")]
    # Path of the Cartesian path planner script the client starts after the user approves a move. Empty: the
    # client's own default (Cartesian_Path_Planner.py next to the client).
    if arg("planner_script"):
        cmd += ["--planner-script", os.path.abspath(os.path.expanduser(arg("planner_script")))]
    # Anything else for the client, e.g. client_args:="--min-score 0.3 --move-speed 0.1 --approval-timeout 60".
    # (Do not add --swap-rb here if camera_swap_rb:=true already swaps the colours in l515_node.)
    cmd += shlex.split(arg("client_args"))

    # ---- 4. The process ----------------------------------------------------------------------------------
    client = ExecuteProcess(
        cmd=cmd,
        name="gungnir_client",
        # Run from the script's folder, exactly like typing `python Gungnir_Client_ROS2.py` there. In particular
        # the FastSAM weights (FastSAM-s.pt) are downloaded to / looked up in the working directory.
        cwd=project_dir,
        output="screen",
        # A pseudo-terminal makes Python flush its output line by line, so the log appears live.
        emulate_tty=True,
        additional_env={"PYTHONUNBUFFERED": "1"},
        # Ctrl-C sends SIGINT; the client then closes its ROS node and executor, which can take ~10 s. The
        # default 5 s before escalating to SIGTERM/SIGKILL would cut that cleanup short.
        sigterm_timeout="15",
        sigkill_timeout="15",
    )

    # ---- 5. Optional start delay -------------------------------------------------------------------------
    # With mode:=sim, Gazebo uses software rendering and is slow to come up; loading FastSAM at the same time
    # competes for the CPU. client_delay:=20 starts the client 20 s after launch. 0 = start immediately.
    delay = float(arg("client_delay"))
    start_client = TimerAction(period=delay, actions=[client]) if delay > 0 else client

    # ---- 6. Helpful messages -----------------------------------------------------------------------------
    # Does a camera exist? Same resolution rule as gungnir.launch.py (auto: sim->sim, real->real, otherwise none).
    # The defaults below mirror that file's defaults and are used only when the user did not pass the argument:
    # configuration values set inside the included file are not visible out here.
    mode = context.launch_configurations.get("mode", "mock")
    camera = context.launch_configurations.get("camera", "auto")
    if camera == "auto":
        camera = {"sim": "sim", "real": "real"}.get(mode, "none")
    if camera == "none":
        messages.append(LogInfo(msg=(
            f"[gungnir] WARNING: mode:={mode} has no camera (camera:=none), so the client will show 'camera "
            "starting...' and its Capture button will stay disabled. Use camera:=sim (with mode:=sim) or "
            "camera:=real.")))
    messages.append(LogInfo(msg=f"[gungnir] Starting the Gungnir client; web page: http://localhost:{arg('client_port')}"
                                " (default bind address is localhost only)"))

    # Tell the user when the client stops. The rest of the launch keeps running (a client crash must not take
    # the simulation or the robot stack down), so without this the exit is easy to miss in the log.
    on_client_exit = RegisterEventHandler(OnProcessExit(
        target_action=client,
        on_exit=[LogInfo(msg="[gungnir] The Gungnir client exited (see its output above). "
                             "The rest of the launch is still running.")]))

    return [*messages, start_client, on_client_exit]


def generate_launch_description():
    # Arguments of THIS file. Those of gungnir.launch.py (mode, camera, robot_model, ...) are declared there.
    args = [
        DeclareLaunchArgument("launch_client", default_value="true", choices=["true", "false"],
                              description="Start the Gungnir client web UI (false: behave like gungnir.launch.py)."),
        DeclareLaunchArgument("client_script", default_value="",
                              description="Path of Gungnir_Client_ROS2.py. Empty: $GUNGNIR_CLIENT_SCRIPT, then the "
                                          "source tree the package was built from (src/gungnir/scripts), the "
                                          "configured WSL path, the package lib/ and share/ folders, and the "
                                          "current directory."),
        DeclareLaunchArgument("client_python", default_value="",
                              description="Python interpreter for the client. Empty: $GUNGNIR_PYTHON, then the "
                                          ".venv/ or venv/ in the client script's folder or the project folders "
                                          "above it, then the launch's own python3."),
        DeclareLaunchArgument("grasp_server", default_value="",
                              description="ContactGraspnet_Server address, e.g. 192.168.1.50 or 192.168.1.50:8765. "
                                          "Empty: the client uses $GUNGNIR_SERVER, then 127.0.0.1."),
        DeclareLaunchArgument("client_port", default_value="8000", description="Port of the client's web page."),
        DeclareLaunchArgument("move_frame", default_value="",
                              description="TF frame the grasp poses are expressed in (the camera's frame_id). "
                                          "Empty: the client's default, camera_depth_optical_frame."),
        DeclareLaunchArgument("planner_script", default_value="",
                              description="Path of the Cartesian path planner script. Empty: the client's default "
                                          "(Cartesian_Path_Planner.py next to the client)."),
        DeclareLaunchArgument("client_args", default_value="",
                              description='Extra client options as one string, e.g. "--min-score 0.3 --move-speed 0.1".'),
        DeclareLaunchArgument("client_delay", default_value="0.0",
                              description="Seconds to wait before starting the client (e.g. 20 while Gazebo starts)."),
    ]
    # Include the existing launch file. No launch_arguments are given on purpose: arguments typed on the
    # command line (mode:=sim, camera:=sim, ...) are visible to the included file and override its defaults.
    base = IncludeLaunchDescription(PythonLaunchDescriptionSource(_base_launch_file()))
    # Order matters only for the log: the base stack is listed first, then the client.
    return LaunchDescription(args + [base, OpaqueFunction(function=client_setup)])