# A3 humanoid simulation – C++ version

This folder is a C++ port of the Python project, with the same behaviour, options, keys and log files. It uses MuJoCo's C library (3.15.0, the same version as the Python package), OpenCV, GLFW and yaml-cpp.

## What was ported

| Python | C++ | Notes |
| --- | --- | --- |
| `robot/interface.py`, `robot/mock_robot.py`, `robot/kinematic_sim_robot.py` | `include/a3/robot.hpp`, `src/robot.cpp` | `RobotInterface`, `MockRobot`, `KinematicSimRobot` (hills and steps included) |
| `safety/supervisor.py` | `include/a3/safety.hpp`, `src/safety.cpp` | Same 5 rules |
| `simulation/scene.py` | `include/a3/scene.hpp`, `src/scene.cpp` | Built with MuJoCo's C++ spec API (`mjSpec`); AGIBOT's files are not changed |
| `simulation/course.py` | `include/a3/course.hpp`, `src/course.cpp` | Same hill, steps, road, people, animals, vehicles |
| `perception/detector.py` | `include/a3/detector.hpp`, `src/detector.cpp` | Colour check + YOLO11n via OpenCV DNN (ONNX file) |
| `perception/spatial.py` | `include/a3/spatial.hpp`, `src/spatial.cpp` | Pixel + depth to room position |
| `navigation/obstacles.py`, `tracker.py`, `grid_planner.py` | `src/obstacles.cpp`, `src/tracker.cpp`, `src/grid_planner.cpp` | Same A* (same tie-breaking, identical paths) |
| `manipulation/arm.py` | `include/a3/arm.hpp`, `src/arm.cpp` | Same IK and magic grasp |
| `mujoco.viewer`, `mujoco.Renderer` | `include/a3/render.hpp`, `src/render.cpp` | 3D window (mouse rotate/zoom) and off-screen camera pictures |
| `task_demo.py` | `apps/task_demo.cpp` | Instructions, field mission, city course |
| `pick_demo.py` | `apps/pick_demo.cpp` | |
| `play_sim.py` | `apps/play_sim.cpp` | Same keys, same logs |
| `mission_sim.py` | `apps/mission_sim.cpp` | Same route files, same logs |
| `plot_log.py` | `apps/plot_log.cpp` | Picture drawn with OpenCV instead of matplotlib |
| `camera_view.py` | `apps/camera_view.cpp` | |
| `demo_mock_walk.py` | `apps/demo_mock_walk.cpp` | |
| `tools/detection_test.py`, `finder_test.py`, `export_joint_limits.py` | `apps/detection_test.cpp`, `finder_test.cpp`, `export_joint_limits.cpp` | |
| `tests/test_safety.py` | `tests/test_safety.cpp` | The same 5 tests |
| `teleop.py` (ROS 2) | `ros2_ws/src/a3_teleop_cpp` | ROS 2 C++ package (rclcpp), publishes `/cmd_vel` |

## How it was checked against Python

The same steps were run in both versions on the same machine:

| Check | Result |
| --- | --- |
| Scene size, robot height, camera position, 100 walking steps | Identical to 6 decimals |
| Arm IK (steps needed, final palm position) | Identical |
| Terrain heights, A* path | Identical |
| Head camera picture | Pixel-for-pixel identical |
| Depth picture | Identical (same reversed-Z depth handling as Python's Renderer) |
| Bottle detection boxes, 3D bottle position | Identical |
| `task_demo` instruction run, without and with camera | Every output line identical |
| `pick_demo` (3 cases, including searching and forced hand) | Every output line identical |
| `mission_sim` route | Log files byte-identical |
| `demo_mock_walk`, 5 safety tests | Identical / 5 passed |

## Differences from the Python version

- **Random layouts:** Python uses NumPy's random generator, C++ uses `std::mt19937`. With the same `--seed`, stones (field mission, city course), rest pauses of people and animals, and `finder_test` layouts land in different random places. The rules, sizes, zones and counts are the same.
- **3D window:** Python's `mujoco.viewer` has many built-in menus and shortcuts; the C++ window is simpler (mouse: left = rotate, right = move, scroll = zoom; our keys work the same). It also does not keep drawing while a program waits for typed input.
- **YOLO:** runs through OpenCV DNN on the CPU (Python used PyTorch on the GPU). Without the ONNX file, or if OpenCV cannot run it, the colour check is used, exactly like Python without YOLO.
- **Speed:** the robot logic is about 10× faster (the 85 m course without camera: about 5 s instead of 46 s). With the camera, rendering dominates, as in Python.
- **ROS 2 teleop:** written and its logic tested, but not compiled against real ROS 2 here; build it with `colcon` (below).

## 1. Install (once)

On Ubuntu 24.04:

```bash
sudo apt install -y build-essential cmake libglfw3-dev libopencv-dev libyaml-cpp-dev
```

MuJoCo C library, version 3.15.0 (same as the Python package):

```bash
cd ~
wget https://github.com/google-deepmind/mujoco/releases/download/3.15.0/mujoco-3.15.0-linux-x86_64.tar.gz
tar -xzf mujoco-3.15.0-linux-x86_64.tar.gz
```

On a Jetson (ARM), use `mujoco-3.15.0-linux-aarch64.tar.gz` instead.

## 2. Build

```bash
cd ~/HDD/Sudarshan_Humanoid/cpp
mkdir -p build && cd build
cmake .. -DMUJOCO_DIR=$HOME/mujoco-3.15.0
make -j$(nproc)
```

All programs are created in `cpp/build/`.

## 3. YOLO model for C++ (optional)

Convert the YOLO11n model to ONNX once, using the Python environment:

```bash
cd ~/HDD/Sudarshan_Humanoid
source ~/robot_venv/bin/activate
mkdir -p models
yolo export model=yolo11n.pt format=onnx opset=12 imgsz=640
mv yolo11n.onnx models/
```

Without this file the C++ programs print `YOLO not available ...` and use the colour check.

## 4. Run

Always run from the project folder (the programs read `config/robot.yaml` and the A3 model from there):

```bash
cd ~/HDD/Sudarshan_Humanoid
./cpp/build/test_safety
./cpp/build/task_demo --course --speedup 5 --say "pick the red bottle from the table and keep it on the empty table"
```

| I want to… | C++ command |
| --- | --- |
| Safety tests | `./cpp/build/test_safety` |
| Mock walk demo | `./cpp/build/demo_mock_walk` |
| Drive with the keyboard | `./cpp/build/play_sim --no-eyes` |
| Drive with the eyes and detection | `./cpp/build/play_sim --objects --detect --target "blue bottle"` |
| Camera + overview window | `./cpp/build/camera_view` |
| Scripted route | `./cpp/build/mission_sim --speedup 5` |
| Path picture of the last run | `./cpp/build/plot_log` |
| Pick up and put back a bottle | `./cpp/build/pick_demo --target blue_bottle` |
| Give an instruction | `./cpp/build/task_demo --say "move the yellow bottle to table 2"` |
| Several instructions | `./cpp/build/task_demo` |
| 70 m field mission | `./cpp/build/task_demo --start-distance 70 --speedup 5 --say "..."` |
| 85 m city course | `./cpp/build/task_demo --course --speedup 5 --say "..."` |
| Quick test, no windows | add `--no-viewer --no-vision` |
| Detection rate on saved pictures | `./cpp/build/detection_test --target "red bottle"` |
| Day 14 measurement | `./cpp/build/finder_test` |
| Export joint limits | `./cpp/build/export_joint_limits` |

All options are the same as in Python (`--help` shows them).

**Without a screen** (SSH, headless Jetson): programs with the camera still need an OpenGL context. Use `--no-viewer` and start them with a virtual screen:

```bash
sudo apt install -y xvfb
xvfb-run -a ./cpp/build/task_demo --no-viewer --course --say "..."
```

## 5. ROS 2 teleop in C++

```bash
cd ~/HDD/Sudarshan_Humanoid/ros2_ws
colcon build --packages-select a3_teleop_cpp
source install/setup.bash
cd ..
ros2 run a3_teleop_cpp teleop
```

Keys: W/S faster/slower, A/D turn, Q stop, Space emergency stop, R reset, X exit. It publishes the allowed velocity on `/cmd_vel`, so `ros2 bag record /cmd_vel` works as before.
