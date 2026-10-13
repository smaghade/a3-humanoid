# A3 Humanoid Project - Progress Report

Generated: 2026-10-13 15:37  |  Author(s): smaghade

## Summary

| Item | Value |
| --- | --- |
| First commit | 2026-10-12 12:13 |
| Latest commit | 2026-10-13 15:34 |
| Number of commits (saves) | 14 |
| Files in the project | 54 |
| Python files | 36 |
| Lines of Python code | 2020 |
| Milestones (tags) | none yet |

## Timeline: what was added in each step

### 2026-10-12 12:13 - Day 1-2: environment setup, talker/listener, rosbag test

Commit `0bb7bd7`

Added:
- `.gitignore`
- `requirements.txt`
- `ros2_ws/src/day2_basics/day2_basics/__init__.py`
- `ros2_ws/src/day2_basics/day2_basics/listener.py`
- `ros2_ws/src/day2_basics/day2_basics/talker.py`
- `ros2_ws/src/day2_basics/package.xml`
- `ros2_ws/src/day2_basics/resource/day2_basics`
- `ros2_ws/src/day2_basics/setup.cfg`
- `ros2_ws/src/day2_basics/setup.py`
- `ros2_ws/src/day2_basics/test/test_copyright.py`
- `ros2_ws/src/day2_basics/test/test_flake8.py`
- `ros2_ws/src/day2_basics/test/test_pep257.py`
- `ros2_ws/src/my_first_package/my_first_package/__init__.py`
- `ros2_ws/src/my_first_package/my_first_package/talker.py`
- `ros2_ws/src/my_first_package/package.xml`
- `ros2_ws/src/my_first_package/resource/my_first_package`
- `ros2_ws/src/my_first_package/setup.cfg`
- `ros2_ws/src/my_first_package/setup.py`
- `ros2_ws/src/my_first_package/test/test_copyright.py`
- `ros2_ws/src/my_first_package/test/test_flake8.py`
- `ros2_ws/src/my_first_package/test/test_pep257.py`

### 2026-10-12 14:32 - Day 6: A3 joint limits exported to config/robot.yaml

Commit `8c77e0c`

Added:
- `config/robot.yaml`
- `tools/export_joint_limits.py`

Changed:
- `.gitignore`

### 2026-10-12 14:35 - Day 5: RobotInterface, MockRobot, safety supervisor + tests

Commit `0443ccb`

Added:
- `demo_mock_walk.py`
- `robot/__init__.py`
- `robot/__pycache__/__init__.cpython-312.pyc`
- `robot/__pycache__/interface.cpython-312.pyc`
- `robot/__pycache__/mock_robot.cpython-312.pyc`
- `robot/interface.py`
- `robot/mock_robot.py`
- `safety/__init__.py`
- `safety/__pycache__/__init__.cpython-312.pyc`
- `safety/__pycache__/supervisor.cpython-312.pyc`
- `safety/supervisor.py`
- `tests/__pycache__/test_safety.cpython-312-pytest-7.4.4.pyc`
- `tests/test_safety.py`

### 2026-10-12 14:36 - Ignore Python cache files

Commit `3468122`

Changed:
- `.gitignore`

### 2026-10-12 15:31 - Add movement logging (events + path CSV) and path plot

Commit `a4cd2bb`

Added:
- `play_sim.py`
- `plot_log.py`

Changed:
- `.gitignore`

### 2026-10-12 18:21 - play_sim: Space = pause, R = resume in MuJoCo window and terminal

Commit `16555f8`

Changed:
- `play_sim.py`

### 2026-10-12 18:23 - Scripted missions, preview robot, teleop, docs pictures

Commit `f35b58d`

Added:
- `docs/videos/001_walk_front_slow_metrics.json`
- `docs/videos/walk_front_slow_metrics.json`
- `mission_sim.py`
- `missions/demo_route.yaml`
- `robot/kinematic_sim_robot.py`
- `teleop.py`

### 2026-10-12 18:43 - Save full codebase: missions, preview robot, teleop, docs

Commit `98f83b0`

Changed:
- `.gitignore`

### 2026-10-13 12:47 - Give the A3 eyes: head camera + EYES window in play_sim

Commit `1d16126`

Added:
- `simulation/__init__.py`
- `simulation/scene.py`

Changed:
- `.gitignore`
- `play_sim.py`
- `robot/kinematic_sim_robot.py`

### 2026-10-13 13:21 - Pick-and-place demo working: walk, reach, magic grasp, lift, place back

Commit `9c5ae7d`

Added:
- `manipulation/__init__.py`
- `manipulation/arm.py`
- `pick_demo.py`

Changed:
- `robot/kinematic_sim_robot.py`

### 2026-10-13 14:41 - Days 11-12: depth + intrinsics + camera pose -> 3D bottle position; pick demo finds bottle with its eyes

Commit `8314e58`

Added:
- `perception/spatial.py`

Changed:
- `pick_demo.py`

### 2026-10-13 14:47 - Day X: ...

Commit `7601ff7`

Added:
- `MJDATA.TXT`
- `MUJOCO_LOG.TXT`
- `camera_view.py`
- `config/camera.yaml`
- `mjmodel.xml`
- `perception/__init__.py`
- `perception/detector.py`
- `screenshot.png`
- `tools/detection_test.py`
- `yolo11n.pt`

Changed:
- `play_sim.py`

### 2026-10-13 15:01 - Add yellow bottle to scene and yellow colour to detector

Commit `17191df`

Changed:
- `perception/detector.py`
- `simulation/scene.py`

### 2026-10-13 15:34 - Fix: floor no longer seen as blue; only requested colour; height check; correct/selectable hand

Commit `1cef14c`

Changed:
- `perception/detector.py`
- `pick_demo.py`

## All files in the project now

```text
config/                   2 files
docs/                     2 files
manipulation/             2 files
missions/                 1 files
perception/               3 files
robot/                    4 files
ros2_ws/                  19 files
safety/                   2 files
simulation/               2 files
tests/                    1 files
tools/                    2 files
top-level-files           14 files
```

## Total change from the first commit to now

```text
 34 files changed, 5454 insertions(+)
```
