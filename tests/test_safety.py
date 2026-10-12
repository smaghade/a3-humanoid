from robot.mock_robot import MockRobot
from safety.supervisor import SafetySupervisor


def make():
    robot = MockRobot()
    return robot, SafetySupervisor.from_yaml(robot)


def test_walking_speed_is_limited():
    robot, safe = make()
    safe.set_velocity(5.0, 0.0, 0.0)              # ask for 5 m/s (far too fast)
    assert robot.velocity[0] <= 0.3 + 1e-9        # robot only gets 0.3 m/s


def test_turning_speed_is_limited():
    robot, safe = make()
    safe.set_velocity(0.0, 0.0, -10.0)
    assert robot.velocity[2] == -0.5


def test_emergency_stop_blocks_motion():
    robot, safe = make()
    safe.emergency_stop()
    safe.set_velocity(0.2, 0.0, 0.0)
    assert robot.velocity == (0.0, 0.0, 0.0)


def test_timeout_stops_robot():
    robot, safe = make()
    safe.set_velocity(0.2, 0.0, 0.0)
    stopped = safe.check_timeout(now=safe.last_cmd_time + 1.0)
    assert stopped and robot.velocity == (0.0, 0.0, 0.0)


def test_joint_limits_with_margin():
    _, safe = make()
    assert safe.is_joint_target_safe("left_knee_joint", 90)       # normal bend: OK
    assert not safe.is_joint_target_safe("left_knee_joint", 140)  # too close to 143 limit
    assert not safe.is_joint_target_safe("unknown_joint", 0)      # unknown joint: rejected
