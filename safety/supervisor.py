"""SafetySupervisor: checks every command before it reaches the robot."""
import math
import time
import yaml
from robot.interface import RobotInterface


class SafetySupervisor(RobotInterface):
    def __init__(self, robot: RobotInterface, config: dict):
        s = config["safety"]
        self.robot = robot
        self.max_v = s["max_linear_velocity_mps"]
        self.max_w = s["max_angular_velocity_radps"]
        self.timeout = s["command_timeout_s"]
        self.margin_deg = s["joint_limit_margin_deg"]
        self.joint_limits = config["joint_limits_deg"]
        self.estopped = False
        self.last_cmd_time = None

    @classmethod
    def from_yaml(cls, robot, path="config/robot.yaml"):
        with open(path) as f:
            return cls(robot, yaml.safe_load(f))

    def stand(self):
        if self.estopped:
            raise RuntimeError("Emergency stop is active")
        self.robot.stand()

    def stop(self):
        self.robot.stop()

    def set_velocity(self, vx, vy, wz):
        if self.estopped:                    # rule 1: no motion during emergency stop
            self.robot.stop()
            return
        speed = math.hypot(vx, vy)
        if speed > self.max_v:               # rule 2: limit walking speed
            k = self.max_v / speed
            vx, vy = vx * k, vy * k
        wz = max(-self.max_w, min(self.max_w, wz))  # rule 3: limit turning speed
        self.last_cmd_time = time.monotonic()
        self.robot.set_velocity(vx, vy, wz)

    def check_timeout(self, now=None):
        """Rule 4: stop the robot if no new command arrived in time."""
        now = time.monotonic() if now is None else now
        if self.last_cmd_time is not None and now - self.last_cmd_time > self.timeout:
            self.robot.stop()
            self.last_cmd_time = None
            return True
        return False

    def is_joint_target_safe(self, joint, angle_deg):
        """Rule 5: joint targets must stay inside the limits, with a safety margin."""
        if joint not in self.joint_limits:
            return False
        lo, hi = self.joint_limits[joint]
        return lo + self.margin_deg <= angle_deg <= hi - self.margin_deg

    def get_pose(self):
        return self.robot.get_pose()

    def emergency_stop(self):
        self.estopped = True
        self.robot.emergency_stop()

    def reset_emergency_stop(self):
        self.estopped = False
