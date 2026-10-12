"""MockRobot: a fake robot for testing without any simulator."""
import math
from robot.interface import RobotInterface, Pose2D


class MockRobot(RobotInterface):
    def __init__(self):
        self.pose = Pose2D()
        self.velocity = (0.0, 0.0, 0.0)
        self.standing = False
        self.estopped = False
        self.log = []                       # remembers every command, useful in tests

    def stand(self):
        self.standing = True
        self.log.append("stand")

    def stop(self):
        self.velocity = (0.0, 0.0, 0.0)
        self.log.append("stop")

    def set_velocity(self, vx, vy, wz):
        self.velocity = (vx, vy, wz)
        self.log.append(("set_velocity", vx, vy, wz))

    def get_pose(self):
        return self.pose

    def emergency_stop(self):
        self.velocity = (0.0, 0.0, 0.0)
        self.estopped = True
        self.log.append("emergency_stop")

    def step(self, dt):
        """Move the fake robot forward in time by dt seconds."""
        vx, vy, wz = self.velocity
        c, s = math.cos(self.pose.yaw), math.sin(self.pose.yaw)
        self.pose.x += (vx * c - vy * s) * dt
        self.pose.y += (vx * s + vy * c) * dt
        self.pose.yaw += wz * dt
