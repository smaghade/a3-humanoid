"""RobotInterface: the one standard way to talk to any robot (mock, simulation, real A3)."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Pose2D:
    x: float = 0.0    # metres forward in the room
    y: float = 0.0    # metres left in the room
    yaw: float = 0.0  # direction the robot faces, in radians


class RobotInterface(ABC):
    @abstractmethod
    def stand(self): ...

    @abstractmethod
    def stop(self): ...

    @abstractmethod
    def set_velocity(self, vx: float, vy: float, wz: float):
        """vx = forward speed, vy = sideways speed (m/s), wz = turning speed (rad/s)."""

    @abstractmethod
    def get_pose(self) -> Pose2D: ...

    @abstractmethod
    def emergency_stop(self): ...

    # Added in later days of the plan:
    def walk_to(self, pose: Pose2D): raise NotImplementedError
    def look_at(self, target): raise NotImplementedError
    def reach_to(self, pose, arm="right"): raise NotImplementedError
    def grasp(self): raise NotImplementedError
    def release(self): raise NotImplementedError
