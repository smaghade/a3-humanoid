"""KinematicSimRobot: shows the A3 in MuJoCo and moves it from velocity commands.

PREVIEW ONLY - not real physics walking. There is no balance controller yet, so the
body glides and the legs/arms swing as an animation. The feet are kept on the floor.
Replace with SimulationRobot (AimSim) later; the rest of the application stays the same.
"""
import math
import mujoco
import numpy as np
from robot.interface import RobotInterface, Pose2D

# the 8 corners of a box, used to find the lowest point of each body part
_CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)


class KinematicSimRobot(RobotInterface):
    def __init__(self, mjcf_path):
        self.m = mujoco.MjModel.from_xml_path(mjcf_path)
        self.m.opt.gravity[:] = 0                                 # no falling in preview mode
        self.d = mujoco.MjData(self.m)
        self.base = self.m.joint("free_joint").qposadr[0]
        self.robot_geoms = np.where(self.m.geom_bodyid != 0)[0]   # robot parts, not the floor
        self.pose = Pose2D()
        self.velocity = (0.0, 0.0, 0.0)
        self.phase = 0.0
        self.joint = {name: self.m.joint(name).qposadr[0] for name in (
            "left_hip_pitch_joint", "right_hip_pitch_joint",
            "left_knee_joint", "right_knee_joint",
            "left_shoulder_pitch_joint", "right_shoulder_pitch_joint")}
        self.step(0.0)                                            # place the robot on the floor
        self.height = float(self.d.qpos[self.base + 2])

    # ---- RobotInterface commands ----
    def stand(self):
        self.velocity = (0.0, 0.0, 0.0)

    def stop(self):
        self.velocity = (0.0, 0.0, 0.0)

    def set_velocity(self, vx, vy, wz):
        self.velocity = (vx, vy, wz)

    def get_pose(self):
        return self.pose

    def emergency_stop(self):
        self.velocity = (0.0, 0.0, 0.0)

    # ---- helpers ----
    def _lowest_point_z(self):
        """Lowest z (height) of any robot part, using each part's bounding box."""
        ids = self.robot_geoms
        aabb = self.m.geom_aabb[ids]                              # centre + half-size, local frame
        local = aabb[:, None, :3] + _CORNERS[None] * aabb[:, None, 3:]
        rot_z = self.d.geom_xmat[ids].reshape(-1, 3, 3)[:, 2, :]  # how local axes map to world z
        world_z = self.d.geom_xpos[ids][:, None, 2] + np.einsum("nj,nkj->nk", rot_z, local)
        return float(world_z.min())

    def step(self, dt):
        vx, vy, wz = self.velocity
        c, s = math.cos(self.pose.yaw), math.sin(self.pose.yaw)
        self.pose.x += (vx * c - vy * s) * dt
        self.pose.y += (vx * s + vy * c) * dt
        self.pose.yaw += wz * dt

        q, b = self.d.qpos, self.base
        q[b:b + 2] = (self.pose.x, self.pose.y)                                     # position on floor
        q[b + 3:b + 7] = (math.cos(self.pose.yaw / 2), 0, 0, math.sin(self.pose.yaw / 2))

        moving = abs(vx) + abs(vy) + 0.3 * abs(wz)
        if moving > 1e-3:
            self.phase += dt * 2 * math.pi * 1.5                                    # ~1.5 steps per second
        amp = min(1.0, moving / 0.3)
        swing = math.sin(self.phase)
        j = self.joint
        q[j["left_hip_pitch_joint"]] = -0.35 * amp * swing
        q[j["right_hip_pitch_joint"]] = 0.35 * amp * swing
        q[j["left_knee_joint"]] = 0.6 * amp * max(0.0, swing)
        q[j["right_knee_joint"]] = 0.6 * amp * max(0.0, -swing)
        q[j["left_shoulder_pitch_joint"]] = 0.3 * amp * swing
        q[j["right_shoulder_pitch_joint"]] = -0.3 * amp * swing

        mujoco.mj_forward(self.m, self.d)
        q[b + 2] -= self._lowest_point_z()                                          # feet touch the floor
        mujoco.mj_forward(self.m, self.d)
