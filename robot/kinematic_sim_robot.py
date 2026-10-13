"""KinematicSimRobot: shows the A3 in MuJoCo and moves it from velocity commands.

PREVIEW ONLY - not real physics walking. The body glides and the legs/arms swing as an
animation, with the feet kept on the floor. Replace with SimulationRobot (AimSim) later.

Use either a model file:   KinematicSimRobot("path/to/robot.xml")
or a ready-built model:    KinematicSimRobot(model=scene_model)
"""
import math
import mujoco
import numpy as np
from robot.interface import RobotInterface, Pose2D

_CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)


class KinematicSimRobot(RobotInterface):
    def __init__(self, mjcf_path=None, model=None, root_body="pelvis_link"):
        self.m = model if model is not None else mujoco.MjModel.from_xml_path(mjcf_path)
        self.m.opt.gravity[:] = 0
        self.d = mujoco.MjData(self.m)
        self.base = self.m.joint("free_joint").qposadr[0]
        root_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, root_body)
        self.robot_geoms = np.where(self.m.body_rootid[self.m.geom_bodyid] == root_id)[0]
        self.pose = Pose2D()
        self.velocity = (0.0, 0.0, 0.0)
        self.phase = 0.0
        self.animate_arms = True        # set False while the arms are used for grasping
        self.joint = {name: self.m.joint(name).qposadr[0] for name in (
            "left_hip_pitch_joint", "right_hip_pitch_joint",
            "left_knee_joint", "right_knee_joint",
            "left_shoulder_pitch_joint", "right_shoulder_pitch_joint")}
        self.step(0.0)
        self.height = float(self.d.qpos[self.base + 2])

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

    def set_head(self, yaw_deg=None, pitch_deg=None):
        """Turn the head (within its joint limits). Positive pitch = look down, positive yaw = left."""
        for name, value in (("head_yaw_joint", yaw_deg), ("head_pitch_joint", pitch_deg)):
            if value is None:
                continue
            j = self.m.joint(name)
            lo, hi = j.range
            self.d.qpos[j.qposadr[0]] = float(np.clip(math.radians(value), lo, hi))

    def get_head(self):
        return tuple(math.degrees(self.d.qpos[self.m.joint(n).qposadr[0]])
                     for n in ("head_yaw_joint", "head_pitch_joint"))

    def _lowest_point_z(self):
        ids = self.robot_geoms
        aabb = self.m.geom_aabb[ids]
        local = aabb[:, None, :3] + _CORNERS[None] * aabb[:, None, 3:]
        rot_z = self.d.geom_xmat[ids].reshape(-1, 3, 3)[:, 2, :]
        world_z = self.d.geom_xpos[ids][:, None, 2] + np.einsum("nj,nkj->nk", rot_z, local)
        return float(world_z.min())

    def step(self, dt):
        vx, vy, wz = self.velocity
        c, s = math.cos(self.pose.yaw), math.sin(self.pose.yaw)
        self.pose.x += (vx * c - vy * s) * dt
        self.pose.y += (vx * s + vy * c) * dt
        self.pose.yaw += wz * dt
        q, b = self.d.qpos, self.base
        q[b:b + 2] = (self.pose.x, self.pose.y)
        q[b + 3:b + 7] = (math.cos(self.pose.yaw / 2), 0, 0, math.sin(self.pose.yaw / 2))
        moving = abs(vx) + abs(vy) + 0.3 * abs(wz)
        if moving > 1e-3:
            self.phase += dt * 2 * math.pi * 1.5
        amp = min(1.0, moving / 0.3)
        swing = math.sin(self.phase)
        j = self.joint
        q[j["left_hip_pitch_joint"]] = -0.35 * amp * swing
        q[j["right_hip_pitch_joint"]] = 0.35 * amp * swing
        q[j["left_knee_joint"]] = 0.6 * amp * max(0.0, swing)
        q[j["right_knee_joint"]] = 0.6 * amp * max(0.0, -swing)
        if self.animate_arms:
            q[j["left_shoulder_pitch_joint"]] = 0.3 * amp * swing
            q[j["right_shoulder_pitch_joint"]] = -0.3 * amp * swing
        mujoco.mj_forward(self.m, self.d)
        q[b + 2] -= self._lowest_point_z()
        mujoco.mj_forward(self.m, self.d)
