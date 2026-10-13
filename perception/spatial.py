"""Days 11-12: from a pixel in the camera picture to a real 3D position in the room.

Day 11 (camera geometry): depth picture + intrinsics (fx, fy, cx, cy)
    X = (u - cx) * Z / fx        how far right of the camera centre
    Y = (v - cy) * Z / fy        how far below the camera centre
    Z = depth                    how far in front of the camera
Day 12 (frames / TF): the camera's position and orientation in the room
    point_in_room = camera_position + camera_rotation @ point_in_camera
MuJoCo knows the camera's pose at every moment (it follows the head, the body and the
walking), which is exactly the job TF2 does on the real robot.
"""
import math

import mujoco
import numpy as np


class DepthCamera:
    def __init__(self, model, camera, width=640, height=480):
        self.m = model
        self.cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        fovy = math.radians(model.cam_fovy[self.cam])
        self.fy = self.fx = (height / 2) / math.tan(fovy / 2)
        self.cx, self.cy = width / 2, height / 2
        self.width, self.height = width, height

    def capture(self, data):
        """Colour picture (RGB) + depth picture (metres along the camera's viewing direction)."""
        self.renderer.update_scene(data, camera=self.cam)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=self.cam)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return rgb, depth

    def pixel_to_camera(self, u, v, z):
        """Day 11: pixel (u, v) + depth z -> point in the camera frame (x right, y down, z forward)."""
        return np.array([(u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z])

    def camera_to_world(self, data, p_cam):
        """Day 12: camera frame -> room frame. MuJoCo cameras look along -z with +y up."""
        p_mj = np.array([p_cam[0], -p_cam[1], -p_cam[2]])
        rot = data.cam_xmat[self.cam].reshape(3, 3)
        return data.cam_xpos[self.cam] + rot @ p_mj

    def world_to_pixel(self, data, p_world):
        """The opposite direction (useful for checking): room point -> pixel (u, v), depth z."""
        rot = data.cam_xmat[self.cam].reshape(3, 3)
        p_mj = rot.T @ (np.asarray(p_world) - data.cam_xpos[self.cam])
        z = -p_mj[2]
        return self.cx + self.fx * p_mj[0] / z, self.cy - self.fy * p_mj[1] / z, z

    def locate(self, data, depth, u, v, window=3, object_radius=0.0):
        """Room position of the object seen at pixel (u, v).

        Uses the median depth in a small window (robust to edge pixels). The depth hits the
        FRONT surface; object_radius moves the point back to the object's centre.
        """
        u, v = int(round(u)), int(round(v))
        patch = depth[max(0, v - window):v + window + 1, max(0, u - window):u + window + 1]
        patch = patch[(patch > 0.05) & (patch < 20)]
        if patch.size == 0:
            return None
        p = self.camera_to_world(data, self.pixel_to_camera(u, v, float(np.median(patch))))
        if object_radius:
            cam = data.cam_xpos[self.cam]
            flat = np.array([p[0] - cam[0], p[1] - cam[1], 0.0])
            p = p + object_radius * flat / (np.linalg.norm(flat) + 1e-9)
        return p
