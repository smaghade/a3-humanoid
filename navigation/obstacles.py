"""Seeing obstacles (stones, boxes, tables) with the depth camera.

How the robot recognises an obstacle, step by step:
  1. Depth picture: distance for every pixel (Day 11).
  2. Every 4th pixel is turned into a 3D point in the room (Days 11-12 maths).
  3. Points that stick up from the floor (higher than 5 cm, lower than 1.2 m) and are
     within 5 m are OBSTACLE points. The floor itself is ignored.
  4. Neighbouring obstacle pixels are grouped into objects; each object gets a box,
     its height and its distance. Low and wide = "stone", taller = "obstacle".
No AI model is needed for this: it works for any shape, even ones never seen before.
"""
import cv2
import numpy as np


class ObstacleDetector:
    def __init__(self, camera, step=4, min_height=0.05, max_height=1.2, max_range=5.0,
                 ignore_radius=0.6):
        self.cam, self.step = camera, step
        self.min_h, self.max_h, self.max_range = min_height, max_height, max_range
        self.ignore_radius = ignore_radius            # the robot's own body, arms and hands
        vs, us = np.mgrid[0:camera.height:step, 0:camera.width:step]
        self.us, self.vs = us.astype(float), vs.astype(float)
        self.grid_shape = us.shape

    def points(self, data, depth):
        """3D room points for the sampled pixels. Returns (points Nx3 or NaN, depth sample)."""
        z = depth[self.vs.astype(int), self.us.astype(int)]
        x = (self.us - self.cam.cx) * z / self.cam.fx
        y = (self.vs - self.cam.cy) * z / self.cam.fy
        p_mj = np.stack([x, -y, -z], axis=-1)                      # MuJoCo camera frame
        rot = data.cam_xmat[self.cam.cam].reshape(3, 3)
        world = data.cam_xpos[self.cam.cam] + p_mj @ rot.T
        return world, z

    def detect(self, data, depth, robot_xy):
        """Returns (obstacle points Mx3, list of objects for display/logging)."""
        world, z = self.points(data, depth)
        dist = np.hypot(world[..., 0] - robot_xy[0], world[..., 1] - robot_xy[1])
        mask = ((z > 0.05) & (z < self.max_range) & (world[..., 2] > self.min_h) &
                (world[..., 2] < self.max_h) & (dist > self.ignore_radius))
        pts = world[mask]
        objects = []
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if area < 3:
                continue
            obj_pts = world[labels == k]
            height = float(obj_pts[:, 2].max())
            centre = obj_pts[:, :2].mean(axis=0)
            d = float(np.hypot(*(centre - np.asarray(robot_xy))))
            kind = "stone" if height < 0.6 else "obstacle"
            s = self.step
            objects.append({"kind": kind, "height": height, "distance": d, "centre": centre,
                            "box": (x * s, y * s, (x + w) * s, (y + h) * s)})
        return pts, objects


def draw_obstacles(bgr, objects):
    for o in objects:
        x1, y1, x2, y2 = o["box"]
        colour = (0, 165, 255) if o["kind"] == "stone" else (0, 0, 200)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), colour, 2)
        text = f"{o['kind']} {o['height']:.2f}m high, {o['distance']:.1f}m away"
        cv2.putText(bgr, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bgr, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    colour, 1, cv2.LINE_AA)
    return bgr
