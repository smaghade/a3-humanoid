"""Seeing obstacles (stones, boxes, tables, people, vehicles, animals) with the depth camera.

How the robot recognises an obstacle, step by step:
  1. Depth picture: distance for every pixel (Day 11).
  2. Every 4th pixel is turned into a 3D point in the room (Days 11-12 maths).
  3. Points that stick up from the GROUND (more than 10 cm above it, below 2.2 m) are
     obstacle points. On flat floor the ground is 0; on the hill, steps and road the robot
     uses the known ground height, so slopes and steps are not mistaken for obstacles.
  4. Neighbouring obstacle pixels are grouped into objects; each object gets a box, its
     height, its size on the floor and its distance.
  5. (tracker.py) Objects are followed from picture to picture: if they move, they are
     moving things (people, animals, vehicles); if they stay, they go on the map.
No AI model is needed for this: it works for any shape, even ones never seen before.
"""
import cv2
import numpy as np


class ObstacleDetector:
    def __init__(self, camera, step=4, min_height=0.10, max_height=2.2, max_range=5.0,
                 ignore_radius=0.6):
        self.cam, self.step = camera, step
        self.min_h, self.max_h, self.max_range = min_height, max_height, max_range
        self.ignore_radius = ignore_radius            # the robot's own body, arms and hands
        vs, us = np.mgrid[0:camera.height:step, 0:camera.width:step]
        self.us, self.vs = us.astype(float), vs.astype(float)

    def points(self, data, depth):
        """3D room points for the sampled pixels. Returns (points HxWx3, depth sample)."""
        z = depth[self.vs.astype(int), self.us.astype(int)]
        x = (self.us - self.cam.cx) * z / self.cam.fx
        y = (self.vs - self.cam.cy) * z / self.cam.fy
        p_mj = np.stack([x, -y, -z], axis=-1)                      # MuJoCo camera frame
        rot = data.cam_xmat[self.cam.cam].reshape(3, 3)
        world = data.cam_xpos[self.cam.cam] + p_mj @ rot.T
        return world, z

    def detect(self, data, depth, robot_xy, ground=None, track_range=None):
        """Returns (obstacle points within max_range, objects).

        ground(x, y): known ground height (None = flat floor at 0).
        track_range: also report objects up to this distance (for tracking moving things).
        Each object: kind, height (above ground), size (on the floor), distance, centre,
        box (pixels), points (3D), near (within max_range)."""
        world, z = self.points(data, depth)
        if ground is None:
            g = 0.0
        else:   # highest ground within 10 cm, so the edge of a step is not seen as an obstacle
            gx, gy = world[..., 0], world[..., 1]
            g = np.maximum(np.maximum(ground(gx - 0.1, gy), ground(gx, gy)), ground(gx + 0.1, gy))
        above = world[..., 2] - g
        dist = np.hypot(world[..., 0] - robot_xy[0], world[..., 1] - robot_xy[1])
        reach = track_range or self.max_range
        mask = ((z > 0.05) & (dist < reach) & (above > self.min_h) & (above < self.max_h) &
                (dist > self.ignore_radius))
        near_mask = mask & (dist < self.max_range)
        pts = world[near_mask]
        objects = []
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if area < 3:
                continue
            sel = labels == k
            obj_pts = world[sel]
            height = float(above[sel].max())
            centre = obj_pts[:, :2].mean(axis=0)
            size = float(np.ptp(obj_pts[:, 0]) ** 2 + np.ptp(obj_pts[:, 1]) ** 2) ** 0.5
            d = float(np.hypot(*(centre - np.asarray(robot_xy))))
            kind = "stone" if height < 0.6 else "obstacle"
            s = self.step
            objects.append({"kind": kind, "height": height, "size": size, "distance": d,
                            "centre": centre, "points": obj_pts, "near": d < self.max_range,
                            "box": (x * s, y * s, (x + w) * s, (y + h) * s)})
        return pts, objects


def draw_obstacles(bgr, objects):
    for o in objects:
        x1, y1, x2, y2 = o["box"]
        moving = o.get("moving", False)
        colour = (255, 0, 255) if moving else ((0, 165, 255) if o["kind"] == "stone" else (0, 0, 200))
        label = o.get("label", o["kind"])
        text = (f"{label} moving {o['speed']:.1f} m/s, {o['distance']:.1f}m" if moving
                else f"{label} {o['height']:.2f}m high, {o['distance']:.1f}m away")
        cv2.rectangle(bgr, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(bgr, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bgr, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    colour, 1, cv2.LINE_AA)
    return bgr
