"""Day 14 mini project: "Where is the X bottle?" - measured over many random layouts.

For each layout the bottles and cup are placed at random spots on the table, and the
robot starts facing a random direction. It then searches with its camera exactly like
pick_demo.py (head left/right, then turning the body), and we measure:
  - was the requested bottle found?
  - how far (cm) is the robot's estimate from the true position?

Usage:
  python tools/finder_test.py                 # 10 layouts, targets red/blue/yellow in turn
  python tools/finder_test.py --layouts 20 --seed 7
  python tools/finder_test.py --target red_bottle
Results: docs/day14_results.csv, pictures of misses in data/day14/
"""
import argparse
import csv
import math
import os
import sys

import cv2
import mujoco
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perception.detector import ObjectDetector, draw          # noqa: E402
from perception.spatial import DepthCamera                    # noqa: E402
from robot.kinematic_sim_robot import KinematicSimRobot       # noqa: E402
from simulation.scene import CAMERA_NAME, OBJECTS, build_scene  # noqa: E402

HEAD_SCAN = [0, 40, -40]
MIN_HEIGHT, MAX_HEIGHT = 0.3, 1.8
BOTTLE_RADIUS = 0.035
TABLE_X, TABLE_Y = (0.95, 1.40), (-0.50, 0.50)     # where objects may be placed (table top)
MIN_GAP = 0.13                                      # objects at least this far apart (m)


def random_layout(robot, rng):
    """Put every object at a random free spot on the table."""
    placed = []
    for name in OBJECTS:
        body = mujoco.mj_name2id(robot.m, mujoco.mjtObj.mjOBJ_BODY, name)
        adr = robot.m.jnt_qposadr[robot.m.body_jntadr[body]]
        z = robot.d.qpos[adr + 2]                       # keep its height on the table
        for _ in range(200):
            x, y = rng.uniform(*TABLE_X), rng.uniform(*TABLE_Y)
            if all(math.hypot(x - px, y - py) > MIN_GAP for px, py in placed):
                break
        placed.append((x, y))
        robot.d.qpos[adr:adr + 7] = [x, y, z, 1, 0, 0, 0]
    mujoco.mj_forward(robot.m, robot.d)


def search(robot, camera, detector, target_words):
    """Same search as pick_demo.py. Returns (estimate, steps used, last picture)."""
    looks, picture = 0, None
    for turn in range(8):
        for head_yaw in HEAD_SCAN:
            robot.set_head(yaw_deg=head_yaw, pitch_deg=0)
            robot.step(0.0)
            rgb, depth = camera.capture(robot.d)
            looks += 1
            picture = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            found = detector.detect(picture)
            for hit in detector.find_all(found, target_words):
                u, v = hit["centre"]
                point = camera.locate(robot.d, depth, u, v, object_radius=BOTTLE_RADIUS)
                if point is not None and MIN_HEIGHT < point[2] < MAX_HEIGHT:
                    return point, looks, draw(picture, found, target_words)
        robot.pose.yaw += math.radians(45)               # turn the body (instantly, for testing)
    return None, looks, picture


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--target", default=None, help="e.g. red_bottle (default: red/blue/yellow in turn)")
    args = ap.parse_args()

    with open("config/robot.yaml") as f:
        cfg = yaml.safe_load(f)
    model, _ = build_scene(cfg["robot"]["mjcf"], with_objects=True)
    robot = KinematicSimRobot(model=model)
    camera = DepthCamera(model, CAMERA_NAME)
    detector = ObjectDetector()
    rng = np.random.default_rng(args.seed)
    bottles = [o for o in OBJECTS if o.endswith("bottle")]
    os.makedirs("docs", exist_ok=True)
    os.makedirs("data/day14", exist_ok=True)

    rows = []
    print(f"{'#':>3}  {'target':14s} {'start':>6s}  {'found':5s} {'looks':>5s}  {'error':>8s}")
    for i in range(1, args.layouts + 1):
        target = args.target or bottles[(i - 1) % len(bottles)]
        random_layout(robot, rng)
        start_yaw = float(rng.uniform(-180, 180))
        robot.pose.x = robot.pose.y = 0.0
        robot.pose.yaw = math.radians(start_yaw)
        robot.step(0.0)
        truth = robot.d.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target)].copy()

        estimate, looks, picture = search(robot, camera, detector, target.replace("_", " "))
        found = estimate is not None
        error_cm = float(np.linalg.norm(estimate - truth) * 100) if found else None
        if picture is not None:
            tag = "ok" if found else "MISS"
            cv2.imwrite(f"data/day14/layout{i:02d}_{target}_{tag}.png", picture)
        rows.append({"layout": i, "target": target, "start_yaw_deg": round(start_yaw),
                     "found": found, "looks": looks,
                     "error_cm": None if error_cm is None else round(error_cm, 2),
                     "true_x": round(truth[0], 3), "true_y": round(truth[1], 3)})
        err_txt = f"{error_cm:6.1f} cm" if found else "      -"
        print(f"{i:3d}  {target:14s} {start_yaw:+5.0f}°  {'yes' if found else 'NO':5s} {looks:5d}  {err_txt}")

    with open("docs/day14_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    hits = [r for r in rows if r["found"]]
    errors = [r["error_cm"] for r in hits]
    print(f"\nFound {len(hits)} of {len(rows)} ({100 * len(hits) / len(rows):.0f}%)")
    if errors:
        print(f"Position error: average {np.mean(errors):.1f} cm, worst {max(errors):.1f} cm")
    print("Saved: docs/day14_results.csv  and pictures in data/day14/")


if __name__ == "__main__":
    main()
