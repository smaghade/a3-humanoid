"""Pick up the red bottle and put it back (preview mode).

The robot:  1. finds where the bottle is      4. reaches and grasps it
            2. walks to a spot in front of it 5. lifts it and holds it
            3. faces the table                6. puts it back and steps away

Usage:
  python pick_demo.py                         # red bottle
  python pick_demo.py --target blue_bottle
  python pick_demo.py --speedup 2             # faster
  python pick_demo.py --no-viewer             # no window, quick test

Keys in the MuJoCo window: Space = pause (e-stop), R = resume.

NOTE: step 1 uses the bottle position the SIMULATOR knows. From Day 11 the robot will
get it from its own camera (depth + intrinsics), replacing only that one function.
Grasping is the plan's "magic grasp": the A3 model has no finger joints.
"""
import argparse
import math
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml

from manipulation.arm import Arm, MagicGrasp
from robot.kinematic_sim_robot import KinematicSimRobot
from safety.supervisor import SafetySupervisor
from simulation.scene import build_scene

DT = 0.02
STAND_BACK = 0.45        # stand this far (m) behind the bottle
SIDE_OFFSET = 0.20       # shift sideways so the bottle is in front of the chosen hand
GRASP_DIST = 0.02        # hand this close (m) -> grasp
LIFT = 0.15              # lift height (m)

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="red_bottle")
ap.add_argument("--speedup", type=float, default=1.0)
ap.add_argument("--no-viewer", action="store_true")
args = ap.parse_args()

with open("config/robot.yaml") as f:
    cfg = yaml.safe_load(f)
model, _ = build_scene(cfg["robot"]["mjcf"], with_objects=True)
robot = KinematicSimRobot(model=model)
safe = SafetySupervisor(robot, cfg)
safe.stand()
viewer = None
t0 = time.time()


def say(text):
    p = robot.pose
    print(f"[{time.time() - t0:6.1f}s] {text:<45s} robot at x={p.x:+.2f} y={p.y:+.2f} "
          f"facing {math.degrees(p.yaw):+.0f} deg")


def on_key(keycode):
    if keycode == 32:
        safe.emergency_stop()
    elif keycode == 82:
        safe.reset_emergency_stop()


def tick(grasp=None):
    """Advance one step; while paused, wait here."""
    while safe.estopped:
        if viewer is not None:
            if not viewer.is_running():
                raise KeyboardInterrupt
            viewer.sync()
        time.sleep(0.05)
    if viewer is not None:
        with viewer.lock():
            robot.step(DT)
            if grasp:
                grasp.update()
            viewer.cam.lookat[:] = (robot.pose.x + 0.3, robot.pose.y, 0.9)
        viewer.sync()
        if not viewer.is_running():
            raise KeyboardInterrupt
        time.sleep(DT / args.speedup)
    else:
        robot.step(DT)
        if grasp:
            grasp.update()


# ---------- walking helpers (same idea as mission_sim) ----------
def turn_to(yaw, grasp=None):
    while True:
        err = math.atan2(math.sin(yaw - robot.pose.yaw), math.cos(yaw - robot.pose.yaw))
        if abs(err) < math.radians(0.5):
            break
        safe.set_velocity(0.0, 0.0, math.copysign(min(2.0, max(0.05, 1.5 * abs(err))), err))
        tick(grasp)
    safe.stop()


def walk_to(x, y, grasp=None):
    dist = math.hypot(x - robot.pose.x, y - robot.pose.y)
    if dist < 0.01:
        return
    heading = math.atan2(y - robot.pose.y, x - robot.pose.x)
    turn_to(heading, grasp)
    while True:
        dx, dy = x - robot.pose.x, y - robot.pose.y
        remaining = math.hypot(dx, dy)
        if remaining < 0.005:
            break
        steer = 2.0 * math.atan2(math.sin(math.atan2(dy, dx) - robot.pose.yaw),
                                 math.cos(math.atan2(dy, dx) - robot.pose.yaw))
        safe.set_velocity(min(0.3, max(0.05, remaining)), 0.0, steer if remaining > 0.05 else 0.0)
        tick(grasp)
    safe.stop()


# ---------- arm helper ----------
ARM_STEP = 0.004          # max joint change per step (rad); smaller = slower, smoother arm


def reach(arm, target, grasp=None, tol=0.01, max_steps=3000):
    for _ in range(max_steps):
        if safe.estopped:
            tick(grasp)
            continue
        if arm.step_towards(target, max_step=ARM_STEP) < tol:
            return True
        tick(grasp)
    return False


def arm_home(arm, grasp=None):
    while arm.step_home(max_step=ARM_STEP) > 1e-3:
        tick(grasp)


# ---------- the task ----------
def run():
    obj = MagicGrasp(model, robot.d, args.target)
    start = (robot.pose.x, robot.pose.y)

    # 1. where is the bottle?
    bottle = obj.position()
    home_pos = bottle.copy()
    side = "left" if bottle[1] >= 0 else "right"
    say(f"1. FIND {args.target}: at x={bottle[0]:.2f} y={bottle[1]:+.2f} z={bottle[2]:.2f}")

    # 2-3. walk to a spot in front of it and face the table
    stand = (bottle[0] - STAND_BACK, bottle[1] - (SIDE_OFFSET if side == "left" else -SIDE_OFFSET))
    say(f"2. NAVIGATE to x={stand[0]:.2f} y={stand[1]:+.2f}")
    walk_to(*stand)
    turn_to(0.0)
    say("3. APPROACH done, facing the table")

    # 4. reach and grasp
    robot.animate_arms = False
    arm = Arm(model, robot.d, side)
    pre_grasp = bottle - np.array([0.035 + 0.10, 0, 0])      # 10 cm in front of the bottle
    grasp_pt = bottle - np.array([0.035 + 0.03, 0, 0])       # touching distance
    say(f"4. REACH with the {side} hand")
    reach(arm, pre_grasp)
    if not reach(arm, grasp_pt, tol=GRASP_DIST):
        say("   could not reach the bottle - stopping safely")
        return False
    obj.attach(arm)
    say("   GRASPED (magic grasp)")

    # 5. lift and hold
    reach(arm, grasp_pt + [0, 0, LIFT], obj)
    say(f"5. LIFTED {LIFT * 100:.0f} cm, holding for 2 s")
    for _ in range(int(2 / DT)):
        tick(obj)

    # 6. put it back where it was
    reach(arm, grasp_pt, obj, tol=0.005)
    obj.release()
    robot.d.qpos[obj.qadr:obj.qadr + 3] = home_pos            # exactly back in place
    mujoco.mj_forward(model, robot.d)
    say("6. PLACED back on the table, released")
    reach(arm, pre_grasp)
    arm_home(arm)
    robot.animate_arms = True
    walk_to(*start)
    turn_to(0.0)
    moved = np.linalg.norm(obj.position() - home_pos) * 100
    say(f"DONE - back at start. Bottle is {moved:.1f} cm from its original place")
    return True


print(__doc__)
try:
    if args.no_viewer:
        run()
    else:
        with mujoco.viewer.launch_passive(model, robot.d, key_callback=on_key) as v:
            viewer = v
            with viewer.lock():
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.8, 150, -20
            run()
            print("\nFinished. Close the MuJoCo window to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
except KeyboardInterrupt:
    safe.stop()
    print("\nStopped by user.")
