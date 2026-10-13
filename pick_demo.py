"""Pick up the red bottle and put it back - the robot FINDS it with its own eyes.

The robot:  1. scans with its camera until it sees the bottle   (Day 10 detection)
            2. turns the picture position into a room position  (Days 11-12: depth + frames)
            3. walks to a spot in front of it, faces the table
            4. reaches and grasps it (magic grasp)
            5. lifts it, holds it, puts it back, walks home

Usage:
  python pick_demo.py                           # red bottle, found with the camera
  python pick_demo.py --target blue_bottle
  python pick_demo.py --start-yaw 120           # start facing away: the robot must search
  python pick_demo.py --no-vision               # old version: ask the simulator instead
  python pick_demo.py --speedup 2               # faster

Keys in the MuJoCo window: Space = pause (e-stop), R = resume.
Grasping is the plan's "magic grasp": the A3 model has no finger joints.
"""
import argparse
import math
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import yaml

from manipulation.arm import Arm, MagicGrasp
from robot.kinematic_sim_robot import KinematicSimRobot
from safety.supervisor import SafetySupervisor
from simulation.scene import CAMERA_NAME, build_scene

DT = 0.02
STAND_BACK = 0.45        # stand this far (m) behind the bottle
SIDE_OFFSET = 0.20       # shift sideways so the bottle is in front of the chosen hand
GRASP_DIST = 0.02        # hand this close (m) -> grasp
LIFT = 0.15              # lift height (m)
ARM_STEP = 0.004         # max joint change per step (rad); smaller = slower, smoother arm
BOTTLE_RADIUS = 0.035    # used to go from the bottle's front surface to its centre
HEAD_SCAN = [0, 40, -40]           # head positions tried before turning the body (deg)
EYES_WINDOW = "A3 EYES - what the robot sees"

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="red_bottle")
ap.add_argument("--start-yaw", type=float, default=0.0, help="robot's starting direction (deg)")
ap.add_argument("--no-vision", action="store_true", help="ask the simulator instead of using the camera")
ap.add_argument("--speedup", type=float, default=1.0)
ap.add_argument("--no-viewer", action="store_true")
args = ap.parse_args()
target_words = args.target.replace("_", " ")          # "red_bottle" -> "red bottle"

with open("config/robot.yaml") as f:
    cfg = yaml.safe_load(f)
model, _ = build_scene(cfg["robot"]["mjcf"], with_objects=True)
robot = KinematicSimRobot(model=model)
robot.pose.yaw = math.radians(args.start_yaw)
robot.step(0.0)
safe = SafetySupervisor(robot, cfg)
safe.stand()

camera = detector = None
if not args.no_vision:
    from perception.detector import ObjectDetector, draw
    from perception.spatial import DepthCamera
    camera = DepthCamera(model, CAMERA_NAME)          # created before the viewer
    detector = ObjectDetector()

viewer = None
frame = 0
t0 = time.time()


def say(text):
    p = robot.pose
    print(f"[{time.time() - t0:6.1f}s] {text:<52s} robot at x={p.x:+.2f} y={p.y:+.2f} "
          f"facing {(math.degrees(p.yaw) + 180) % 360 - 180:+.0f} deg")


def on_key(keycode):
    if keycode == 32:
        safe.emergency_stop()
    elif keycode == 82:
        safe.reset_emergency_stop()


def look(note=""):
    """Take a picture (+ depth), run the detector, show it. Returns (detections, depth)."""
    if camera is None:
        return [], None
    if viewer is not None:
        with viewer.lock():
            rgb, depth = camera.capture(robot.d)
    else:
        rgb, depth = camera.capture(robot.d)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    found = detector.detect(bgr)
    if viewer is not None:
        shown = draw(bgr.copy(), found, target_words)
        if note:
            cv2.putText(shown, note, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(EYES_WINDOW, shown)
        cv2.waitKey(1)
    return found, depth


def tick(grasp=None):
    """Advance one step; while paused, wait here."""
    global frame
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
            viewer.cam.lookat[:] = (robot.pose.x + 0.3 * math.cos(robot.pose.yaw),
                                    robot.pose.y + 0.3 * math.sin(robot.pose.yaw), 0.9)
        viewer.sync()
        if not viewer.is_running():
            raise KeyboardInterrupt
        if camera is not None and frame % 5 == 0:
            look()
        time.sleep(DT / args.speedup)
    else:
        robot.step(DT)
        if grasp:
            grasp.update()
    frame += 1


# ---------- walking helpers ----------
def turn_to(yaw, grasp=None):
    while True:
        err = math.atan2(math.sin(yaw - robot.pose.yaw), math.cos(yaw - robot.pose.yaw))
        if abs(err) < math.radians(0.5):
            break
        safe.set_velocity(0.0, 0.0, math.copysign(min(2.0, max(0.05, 1.5 * abs(err))), err))
        tick(grasp)
    safe.stop()


def walk_to(x, y, grasp=None):
    if math.hypot(x - robot.pose.x, y - robot.pose.y) < 0.01:
        return
    turn_to(math.atan2(y - robot.pose.y, x - robot.pose.x), grasp)
    while True:
        dx, dy = x - robot.pose.x, y - robot.pose.y
        remaining = math.hypot(dx, dy)
        if remaining < 0.005:
            break
        err = math.atan2(math.sin(math.atan2(dy, dx) - robot.pose.yaw),
                         math.cos(math.atan2(dy, dx) - robot.pose.yaw))
        safe.set_velocity(min(0.3, max(0.05, remaining)), 0.0, 2.0 * err if remaining > 0.05 else 0.0)
        tick(grasp)
    safe.stop()


# ---------- arm helpers ----------
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


# ---------- FIND: with the eyes (Days 10-12) ----------
def find_with_eyes():
    """Scan: head left/right, then turn the body 45 deg at a time, until the target is seen."""
    for turn in range(8):
        for head_yaw in HEAD_SCAN:
            robot.set_head(yaw_deg=head_yaw, pitch_deg=0)
            for _ in range(10):                      # let the view settle
                tick()
            found, depth = look(f"searching for {target_words}...")
            hit = detector.find(found, target_words)
            if hit:
                u, v = hit["centre"]
                point = camera.locate(robot.d, depth, u, v, object_radius=BOTTLE_RADIUS)
                say(f"   SEEN at pixel ({u},{v}) [{hit['source']}], head {head_yaw:+d} deg")
                robot.set_head(yaw_deg=0, pitch_deg=0)
                return point
        say("   not seen here - turning 45 deg")
        turn_to(robot.pose.yaw + math.radians(45))
    return None


# ---------- the task ----------
def run():
    obj = MagicGrasp(model, robot.d, args.target)
    truth = obj.position()                     # used only to MEASURE the vision error
    start = (robot.pose.x, robot.pose.y)

    # 1-2. where is the bottle?
    if camera is None:
        bottle = truth.copy()
        say(f"1. FIND {target_words}: asked the simulator")
    else:
        say(f"1. FIND {target_words} with the camera")
        bottle = find_with_eyes()
        if bottle is None:
            say(f"   {target_words} not found anywhere - stopping safely")
            return False
        err = np.linalg.norm(bottle - truth) * 100
        say(f"2. LOCATED at x={bottle[0]:.2f} y={bottle[1]:+.2f} z={bottle[2]:.2f} "
            f"(error vs truth {err:.1f} cm)")
    side = "left" if bottle[1] >= robot.pose.y else "right"

    # 3. walk to a spot in front of it and face the table
    stand = (bottle[0] - STAND_BACK, bottle[1] - (SIDE_OFFSET if side == "left" else -SIDE_OFFSET))
    say(f"3. NAVIGATE to x={stand[0]:.2f} y={stand[1]:+.2f}")
    walk_to(*stand)
    turn_to(0.0)

    # 4. reach and grasp
    robot.animate_arms = False
    arm = Arm(model, robot.d, side)
    pre_grasp = bottle - np.array([BOTTLE_RADIUS + 0.10, 0, 0])
    grasp_pt = bottle - np.array([BOTTLE_RADIUS + 0.03, 0, 0])
    say(f"4. REACH with the {side} hand")
    reach(arm, pre_grasp)
    if not reach(arm, grasp_pt, tol=GRASP_DIST) or \
            np.linalg.norm(arm.palm() - obj.position()) > 0.12:
        say("   hand is not at the bottle - stopping safely")
        arm_home(arm)
        return False
    home_pos = obj.position()
    obj.attach(arm)
    say("   GRASPED (magic grasp)")

    # 5. lift, hold, put back
    reach(arm, grasp_pt + [0, 0, LIFT], obj)
    say(f"5. LIFTED {LIFT * 100:.0f} cm, holding for 2 s")
    for _ in range(int(2 / DT)):
        tick(obj)
    reach(arm, grasp_pt, obj, tol=0.005)
    obj.release()
    robot.d.qpos[obj.qadr:obj.qadr + 3] = home_pos
    mujoco.mj_forward(model, robot.d)
    say("   PLACED back on the table, released")
    reach(arm, pre_grasp)
    arm_home(arm)
    robot.animate_arms = True
    walk_to(*start)
    turn_to(0.0)
    say("DONE - back at start")
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
finally:
    cv2.destroyAllWindows()
