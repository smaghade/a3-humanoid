"""Drive the A3 in the MuJoCo viewer through the safety supervisor - with logging and EYES.

Same robot and keys as before, plus a second window "A3 EYES" showing what the
robot's head camera sees.

Keys work in the MuJoCo window, the A3 EYES window and the terminal:
  Arrows (or W/S/A/D) = walk forward/back, turn left/right
  Enter (or Q)  = stop
  Space         = PAUSE (emergency stop)
  R             = RESUME, then use the arrows again
  I / K         = look up / down        J / L = look left / right   (EYES window or terminal)
  C             = save what the robot sees to data/snapshots/        (EYES window or terminal)
  X (terminal)  = exit

Options:
  python play_sim.py              robot + eyes
  python play_sim.py --objects    also put a table, red bottle, blue bottle and cup in front of it
  python play_sim.py --no-eyes    the old version, without the camera window
"""
import argparse
import csv
import datetime
import math
import os
import queue
import select
import sys
import termios
import time
import tty

import cv2
import mujoco
import mujoco.viewer
import yaml

from robot.kinematic_sim_robot import KinematicSimRobot
from safety.supervisor import SafetySupervisor
from simulation.scene import CAMERA_NAME, build_scene, camera_intrinsics

ap = argparse.ArgumentParser()
ap.add_argument("--objects", action="store_true", help="add a table, bottles and a cup")
ap.add_argument("--no-eyes", action="store_true", help="no camera window")
ap.add_argument("--detect", action="store_true", help="find bottles and cups in the camera picture")
ap.add_argument("--target", default="red bottle", help="object to look for, e.g. \"blue bottle\"")
args = ap.parse_args()

STEP_V, STEP_W, DT = 0.1, 0.25, 0.02
HEAD_STEP = 3.0                                 # degrees per key press
EYE_W, EYE_H, EYE_EVERY = 640, 480, 3           # camera picture size, update every 3 frames
EYES_WINDOW = "A3 EYES - what the robot sees"
LOG_EVERY = 5
WINDOW_KEYS = {265: "w", 264: "s", 263: "a", 262: "d",   # arrows
               257: "q",                                  # Enter = stop
               32: " ",                                   # Space = pause (e-stop)
               82: "r", 268: "r"}                         # R or Home = resume
EVENT_NAMES = {"w": "FORWARD +", "s": "BACK / SLOWER", "a": "TURN LEFT", "d": "TURN RIGHT",
               "q": "STOP", " ": "PAUSE (E-STOP)", "r": "RESUME", "x": "EXIT",
               "i": "LOOK UP", "k": "LOOK DOWN", "j": "LOOK LEFT", "l": "LOOK RIGHT", "c": "SNAPSHOT"}
MOVE_KEYS = {"w", "s", "a", "d"}

keys = queue.Queue()


def on_window_key(keycode):
    if keycode in WINDOW_KEYS:
        keys.put(WINDOW_KEYS[keycode])


def read_terminal_key():
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    return sys.stdin.read(1) if ready else None


def facing_deg(yaw):
    return (math.degrees(yaw) + 180) % 360 - 180


# ---------- setup: same robot as before, now with a camera in its head ----------
with open("config/robot.yaml") as f:
    cfg = yaml.safe_load(f)
model, _ = build_scene(cfg["robot"]["mjcf"], with_objects=args.objects)
robot = KinematicSimRobot(model=model)
safe = SafetySupervisor(robot, cfg)
safe.stand()
state = {"vx": 0.0, "wz": 0.0, "last": "-", "exit": False}

eyes = None
detector = None
if not args.no_eyes:
    eyes = mujoco.Renderer(model, height=EYE_H, width=EYE_W)    # created before the viewer
    intr = camera_intrinsics(model, EYE_W, EYE_H)
    with open("config/camera.yaml", "w") as f:
        yaml.safe_dump({"head_camera": intr}, f, sort_keys=False)
    os.makedirs("data/snapshots", exist_ok=True)
    if args.detect:
        from perception.detector import ObjectDetector, draw
        detector = ObjectDetector()

os.makedirs("logs", exist_ok=True)
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f"logs/run_{stamp}.csv"
events_path = f"logs/run_{stamp}_events.txt"
csv_file = open(csv_path, "w", newline="")
path_log = csv.writer(csv_file)
path_log.writerow(["time_s", "x_m", "y_m", "facing_deg", "speed_mps", "turn_radps", "status"])
events_file = open(events_path, "w")
t_start = time.time()
last_eye_image = None


def log_event(text):
    p = robot.pose
    line = (f"{time.time() - t_start:7.2f}s  {text:<15s} at x={p.x:+.2f} y={p.y:+.2f} "
            f"facing {facing_deg(p.yaw):+4.0f} deg | speed {state['vx']:+.2f} turn {state['wz']:+.2f}")
    events_file.write(line + "\n")
    events_file.flush()


def handle(key):
    k = key.lower()
    if safe.estopped and k in MOVE_KEYS:
        state["last"] = "PAUSED - press R"
        return

    state["last"] = EVENT_NAMES.get(k, repr(key))
    yaw, pitch = robot.get_head()
    if k == "w":
        state["vx"] += STEP_V
    elif k == "s":
        state["vx"] -= STEP_V
    elif k == "a":
        state["wz"] += STEP_W
    elif k == "d":
        state["wz"] -= STEP_W
    elif k == "q":
        state["vx"] = state["wz"] = 0.0
        safe.stop()
    elif k == " ":
        state["vx"] = state["wz"] = 0.0
        safe.emergency_stop()
    elif k == "r":
        state["vx"] = state["wz"] = 0.0
        safe.reset_emergency_stop()
    elif k == "i":
        robot.set_head(pitch_deg=pitch - HEAD_STEP)          # look up
    elif k == "k":
        robot.set_head(pitch_deg=pitch + HEAD_STEP)          # look down
    elif k == "j":
        robot.set_head(yaw_deg=yaw + HEAD_STEP)              # look left
    elif k == "l":
        robot.set_head(yaw_deg=yaw - HEAD_STEP)              # look right
    elif k == "c" and last_eye_image is not None:
        name = f"data/snapshots/eye_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"
        cv2.imwrite(name, last_eye_image)
        state["last"] = "SAVED PICTURE"
    elif k == "x":
        state["exit"] = True

    state["vx"], state["wz"] = round(state["vx"], 2), round(state["wz"], 2)
    safe.set_velocity(state["vx"], 0.0, state["wz"])
    state["vx"], _, state["wz"] = robot.velocity
    if k in EVENT_NAMES:
        log_event(EVENT_NAMES[k])


# ---------- main loop ----------
print(__doc__)
print(f"Logging to: {events_path}\n            {csv_path}\n")
log_event("START")
use_terminal = sys.stdin.isatty()
old_settings = termios.tcgetattr(sys.stdin) if use_terminal else None
if use_terminal:
    tty.setcbreak(sys.stdin.fileno())
frame = 0
try:
    with mujoco.viewer.launch_passive(robot.m, robot.d, key_callback=on_window_key) as viewer:
        with viewer.lock():
            viewer.cam.distance = 3.0
        while viewer.is_running() and not state["exit"]:
            start = time.time()
            if use_terminal:
                k = read_terminal_key()
                if k:
                    keys.put(k)
            while not keys.empty():
                handle(keys.get())

            safe.set_velocity(state["vx"], 0.0, state["wz"])
            state["vx"], _, state["wz"] = robot.velocity
            with viewer.lock():
                robot.step(DT)
                viewer.cam.lookat[:] = (robot.pose.x, robot.pose.y, robot.height * 0.6)
            viewer.sync()

            # ---- EYES: render the head camera into its own window ----
            if eyes is not None and frame % EYE_EVERY == 0:
                with viewer.lock():
                    eyes.update_scene(robot.d, camera=CAMERA_NAME)
                    rgb = eyes.render()
                last_eye_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                shown = last_eye_image.copy()
                if detector is not None:                         # Day 10: find objects
                    found = detector.detect(last_eye_image)
                    draw(shown, found, args.target)
                    hit = detector.find(found, args.target)
                    state["seen"] = (f"{args.target} at pixel {hit['centre']}" if hit
                                     else f"{args.target}: not seen")
                yaw, pitch = robot.get_head()
                cv2.putText(shown, f"head yaw {yaw:+.0f} pitch {pitch:+.0f} deg", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(EYES_WINDOW, shown)
                eye_key = cv2.waitKey(1) & 0xFF                 # keys typed in the EYES window
                if eye_key != 255 and eye_key < 128:
                    keys.put(chr(eye_key))

            p = robot.pose
            status = "PAUSED (E-STOP)" if safe.estopped else "ok"
            if frame % LOG_EVERY == 0:
                path_log.writerow([f"{time.time() - t_start:.2f}", f"{p.x:.3f}", f"{p.y:.3f}",
                                   f"{facing_deg(p.yaw):.1f}", f"{state['vx']:.2f}",
                                   f"{state['wz']:.2f}", "E-STOP" if safe.estopped else "ok"])
            frame += 1
            print(f"\rlast: {state['last']:17s} | speed {state['vx']:+.2f} m/s "
                  f"turn {state['wz']:+.2f} rad/s | x={p.x:+.2f} y={p.y:+.2f} "
                  f"facing {facing_deg(p.yaw):+4.0f} deg | {status:15s} {state.get('seen', ''):34s}", end="", flush=True)
            time.sleep(max(0.0, DT - (time.time() - start)))
finally:
    log_event("END")
    csv_file.close()
    events_file.close()
    if eyes is not None:
        cv2.destroyAllWindows()
        eyes.close()
    if use_terminal:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    print(f"\nStopped. Logs saved:\n  {events_path}\n  {csv_path}")
