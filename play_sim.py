"""Drive the A3 in the MuJoCo viewer through the safety supervisor - with logging.

Keys work in BOTH the MuJoCo window and the terminal:
  Arrows (or W/S/A/D in terminal) = walk forward/back, turn left/right
  Enter (or Q)  = stop
  Space         = PAUSE (emergency stop)
  R             = RESUME (clear the emergency stop), then use the arrows again
  X (terminal)  = exit

Every run writes two files in logs/:
  run_<date>_events.txt  - each action in plain words
  run_<date>.csv         - position 10 times per second (for plot_log.py)
"""
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

import mujoco.viewer
import yaml

from robot.kinematic_sim_robot import KinematicSimRobot
from safety.supervisor import SafetySupervisor

STEP_V, STEP_W, DT = 0.1, 0.25, 0.02
LOG_EVERY = 5                                   # write a path row every 5 frames = 0.1 s
# MuJoCo window key codes -> same commands as the terminal keys
WINDOW_KEYS = {265: "w", 264: "s", 263: "a", 262: "d",   # arrows
               257: "q",                                  # Enter = stop
               32: " ",                                   # Space = pause (e-stop)
               82: "r", 268: "r"}                         # R or Home = resume
EVENT_NAMES = {"w": "FORWARD +", "s": "BACK / SLOWER", "a": "TURN LEFT", "d": "TURN RIGHT",
               "q": "STOP", " ": "PAUSE (E-STOP)", "r": "RESUME", "x": "EXIT"}
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


# ---------- setup ----------
with open("config/robot.yaml") as f:
    cfg = yaml.safe_load(f)
robot = KinematicSimRobot(cfg["robot"]["mjcf"])
safe = SafetySupervisor(robot, cfg)
safe.stand()
state = {"vx": 0.0, "wz": 0.0, "last": "-", "exit": False}

os.makedirs("logs", exist_ok=True)
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f"logs/run_{stamp}.csv"
events_path = f"logs/run_{stamp}_events.txt"
csv_file = open(csv_path, "w", newline="")
path_log = csv.writer(csv_file)
path_log.writerow(["time_s", "x_m", "y_m", "facing_deg", "speed_mps", "turn_radps", "status"])
events_file = open(events_path, "w")
t_start = time.time()


def log_event(text):
    p = robot.pose
    line = (f"{time.time() - t_start:7.2f}s  {text:<15s} at x={p.x:+.2f} y={p.y:+.2f} "
            f"facing {facing_deg(p.yaw):+4.0f} deg | speed {state['vx']:+.2f} turn {state['wz']:+.2f}")
    events_file.write(line + "\n")
    events_file.flush()


def handle(key):
    k = key.lower()

    # While paused, movement keys are blocked - tell the user how to resume
    if safe.estopped and k in MOVE_KEYS:
        state["last"] = "PAUSED - press R"
        return

    state["last"] = EVENT_NAMES.get(k, repr(key))
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
    elif k == " ":                                   # pause = emergency stop
        state["vx"] = state["wz"] = 0.0
        safe.emergency_stop()
    elif k == "r":                                   # resume: clear e-stop, start from standstill
        state["vx"] = state["wz"] = 0.0
        safe.reset_emergency_stop()
    elif k == "x":
        state["exit"] = True

    state["vx"], state["wz"] = round(state["vx"], 2), round(state["wz"], 2)
    safe.set_velocity(state["vx"], 0.0, state["wz"])
    state["vx"], _, state["wz"] = robot.velocity      # what safety allowed
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
            robot.step(DT)

            with viewer.lock():                       # camera follows the robot
                viewer.cam.lookat[:] = (robot.pose.x, robot.pose.y, robot.height * 0.6)
            viewer.sync()

            p = robot.pose
            status = "PAUSED (E-STOP)" if safe.estopped else "ok"
            if frame % LOG_EVERY == 0:
                path_log.writerow([f"{time.time() - t_start:.2f}", f"{p.x:.3f}", f"{p.y:.3f}",
                                   f"{facing_deg(p.yaw):.1f}", f"{state['vx']:.2f}",
                                   f"{state['wz']:.2f}", "E-STOP" if safe.estopped else "ok"])
            frame += 1
            print(f"\rlast: {state['last']:17s} | speed {state['vx']:+.2f} m/s "
                  f"turn {state['wz']:+.2f} rad/s | x={p.x:+.2f} y={p.y:+.2f} "
                  f"facing {facing_deg(p.yaw):+4.0f} deg | {status:15s}", end="", flush=True)
            time.sleep(max(0.0, DT - (time.time() - start)))
finally:
    log_event("END")
    csv_file.close()
    events_file.close()
    if use_terminal:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    print(f"\nStopped. Logs saved:\n  {events_path}\n  {csv_path}")
