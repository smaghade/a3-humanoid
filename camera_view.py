"""Week 2, Days 8-9: see through the robot's eyes.

One window, two pictures:
  LEFT  = what the A3's head camera sees
  RIGHT = an overview camera following the robot

Click the window, then use these keys:
  W / S = walk forward / back      A / D = turn left / right
  I / K = look up / down           J / L = look left / right
  Q = stop      SPACE = pause (e-stop)      R = resume
  C = save the camera picture to data/snapshots/
  X or Esc = exit
"""
import datetime
import math
import os
import time

import cv2
import mujoco
import numpy as np
import yaml

from robot.kinematic_sim_robot import KinematicSimRobot
from safety.supervisor import SafetySupervisor
from simulation.scene import CAMERA_NAME, build_scene, camera_intrinsics

W, H, DT = 640, 480, 0.02
STEP_V, STEP_W, HEAD_STEP = 0.1, 0.25, 3.0
WINDOW = "A3 camera (left) | overview (right)"

# ---------- build the scene ----------
with open("config/robot.yaml") as f:
    cfg = yaml.safe_load(f)
model, _ = build_scene(cfg["robot"]["mjcf"])
robot = KinematicSimRobot(model=model)
safe = SafetySupervisor(robot, cfg)
safe.stand()
renderer = mujoco.Renderer(model, height=H, width=W)

overview = mujoco.MjvCamera()
overview.distance, overview.azimuth, overview.elevation = 3.5, 210, -20

# ---------- Day 8: camera intrinsics (camera_info) ----------
intr = camera_intrinsics(model, W, H)
with open("config/camera.yaml", "w") as f:
    yaml.safe_dump({"head_camera": intr}, f, sort_keys=False)
print(__doc__)
print(f"Camera intrinsics saved to config/camera.yaml: fx={intr['fx']:.1f} fy={intr['fy']:.1f} "
      f"cx={intr['cx']:.0f} cy={intr['cy']:.0f} ({W}x{H}, fovy {intr['fovy_deg']:.0f} deg)")


def render(camera):
    renderer.update_scene(robot.d, camera=camera)
    return cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)


def label(img, lines, color=(255, 255, 255)):
    for i, text in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


vx = wz = 0.0
message = "ready"
os.makedirs("data/snapshots", exist_ok=True)
cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

while True:
    start = time.time()
    key = cv2.waitKey(1) & 0xFF
    if key != 255:
        k = chr(key).lower() if key < 128 else ""
        yaw, pitch = robot.get_head()
        if key == 27 or k == "x":
            break
        if safe.estopped and k in "wsad" and k:
            message = "PAUSED - press R"
        elif k == "w":
            vx += STEP_V
        elif k == "s":
            vx -= STEP_V
        elif k == "a":
            wz += STEP_W
        elif k == "d":
            wz -= STEP_W
        elif k == "q":
            vx = wz = 0.0
            message = "stopped"
        elif k == " ":
            vx = wz = 0.0
            safe.emergency_stop()
            message = "PAUSED (e-stop)"
        elif k == "r":
            vx = wz = 0.0
            safe.reset_emergency_stop()
            message = "resumed"
        elif k == "i":
            robot.set_head(pitch_deg=pitch - HEAD_STEP)      # negative pitch = look up
        elif k == "k":
            robot.set_head(pitch_deg=pitch + HEAD_STEP)      # positive pitch = look down
        elif k == "j":
            robot.set_head(yaw_deg=yaw + HEAD_STEP)          # positive yaw = look left
        elif k == "l":
            robot.set_head(yaw_deg=yaw - HEAD_STEP)
        elif k == "c":
            name = f"data/snapshots/eye_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"
            cv2.imwrite(name, render(CAMERA_NAME))
            message = f"saved {name}"
        vx, wz = round(vx, 2), round(wz, 2)

    safe.set_velocity(vx, 0.0, wz)
    vx, _, wz = robot.velocity                              # what safety allowed
    robot.step(DT)

    p = robot.pose
    overview.lookat[:] = (p.x, p.y, 0.8)
    eye = render(CAMERA_NAME)
    world = render(overview)

    yaw, pitch = robot.get_head()
    label(eye, ["ROBOT CAMERA", f"head yaw {yaw:+.0f}  pitch {pitch:+.0f} deg"])
    label(world, ["OVERVIEW",
                  f"speed {vx:+.2f} m/s  turn {wz:+.2f} rad/s",
                  f"x={p.x:+.2f} y={p.y:+.2f} facing {math.degrees(p.yaw):+.0f} deg",
                  "PAUSED - press R" if safe.estopped else message],
          (0, 0, 255) if safe.estopped else (255, 255, 255))
    cv2.imshow(WINDOW, np.hstack([eye, world]))
    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
        break
    time.sleep(max(0.0, DT - (time.time() - start)))

cv2.destroyAllWindows()
renderer.close()
print("Closed.")
