"""Give the robot an instruction in plain words; it plans and carries it out.

Example instructions:
  "pick the red bottle from the table and keep it on the empty table"
  "move the blue bottle to table 2"
  "put the red bottle back on table 1"
  "bring the yellow bottle to the other table"

What happens (a first version of plan Days 15-16):
  1. PLANNER  : the sentence -> a list of actions from a fixed menu (checked)
  2. EXECUTOR : runs each action, reports SUCCESS / FAILURE + reason, stops safely on failure
  Actions: FIND, APPROACH, GRASP, LIFT, GO_TO_TABLE, PLACE, RETURN

Usage:
  python task_demo.py --say "pick the red bottle and keep it on the empty table"
  python task_demo.py                 # asks you for instructions, one after another
  python task_demo.py --no-vision     # simulator gives object positions
  python task_demo.py --speedup 3 | --no-viewer

Tables are known places (the robot's map): table_1 has the bottles, table_2 is empty (grey).
Keys in the MuJoCo window: Space = pause (e-stop), R = resume.
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
from simulation.scene import CAMERA_NAME, OBJECTS, TABLE_HEIGHT, TABLES, build_scene

DT = 0.02
STAND_BACK = 0.45        # stand this far (m) from the object/spot
SIDE_OFFSET = 0.20       # shift sideways so the object is in front of the chosen hand
GRASP_DIST = 0.02
LIFT = 0.15
ARM_STEP = 0.004
BOTTLE_RADIUS = 0.035
BOTTLE_Z = TABLE_HEIGHT + 0.12            # bottle centre height when standing on a table
HEAD_SCAN = [0, 40, -40]
MIN_HEIGHT, MAX_HEIGHT = 0.3, 1.8
SPOT_SHIFTS = [0.0, -0.15, 0.15, -0.30, 0.30]   # free places along a table, from its middle
EYES_WINDOW = "A3 EYES - what the robot sees"
COLOURS = ["red", "blue", "yellow"]

ap = argparse.ArgumentParser()
ap.add_argument("--say", default=None, help="the instruction, in quotes")
ap.add_argument("--no-vision", action="store_true")
ap.add_argument("--speedup", type=float, default=1.0)
ap.add_argument("--no-viewer", action="store_true")
args = ap.parse_args()


# =====================================================================================
# PLANNER: sentence -> checked list of actions
# =====================================================================================
ALLOWED = {"FIND", "APPROACH", "GRASP", "LIFT", "GO_TO_TABLE", "PLACE", "RETURN"}
TABLE_WORDS = {
    "table_2": ["table 2", "table two", "second table", "other table", "empty table", "new table",
                "grey table", "gray table", "another table"],
    "table_1": ["table 1", "table one", "first table", "same table", "original table", "back"],
}


def make_plan(sentence):
    """Returns (plan, message). plan is None if the instruction is not understood."""
    t = " " + sentence.lower() + " "
    colours = [c for c in COLOURS if f" {c} " in t or f" {c}-" in t]
    if "cup" in t and "bottle" not in t:
        return None, "Sorry, I can only move bottles for now (red, blue or yellow)."
    if len(colours) != 1:
        return None, "Which bottle? Please say exactly one colour: red, blue or yellow."
    target = f"{colours[0]}_bottle"
    dest = None
    for table, words in TABLE_WORDS.items():          # table_2 words are checked first
        if any(w in t for w in words):
            dest = table
            break
    if dest is None and any(w in t for w in [" keep ", " put ", " place ", " move ", " bring ", " take "]):
        dest = "table_2"                              # "move it" with no table named -> the other table
    if dest is None:
        return None, "Where should I put it? Say for example 'on table 2' or 'on the empty table'."
    plan = [("FIND", target), ("APPROACH", target), ("GRASP", target), ("LIFT", target),
            ("GO_TO_TABLE", dest), ("PLACE", dest), ("RETURN", "start")]
    assert all(a in ALLOWED for a, _ in plan)        # only actions from the fixed menu
    return plan, f"Plan: move the {colours[0]} bottle to {dest.replace('_', ' ')}, then come back."


# =====================================================================================
# Robot, scene, eyes
# =====================================================================================
with open("config/robot.yaml") as f:
    cfg = yaml.safe_load(f)
model, _ = build_scene(cfg["robot"]["mjcf"], with_objects=True)
robot = KinematicSimRobot(model=model)
safe = SafetySupervisor(robot, cfg)
safe.stand()
START = (robot.pose.x, robot.pose.y, robot.pose.yaw)

camera = detector = draw = None
if not args.no_vision:
    from perception.detector import ObjectDetector, draw
    from perception.spatial import DepthCamera
    camera = DepthCamera(model, CAMERA_NAME)
    detector = ObjectDetector()

viewer = None
frame = 0
t0 = time.time()
grasps = {name: MagicGrasp(model, robot.d, name) for name in OBJECTS if name.endswith("bottle")}
holding = {"grasp": None, "arm": None}          # what the robot currently holds
HOME = {name: g.position() for name, g in grasps.items()}   # where each bottle started


def say(text):
    p = robot.pose
    print(f"[{time.time() - t0:6.1f}s] {text:<54s} robot at x={p.x:+.2f} y={p.y:+.2f} "
          f"facing {(math.degrees(p.yaw) + 180) % 360 - 180:+.0f} deg")


def on_key(keycode):
    if keycode == 32:
        safe.emergency_stop()
    elif keycode == 82:
        safe.reset_emergency_stop()


def look(note=""):
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
        shown = draw(bgr.copy(), found)
        if note:
            cv2.putText(shown, note, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(EYES_WINDOW, shown)
        cv2.waitKey(1)
    return found, depth


def tick():
    global frame
    while safe.estopped:
        if viewer is not None:
            if not viewer.is_running():
                raise KeyboardInterrupt
            viewer.sync()
        time.sleep(0.05)
    g = holding["grasp"]
    if viewer is not None:
        with viewer.lock():
            robot.step(DT)
            if g:
                g.update()
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
        if g:
            g.update()
    frame += 1


# ---------- movement helpers ----------
def turn_to(yaw):
    while True:
        err = math.atan2(math.sin(yaw - robot.pose.yaw), math.cos(yaw - robot.pose.yaw))
        if abs(err) < math.radians(0.5):
            break
        safe.set_velocity(0.0, 0.0, math.copysign(min(2.0, max(0.05, 1.5 * abs(err))), err))
        tick()
    safe.stop()


def walk_to(x, y):
    if math.hypot(x - robot.pose.x, y - robot.pose.y) < 0.01:
        return
    turn_to(math.atan2(y - robot.pose.y, x - robot.pose.x))
    while True:
        dx, dy = x - robot.pose.x, y - robot.pose.y
        remaining = math.hypot(dx, dy)
        if remaining < 0.005:
            break
        err = math.atan2(math.sin(math.atan2(dy, dx) - robot.pose.yaw),
                         math.cos(math.atan2(dy, dx) - robot.pose.yaw))
        safe.set_velocity(min(0.3, max(0.05, remaining)), 0.0, 2.0 * err if remaining > 0.05 else 0.0)
        tick()
    safe.stop()


def reach(arm, target, tol=0.01, tool=None, max_steps=3000):
    for _ in range(max_steps):
        if safe.estopped:
            tick()
            continue
        dist = arm.step_towards(target, max_step=ARM_STEP) if tool is None else \
            arm.step_towards(target, max_step=ARM_STEP, tool=tool)
        if dist < tol:
            return True
        tick()
    return False


def arm_home(arm):
    while arm.step_home(max_step=ARM_STEP) > 1e-3:
        tick()


def table_of(point):
    """Which known table is this point on (or None)."""
    for name, t in TABLES.items():
        (cx, cy), (hx, hy) = t["center"], t["half"]
        if abs(point[0] - cx) <= hx + 0.05 and abs(point[1] - cy) <= hy + 0.05:
            return name
    return None


def stand_spot(point, table, side):
    """Where to stand (x, y) and which way to face, to use `side` hand on `point`."""
    fx, fy = TABLES[table]["front"]
    left = np.array([-fy, fx])                       # robot's left when facing the table
    shift = -SIDE_OFFSET if side == "left" else SIDE_OFFSET
    xy = np.array(point[:2]) - STAND_BACK * np.array([fx, fy]) + shift * left
    return xy, math.atan2(fy, fx)


def object_positions(skip=None):
    """Positions of every object (bottles AND cup), except the one being carried."""
    out = []
    for name in OBJECTS:
        if skip is not None and name == skip.name:
            continue
        out.append(robot.d.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy())
    return out


def free_spot(table, target=None):
    """A free place on a table: the object's ORIGINAL spot if that is on this table and free,
    otherwise a free place 20 cm in from the table's front edge."""
    others = object_positions(skip=holding["grasp"])
    if target is not None and table_of(HOME[target]) == table:
        home = HOME[target]
        if all(np.hypot(*(home[:2] - o[:2])) > 0.12 for o in others):
            return home.copy()
    t = TABLES[table]
    (cx, cy), (hx, hy), (fx, fy) = t["center"], t["half"], t["front"]
    depth = hx if fx else hy
    along = np.array([-fy, fx])
    base = np.array([cx, cy]) - (depth - 0.20) * np.array([fx, fy])
    for s in SPOT_SHIFTS:
        p = base + s * along
        if all(np.hypot(*(p - o[:2])) > 0.12 for o in others):
            return np.array([p[0], p[1], BOTTLE_Z])
    return None


# =====================================================================================
# EXECUTOR: each action returns (ok, reason)
# =====================================================================================
memory = {}            # what the robot learned during the task (object position, hand...)


def act_find(target):
    words = target.replace("_", " ")
    truth = grasps[target].position()
    if camera is None:
        memory["object"] = truth
        return True, "position from the simulator"
    for _ in range(8):
        for head_yaw in HEAD_SCAN:
            robot.set_head(yaw_deg=head_yaw, pitch_deg=0)
            for _ in range(10):
                tick()
            found, depth = look(f"searching for {words}...")
            for hit in detector.find_all(found, words):
                u, v = hit["centre"]
                p = camera.locate(robot.d, depth, u, v, object_radius=BOTTLE_RADIUS)
                if p is None or not (MIN_HEIGHT < p[2] < MAX_HEIGHT):
                    continue
                robot.set_head(yaw_deg=0, pitch_deg=0)
                memory["object"] = p
                err = np.linalg.norm(p - truth) * 100
                return True, f"seen at x={p[0]:.2f} y={p[1]:+.2f} (error {err:.1f} cm)"
        turn_to(robot.pose.yaw + math.radians(45))
    robot.set_head(yaw_deg=0, pitch_deg=0)
    return False, f"{words} not seen anywhere"


def act_approach(target):
    p = memory["object"]
    table = table_of(p)
    if table is None:
        return False, "the object is not on a known table"
    fx, fy = TABLES[table]["front"]
    left = np.array([-fy, fx])
    rel = np.array(p[:2]) - np.array(TABLES[table]["center"])
    side = "left" if rel @ left >= 0 else "right"
    xy, yaw = stand_spot(p, table, side)
    walk_to(*xy)
    turn_to(yaw)
    memory.update(side=side, from_table=table, face=np.array([fx, fy, 0.0]))
    return True, f"at {table.replace('_', ' ')}, using the {side} hand"


def act_grasp(target):
    p, face = memory["object"], memory["face"]
    robot.animate_arms = False
    arm = Arm(model, robot.d, memory["side"])
    reach(arm, p - (BOTTLE_RADIUS + 0.10) * face)
    ok = reach(arm, p - (BOTTLE_RADIUS + 0.03) * face, tol=GRASP_DIST)
    g = grasps[target]
    if not ok or np.linalg.norm(arm.palm() - g.position()) > 0.12:
        arm_home(arm)
        robot.animate_arms = True
        return False, "hand did not reach the bottle"
    g.attach(arm)
    holding.update(grasp=g, arm=arm)
    return True, "grasped (magic grasp)"


def act_lift(target):
    g, arm = holding["grasp"], holding["arm"]
    up = g.position() + [0, 0, LIFT]
    reach(arm, up, tool=g.local)
    return True, f"lifted {LIFT * 100:.0f} cm"


def act_go_to_table(table):
    spot = free_spot(table, holding["grasp"].name if holding["grasp"] else None)
    if spot is None:
        return False, f"no free place on {table.replace('_', ' ')}"
    xy, yaw = stand_spot(spot, table, memory["side"])
    walk_to(*xy)
    turn_to(yaw)
    memory["spot"] = spot
    memory["face"] = np.array([*TABLES[table]["front"], 0.0])
    return True, f"at {table.replace('_', ' ')}, place spot x={spot[0]:.2f} y={spot[1]:+.2f}"


def act_place(table):
    g, arm, spot = holding["grasp"], holding["arm"], memory["spot"]
    reach(arm, spot + [0, 0, 0.10], tool=g.local)                  # above the spot
    reach(arm, spot, tol=0.005, tool=g.local)                     # lower it
    g.release(put_at=spot)                                         # stand it upright exactly there
    holding.update(grasp=None, arm=None)
    reach(arm, arm.palm() - 0.10 * memory["face"])                # pull the hand back
    arm_home(arm)
    robot.animate_arms = True
    return True, f"placed on {table.replace('_', ' ')} at x={spot[0]:.2f} y={spot[1]:+.2f}"


def act_return(_):
    walk_to(START[0], START[1])
    turn_to(START[2])
    return True, "back at the start position"


ACTIONS = {"FIND": act_find, "APPROACH": act_approach, "GRASP": act_grasp, "LIFT": act_lift,
           "GO_TO_TABLE": act_go_to_table, "PLACE": act_place, "RETURN": act_return}


def recover():
    """After a failure: put down anything held (where the arm is), arm home, go back."""
    g, arm = holding["grasp"], holding["arm"]
    if g is not None:
        p = g.position()
        g.release(put_at=[p[0], p[1], BOTTLE_Z] if table_of(p) else None)
        holding.update(grasp=None, arm=None)
    if arm is not None:
        arm_home(arm)
    robot.animate_arms = True
    act_return(None)


def execute(plan):
    memory.clear()
    for i, (action, arg) in enumerate(plan, 1):
        say(f"[{i}/{len(plan)}] {action} {arg}")
        ok, reason = ACTIONS[action](arg)
        say(f"      {'SUCCESS' if ok else 'FAILURE'}: {reason}")
        if not ok:
            say("Stopping safely and going back to the start.")
            recover()
            return False
    say("TASK COMPLETE")
    return True


def run():
    sentences = [args.say] if args.say else None
    while True:
        sentence = sentences.pop(0) if sentences else (None if args.say else input("\nInstruction (or 'quit'): "))
        if sentence is None or sentence.strip().lower() in ("quit", "exit", "q"):
            return
        plan, message = make_plan(sentence)
        print(f'\n"{sentence}"\n{message}')
        if plan:
            for i, (a, x) in enumerate(plan, 1):
                print(f"   {i}. {a} {x}")
            execute(plan)
        if args.say:
            return


print(__doc__)
try:
    if args.no_viewer:
        run()
    else:
        with mujoco.viewer.launch_passive(model, robot.d, key_callback=on_key) as v:
            viewer = v
            with viewer.lock():
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.5, 200, -25
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
