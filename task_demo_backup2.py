"""Give the robot an instruction in plain words; it plans and carries it out.

Example instructions:
  "pick the red bottle from the table and keep it on the empty table"
  "move the blue bottle to table 2"
  "put the red bottle back on table 1"
  "bring the yellow bottle to the other table"

What happens (a first version of plan Days 15-16):
  1. PLANNER  : the sentence -> a list of actions from a fixed menu (checked)
  2. EXECUTOR : runs each action, reports SUCCESS / FAILURE + reason, stops safely on failure
  Actions: GO_TO_AREA, FIND, APPROACH, GRASP, LIFT, GO_TO_TABLE, PLACE, RETURN

Usage:
  python task_demo.py --say "pick the red bottle and keep it on the empty table"
  python task_demo.py                 # asks you for instructions, one after another
  python task_demo.py --no-vision     # simulator gives object positions
  python task_demo.py --speedup 3 | --no-viewer

Field mode (real-world style): start far away with stones in between
  python task_demo.py --start-distance 70 --stones 15 --speedup 5 \
         --say "pick the red bottle from the table and keep it on the empty table"
  While walking, the robot recognises the stones with its depth camera, draws them on its
  own map (window "A3 MAP") and plans a path around them (A*). It walks back the same way.

City course (85 m): hill, steps, a road with traffic and a zebra crossing, people and animals
  python task_demo.py --course --speedup 5 \
         --say "pick the red bottle from the table and keep it on the empty table"
  The robot also tracks MOVING things (people, dog, cat, car, motorcycle, bicycle), gives way
  to them, stops at the road to look left and right, and crosses when the road is clear.

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
MAP_WINDOW = "A3 MAP - what the robot knows"
STAGING_BACK = 1.5       # stop this far in front of a table before searching it (m)
FAR = 3.0                # farther than this -> use the map + path planner to walk

ap = argparse.ArgumentParser()
ap.add_argument("--say", default=None, help="the instruction, in quotes")
ap.add_argument("--no-vision", action="store_true")
ap.add_argument("--speedup", type=float, default=1.0)
ap.add_argument("--no-viewer", action="store_true")
ap.add_argument("--start-distance", type=float, default=0.0,
                help="start this many metres in front of the bottle table (e.g. 70)")
ap.add_argument("--stones", type=int, default=None,
                help="number of stones/boxes on the way (default: 1 per 5 m when far away)")
ap.add_argument("--seed", type=int, default=3, help="changes where the stones are")
ap.add_argument("--course", action="store_true",
                help="city course: hill, steps, road with traffic, people and animals (85 m)")
args = ap.parse_args()
if args.course and args.start_distance <= 0:
    args.start_distance = 85.0


# =====================================================================================
# PLANNER: sentence -> checked list of actions
# =====================================================================================
ALLOWED = {"GO_TO_AREA", "FIND", "APPROACH", "GRASP", "LIFT", "GO_TO_TABLE", "PLACE", "RETURN"}
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
    plan = [("GO_TO_AREA", target), ("FIND", target), ("APPROACH", target), ("GRASP", target), ("LIFT", target),
            ("GO_TO_TABLE", dest), ("PLACE", dest), ("RETURN", "start")]
    assert all(a in ALLOWED for a, _ in plan)        # only actions from the fixed menu
    return plan, f"Plan: move the {colours[0]} bottle to {dest.replace('_', ' ')}, then come back."


# =====================================================================================
# Robot, scene, eyes
# =====================================================================================
with open("config/robot.yaml") as f:
    cfg = yaml.safe_load(f)


def make_stones(start_x, n, seed):
    """Random stones and boxes between the start and the tables (like a real room/yard)."""
    rng = np.random.default_rng(seed)
    stones, end_x = [], TABLES["table_1"]["center"][0] - TABLES["table_1"]["half"][0] - STAGING_BACK - 1.5
    for k in range(n):
        x = rng.uniform(start_x + 4.0, end_x)
        y = rng.normal(0.0, 0.8) if k % 3 else rng.uniform(-0.2, 0.2)    # every 3rd right on the way
        grey = rng.uniform(0.30, 0.55)
        rgba = [grey + rng.uniform(0, 0.06), grey, grey - rng.uniform(0, 0.06), 1]
        if k % 4 == 3:
            half = rng.uniform(0.15, 0.30)
            stones.append({"name": f"box_{k}", "shape": "box", "pos": [x, y, half],
                           "size": [half, half * rng.uniform(0.8, 1.5), half], "rgba": rgba})
        else:
            r = rng.uniform(0.15, 0.40)
            stones.append({"name": f"stone_{k}", "shape": "stone",
                           "pos": [x, y, r * 0.5], "size": [r, r * rng.uniform(0.7, 1.2), r * 0.6],
                           "rgba": rgba})
    return stones


START_X = TABLES["table_1"]["center"][0] - TABLES["table_1"]["half"][0] - args.start_distance \
    if args.start_distance > 0 else 0.0
n_stones = args.stones if args.stones is not None else (int(args.start_distance // 5) if args.start_distance > 5 else 0)
course = None
if args.course:
    from simulation.course import Course
    course = Course(START_X, n_stones, args.seed)
    STONES = course.stones
    model, spec = build_scene(cfg["robot"]["mjcf"], with_objects=True)
    course.add_to_spec(spec)
    model = spec.compile()
else:
    STONES = make_stones(START_X, n_stones, args.seed) if n_stones else []
    model, _ = build_scene(cfg["robot"]["mjcf"], with_objects=True, obstacles=STONES)
model.vis.map.znear = 0.02 / model.stat.extent      # camera sees from 2 cm, even in a big scene
robot = KinematicSimRobot(model=model)
robot.pose.x = START_X
if course is not None:
    course.bind(model, robot.d)
    robot.ground = course.ground
robot.step(0.0)
safe = SafetySupervisor(robot, cfg)
safe.stand()
START = (robot.pose.x, robot.pose.y, robot.pose.yaw)

camera = detector = draw = None
if not args.no_vision:
    from perception.detector import ObjectDetector, draw
    from perception.spatial import DepthCamera
    camera = DepthCamera(model, CAMERA_NAME)
    detector = ObjectDetector()

from navigation.grid_planner import OccupancyGrid
from navigation.obstacles import ObstacleDetector, draw_obstacles
grid = OccupancyGrid((min(START_X, 0.0) - 5, 5.0), (-6.0, 6.0))
obstacle_eye = ObstacleDetector(camera) if camera is not None else None
if camera is None:                                    # no camera: give the robot the true map
    for ob in STONES:
        cx, cy, _ = ob["pos"]
        r = max(ob["size"][:2])
        pts = [[cx + dx, cy + dy, 0.1] for dx in np.arange(-r, r + 0.01, 0.05)
               for dy in np.arange(-r, r + 0.01, 0.05)]
        grid.add_points(np.array(pts))
seen_obstacles = []                                   # what the robot recognised on the way
from navigation.tracker import Tracker
tracker = Tracker()
SIM = {"t": 0.0}                                       # simulation clock (seconds)
spotted_ids = set()                                    # moving things already announced
events = {"waits": 0, "crossings": 0, "gave_way": 0}
nav = {"active": False, "path": None, "goal": None}

viewer = None
frame = 0
t0 = time.time()
grasps = {name: MagicGrasp(model, robot.d, name) for name in OBJECTS if name.endswith("bottle")}
holding = {"grasp": None, "arm": None}          # what the robot currently holds
HOME = {name: g.position() for name, g in grasps.items()}   # where each bottle started
LAST_TABLE = {name: "table_1" for name in grasps}           # where the robot last saw/put each bottle


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
    SIM["t"] += DT
    if viewer is not None:
        with viewer.lock():
            if course is not None:
                course.update(DT, (robot.pose.x, robot.pose.y))
            robot.step(DT)
            if g:
                g.update()
            viewer.cam.lookat[:] = (robot.pose.x + 0.3 * math.cos(robot.pose.yaw),
                                    robot.pose.y + 0.3 * math.sin(robot.pose.yaw), 0.9)
        viewer.sync()
        if not viewer.is_running():
            raise KeyboardInterrupt
        if camera is not None and frame % 5 == 0 and not nav["active"]:
            look()
        time.sleep(DT / args.speedup)
    else:
        if course is not None:
            course.update(DT, (robot.pose.x, robot.pose.y))
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
# NAVIGATION with obstacle recognition (Day 17): depth camera -> map -> A* path
# =====================================================================================
def sense():
    """Look with the depth camera: recognise obstacles, follow moving things, update the map."""
    if obstacle_eye is None:
        return []
    if viewer is not None:
        with viewer.lock():
            rgb, depth = camera.capture(robot.d)
    else:
        rgb, depth = camera.capture(robot.d)
    here = (robot.pose.x, robot.pose.y)
    if course is None:                                    # flat floor, nothing moves
        pts, objects = obstacle_eye.detect(robot.d, depth, here)
        grid.add_points(pts)
    else:
        pts, objects = obstacle_eye.detect(robot.d, depth, here, ground=course.ground, track_range=25.0)
        tracker.update(objects, SIM["t"])
        for o in objects:
            tr = o["track"]
            o.update(label=tr.label, moving=tr.moving, speed=tr.speed)
            if o["near"] and tr.static:                   # only things that stay still go on the map
                grid.add_points(o["points"])
            if tr.moving and tr.id not in spotted_ids and o["distance"] < 20 \
                    and table_of(tr.pos) is None and not on_known_obstacle(tr.pos):
                spotted_ids.add(tr.id)
                say(f"   SPOTTED moving {tr.label}: {tr.speed:.1f} m/s, {o['distance']:.1f} m away")
    for o in objects:                                     # log each NEW still obstacle once
        if o["distance"] > 4.0 or o.get("moving") or (course is not None and not o["track"].static):
            continue
        if all(np.hypot(*(o["centre"] - k["centre"])) > 0.7 for k in seen_obstacles):
            seen_obstacles.append(o)
            say(f"   RECOGNISED {o['kind']} #{len(seen_obstacles)}: {o['height']:.2f} m high, "
                f"seen {o['distance']:.1f} m away at x={o['centre'][0]:.1f} y={o['centre'][1]:+.1f}")
    if viewer is not None:
        bgr = draw_obstacles(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), objects)
        cv2.putText(bgr, f"obstacles recognised: {len(seen_obstacles)}   moving things seen: {len(spotted_ids)}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(EYES_WINDOW, bgr)
        draw_map()
        cv2.waitKey(1)
    return objects


def draw_map(half=7.0, ppm=40):
    """Top-down map around the robot: terrain, obstacles, safety margin, path, moving things."""
    size = int(2 * half * ppm)
    img = np.full((size, size, 3), 235, np.uint8)
    rx, ry = robot.pose.x, robot.pose.y
    to_px = lambda x, y: (int((y - ry + half) * ppm), int((rx - x + half) * ppm))   # forward = up
    if course is not None:                                # known terrain from the map
        from simulation.course import ROAD, STEPS, STEPS_END, HILL
        for (x0, x1), colour, name in [(ROAD, (90, 90, 90), "ROAD"), ((STEPS["start"], STEPS_END), (150, 170, 190), "STEPS"),
                                       ((HILL["xc"] - 2 * HILL["sigma"], HILL["xc"] + 2 * HILL["sigma"]), (170, 205, 170), "HILL")]:
            a, b = to_px(x1, ry - half)[1], to_px(x0, ry - half)[1]
            if b >= 0 and a <= size:
                cv2.rectangle(img, (0, max(a, 0)), (size, min(b, size)), colour, -1)
                cv2.putText(img, name, (size - 70, max(a, 0) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
    i0, j0 = grid.cell(rx - half, ry - half)
    i1, j1 = grid.cell(rx + half, ry + half)
    for i in range(max(i0, 0), min(i1 + 1, grid.nx)):
        for j in range(max(j0, 0), min(j1 + 1, grid.ny)):
            if grid.blocked[i, j]:
                x, y = grid.centre(i, j)
                c = (40, 40, 40) if grid.occ[i, j] else (190, 190, 205)
                u, v = to_px(x, y)
                cv2.rectangle(img, (u - 2, v - 2), (u + 2, v + 2), c, -1)
    if nav["path"]:
        pts = [to_px(x, y) for x, y in [(rx, ry)] + nav["path"]]
        cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 160, 0), 2)
    if nav["goal"] is not None:
        cv2.circle(img, to_px(*nav["goal"]), 7, (0, 0, 220), -1)
    for k, o in enumerate(seen_obstacles, 1):
        u, v = to_px(*o["centre"])
        cv2.putText(img, f"#{k}", (u + 6, v), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 120, 255), 1, cv2.LINE_AA)
    for pos, vel, label in moving_things():               # moving things with their direction
        u, v = to_px(*pos)
        cv2.circle(img, (u, v), 9, (255, 0, 255), 2)
        e = to_px(*(pos + vel * 1.5))
        cv2.arrowedLine(img, (u, v), e, (255, 0, 255), 2, tipLength=0.3)
        cv2.putText(img, label, (u + 10, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 0, 160), 1, cv2.LINE_AA)
    u, v = to_px(rx, ry)
    cv2.circle(img, (u, v), 8, (200, 80, 0), -1)
    hx, hy = to_px(rx + 0.5 * math.cos(robot.pose.yaw), ry + 0.5 * math.sin(robot.pose.yaw))
    cv2.line(img, (u, v), (hx, hy), (200, 80, 0), 2)
    for k, line in enumerate([f"robot x={rx:+.1f} y={ry:+.1f}  status: {nav.get('status', '')}",
                              "dark = obstacle, light = safety margin, green = path",
                              "pink = moving thing + direction, red = goal"]):
        cv2.putText(img, line, (8, 18 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.imshow(MAP_WINDOW, img)


def on_known_obstacle(pos, radius=0.4):
    """Is there a still obstacle on the robot's map within `radius` of pos?"""
    i, j = grid.cell(*pos)
    r = int(radius / grid.res) + 1
    i0, i1, j0, j1 = max(i - r, 0), min(i + r + 1, grid.nx), max(j - r, 0), min(j + r + 1, grid.ny)
    return i0 < i1 and j0 < j1 and bool(grid.occ[i0:i1, j0:j1].any())


def moving_things():
    """(position, velocity, label) of the moving things the robot knows about right now."""
    if obstacle_eye is not None:
        out = []
        for tr in tracker.moving():
            if table_of(tr.pos) is not None or on_known_obstacle(tr.pos):
                continue          # part of a table/stone seen from a new angle - not really moving
            out.append((tr.pos, tr.vel, tr.label))
        return out
    if course is not None:                                # no camera: true positions
        here = np.array([robot.pose.x, robot.pose.y])
        return [(p, v, mv.kind) for mv, p, v in course.visible_movers() if np.hypot(*(p - here)) < 20]
    return []


_T = np.arange(0.0, 4.01, 0.25)


def closest_gap(here, heading, speed):
    """If the robot walks at `speed` (0 = stands, negative = backs up) for the next 4 s, how
    close does the nearest moving thing come? Only counts things that get CLOSER than they
    are now. Returns (gap m, label, distance now)."""
    things = moving_things()
    if not things:
        return (1e9, None, None)
    here = np.asarray(here, float)
    pos = np.array([p for p, _, _ in things])                      # (N, 2)
    vel = np.array([v for _, v, _ in things])
    fwd = np.array([math.cos(heading), math.sin(heading)]) * speed
    future_things = pos[:, None, :] + vel[:, None, :] * _T[None, :, None]     # (N, T, 2)
    future_robot = here[None, None, :] + fwd[None, None, :] * _T[None, :, None]
    gaps = np.hypot(*(future_things - future_robot).transpose(2, 0, 1))      # (N, T)
    # how much CLOSER than now each thing gets (moving away from something that is already
    # near is fine; moving towards it is not)
    gmin = gaps.min(axis=1)
    worst = np.where(gmin < gaps[:, 0] - 0.05, gmin, 1e9)
    k = int(np.argmin(worst))
    return float(worst[k]), things[k][2], float(gaps[k, 0])


CLEARANCE = 1.2          # keep at least this far from people, animals and vehicles (m)


def look_both_ways():
    """At the edge of the road: look left and right, wait until no vehicle is coming."""
    safe.stop()
    nav["status"] = "at the road: looking left and right"
    say("   at the road - stopping to look left and right")
    while True:
        for yaw in (60, -60, 0):
            robot.set_head(yaw_deg=yaw, pitch_deg=-10)       # look a little up: see further
            for k in range(int(1.2 / DT)):
                if k % 5 == 0:
                    sense()
                tick()
        here = np.array([robot.pose.x, robot.pose.y])
        coming = [(label, abs(pos[1] - here[1])) for pos, vel, label in moving_things()
                  if course.road[0] - 1 <= pos[0] <= course.road[1] + 1
                  and (here[1] - pos[1]) * vel[1] > 0 and abs(pos[1] - here[1]) < 25]
        if not coming:
            break
        label, d = min(coming, key=lambda c: c[1])
        say(f"   {label} coming, {d:.0f} m away - waiting for it to pass")
        events["waits"] += 1
        for k in range(int(2.0 / DT)):
            if k % 5 == 0:
                sense()
            tick()
    robot.set_head(yaw_deg=0, pitch_deg=0)
    events["crossings"] += 1
    say("   road is clear - crossing on the zebra crossing")


def navigate(goal, final_tol=0.05):
    """Walk to goal (x, y) smoothly: sense -> map -> plan -> follow; give way to moving things;
    look both ways before crossing a road."""
    nav.update(active=True, goal=tuple(goal), path=None, status="walking")
    replans, steps, last_logged, last_plan_t = 0, 0, None, -1e9
    cmd = [0.0, 0.0]                                       # current (forward speed, turn speed)
    waiting, clear_for, crossing_ok = False, 0.0, False

    def drive(vx, wz):                                     # smooth speed changes
        cmd[0] += max(-0.6 * DT, min(0.6 * DT, vx - cmd[0]))
        cmd[1] += max(-2.0 * DT, min(2.0 * DT, wz - cmd[1]))
        safe.set_velocity(cmd[0], 0.0, cmd[1])

    try:
        while True:
            if steps % 5 == 0:
                sense()
            here = (robot.pose.x, robot.pose.y)
            if math.hypot(goal[0] - here[0], goal[1] - here[1]) < final_tol:
                break
            if nav["path"] is None or (steps % 5 == 0 and SIM["t"] - last_plan_t > 1.0
                                       and grid.path_blocked(nav["path"], start=here)):
                path = grid.astar(here, goal)
                last_plan_t = SIM["t"]
                if path is None:
                    safe.stop()
                    return False, "no free path to the goal"
                nav["path"] = path[1:] if len(path) > 1 else [tuple(goal)]
                nav["path"][-1] = tuple(goal)
                replans += 1
                if replans > 1 and (last_logged is None or math.hypot(here[0] - last_logged[0],
                                                                       here[1] - last_logged[1]) > 1.0):
                    say("   new obstacle on the path - planning a way around it")
                    last_logged = here
            wx, wy = nav["path"][0]
            d = math.hypot(wx - here[0], wy - here[1])
            if d < (final_tol if len(nav["path"]) == 1 else 0.3):
                nav["path"].pop(0)
                if not nav["path"]:
                    break
                continue
            heading = math.atan2(wy - here[1], wx - here[0])

            # ---- road: stop at the edge, look both ways, then cross ----
            if course is not None:
                ahead_x = here[0] + 0.8 * math.cos(heading)
                if not course.on_road(here[0]) and course.on_road(ahead_x) and not crossing_ok:
                    cmd[:] = [0.0, 0.0]
                    look_both_ways()
                    crossing_ok = True
                    nav["status"] = "crossing the road"
                    continue
                if crossing_ok and not course.on_road(here[0]) and not course.on_road(ahead_x):
                    crossing_ok = False
                    nav["status"] = "walking"

            # ---- give way to moving things: walk on if safe, otherwise wait (or step back) ----
            nominal = min(0.3, max(0.05, d if len(nav["path"]) == 1 else 1.0))
            go = closest_gap(here, robot.pose.yaw, nominal)
            if waiting:                                            # only walk on when clearly safe
                if go[0] >= CLEARANCE + 0.3:
                    clear_for += DT
                else:
                    clear_for = 0.0
                if clear_for < 0.5:
                    stay = closest_gap(here, robot.pose.yaw, 0.0)
                    if stay[0] < 0.9 and closest_gap(here, robot.pose.yaw, -0.2)[0] > stay[0]:
                        safe.set_velocity(-0.2, 0.0, 0.0)          # too close even standing: back off
                        cmd[:] = [-0.2, 0.0]
                    else:
                        safe.stop()
                        cmd[:] = [0.0, 0.0]
                    tick()
                    steps += 1
                    continue
                waiting = False
                say("   way is clear - walking on")
                nav["status"] = "crossing the road" if crossing_ok else "walking"
            elif go[0] < CLEARANCE:
                stay = closest_gap(here, robot.pose.yaw, 0.0)
                if stay[0] >= go[0]:                               # standing still is safer: wait
                    say(f"   WAITING: {go[1]} {go[2]:.1f} m away is crossing the way")
                    events["gave_way"] += 1
                    nav["status"] = f"waiting for {go[1]}"
                    waiting, clear_for = True, 0.0
                    safe.stop()
                    cmd[:] = [0.0, 0.0]
                    tick()
                    steps += 1
                    continue
                # walking on is safer than standing (already in its way): keep going

            # ---- follow the path smoothly ----
            err = math.atan2(math.sin(heading - robot.pose.yaw), math.cos(heading - robot.pose.yaw))
            if abs(err) > 1.0:                                    # big turn: turn on the spot
                drive(0.0, math.copysign(0.5, err))
            else:
                drive(nominal * max(0.0, math.cos(err)), 2.0 * err)
            tick()
            steps += 1
    finally:
        safe.stop()
        nav.update(active=False, path=None, status="")
    return True, f"arrived ({replans - 1 if replans else 0} replans)"


def staging_point(table):
    t = TABLES[table]
    (cx, cy), (hx, hy), (fx, fy) = t["center"], t["half"], t["front"]
    depth = hx if fx else hy
    return (cx - (depth + STAGING_BACK) * fx, cy - (depth + STAGING_BACK) * fy), math.atan2(fy, fx)


# =====================================================================================
# EXECUTOR: each action returns (ok, reason)
# =====================================================================================
memory = {}            # what the robot learned during the task (object position, hand...)


def act_go_to_area(target):
    """Walk to a point in front of the table where the bottle was last seen."""
    table = LAST_TABLE[target]
    (sx, sy), yaw = staging_point(table)
    if math.hypot(sx - robot.pose.x, sy - robot.pose.y) < FAR:
        return True, f"already near {table.replace('_', ' ')}"
    say(f"   walking {math.hypot(sx - robot.pose.x, sy - robot.pose.y):.1f} m to {table.replace('_', ' ')}"
        f" (map starts empty; obstacles are added as they are seen)")
    ok, reason = navigate((sx, sy))
    if ok:
        turn_to(yaw)
    return ok, f"{reason}; {len(seen_obstacles)} obstacles recognised so far"


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
    LAST_TABLE[target] = table
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
    LAST_TABLE[g.name] = table
    holding.update(grasp=None, arm=None)
    reach(arm, arm.palm() - 0.10 * memory["face"])                # pull the hand back
    arm_home(arm)
    robot.animate_arms = True
    return True, f"placed on {table.replace('_', ' ')} at x={spot[0]:.2f} y={spot[1]:+.2f}"


def act_return(_):
    if math.hypot(START[0] - robot.pose.x, START[1] - robot.pose.y) > FAR:
        ok, reason = navigate((START[0], START[1]))
        if not ok:
            return False, reason
    else:
        walk_to(START[0], START[1])
    turn_to(START[2])
    return True, "back at the start position"


ACTIONS = {"GO_TO_AREA": act_go_to_area, "FIND": act_find, "APPROACH": act_approach, "GRASP": act_grasp, "LIFT": act_lift,
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
    if STONES:
        found = sum(1 for ob in STONES
                    if any(np.hypot(ob["pos"][0] - o["centre"][0], ob["pos"][1] - o["centre"][1]) < 0.8
                           for o in seen_obstacles))
        near = [ob for ob in STONES if abs(ob["pos"][1]) < 2.5]
        print(f"\nObstacle report: recognised {len(seen_obstacles)} obstacles on the way; "
              f"{found} of {len(STONES)} placed stones/boxes were recognised "
              f"({len(near)} of them lay within 2.5 m of the direct line).")
    if course is not None:
        print(f"Moving things: {len(spotted_ids)} sightings by the camera; the robot gave way "
              f"{events['gave_way']} times, crossed the road {events['crossings']} times "
              f"and waited for traffic {events['waits']} times.")
        print("Closest each moving thing came to the robot (centre to centre):")
        for name, d in course.min_dist.items():
            print(f"   {name:11s} {d:5.1f} m")
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
