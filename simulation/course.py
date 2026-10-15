"""A real-world style course between the robot and the tables.

From the start (about 85 m away) to the tables, the robot meets:
  - stones and boxes (static)
  - a HILL (smooth, 0.6 m high)                       around x = -60
  - a ROAD with a zebra crossing and traffic          x = -45 .. -38
      car + motorcycle in one lane, bicycle in the other; vehicles brake for the robot
      when it is on the crossing (like drivers stopping for a pedestrian)
  - STEPS up and down (4 x 12 cm, with a platform)    x = -30 .. -24.8
  - people and animals walking across the robot's way:
      person at x = -72, dog at x = -52, person at x = -20, cat at x = -10
People and animals walk across, rest 3-7 s at each end, and walk back. The robot must give
way to them; they only stop at the very last moment (0.9 m) if the robot is right in their way.
All moving things are simple shapes (boxes, cylinders, capsules) with muted colours, so the
bottle colour detector never mistakes them for bottles.
"""
import math

import mujoco
import numpy as np

ROAD = (-45.0, -38.0)              # x range of the road
LANES = {"lane_1": (-43.25, +1), "lane_2": (-39.75, -1)}   # x of lane centre, driving direction (y)
CROSSING_HALF_WIDTH = 1.5          # zebra crossing: |y| < 1.5
HILL = {"xc": -60.0, "sigma": 3.5, "height": 0.6, "half_x": 12.0, "half_y": 9.0}
STEPS = {"start": -30.0, "n": 4, "rise": 0.12, "depth": 0.4, "platform": 2.0, "half_y": 8.0}
STEPS_END = STEPS["start"] + 2 * STEPS["n"] * STEPS["depth"] + STEPS["platform"]
STONE_ZONES = [(-81, -74), (-70.5, -64.5), (-36, -31), (-24.3, -21.6), (-18.4, -11.6), (-8.4, -4)]

ROT_X90 = [0.7071068, 0.7071068, 0, 0]       # turns a cylinder's axis sideways (wheels)

SHAPES = {   # local frame: x = forward, z = up, ground at z = 0
    "car": [("box", [1.9, 0.85, 0.35], [0, 0, 0.62], None, [0.30, 0.35, 0.50, 1]),
            ("box", [1.0, 0.75, 0.30], [-0.2, 0, 1.27], None, [0.26, 0.30, 0.42, 1]),
            *[("cylinder", [0.33, 0.12], [sx * 1.25, sy * 0.86, 0.33], ROT_X90, [0.1, 0.1, 0.1, 1])
              for sx in (-1, 1) for sy in (-1, 1)]],
    "motorcycle": [("box", [0.8, 0.15, 0.2], [0, 0, 0.62], None, [0.50, 0.32, 0.30, 1]),
                   ("cylinder", [0.32, 0.07], [0.75, 0, 0.32], ROT_X90, [0.1, 0.1, 0.1, 1]),
                   ("cylinder", [0.32, 0.07], [-0.75, 0, 0.32], ROT_X90, [0.1, 0.1, 0.1, 1]),
                   ("capsule", [0.18, 0.33], [-0.1, 0, 1.2], None, [0.35, 0.33, 0.30, 1]),
                   ("sphere", [0.15], [-0.05, 0, 1.75], None, [0.25, 0.25, 0.28, 1])],
    "bicycle": [("box", [0.6, 0.03, 0.03], [0, 0, 0.7], None, [0.40, 0.40, 0.45, 1]),
                ("cylinder", [0.33, 0.02], [0.55, 0, 0.33], ROT_X90, [0.1, 0.1, 0.1, 1]),
                ("cylinder", [0.33, 0.02], [-0.55, 0, 0.33], ROT_X90, [0.1, 0.1, 0.1, 1]),
                ("capsule", [0.17, 0.33], [-0.05, 0, 1.15], None, [0.45, 0.42, 0.36, 1]),
                ("sphere", [0.12], [0.0, 0, 1.65], None, [0.55, 0.47, 0.40, 1])],
    "person": [("capsule", [0.08, 0.25], [0, 0.1, 0.33], None, [0.30, 0.30, 0.35, 1]),
               ("capsule", [0.08, 0.25], [0, -0.1, 0.33], None, [0.30, 0.30, 0.35, 1]),
               ("capsule", [0.2, 0.32], [0, 0, 1.12], None, [0.48, 0.42, 0.36, 1]),
               ("sphere", [0.12], [0, 0, 1.75], None, [0.60, 0.50, 0.42, 1])],
    "dog": [("ellipsoid", [0.40, 0.15, 0.16], [0, 0, 0.45], None, [0.45, 0.36, 0.26, 1]),
            ("sphere", [0.13], [0.45, 0, 0.62], None, [0.42, 0.34, 0.25, 1]),
            *[("capsule", [0.05, 0.14], [sx * 0.28, sy * 0.1, 0.18], None, [0.40, 0.32, 0.24, 1])
              for sx in (-1, 1) for sy in (-1, 1)]],
    "cat": [("ellipsoid", [0.22, 0.08, 0.09], [0, 0, 0.25], None, [0.42, 0.42, 0.40, 1]),
            ("sphere", [0.08], [0.25, 0, 0.35], None, [0.40, 0.40, 0.38, 1]),
            *[("capsule", [0.03, 0.08], [sx * 0.15, sy * 0.05, 0.1], None, [0.38, 0.38, 0.36, 1])
              for sx in (-1, 1) for sy in (-1, 1)]],
}
GEOM = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
        "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
        "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID}
VEHICLES = {"car", "motorcycle", "bicycle"}


class Mover:
    """Something that moves across the robot's way (back and forth, or round a road loop)."""

    def __init__(self, name, kind, x, y_range, speed, y0, direction, loop=False, gap=6.0):
        self.name, self.kind, self.x = name, kind, x
        self.y_min, self.y_max = y_range
        self.speed, self.y, self.dir = speed, y0, direction
        self.loop, self.gap, self.wait = loop, gap, 0.0
        self.hidden = False
        self.stopped_for_robot = False
        self.qadr = None

    def position(self):
        return np.array([self.x, self.y])


class Course:
    def __init__(self, start_x, n_stones, seed=3):
        self.start_x = start_x
        self.rng = np.random.default_rng(seed)
        self.road = ROAD
        self.movers = [
            Mover("person_1", "person", -72.0, (-5, 5), 1.0, -5.0, +1),
            Mover("dog", "dog", -52.0, (-4, 4), 1.5, 4.0, -1),
            Mover("car", "car", LANES["lane_1"][0], (-32, 32), 3.0, -12.0, +1, loop=True, gap=6),
            Mover("motorcycle", "motorcycle", LANES["lane_1"][0], (-32, 32), 4.5, -30.0, +1, loop=True, gap=9),
            Mover("bicycle", "bicycle", LANES["lane_2"][0], (-32, 32), 2.0, 20.0, -1, loop=True, gap=5),
            Mover("person_2", "person", -20.0, (-4, 4), 0.8, -4.0, +1),
            Mover("cat", "cat", -10.0, (-3, 3), 0.6, 3.0, -1),
        ]
        self.stones = self._make_stones(n_stones)
        self.min_dist = {m.name: 1e9 for m in self.movers}

    # ---------------- terrain height (the robot's map knows the hill, steps and road) ----------
    def ground(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        h = np.zeros(np.broadcast(x, y).shape)
        hx = (np.abs(x - HILL["xc"]) <= HILL["half_x"]) & (np.abs(y) <= HILL["half_y"])
        h = np.where(hx, HILL["height"] * np.exp(-0.5 * ((x - HILL["xc"]) / HILL["sigma"]) ** 2), h)
        for k in range(1, STEPS["n"] + 1):
            lo = STEPS["start"] + (k - 1) * STEPS["depth"]
            hi = STEPS_END - (k - 1) * STEPS["depth"]
            on = (x >= lo) & (x <= hi) & (np.abs(y) <= STEPS["half_y"])
            h = np.where(on, np.maximum(h, k * STEPS["rise"]), h)
        on_road = (x >= ROAD[0]) & (x <= ROAD[1])
        h = np.where(on_road, np.maximum(h, 0.01), h)
        return h if h.shape else float(h)

    def on_road(self, x):
        return ROAD[0] <= x <= ROAD[1]

    def _make_stones(self, n):
        stones, zones = [], [z for z in STONE_ZONES if z[0] > self.start_x + 3]
        for k in range(n):
            lo, hi = zones[k % len(zones)]
            x = self.rng.uniform(lo, hi)
            y = self.rng.normal(0.0, 0.8) if k % 3 else self.rng.uniform(-0.2, 0.2)
            g = float(self.ground(x, y))
            grey = self.rng.uniform(0.30, 0.55)
            rgba = [grey + self.rng.uniform(0, 0.06), grey, grey - self.rng.uniform(0, 0.06), 1]
            if k % 4 == 3:
                half = self.rng.uniform(0.15, 0.30)
                stones.append({"name": f"box_{k}", "shape": "box", "pos": [x, y, g + half],
                               "size": [half, half * self.rng.uniform(0.8, 1.5), half], "rgba": rgba})
            else:
                r = self.rng.uniform(0.18, 0.40)
                stones.append({"name": f"stone_{k}", "shape": "stone", "pos": [x, y, g + r * 0.5],
                               "size": [r, r * self.rng.uniform(0.7, 1.2), r * 0.6], "rgba": rgba})
        return stones

    # ---------------- build everything into the MuJoCo scene ----------------
    def add_to_spec(self, spec):
        world = spec.worldbody
        # hill: a smooth height field
        ncol, nrow = 160, 40
        xs = np.linspace(-HILL["half_x"], HILL["half_x"], ncol)
        prof = np.exp(-0.5 * (xs / HILL["sigma"]) ** 2)
        data = np.tile(prof, (nrow, 1)).flatten()
        spec.add_hfield(name="hill", size=[HILL["half_x"], HILL["half_y"], HILL["height"], 0.05],
                        nrow=nrow, ncol=ncol, userdata=data.tolist())
        hill = world.add_body(name="hill", pos=[HILL["xc"], 0, 0])
        hill.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="hill", rgba=[0.42, 0.48, 0.36, 1])
        # steps: stacked slabs (a staircase up, a platform, a staircase down)
        for k in range(1, STEPS["n"] + 1):
            lo = STEPS["start"] + (k - 1) * STEPS["depth"]
            hi = STEPS_END - (k - 1) * STEPS["depth"]
            b = world.add_body(name=f"step_{k}", pos=[(lo + hi) / 2, 0, (k - 0.5) * STEPS["rise"]])
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[(hi - lo) / 2, STEPS["half_y"], STEPS["rise"] / 2],
                       rgba=[0.62 - 0.03 * k, 0.60 - 0.03 * k, 0.56 - 0.03 * k, 1])
        # road, zebra crossing, lane line
        road = world.add_body(name="road", pos=[(ROAD[0] + ROAD[1]) / 2, 0, 0.005])
        road.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[(ROAD[1] - ROAD[0]) / 2, 40, 0.005],
                      rgba=[0.18, 0.18, 0.20, 1])
        for i in range(7):
            road.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.25, CROSSING_HALF_WIDTH, 0.002],
                          pos=[ROAD[0] + 0.5 + i * 1.0 - (ROAD[0] + ROAD[1]) / 2, 0, 0.006],
                          rgba=[0.9, 0.9, 0.9, 1])
        for yy in np.arange(-38, 39, 4.0):
            if abs(yy) > CROSSING_HALF_WIDTH + 1:
                road.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.06, 1.0, 0.002],
                              pos=[0, yy, 0.006], rgba=[0.9, 0.9, 0.9, 1])
        # stones and boxes
        for ob in self.stones:
            body = world.add_body(name=ob["name"], pos=ob["pos"])
            shape = mujoco.mjtGeom.mjGEOM_ELLIPSOID if ob["shape"] == "stone" else mujoco.mjtGeom.mjGEOM_BOX
            body.add_geom(type=shape, size=ob["size"], rgba=ob["rgba"])
        # moving things
        for m in self.movers:
            body = world.add_body(name=m.name, pos=[m.x, m.y, 0])
            body.add_freejoint()
            for geom, size, pos, quat, rgba in SHAPES[m.kind]:
                kw = {"quat": quat} if quat else {}
                size3 = list(size) + [0.0] * (3 - len(size))          # MuJoCo always takes 3 numbers
                body.add_geom(type=GEOM[geom], size=size3, pos=pos, rgba=rgba, **kw)

    def bind(self, model, data):
        self.m, self.d = model, data
        for mv in self.movers:
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mv.name)
            mv.qadr = model.jnt_qposadr[model.body_jntadr[body]]
        self._place_all()

    # ---------------- movement every simulation step ----------------
    def update(self, dt, robot_xy):
        rx, ry = robot_xy
        robot_on_road = self.on_road(rx)
        for mv in self.movers:
            mv.stopped_for_robot = False
            if mv.hidden:                                    # vehicle waiting off-screen to come again
                mv.wait -= dt
                if mv.wait <= 0:
                    mv.hidden, mv.y = False, (mv.y_min if mv.dir > 0 else mv.y_max)
                continue
            if getattr(mv, "pause", 0.0) > 0:               # resting at the end of its walk
                mv.pause -= dt
                self.min_dist[mv.name] = min(self.min_dist[mv.name], math.hypot(mv.x - rx, mv.y - ry))
                continue
            ahead = (ry - mv.y) * mv.dir                     # how far the robot is ahead of it
            if mv.kind in VEHICLES:
                blocked = robot_on_road and abs(rx - mv.x) < 2.5 and 0 < ahead < 9.0
                for other in self.movers:                    # keep distance to a vehicle in front
                    if other is not mv and other.kind in VEHICLES and not other.hidden and \
                            abs(other.x - mv.x) < 0.5 and 0 < (other.y - mv.y) * mv.dir < 7.0:
                        blocked = True
            else:   # people and animals only stop at the last moment, if the robot is right in their way
                blocked = abs(rx - mv.x) < 0.7 and 0 < ahead < 0.9
            if blocked:
                mv.stopped_for_robot = robot_on_road or mv.kind not in VEHICLES
            else:
                mv.y += mv.dir * mv.speed * dt
            if mv.loop and (mv.y > mv.y_max or mv.y < mv.y_min):
                mv.hidden, mv.wait = True, mv.gap
            elif not mv.loop and (mv.y > mv.y_max or mv.y < mv.y_min):
                mv.y = min(max(mv.y, mv.y_min), mv.y_max)
                mv.dir *= -1
                mv.pause = float(self.rng.uniform(3.0, 7.0))       # rest a moment, then walk back
            self.min_dist[mv.name] = min(self.min_dist[mv.name],
                                         math.hypot(mv.x - rx, mv.y - ry) if not mv.hidden else 1e9)
        self._place_all()

    def _place_all(self):
        for mv in self.movers:
            q = self.d.qpos
            if mv.hidden:
                q[mv.qadr:mv.qadr + 7] = [mv.x, 200.0 * mv.dir, -5.0, 1, 0, 0, 0]
                continue
            yaw = math.pi / 2 if mv.dir > 0 else -math.pi / 2
            q[mv.qadr:mv.qadr + 7] = [mv.x, mv.y, self.ground(mv.x, mv.y),
                                      math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]

    def visible_movers(self):
        """True positions and velocities (used only when the robot has no camera, and for reports)."""
        out = []
        for mv in self.movers:
            if not mv.hidden:
                v = 0.0 if (mv.stopped_for_robot or getattr(mv, "pause", 0.0) > 0) else mv.speed * mv.dir
                out.append((mv, np.array([mv.x, mv.y]), np.array([0.0, v])))
        return out
