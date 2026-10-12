"""Scripted mission: the A3 walks a route from a YAML file - no keyboard needed.

Usage:
  python mission_sim.py                          # runs missions/demo_route.yaml
  python mission_sim.py missions/other.yaml      # another route file
  python mission_sim.py --speedup 5              # 5x faster than real time
  python mission_sim.py --no-viewer              # no window, finishes instantly (for testing)
"""
import argparse
import csv
import datetime
import math
import os
import time

import mujoco.viewer
import yaml

from robot.kinematic_sim_robot import KinematicSimRobot
from safety.supervisor import SafetySupervisor

DT = 0.02                      # simulation step (s)
LOG_EVERY = 5                  # write a path row every 0.1 s
POS_TOL = 0.001                # stop within 1 mm of the target distance
YAW_TOL = math.radians(0.1)    # stop turning within 0.1 degrees


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def deg(a):
    return math.degrees(wrap(a))


class MissionRunner:
    def __init__(self, robot, safe, viewer, speedup, log_prefix):
        self.robot, self.safe, self.viewer, self.speedup = robot, safe, viewer, speedup
        self.sim_time = 0.0
        self.frame = 0
        self.current = "-"
        self.heading = robot.pose.yaw   # intended direction
        self.csv_file = open(log_prefix + ".csv", "w", newline="")
        self.path_log = csv.writer(self.csv_file)
        self.path_log.writerow(["time_s", "x_m", "y_m", "facing_deg", "speed_mps", "turn_radps", "status"])
        self.events = open(log_prefix + "_events.txt", "w")

    def event(self, text):
        p = self.robot.pose
        line = (f"{self.sim_time:7.2f}s  {text:<26s} at x={p.x:+6.2f} y={p.y:+6.2f} "
                f"facing {deg(p.yaw):+4.0f} deg")
        self.events.write(line + "\n")
        self.events.flush()
        print(f"\r{line}{' ' * 30}")

    def tick(self):
        """Advance the simulation by one step: move, log, draw."""
        if self.safe.estopped:
            raise RuntimeError("emergency stop active")
        self.robot.step(DT)
        self.sim_time += DT
        p = self.robot.pose
        vx, _, wz = self.robot.velocity
        if self.frame % LOG_EVERY == 0:
            status = "E-STOP" if self.safe.estopped else "ok"
            self.path_log.writerow([f"{self.sim_time:.2f}", f"{p.x:.3f}", f"{p.y:.3f}",
                                    f"{deg(p.yaw):.1f}", f"{vx:.2f}", f"{wz:.2f}", status])
        self.frame += 1
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise KeyboardInterrupt
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = (p.x, p.y, self.robot.height * 0.5)
            self.viewer.sync()
            time.sleep(DT / self.speedup)
            if self.frame % 10 == 0:
                print(f"\r  {self.current:<28s} | speed {vx:+.2f} turn {wz:+.2f} | "
                      f"x={p.x:+6.2f} y={p.y:+6.2f} facing {deg(p.yaw):+4.0f} deg   ", end="", flush=True)

    # ---- the basic moves ----
    def walk(self, distance, vx_sign, vy_sign, speed):
        p = self.robot.pose                      # updated in place as the robot moves
        x0, y0 = p.x, p.y
        while True:
            remaining = distance - math.hypot(p.x - x0, p.y - y0)
            if remaining <= POS_TOL:
                break
            v = min(speed, max(0.05, remaining), remaining / DT)  # slow down, land exactly
            steer = 2.0 * (self.heading - p.yaw)  # keep walking straight
            self.safe.set_velocity(vx_sign * v, vy_sign * v, steer)
            self.tick()
        self.safe.stop()

    def turn(self, angle_deg):
        self.heading += math.radians(angle_deg)   # absolute target: errors don't add up
        while True:
            err = self.heading - self.robot.pose.yaw
            if abs(err) <= YAW_TOL:
                break
            wz = math.copysign(min(2.0, max(0.05, 1.5 * abs(err))), err)  # safety limits this
            self.safe.set_velocity(0.0, 0.0, wz)
            self.tick()
        self.safe.stop()

    def wait(self, seconds):
        self.safe.stop()
        for _ in range(int(round(seconds / DT))):
            self.tick()

    def close(self):
        p = self.robot.pose                        # always save the final position
        self.path_log.writerow([f"{self.sim_time:.2f}", f"{p.x:.3f}", f"{p.y:.3f}",
                                f"{deg(p.yaw):.1f}", "0.00", "0.00", "ok"])
        self.csv_file.close()
        self.events.close()


ACTIONS = {
    "forward":    lambda r, v, s: r.walk(v, +1, 0, s),
    "backward":   lambda r, v, s: r.walk(v, -1, 0, s),
    "side_left":  lambda r, v, s: r.walk(v, 0, +1, s),
    "side_right": lambda r, v, s: r.walk(v, 0, -1, s),
    "turn_left":  lambda r, v, s: r.turn(+v),
    "turn_right": lambda r, v, s: r.turn(-v),
    "wait":       lambda r, v, s: r.wait(v),
    "stop":       lambda r, v, s: r.safe.stop(),
}
DIRECTIONS = {"forward": (1, 0), "backward": (-1, 0), "side_left": (0, 1), "side_right": (0, -1)}


def expected_end(steps):
    """Where the robot SHOULD end up if every step is perfect."""
    x = y = yaw = 0.0
    for step in steps:
        (action, value), = step.items()
        if action in DIRECTIONS:
            dx, dy = DIRECTIONS[action]
            x += value * (dx * math.cos(yaw) - dy * math.sin(yaw))
            y += value * (dx * math.sin(yaw) + dy * math.cos(yaw))
        elif action == "turn_left":
            yaw += math.radians(value)
        elif action == "turn_right":
            yaw -= math.radians(value)
    return x, y, yaw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mission", nargs="?", default="missions/demo_route.yaml")
    ap.add_argument("--speedup", type=float, default=1.0)
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()

    with open("config/robot.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(args.mission) as f:
        mission = yaml.safe_load(f)
    speed, steps = mission.get("speed", 0.3), mission["steps"]
    for step in steps:
        (action, _), = step.items()
        if action not in ACTIONS:
            raise SystemExit(f"Unknown step '{action}' in {args.mission}. Allowed: {list(ACTIONS)}")

    robot = KinematicSimRobot(cfg["robot"]["mjcf"])
    safe = SafetySupervisor(robot, cfg)
    safe.stand()

    os.makedirs("logs", exist_ok=True)
    prefix = f"logs/run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ex, ey, eyaw = expected_end(steps)
    print(f"Mission: {args.mission}  ({len(steps)} steps, speed {speed} m/s)")
    print(f"Expected end: x={ex:+.2f} y={ey:+.2f} facing {deg(eyaw):+.0f} deg\n")

    def run(viewer):
        runner = MissionRunner(robot, safe, viewer, args.speedup, prefix)
        runner.event("MISSION START")
        try:
            for i, step in enumerate(steps, 1):
                (action, value), = step.items()
                runner.current = f"step {i}/{len(steps)}: {action} {value}"
                runner.event(f"START {action} {value}")
                ACTIONS[action](runner, value, speed)
                runner.event(f"DONE  {action} {value}")
            runner.event("MISSION COMPLETE")
        except KeyboardInterrupt:
            runner.event("MISSION ABORTED (stopped by user)")
        except RuntimeError as err:
            runner.event(f"MISSION ABORTED: {err}")
        finally:
            safe.stop()
            runner.close()

        p = robot.pose
        error = math.hypot(p.x - ex, p.y - ey)
        print(f"\nExpected end: x={ex:+.2f} y={ey:+.2f} facing {deg(eyaw):+.0f} deg")
        print(f"Actual end:   x={p.x:+.2f} y={p.y:+.2f} facing {deg(p.yaw):+.0f} deg")
        print(f"Position error: {error * 100:.1f} cm   |   time: {runner.sim_time:.1f} s")
        print(f"Logs: {prefix}_events.txt  and  {prefix}.csv")

    if args.no_viewer:
        run(None)
    else:
        with mujoco.viewer.launch_passive(robot.m, robot.d) as viewer:
            with viewer.lock():
                viewer.cam.distance = 8.0       # zoom out to see the turns
                viewer.cam.elevation = -45
            run(viewer)
            print("\nMission finished. Close the MuJoCo window to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)


if __name__ == "__main__":
    main()
