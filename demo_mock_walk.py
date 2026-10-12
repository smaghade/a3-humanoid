"""Day 5 demo: stand, walk, turn, walk back, stop - through the safety supervisor."""
import math
from robot.mock_robot import MockRobot
from safety.supervisor import SafetySupervisor

robot = MockRobot()
safe = SafetySupervisor.from_yaml(robot)


def run(seconds, vx=0.0, vy=0.0, wz=0.0):
    safe.set_velocity(vx, vy, wz)
    for _ in range(int(seconds / 0.1)):
        robot.step(0.1)
    p = safe.get_pose()
    print(f"asked vx={vx}, wz={wz} for {seconds}s -> x={p.x:.2f} m, y={p.y:.2f} m, "
          f"facing {math.degrees(p.yaw):.0f} deg")


safe.stand()
run(2, vx=1.0)    # asks 1.0 m/s, safety limits it to 0.3 m/s
run(2, wz=0.5)    # turn left
run(2, vx=-0.2)   # walk backward
safe.stop()
print("Stopped.")
