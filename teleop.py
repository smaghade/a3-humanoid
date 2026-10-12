"""Day 7: keyboard -> SafetySupervisor -> RobotInterface (MockRobot for now)."""
import math
import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist

from robot.mock_robot import MockRobot
from safety.supervisor import SafetySupervisor

HELP = """
  W / S  = faster forward / backward      A / D = turn left / right
  Q      = stop                           SPACE = EMERGENCY STOP
  R      = reset emergency stop           X     = exit
"""
STEP_V, STEP_W, DT = 0.1, 0.25, 0.1   # speed change per key press, loop time (s)


def get_key(timeout):
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.read(1) if ready else None


def main():
    rclpy.init()
    node = rclpy.create_node("teleop")
    pub = node.create_publisher(Twist, "cmd_vel", 10)

    robot = MockRobot()                          # later: SimulationRobot / PhysicalRobot
    safe = SafetySupervisor.from_yaml(robot)
    safe.stand()
    vx = wz = 0.0

    print(HELP)
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())            # read keys without pressing Enter
    try:
        while True:
            key = get_key(DT)
            if key:
                k = key.lower()
                if k == "w":
                    vx += STEP_V
                elif k == "s":
                    vx -= STEP_V
                elif k == "a":
                    wz += STEP_W
                elif k == "d":
                    wz -= STEP_W
                elif k == "q":
                    vx = wz = 0.0
                    safe.stop()
                elif key == " ":
                    vx = wz = 0.0
                    safe.emergency_stop()
                elif k == "r":
                    safe.reset_emergency_stop()
                elif k == "x":
                    break

            safe.set_velocity(vx, 0.0, wz)       # sent every loop, so the timeout stays fresh
            vx, _, wz = robot.velocity           # what safety actually allowed
            robot.step(DT)

            msg = Twist()
            msg.linear.x, msg.angular.z = vx, wz
            pub.publish(msg)                     # so rosbag2 can record it

            p = robot.pose
            state = "EMERGENCY STOP" if safe.estopped else "ok"
            print(f"\rspeed {vx:+.2f} m/s  turn {wz:+.2f} rad/s | "
                  f"x={p.x:+.2f} y={p.y:+.2f} facing {math.degrees(p.yaw):+5.0f} deg | {state}      ",
                  end="", flush=True)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        safe.stop()
        node.destroy_node()
        rclpy.shutdown()
        print("\nTeleop closed. Robot stopped.")


if __name__ == "__main__":
    main()
