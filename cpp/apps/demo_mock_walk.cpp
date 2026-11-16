// Day 5 demo: stand, walk, turn, walk back, stop - through the safety supervisor.
#include <cstdio>
#include "a3/config.hpp"
#include "a3/robot.hpp"
#include "a3/safety.hpp"

using namespace a3;
int main() {
  MockRobot robot;
  SafetySupervisor safe(robot, load_config("config/robot.yaml"));
  auto run = [&](double seconds, double vx, double vy, double wz) {
    safe.set_velocity(vx, vy, wz);
    for (int i = 0; i < (int)(seconds / 0.1); i++) robot.step(0.1);
    Pose2D p = safe.get_pose();
    std::printf("asked vx=%g, wz=%g for %gs -> x=%.2f m, y=%.2f m, facing %.0f deg\n", vx, wz, seconds, p.x, p.y,
                rad2deg(p.yaw));
  };
  safe.stand();
  run(2, 1.0, 0, 0);     // asks 1.0 m/s, safety limits it to 0.3 m/s
  run(2, 0, 0, 0.5);     // turn left
  run(2, -0.2, 0, 0);    // walk backward
  safe.stop();
  std::printf("Stopped.\n");
}
