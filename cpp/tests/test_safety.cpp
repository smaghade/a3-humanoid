// The 5 safety tests (same as tests/test_safety.py). Run from the project folder:
//   ./cpp/build/test_safety          (or: cd cpp/build && ctest)
#include <cmath>
#include <cstdio>
#include <functional>
#include "a3/config.hpp"
#include "a3/robot.hpp"
#include "a3/safety.hpp"

using namespace a3;
static int failed = 0, passed = 0;
static void check(const char* name, const std::function<bool()>& test) {
  bool ok = false;
  try { ok = test(); } catch (const std::exception& e) { std::printf("  error: %s\n", e.what()); }
  std::printf("%s %s\n", ok ? "PASSED" : "FAILED", name);
  (ok ? passed : failed)++;
}

int main() {
  Config cfg = load_config("config/robot.yaml");
  check("test_walking_speed_is_limited", [&] {
    MockRobot r; SafetySupervisor s(r, cfg);
    s.set_velocity(5.0, 0.0, 0.0);                       // ask for 5 m/s (far too fast)
    return r.velocity[0] <= 0.3 + 1e-9;                  // robot only gets 0.3 m/s
  });
  check("test_turning_speed_is_limited", [&] {
    MockRobot r; SafetySupervisor s(r, cfg);
    s.set_velocity(0.0, 0.0, -10.0);
    return r.velocity[2] == -0.5;
  });
  check("test_emergency_stop_blocks_motion", [&] {
    MockRobot r; SafetySupervisor s(r, cfg);
    s.emergency_stop();
    s.set_velocity(0.2, 0.0, 0.0);
    return r.velocity == Vec3{0, 0, 0};
  });
  check("test_timeout_stops_robot", [&] {
    MockRobot r; SafetySupervisor s(r, cfg);
    s.set_velocity(0.2, 0.0, 0.0);
    bool stopped = s.check_timeout(*s.last_cmd_time + 1.0);
    return stopped && r.velocity == Vec3{0, 0, 0};
  });
  check("test_joint_limits_with_margin", [&] {
    MockRobot r; SafetySupervisor s(r, cfg);
    return s.is_joint_target_safe("left_knee_joint", 90) &&       // normal bend: OK
           !s.is_joint_target_safe("left_knee_joint", 140) &&     // too close to the 143 limit
           !s.is_joint_target_safe("unknown_joint", 0);           // unknown joint: rejected
  });
  std::printf("%d passed, %d failed\n", passed, failed);
  return failed ? 1 : 0;
}
