#include "a3/safety.hpp"
#include <algorithm>
#include <cmath>

namespace a3 {
SafetySupervisor::SafetySupervisor(RobotInterface& robot, const Config& c)
    : max_v(c.max_linear_velocity_mps), max_w(c.max_angular_velocity_radps),
      timeout(c.command_timeout_s), margin_deg(c.joint_limit_margin_deg),
      joint_limits(c.joint_limits_deg), robot_(robot) {}

void SafetySupervisor::stand() {
  if (estopped) throw std::runtime_error("Emergency stop is active");
  robot_.stand();
}
void SafetySupervisor::set_velocity(double vx, double vy, double wz) {
  if (estopped) {                                  // rule 1: no motion during emergency stop
    robot_.stop();
    return;
  }
  double speed = std::hypot(vx, vy);
  if (speed > max_v) {                             // rule 2: limit walking speed
    double k = max_v / speed;
    vx *= k;
    vy *= k;
  }
  wz = std::max(-max_w, std::min(max_w, wz));      // rule 3: limit turning speed
  last_cmd_time = monotonic_seconds();
  robot_.set_velocity(vx, vy, wz);
}
bool SafetySupervisor::check_timeout(std::optional<double> now) {
  double t = now ? *now : monotonic_seconds();
  if (last_cmd_time && t - *last_cmd_time > timeout) {
    robot_.stop();
    last_cmd_time.reset();
    return true;
  }
  return false;
}
bool SafetySupervisor::is_joint_target_safe(const std::string& joint, double angle_deg) const {
  auto it = joint_limits.find(joint);
  if (it == joint_limits.end()) return false;
  return it->second.first + margin_deg <= angle_deg && angle_deg <= it->second.second - margin_deg;
}
void SafetySupervisor::emergency_stop() {
  estopped = true;
  robot_.emergency_stop();
}
}  // namespace a3
