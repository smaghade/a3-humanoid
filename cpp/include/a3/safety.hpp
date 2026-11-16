// SafetySupervisor: checks every command before it reaches the robot.
#pragma once
#include <optional>
#include <stdexcept>
#include "a3/config.hpp"
#include "a3/robot.hpp"

namespace a3 {
class SafetySupervisor : public RobotInterface {
 public:
  SafetySupervisor(RobotInterface& robot, const Config& cfg);
  void stand() override;
  void stop() override { robot_.stop(); }
  void set_velocity(double vx, double vy, double wz) override;
  Pose2D get_pose() const override { return robot_.get_pose(); }
  void emergency_stop() override;
  void reset_emergency_stop() { estopped = false; }
  bool check_timeout(std::optional<double> now = std::nullopt);       // rule 4
  bool is_joint_target_safe(const std::string& joint, double angle_deg) const;   // rule 5

  double max_v, max_w, timeout, margin_deg;
  std::map<std::string, std::pair<double, double>> joint_limits;
  bool estopped = false;
  std::optional<double> last_cmd_time;

 private:
  RobotInterface& robot_;
};
}  // namespace a3
