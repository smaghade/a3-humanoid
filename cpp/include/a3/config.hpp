// config/robot.yaml (same file as the Python version).
#pragma once
#include <map>
#include <string>
#include <utility>

namespace a3 {
struct Config {
  std::string backend = "simulation";
  std::string model, mjcf;
  double max_linear_velocity_mps = 0.3, max_angular_velocity_radps = 0.5;
  double joint_limit_margin_deg = 5.0, command_timeout_s = 0.5;
  std::map<std::string, std::pair<double, double>> joint_limits_deg;
};
Config load_config(const std::string& path = "config/robot.yaml");
}  // namespace a3
