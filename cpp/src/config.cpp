#include "a3/config.hpp"
#include <yaml-cpp/yaml.h>
#include <stdexcept>

namespace a3 {
Config load_config(const std::string& path) {
  YAML::Node y;
  try {
    y = YAML::LoadFile(path);
  } catch (const std::exception& e) {
    throw std::runtime_error("Cannot read " + path + " (run from the project folder): " + e.what());
  }
  Config c;
  if (y["backend"]) c.backend = y["backend"].as<std::string>();
  c.model = y["robot"]["model"].as<std::string>("");
  c.mjcf = y["robot"]["mjcf"].as<std::string>();
  auto s = y["safety"];
  c.max_linear_velocity_mps = s["max_linear_velocity_mps"].as<double>();
  c.max_angular_velocity_radps = s["max_angular_velocity_radps"].as<double>();
  c.joint_limit_margin_deg = s["joint_limit_margin_deg"].as<double>();
  c.command_timeout_s = s["command_timeout_s"].as<double>();
  for (const auto& kv : y["joint_limits_deg"])
    c.joint_limits_deg[kv.first.as<std::string>()] = {kv.second[0].as<double>(), kv.second[1].as<double>()};
  return c;
}
}  // namespace a3
