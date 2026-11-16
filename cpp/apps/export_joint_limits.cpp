// Read joint limits from the A3 MuJoCo model and write config/robot.yaml.
#include <mujoco/mujoco.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include "a3/util.hpp"

using namespace a3;
int main() {
  const std::string MODEL = "a3_t3d0", MJCF = "A3-A3U-robot-model/" + MODEL + "/mjcf/" + MODEL + ".xml";
  char err[1000] = "";
  mjModel* m = mj_loadXML(MJCF.c_str(), nullptr, err, sizeof(err));
  if (!m) { std::printf("Cannot load %s: %s\n", MJCF.c_str(), err); return 1; }
  std::ofstream f("config/robot.yaml");
  f << "backend: simulation\n"
    << "robot: {model: " << MODEL << ", mjcf: " << MJCF << "}\n"
    << "safety: {max_linear_velocity_mps: 0.3, max_angular_velocity_radps: 0.5, joint_limit_margin_deg: 5.0,\n"
    << "  command_timeout_s: 0.5}\n"
    << "joint_limits_deg:\n";
  int n = 0;
  for (int i = 0; i < m->njnt; i++) {
    if (!m->jnt_limited[i] || m->jnt_type[i] != mjJNT_HINGE) continue;
    double lo = std::round(rad2deg(m->jnt_range[2 * i]) * 10) / 10, hi = std::round(rad2deg(m->jnt_range[2 * i + 1]) * 10) / 10;
    f << "  " << mj_id2name(m, mjOBJ_JOINT, i) << ": [" << fmt("%.1f", lo) << ", " << fmt("%.1f", hi) << "]\n";
    n++;
  }
  std::printf("Saved %d joint limits to config/robot.yaml\n", n);
  mj_deleteModel(m);
}
