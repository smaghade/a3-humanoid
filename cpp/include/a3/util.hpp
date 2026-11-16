// Small helpers shared by all modules (maths, names, time).
#pragma once
#include <mujoco/mujoco.h>
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace a3 {
constexpr double PI = 3.14159265358979323846;
inline double deg2rad(double d) { return d * PI / 180.0; }
inline double rad2deg(double r) { return r * 180.0 / PI; }
inline double clamp(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline double wrap_angle(double a) { return std::atan2(std::sin(a), std::cos(a)); }
inline double facing_deg(double yaw) {            // keep between -180 and +180
  double d = std::fmod(rad2deg(yaw) + 180.0, 360.0);
  if (d < 0) d += 360.0;
  return d - 180.0;
}
using Vec2 = std::array<double, 2>;
using Vec3 = std::array<double, 3>;

// MuJoCo name lookups that stop the program with a clear message when a name is missing.
int body_id(const mjModel* m, const std::string& name);
int joint_id(const mjModel* m, const std::string& name);
int camera_id(const mjModel* m, const std::string& name);
int joint_qadr(const mjModel* m, const std::string& name);
double monotonic_seconds();
std::string now_stamp();                            // e.g. 20261012_153012
std::string fmt(const char* f, ...);               // printf-style formatting into a std::string
}  // namespace a3
