// The world: the A3 + a head camera (+ tables, bottles, cup and optional stones).
// AGIBOT's original files are not changed: everything is added in memory with mjSpec.
#pragma once
#include <mujoco/mujoco.h>
#include <array>
#include <map>
#include <string>
#include <vector>

namespace a3 {
extern const char* HEAD_BODY;
extern const char* CAMERA_NAME;
extern const std::vector<std::string> OBJECTS;     // red_bottle, blue_bottle, yellow_bottle, cup
constexpr double TABLE_HEIGHT = 0.75;

struct Table {
  double cx, cy, hx, hy, fx, fy;                     // centre, half size, which way to face it
};
extern const std::map<std::string, Table> TABLES;   // table_1 (bottles), table_2 (empty)

struct Obstacle {                                    // a stone or box on the floor
  std::string name, shape;                           // shape: "stone" (ellipsoid) or "box"
  std::array<double, 3> pos, size;
  std::array<float, 4> rgba;
};

struct Scene {
  mjSpec* spec = nullptr;
  mjModel* model = nullptr;
};

// Build the scene. Call compile_scene() again after adding more things to scene.spec.
Scene build_scene(const std::string& mjcf_path, bool with_objects = true, double camera_fovy = 70.0,
                  double camera_tilt_deg = 30.0, const std::vector<Obstacle>& obstacles = {});
mjModel* compile_scene(mjSpec* spec);               // throws with MuJoCo's message on error

struct Intrinsics {
  double fx, fy, cx, cy, fovy_deg;
  int width, height;
};
Intrinsics camera_intrinsics(const mjModel* m, int width, int height, const std::string& cam = CAMERA_NAME);
void save_camera_yaml(const Intrinsics& in, const std::string& path = "config/camera.yaml");

// small mjSpec helpers (used by the course too)
mjsBody* add_body(mjSpec* spec, mjsBody* parent, const std::string& name, const std::array<double, 3>& pos);
mjsGeom* add_geom(mjsBody* body, mjtGeom type, const std::array<double, 3>& size,
                  const std::array<float, 4>& rgba, const std::array<double, 3>& pos = {0, 0, 0},
                  const double* quat = nullptr);
}  // namespace a3
