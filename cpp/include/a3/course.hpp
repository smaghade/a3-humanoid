// The 85 m city course: stones, hill, steps, road with a zebra crossing and traffic,
// people and animals that walk across the robot's way.
#pragma once
#include <mujoco/mujoco.h>
#include <map>
#include <random>
#include <string>
#include <vector>
#include "a3/scene.hpp"
#include "a3/util.hpp"

namespace a3 {
constexpr double ROAD_X0 = -45.0, ROAD_X1 = -38.0;
constexpr double CROSSING_HALF_WIDTH = 1.5;
struct HillSpec { double xc = -60.0, sigma = 3.5, height = 0.6, half_x = 12.0, half_y = 9.0; };
struct StepsSpec { double start = -30.0; int n = 4; double rise = 0.12, depth = 0.4, platform = 2.0, half_y = 8.0; };
extern const HillSpec HILL;
extern const StepsSpec STEPS;
extern const double STEPS_END;

struct Mover {                                      // something that moves across the robot's way
  std::string name, kind;
  double x, y_min, y_max, speed, y;
  int dir;
  bool loop = false;
  double gap = 6.0, wait = 0.0, pause = 0.0;
  bool hidden = false, stopped_for_robot = false;
  int qadr = -1;
};
struct MoverState {                                 // true position/velocity (no-camera mode, reports)
  const Mover* mover;
  Vec2 pos, vel;
};

class Course {
 public:
  Course(double start_x, int n_stones, unsigned seed = 3);
  double ground(double x, double y) const;          // known ground height (hill, steps, road)
  bool on_road(double x) const { return ROAD_X0 <= x && x <= ROAD_X1; }
  void add_to_spec(mjSpec* spec) const;
  void bind(const mjModel* m, mjData* d);
  void update(double dt, Vec2 robot_xy);
  std::vector<MoverState> visible_movers() const;

  double start_x;
  std::vector<Obstacle> stones;
  std::vector<Mover> movers;
  std::map<std::string, double> min_dist;           // closest each mover came to the robot

 private:
  void make_stones(int n);
  void place_all();
  std::mt19937 rng_;
  double uniform(double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(rng_); }
  double normal(double mu, double sd) { return std::normal_distribution<double>(mu, sd)(rng_); }
  const mjModel* m_ = nullptr;
  mjData* d_ = nullptr;
};
}  // namespace a3
