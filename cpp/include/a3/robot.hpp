// RobotInterface ("universal remote"), MockRobot (tests) and KinematicSimRobot (MuJoCo preview).
#pragma once
#include <mujoco/mujoco.h>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include "a3/util.hpp"

namespace a3 {
struct Pose2D {
  double x = 0, y = 0, yaw = 0;     // metres forward, metres left, radians
};

class RobotInterface {
 public:
  virtual ~RobotInterface() = default;
  virtual void stand() = 0;
  virtual void stop() = 0;
  virtual void set_velocity(double vx, double vy, double wz) = 0;   // m/s, m/s, rad/s
  virtual Pose2D get_pose() const = 0;
  virtual void emergency_stop() = 0;
};

// A fake robot for testing without any simulator.
class MockRobot : public RobotInterface {
 public:
  Pose2D pose;
  Vec3 velocity{0, 0, 0};
  bool standing = false, estopped = false;
  std::vector<std::string> log;     // every command, useful in tests
  void stand() override { standing = true; log.push_back("stand"); }
  void stop() override { velocity = {0, 0, 0}; log.push_back("stop"); }
  void set_velocity(double vx, double vy, double wz) override {
    velocity = {vx, vy, wz};
    log.push_back("set_velocity");
  }
  Pose2D get_pose() const override { return pose; }
  void emergency_stop() override { velocity = {0, 0, 0}; estopped = true; log.push_back("emergency_stop"); }
  void step(double dt);              // move the fake robot forward in time by dt seconds
};

// The A3 in MuJoCo, PREVIEW ONLY: the body glides and the legs/arms swing as an animation,
// with the feet kept on the ground (also on hills and steps). Replace with SimulationRobot later.
class KinematicSimRobot : public RobotInterface {
 public:
  KinematicSimRobot(mjModel* model, const std::string& root_body = "pelvis_link");
  ~KinematicSimRobot() override;
  KinematicSimRobot(const KinematicSimRobot&) = delete;
  KinematicSimRobot& operator=(const KinematicSimRobot&) = delete;

  void stand() override { velocity = {0, 0, 0}; }
  void stop() override { velocity = {0, 0, 0}; }
  void set_velocity(double vx, double vy, double wz) override { velocity = {vx, vy, wz}; }
  Pose2D get_pose() const override { return pose; }
  void emergency_stop() override { velocity = {0, 0, 0}; }

  // Turn the head within its joint limits. Positive pitch = look down, positive yaw = left.
  void set_head(const double* yaw_deg, const double* pitch_deg);
  void set_head_deg(double yaw_deg, double pitch_deg) { set_head(&yaw_deg, &pitch_deg); }
  std::pair<double, double> get_head() const;
  void step(double dt);

  mjModel* m;
  mjData* d;
  Pose2D pose;
  Vec3 velocity{0, 0, 0};
  bool animate_arms = true;                       // false while an arm is used for grasping
  std::function<double(double, double)> ground;   // optional ground height for hills/steps
  double height = 0;

 private:
  double lowest_point_z() const;
  int base_;
  double phase_ = 0, gz_ = 0;
  std::vector<int> robot_geoms_;
  std::map<std::string, int> joint_;
};
}  // namespace a3
