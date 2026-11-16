// Arm IK (damped least squares with MuJoCo's Jacobian, joint limits respected, waist helps reach)
// and "magic grasp" (the A3 model has no finger joints: the object is attached to the hand).
#pragma once
#include <mujoco/mujoco.h>
#include <optional>
#include <string>
#include <vector>
#include "a3/util.hpp"

namespace a3 {
extern const Vec3 PALM_OFFSET;          // palm centre in the wrist_yaw frame (0.12 m forward)

class Arm {
 public:
  Arm(const mjModel* m, mjData* d, const std::string& side = "left", bool use_waist = true);
  Vec3 point(const Vec3& local = PALM_OFFSET) const;           // a point fixed to the hand, in the world
  Vec3 palm() const { return point(PALM_OFFSET); }
  double step_towards(const Vec3& target, double max_step = 0.03, double damping = 0.05,
                      const Vec3& tool = PALM_OFFSET);          // one IK step; remaining distance (m)
  double step_home(double max_step = 0.03);                     // remaining joint error (rad)
  std::string side;
  int hand;

 private:
  const mjModel* m_;
  mjData* d_;
  std::vector<int> qadr_, dadr_;
  std::vector<double> lo_, hi_, home_;
};

class MagicGrasp {
 public:
  MagicGrasp(const mjModel* m, mjData* d, const std::string& obj_name);
  Vec3 position() const;
  void attach(Arm* arm);
  void update();                                                // call every step while held
  void release(const std::optional<Vec3>& put_at = std::nullopt);
  std::string name;
  Arm* attached_to = nullptr;
  Vec3 local{0, 0, 0};                                          // object centre in the wrist frame
  int qadr;

 private:
  const mjModel* m_;
  mjData* d_;
  int body_;
};
}  // namespace a3
