#include "a3/robot.hpp"
#include <algorithm>
#include <limits>

namespace a3 {
void MockRobot::step(double dt) {
  double c = std::cos(pose.yaw), s = std::sin(pose.yaw);
  pose.x += (velocity[0] * c - velocity[1] * s) * dt;
  pose.y += (velocity[0] * s + velocity[1] * c) * dt;
  pose.yaw += velocity[2] * dt;
}

KinematicSimRobot::KinematicSimRobot(mjModel* model, const std::string& root_body) : m(model) {
  m->opt.gravity[0] = m->opt.gravity[1] = m->opt.gravity[2] = 0;   // no falling in preview mode
  d = mj_makeData(m);
  base_ = joint_qadr(m, "free_joint");
  int root = body_id(m, root_body);
  for (int g = 0; g < m->ngeom; g++)                 // only the robot's own parts
    if (m->body_rootid[m->geom_bodyid[g]] == root) robot_geoms_.push_back(g);
  for (const char* n : {"left_hip_pitch_joint", "right_hip_pitch_joint", "left_knee_joint",
                        "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint"})
    joint_[n] = joint_qadr(m, n);
  step(0.0);
  height = d->qpos[base_ + 2];
}
KinematicSimRobot::~KinematicSimRobot() { mj_deleteData(d); }

void KinematicSimRobot::set_head(const double* yaw_deg, const double* pitch_deg) {
  const char* names[2] = {"head_yaw_joint", "head_pitch_joint"};
  const double* vals[2] = {yaw_deg, pitch_deg};
  for (int i = 0; i < 2; i++) {
    if (!vals[i]) continue;
    int j = joint_id(m, names[i]);
    d->qpos[m->jnt_qposadr[j]] = clamp(deg2rad(*vals[i]), m->jnt_range[2 * j], m->jnt_range[2 * j + 1]);
  }
}
std::pair<double, double> KinematicSimRobot::get_head() const {
  return {rad2deg(d->qpos[joint_qadr(m, "head_yaw_joint")]), rad2deg(d->qpos[joint_qadr(m, "head_pitch_joint")])};
}

double KinematicSimRobot::lowest_point_z() const {
  // lowest corner of every robot part's bounding box, in the world
  double low = std::numeric_limits<double>::infinity();
  for (int g : robot_geoms_) {
    const double* aabb = m->geom_aabb + 6 * g;
    const double* R = d->geom_xmat + 9 * g;          // row 2 = how local axes map to world z
    for (int sx = -1; sx <= 1; sx += 2)
      for (int sy = -1; sy <= 1; sy += 2)
        for (int sz = -1; sz <= 1; sz += 2) {
          double lx = aabb[0] + sx * aabb[3], ly = aabb[1] + sy * aabb[4], lz = aabb[2] + sz * aabb[5];
          double z = d->geom_xpos[3 * g + 2] + R[6] * lx + R[7] * ly + R[8] * lz;
          low = std::min(low, z);
        }
  }
  return low;
}

void KinematicSimRobot::step(double dt) {
  double vx = velocity[0], vy = velocity[1], wz = velocity[2];
  double c = std::cos(pose.yaw), s = std::sin(pose.yaw);
  pose.x += (vx * c - vy * s) * dt;
  pose.y += (vx * s + vy * c) * dt;
  pose.yaw += wz * dt;
  double* q = d->qpos;
  int b = base_;
  q[b] = pose.x;
  q[b + 1] = pose.y;
  q[b + 3] = std::cos(pose.yaw / 2); q[b + 4] = 0; q[b + 5] = 0; q[b + 6] = std::sin(pose.yaw / 2);
  double moving = std::fabs(vx) + std::fabs(vy) + 0.3 * std::fabs(wz);
  if (moving > 1e-3) phase_ += dt * 2 * PI * 1.5;    // about 1.5 steps per second
  double amp = std::min(1.0, moving / 0.3), swing = std::sin(phase_);
  q[joint_["left_hip_pitch_joint"]] = -0.35 * amp * swing;
  q[joint_["right_hip_pitch_joint"]] = 0.35 * amp * swing;
  q[joint_["left_knee_joint"]] = 0.6 * amp * std::max(0.0, swing);
  q[joint_["right_knee_joint"]] = 0.6 * amp * std::max(0.0, -swing);
  if (animate_arms) {
    q[joint_["left_shoulder_pitch_joint"]] = 0.3 * amp * swing;
    q[joint_["right_shoulder_pitch_joint"]] = -0.3 * amp * swing;
  }
  mj_forward(m, d);
  double gz = 0.0;
  if (ground) {                                       // walk up/down hills and steps smoothly
    double target = ground(pose.x, pose.y), change = target - gz_;
    gz_ += dt <= 0 ? change : std::max(-0.6 * dt, std::min(0.6 * dt, change));
    gz = gz_;
  }
  q[b + 2] -= lowest_point_z() - gz;                  // feet touch the ground
  mj_forward(m, d);
}
}  // namespace a3
