#include "a3/arm.hpp"
#include <algorithm>
#include <cmath>

namespace a3 {
const Vec3 PALM_OFFSET = {0.12, 0.0, 0.0};

Arm::Arm(const mjModel* m, mjData* d, const std::string& s, bool use_waist) : side(s), m_(m), d_(d) {
  std::vector<std::string> names = {s + "_shoulder_pitch_joint", s + "_shoulder_roll_joint", s + "_shoulder_yaw_joint",
                                    s + "_elbow_joint", s + "_wrist_roll_joint", s + "_wrist_pitch_joint",
                                    s + "_wrist_yaw_joint"};
  if (use_waist) names.insert(names.begin(), "waist_pitch_joint");
  for (const auto& n : names) {
    int j = joint_id(m, n);
    qadr_.push_back(m->jnt_qposadr[j]);
    dadr_.push_back(m->jnt_dofadr[j]);
    lo_.push_back(m->jnt_range[2 * j]);
    hi_.push_back(m->jnt_range[2 * j + 1]);
    home_.push_back(d->qpos[m->jnt_qposadr[j]]);
  }
  hand = body_id(m, s + "_wrist_yaw_Link");
}

Vec3 Arm::point(const Vec3& l) const {
  const double* R = d_->xmat + 9 * hand;
  const double* p = d_->xpos + 3 * hand;
  Vec3 out;
  for (int i = 0; i < 3; i++) out[i] = p[i] + R[3 * i] * l[0] + R[3 * i + 1] * l[1] + R[3 * i + 2] * l[2];
  return out;
}

double Arm::step_towards(const Vec3& target, double max_step, double damping, const Vec3& tool) {
  mj_forward(m_, d_);
  Vec3 p = point(tool);
  double err[3] = {target[0] - p[0], target[1] - p[1], target[2] - p[2]};
  std::vector<double> jacp(3 * m_->nv);
  mj_jac(m_, d_, jacp.data(), nullptr, p.data(), hand);
  int n = (int)dadr_.size();
  std::vector<double> J(3 * n);                       // columns of the Jacobian for our joints
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < n; c++) J[r * n + c] = jacp[r * m_->nv + dadr_[c]];
  double A[9];                                        // A = J J^T + damping^2 I
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++) {
      double s = 0;
      for (int k = 0; k < n; k++) s += J[r * n + k] * J[c * n + k];
      A[3 * r + c] = s + (r == c ? damping * damping : 0.0);
    }
  double det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
  double inv[9] = {(A[4] * A[8] - A[5] * A[7]) / det, (A[2] * A[7] - A[1] * A[8]) / det, (A[1] * A[5] - A[2] * A[4]) / det,
                   (A[5] * A[6] - A[3] * A[8]) / det, (A[0] * A[8] - A[2] * A[6]) / det, (A[2] * A[3] - A[0] * A[5]) / det,
                   (A[3] * A[7] - A[4] * A[6]) / det, (A[1] * A[6] - A[0] * A[7]) / det, (A[0] * A[4] - A[1] * A[3]) / det};
  double y[3];                                        // y = A^-1 err
  for (int r = 0; r < 3; r++) y[r] = inv[3 * r] * err[0] + inv[3 * r + 1] * err[1] + inv[3 * r + 2] * err[2];
  for (int c = 0; c < n; c++) {
    double dq = J[c] * y[0] + J[n + c] * y[1] + J[2 * n + c] * y[2];      // J^T y
    dq = clamp(dq, -max_step, max_step);                                  // small, smooth moves
    d_->qpos[qadr_[c]] = clamp(d_->qpos[qadr_[c]] + dq, lo_[c], hi_[c]);   // never past a limit
  }
  mj_forward(m_, d_);
  Vec3 q = point(tool);
  return std::sqrt(std::pow(target[0] - q[0], 2) + std::pow(target[1] - q[1], 2) + std::pow(target[2] - q[2], 2));
}

double Arm::step_home(double max_step) {
  double worst = 0;
  for (size_t c = 0; c < qadr_.size(); c++) {
    double q = d_->qpos[qadr_[c]];
    d_->qpos[qadr_[c]] = q + clamp(home_[c] - q, -max_step, max_step);
  }
  mj_forward(m_, d_);
  for (size_t c = 0; c < qadr_.size(); c++) worst = std::max(worst, std::fabs(home_[c] - d_->qpos[qadr_[c]]));
  return worst;
}

MagicGrasp::MagicGrasp(const mjModel* m, mjData* d, const std::string& obj) : name(obj), m_(m), d_(d) {
  body_ = body_id(m, obj);
  qadr = m->jnt_qposadr[m->body_jntadr[body_]];
}
Vec3 MagicGrasp::position() const { return {d_->xpos[3 * body_], d_->xpos[3 * body_ + 1], d_->xpos[3 * body_ + 2]}; }
void MagicGrasp::attach(Arm* arm) {
  attached_to = arm;
  const double* R = d_->xmat + 9 * arm->hand;
  const double* h = d_->xpos + 3 * arm->hand;
  Vec3 p = position();
  double r[3] = {p[0] - h[0], p[1] - h[1], p[2] - h[2]};
  for (int i = 0; i < 3; i++) local[i] = R[i] * r[0] + R[3 + i] * r[1] + R[6 + i] * r[2];   // R^T r
}
void MagicGrasp::update() {
  if (!attached_to) return;
  Vec3 p = attached_to->point(local);
  double* q = d_->qpos + qadr;
  q[0] = p[0]; q[1] = p[1]; q[2] = p[2];
  q[3] = 1; q[4] = 0; q[5] = 0; q[6] = 0;            // stays upright
  mj_forward(m_, d_);
}
void MagicGrasp::release(const std::optional<Vec3>& put_at) {
  attached_to = nullptr;
  if (put_at) {
    double* q = d_->qpos + qadr;
    q[0] = (*put_at)[0]; q[1] = (*put_at)[1]; q[2] = (*put_at)[2];
    q[3] = 1; q[4] = 0; q[5] = 0; q[6] = 0;
    mj_forward(m_, d_);
  }
}
}  // namespace a3
