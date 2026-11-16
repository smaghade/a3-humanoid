#include "a3/spatial.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace a3 {
DepthCamera::DepthCamera(const mjModel* m, Graphics& gfx, const std::string& camera, int w, int h)
    : cam(camera_id(m, camera)), width(w), height(h), gfx_(gfx) {
  double fovy = deg2rad(m->cam_fovy[cam]);
  fy = fx = (h / 2.0) / std::tan(fovy / 2);
  cx = w / 2.0;
  cy = h / 2.0;
}
void DepthCamera::capture(const mjData* d, cv::Mat& bgr, cv::Mat& depth) { gfx_.capture(d, cam, bgr, depth, width, height); }

Vec3 DepthCamera::pixel_to_camera(double u, double v, double z) const {
  return {(u - cx) * z / fx, (v - cy) * z / fy, z};
}
Vec3 DepthCamera::camera_to_world(const mjData* d, const Vec3& p) const {
  const double* R = d->cam_xmat + 9 * cam;          // MuJoCo cameras look along -z with +y up
  const double* t = d->cam_xpos + 3 * cam;
  double pm[3] = {p[0], -p[1], -p[2]};
  Vec3 out;
  for (int i = 0; i < 3; i++) out[i] = t[i] + R[3 * i] * pm[0] + R[3 * i + 1] * pm[1] + R[3 * i + 2] * pm[2];
  return out;
}
Vec3 DepthCamera::world_to_pixel(const mjData* d, const Vec3& w) const {
  const double* R = d->cam_xmat + 9 * cam;
  const double* t = d->cam_xpos + 3 * cam;
  double r[3] = {w[0] - t[0], w[1] - t[1], w[2] - t[2]}, pm[3];
  for (int i = 0; i < 3; i++) pm[i] = R[i] * r[0] + R[3 + i] * r[1] + R[6 + i] * r[2];   // R^T * r
  double z = -pm[2];
  return {cx + fx * pm[0] / z, cy - fy * pm[1] / z, z};
}
std::optional<Vec3> DepthCamera::locate(const mjData* d, const cv::Mat& depth, double u_, double v_, int window,
                                        double object_radius) const {
  int u = (int)std::lround(u_), v = (int)std::lround(v_);
  std::vector<float> patch;                          // median depth in a small window
  for (int r = std::max(0, v - window); r <= std::min(depth.rows - 1, v + window); r++)
    for (int c = std::max(0, u - window); c <= std::min(depth.cols - 1, u + window); c++) {
      float z = depth.at<float>(r, c);
      if (z > 0.05f && z < 20.0f) patch.push_back(z);
    }
  if (patch.empty()) return std::nullopt;
  std::sort(patch.begin(), patch.end());
  size_t n = patch.size();
  double med = n % 2 ? patch[n / 2] : 0.5 * (patch[n / 2 - 1] + patch[n / 2]);   // like numpy.median
  Vec3 p = camera_to_world(d, pixel_to_camera(u, v, med));
  if (object_radius != 0.0) {                        // front surface -> object centre
    const double* c = d->cam_xpos + 3 * cam;
    double fx_ = p[0] - c[0], fy_ = p[1] - c[1], nrm = std::hypot(fx_, fy_) + 1e-9;
    p[0] += object_radius * fx_ / nrm;
    p[1] += object_radius * fy_ / nrm;
  }
  return p;
}
}  // namespace a3
