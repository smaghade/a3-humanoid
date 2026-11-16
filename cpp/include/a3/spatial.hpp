// Days 11-12: from a pixel in the camera picture to a real 3D position in the room.
//   Day 11: X = (u - cx) Z / fx,  Y = (v - cy) Z / fy,  Z = depth
//   Day 12: point_in_room = camera_position + camera_rotation * point_in_camera
#pragma once
#include <mujoco/mujoco.h>
#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include "a3/render.hpp"
#include "a3/util.hpp"

namespace a3 {
class DepthCamera {
 public:
  DepthCamera(const mjModel* m, Graphics& gfx, const std::string& camera, int width = 640, int height = 480);
  void capture(const mjData* d, cv::Mat& bgr, cv::Mat& depth);       // colour + depth (metres)
  Vec3 pixel_to_camera(double u, double v, double z) const;          // x right, y down, z forward
  Vec3 camera_to_world(const mjData* d, const Vec3& p_cam) const;
  Vec3 world_to_pixel(const mjData* d, const Vec3& p_world) const;   // returns (u, v, z)
  std::optional<Vec3> locate(const mjData* d, const cv::Mat& depth, double u, double v, int window = 3,
                             double object_radius = 0.0) const;
  int cam;
  double fx, fy, cx, cy;
  int width, height;

 private:
  Graphics& gfx_;
};
}  // namespace a3
