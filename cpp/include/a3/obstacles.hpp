// Seeing obstacles (stones, boxes, tables, people, vehicles, animals) with the depth camera:
// every 4th pixel -> 3D point; points 10 cm - 2.2 m above the GROUND are obstacle points;
// neighbouring ones are grouped into objects (height, size, distance). No AI needed.
#pragma once
#include <functional>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include "a3/spatial.hpp"

namespace a3 {
struct Track;
struct ObstacleObj {
  std::string kind;                     // "stone" (< 0.6 m high) or "obstacle"
  double height = 0, size = 0, distance = 0;
  Vec2 centre{0, 0};
  std::vector<Vec3> points;
  bool near = false;
  int box[4] = {0, 0, 0, 0};            // x1, y1, x2, y2 in pixels
  // filled by the tracker / task program
  std::shared_ptr<Track> track;
  std::string label;
  bool moving = false;
  double speed = 0;
};

class ObstacleDetector {
 public:
  ObstacleDetector(const DepthCamera& cam, int step = 4, double min_height = 0.10, double max_height = 2.2,
                   double max_range = 5.0, double ignore_radius = 0.6);
  // Returns the obstacle points within max_range; fills `objects`.
  std::vector<Vec3> detect(const mjData* d, const cv::Mat& depth, Vec2 robot_xy, std::vector<ObstacleObj>& objects,
                           const std::function<double(double, double)>& ground = nullptr,
                           double track_range = 0.0) const;
  double max_range;

 private:
  const DepthCamera& cam_;
  int step_;
  double min_h_, max_h_, ignore_radius_;
};

cv::Mat& draw_obstacles(cv::Mat& bgr, const std::vector<ObstacleObj>& objects);
}  // namespace a3
