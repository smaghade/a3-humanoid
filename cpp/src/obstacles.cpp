#include "a3/obstacles.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/imgproc.hpp>

namespace a3 {
ObstacleDetector::ObstacleDetector(const DepthCamera& cam, int step, double min_height, double max_height,
                                   double max_range_, double ignore_radius)
    : max_range(max_range_), cam_(cam), step_(step), min_h_(min_height), max_h_(max_height),
      ignore_radius_(ignore_radius) {}

std::vector<Vec3> ObstacleDetector::detect(const mjData* d, const cv::Mat& depth, Vec2 robot,
                                           std::vector<ObstacleObj>& objects,
                                           const std::function<double(double, double)>& ground,
                                           double track_range) const {
  objects.clear();
  int H = (cam_.height + step_ - 1) / step_, W = (cam_.width + step_ - 1) / step_;
  std::vector<Vec3> world(H * W);
  std::vector<double> above(H * W), dist(H * W), zs(H * W);
  const double* R = d->cam_xmat + 9 * cam_.cam;
  const double* t = d->cam_xpos + 3 * cam_.cam;
  for (int r = 0; r < H; r++)
    for (int c = 0; c < W; c++) {
      int k = r * W + c;
      double u = c * step_, v = r * step_, z = depth.at<float>(r * step_, c * step_);
      double pm[3] = {(u - cam_.cx) * z / cam_.fx, -((v - cam_.cy) * z / cam_.fy), -z};   // MuJoCo camera frame
      for (int i = 0; i < 3; i++) world[k][i] = t[i] + R[3 * i] * pm[0] + R[3 * i + 1] * pm[1] + R[3 * i + 2] * pm[2];
      double g = 0.0;
      if (ground) {   // highest ground within 10 cm, so the edge of a step is not an obstacle
        double gx = world[k][0], gy = world[k][1];
        g = std::max(std::max(ground(gx - 0.1, gy), ground(gx, gy)), ground(gx + 0.1, gy));
      }
      above[k] = world[k][2] - g;
      dist[k] = std::hypot(world[k][0] - robot[0], world[k][1] - robot[1]);
      zs[k] = z;
    }
  double reach = track_range > 0 ? track_range : max_range;
  cv::Mat mask(H, W, CV_8U, cv::Scalar(0));
  std::vector<Vec3> pts;
  for (int k = 0; k < H * W; k++) {
    bool ok = zs[k] > 0.05 && dist[k] < reach && above[k] > min_h_ && above[k] < max_h_ && dist[k] > ignore_radius_;
    if (!ok) continue;
    mask.at<unsigned char>(k / W, k % W) = 1;
    if (dist[k] < max_range) pts.push_back(world[k]);
  }
  cv::Mat labels, stats, cents;
  int n = cv::connectedComponentsWithStats(mask, labels, stats, cents, 8);
  std::vector<std::vector<int>> members(n);
  for (int r = 0; r < H; r++)
    for (int c = 0; c < W; c++) {
      int l = labels.at<int>(r, c);
      if (l > 0) members[l].push_back(r * W + c);
    }
  for (int l = 1; l < n; l++) {
    int area = stats.at<int>(l, cv::CC_STAT_AREA);
    if (area < 3) continue;
    ObstacleObj o;
    double hmax = -1e18, sx = 0, sy = 0;
    double xmin = 1e18, xmax = -1e18, ymin = 1e18, ymax = -1e18;
    for (int k : members[l]) {
      hmax = std::max(hmax, above[k]);
      sx += world[k][0];
      sy += world[k][1];
      xmin = std::min(xmin, world[k][0]); xmax = std::max(xmax, world[k][0]);
      ymin = std::min(ymin, world[k][1]); ymax = std::max(ymax, world[k][1]);
      o.points.push_back(world[k]);
    }
    double cnt = (double)members[l].size();
    o.height = hmax;
    o.centre = {sx / cnt, sy / cnt};
    o.size = std::sqrt((xmax - xmin) * (xmax - xmin) + (ymax - ymin) * (ymax - ymin));
    o.distance = std::hypot(o.centre[0] - robot[0], o.centre[1] - robot[1]);
    o.kind = o.height < 0.6 ? "stone" : "obstacle";
    o.near = o.distance < max_range;
    int x = stats.at<int>(l, cv::CC_STAT_LEFT), y = stats.at<int>(l, cv::CC_STAT_TOP);
    int w = stats.at<int>(l, cv::CC_STAT_WIDTH), h = stats.at<int>(l, cv::CC_STAT_HEIGHT);
    o.box[0] = x * step_; o.box[1] = y * step_; o.box[2] = (x + w) * step_; o.box[3] = (y + h) * step_;
    objects.push_back(std::move(o));
  }
  return pts;
}

cv::Mat& draw_obstacles(cv::Mat& bgr, const std::vector<ObstacleObj>& objects) {
  for (const auto& o : objects) {
    cv::Scalar colour = o.moving ? cv::Scalar(255, 0, 255) : (o.kind == "stone" ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 0, 200));
    std::string label = o.label.empty() ? o.kind : o.label;
    char text[200];
    if (o.moving)
      std::snprintf(text, sizeof(text), "%s moving %.1f m/s, %.1fm", label.c_str(), o.speed, o.distance);
    else
      std::snprintf(text, sizeof(text), "%s %.2fm high, %.1fm away", label.c_str(), o.height, o.distance);
    cv::rectangle(bgr, {o.box[0], o.box[1]}, {o.box[2], o.box[3]}, colour, 2);
    cv::Point org(o.box[0], std::max(15, o.box[1] - 6));
    cv::putText(bgr, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.45, {255, 255, 255}, 2, cv::LINE_AA);
    cv::putText(bgr, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv::LINE_AA);
  }
  return bgr;
}
}  // namespace a3
