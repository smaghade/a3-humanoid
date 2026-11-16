// plot_log: draw the robot's path from the newest log (or a given CSV file) and print a summary.
// C++ port of plot_log.py; the picture is drawn with OpenCV instead of matplotlib.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <string>
#include <vector>
#include "a3/util.hpp"

using namespace a3;
int main(int argc, char** argv) {
  std::string path;
  if (argc > 1) path = argv[1];
  else {
    std::vector<std::string> logs;
    if (std::filesystem::exists("logs"))
      for (const auto& e : std::filesystem::directory_iterator("logs")) {
        std::string n = e.path().string();
        if (n.find("logs/run_") == 0 && n.size() > 4 && n.substr(n.size() - 4) == ".csv") logs.push_back(n);
      }
    if (logs.empty()) { std::fprintf(stderr, "No logs/run_*.csv files found.\n"); return 1; }
    std::sort(logs.begin(), logs.end());
    path = logs.back();
  }
  std::ifstream f(path);
  if (!f) { std::fprintf(stderr, "Cannot open %s\n", path.c_str()); return 1; }
  std::string line;
  std::getline(f, line);                            // header
  std::vector<double> t, x, y, yaw;
  std::vector<bool> estop;
  std::string last_facing;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();     // CRLF files from Python's csv
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string c[7];
    for (auto& s : c) std::getline(ss, s, ',');
    t.push_back(std::stod(c[0]));
    x.push_back(std::stod(c[1]));
    y.push_back(std::stod(c[2]));
    yaw.push_back(deg2rad(std::stod(c[3])));
    estop.push_back(c[6] == "E-STOP");
    last_facing = c[3];
  }
  if (x.empty()) { std::fprintf(stderr, "%s has no data rows.\n", path.c_str()); return 1; }
  double distance = 0;
  for (size_t i = 1; i < x.size(); i++) distance += std::hypot(x[i] - x[i - 1], y[i] - y[i - 1]);
  std::printf("File:      %s\n", path.c_str());
  std::printf("Duration:  %.1f s\n", t.back());
  std::printf("Distance:  %.2f m walked\n", distance);
  std::printf("Start:     x=%+.2f y=%+.2f\n", x[0], y[0]);
  std::printf("End:       x=%+.2f y=%+.2f facing %+.0f deg\n", x.back(), y.back(), std::stod(last_facing));

  // ---- picture: 840 x 840, equal scale on both axes, grid, path, arrows, start/end ----
  const int W = 840, H = 840, L = 90, R = 30, T = 60, B = 80;
  double xmin = *std::min_element(x.begin(), x.end()), xmax = *std::max_element(x.begin(), x.end());
  double ymin = *std::min_element(y.begin(), y.end()), ymax = *std::max_element(y.begin(), y.end());
  double span = std::max({xmax - xmin, ymax - ymin, 1.0}) * 1.15;
  double cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2;
  double s = std::min(W - L - R, H - T - B) / span;
  int ox = L + (W - L - R) / 2, oy = T + (H - T - B) / 2;
  auto px = [&](double X, double Y) { return cv::Point((int)(ox + (X - cx) * s), (int)(oy - (Y - cy) * s)); };
  cv::Mat img(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
  double tick = std::pow(10, std::floor(std::log10(span / 2)));
  if (span / tick > 10) tick *= 2;
  for (double g = std::floor((cx - span / 2) / tick) * tick; g <= cx + span / 2; g += tick) {
    cv::Point a = px(g, cy - span / 2), b = px(g, cy + span / 2);
    if (a.x < L || a.x > W - R) continue;
    cv::line(img, {a.x, T}, {a.x, H - B}, {225, 225, 225}, 1);
    cv::putText(img, fmt("%g", g), {a.x - 12, H - B + 20}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {60, 60, 60}, 1, cv::LINE_AA);
  }
  for (double g = std::floor((cy - span / 2) / tick) * tick; g <= cy + span / 2; g += tick) {
    cv::Point a = px(cx - span / 2, g);
    if (a.y < T || a.y > H - B) continue;
    cv::line(img, {L, a.y}, {W - R, a.y}, {225, 225, 225}, 1);
    cv::putText(img, fmt("%g", g), {L - 45, a.y + 5}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {60, 60, 60}, 1, cv::LINE_AA);
  }
  cv::rectangle(img, {L, T}, {W - R, H - B}, {0, 0, 0}, 1);
  std::vector<cv::Point> pts;
  for (size_t i = 0; i < x.size(); i++) pts.push_back(px(x[i], y[i]));
  cv::polylines(img, pts, false, {180, 119, 31}, 2, cv::LINE_AA);              // path (blue)
  size_t step = std::max<size_t>(1, x.size() / 25);                            // direction arrows
  for (size_t i = 0; i < x.size(); i += step) {
    cv::Point a = pts[i], b(a.x + (int)(22 * std::cos(yaw[i])), a.y - (int)(22 * std::sin(yaw[i])));
    cv::arrowedLine(img, a, b, {128, 128, 128}, 1, cv::LINE_AA, 0, 0.35);
  }
  cv::circle(img, pts.front(), 9, {44, 160, 44}, -1, cv::LINE_AA);             // start (green)
  cv::rectangle(img, {pts.back().x - 8, pts.back().y - 8}, {pts.back().x + 8, pts.back().y + 8}, {40, 39, 214}, -1);
  bool any_estop = false;
  for (size_t i = 0; i < x.size(); i++)
    if (estop[i]) {
      any_estop = true;
      cv::drawMarker(img, pts[i], {0, 0, 0}, cv::MARKER_TILTED_CROSS, 12, 2);
    }
  cv::putText(img, fmt("Robot path (%.1f m in %.0f s)", distance, t.back()), {L, 38}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
              {0, 0, 0}, 2, cv::LINE_AA);
  cv::putText(img, "x (m)  - forward at start", {W / 2 - 110, H - 30}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {0, 0, 0}, 1, cv::LINE_AA);
  cv::Mat ylabel(30, 260, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::putText(ylabel, "y (m)  - left at start", {5, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {0, 0, 0}, 1, cv::LINE_AA);
  cv::rotate(ylabel, ylabel, cv::ROTATE_90_COUNTERCLOCKWISE);
  ylabel.copyTo(img(cv::Rect(5, H / 2 - 130, 30, 260)));
  int lx = W - R - 175, ly = T + 12;                                            // legend
  cv::rectangle(img, {lx - 8, ly - 4}, {W - R - 8, ly + (any_estop ? 92 : 70)}, {200, 200, 200}, 1);
  cv::line(img, {lx, ly + 10}, {lx + 25, ly + 10}, {180, 119, 31}, 2);
  cv::putText(img, "path", {lx + 35, ly + 15}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1, cv::LINE_AA);
  cv::circle(img, {lx + 12, ly + 32}, 7, {44, 160, 44}, -1);
  cv::putText(img, "start", {lx + 35, ly + 37}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1, cv::LINE_AA);
  cv::rectangle(img, {lx + 5, ly + 48}, {lx + 19, ly + 62}, {40, 39, 214}, -1);
  cv::putText(img, "end", {lx + 35, ly + 59}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1, cv::LINE_AA);
  if (any_estop) {
    cv::drawMarker(img, {lx + 12, ly + 77}, {0, 0, 0}, cv::MARKER_TILTED_CROSS, 10, 2);
    cv::putText(img, "emergency stop", {lx + 35, ly + 81}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1, cv::LINE_AA);
  }
  std::string png = path.substr(0, path.size() - 4) + ".png";
  cv::imwrite(png, img);
  std::printf("Picture:   %s\n", png.c_str());
  if (std::getenv("DISPLAY")) {                     // like plt.show()
    cv::imshow("Robot path", img);
    cv::waitKey(0);
  }
  return 0;
}
