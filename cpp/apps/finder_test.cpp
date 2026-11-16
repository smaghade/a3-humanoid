// Day 14 mini project: "Where is the X bottle?" - measured over many random layouts.
// C++ port of tools/finder_test.py. For each layout the bottles and cup are placed at random spots
// on the table and the robot starts facing a random direction; it searches like pick_demo and we
// measure whether the bottle was found and the position error.
//   ./cpp/build/finder_test                    (10 layouts, red/blue/yellow in turn)
//   ./cpp/build/finder_test --layouts 20 --seed 7 | --target red_bottle
// (Random layouts differ from the Python version for the same --seed: different random generator.)
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "a3/config.hpp"
#include "a3/detector.hpp"
#include "a3/render.hpp"
#include "a3/robot.hpp"
#include "a3/scene.hpp"
#include "a3/spatial.hpp"

using namespace a3;
static const int HEAD_SCAN[3] = {0, 40, -40};
static const double MIN_HEIGHT = 0.3, MAX_HEIGHT = 1.8, BOTTLE_RADIUS = 0.035, MIN_GAP = 0.13;
static const double TABLE_X[2] = {0.95, 1.40}, TABLE_Y[2] = {-0.50, 0.50};

int main(int argc, char** argv) {
  int layouts = 10;
  unsigned seed = 1;
  std::string fixed_target;
  for (int i = 1; i < argc; i++) {
    std::string s = argv[i];
    if (s == "--layouts" && i + 1 < argc) layouts = std::stoi(argv[++i]);
    else if (s == "--seed" && i + 1 < argc) seed = (unsigned)std::stoul(argv[++i]);
    else if (s == "--target" && i + 1 < argc) fixed_target = argv[++i];
    else { std::fprintf(stderr, "unrecognized argument: %s\n", s.c_str()); return 1; }
  }
  try {
    Config cfg = load_config();
    mjModel* model = build_scene(cfg.mjcf, true).model;
    KinematicSimRobot robot(model);
    Graphics gfx(model, false);
    DepthCamera camera(model, gfx, CAMERA_NAME);
    ObjectDetector detector;
    std::mt19937 rng(seed);
    auto U = [&](double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(rng); };
    std::vector<std::string> bottles;
    for (const auto& o : OBJECTS)
      if (o.size() > 6 && o.compare(o.size() - 6, 6, "bottle") == 0) bottles.push_back(o);
    std::filesystem::create_directories("docs");
    std::filesystem::create_directories("data/day14");
    std::ofstream csv("docs/day14_results.csv");
    csv << "layout,target,start_yaw_deg,found,looks,error_cm,true_x,true_y\r\n";
    int hits = 0;
    double err_sum = 0, err_worst = 0;
    std::printf("%3s  %-14s %6s  %-5s %5s  %8s\n", "#", "target", "start", "found", "looks", "error");
    for (int i = 1; i <= layouts; i++) {
      std::string target = fixed_target.empty() ? bottles[(i - 1) % bottles.size()] : fixed_target;
      // random layout: every object at a random free spot on the table
      std::vector<std::pair<double, double>> placed;
      for (const auto& name : OBJECTS) {
        int adr = model->jnt_qposadr[model->body_jntadr[body_id(model, name)]];
        double z = robot.d->qpos[adr + 2], x = 0, y = 0;
        for (int k = 0; k < 200; k++) {
          x = U(TABLE_X[0], TABLE_X[1]);
          y = U(TABLE_Y[0], TABLE_Y[1]);
          bool ok = true;
          for (auto [px, py] : placed)
            if (std::hypot(x - px, y - py) <= MIN_GAP) { ok = false; break; }
          if (ok) break;
        }
        placed.push_back({x, y});
        double q[7] = {x, y, z, 1, 0, 0, 0};
        std::copy(q, q + 7, robot.d->qpos + adr);
      }
      mj_forward(model, robot.d);
      double start_yaw = U(-180, 180);
      robot.pose = Pose2D{0, 0, deg2rad(start_yaw)};
      robot.step(0.0);
      int tb = body_id(model, target);
      Vec3 truth{robot.d->xpos[3 * tb], robot.d->xpos[3 * tb + 1], robot.d->xpos[3 * tb + 2]};
      // same search as pick_demo
      std::string words = target;
      for (auto& c : words)
        if (c == '_') c = ' ';
      std::optional<Vec3> estimate;
      int looks = 0;
      cv::Mat picture;
      for (int turn = 0; turn < 8 && !estimate; turn++) {
        for (int head_yaw : HEAD_SCAN) {
          robot.set_head_deg(head_yaw, 0);
          robot.step(0.0);
          cv::Mat depth;
          camera.capture(robot.d, picture, depth);
          looks++;
          auto found = detector.detect(picture);
          for (const Detection* hit : ObjectDetector::find_all(found, words)) {
            auto p = camera.locate(robot.d, depth, hit->centre.x, hit->centre.y, 3, BOTTLE_RADIUS);
            if (p && MIN_HEIGHT < (*p)[2] && (*p)[2] < MAX_HEIGHT) {
              estimate = p;
              draw(picture, found, words);
              break;
            }
          }
          if (estimate) break;
        }
        if (!estimate) robot.pose.yaw += deg2rad(45);       // turn the body (instantly, for testing)
      }
      bool found = estimate.has_value();
      double error_cm = 0;
      if (found) {
        Vec3 e = *estimate;
        error_cm = std::sqrt(std::pow(e[0] - truth[0], 2) + std::pow(e[1] - truth[1], 2) + std::pow(e[2] - truth[2], 2)) * 100;
        hits++;
        err_sum += error_cm;
        err_worst = std::max(err_worst, error_cm);
      }
      if (!picture.empty())
        cv::imwrite(fmt("data/day14/layout%02d_%s_%s.png", i, target.c_str(), found ? "ok" : "MISS"), picture);
      csv << i << "," << target << "," << std::lround(start_yaw) << "," << (found ? "True" : "False") << "," << looks << ","
          << (found ? fmt("%g", std::round(error_cm * 100) / 100) : "") << "," << fmt("%g", std::round(truth[0] * 1000) / 1000)
          << "," << fmt("%g", std::round(truth[1] * 1000) / 1000) << "\r\n";
      std::string err_txt = found ? fmt("%6.1f cm", error_cm) : "      -";
      std::printf("%3d  %-14s %+5.0f°  %-5s %5d  %s\n", i, target.c_str(), start_yaw, found ? "yes" : "NO", looks, err_txt.c_str());
      std::fflush(stdout);
    }
    std::printf("\nFound %d of %d (%.0f%%)\n", hits, layouts, 100.0 * hits / layouts);
    if (hits) std::printf("Position error: average %.1f cm, worst %.1f cm\n", err_sum / hits, err_worst);
    std::printf("Saved: docs/day14_results.csv  and pictures in data/day14/\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
  return 0;
}
