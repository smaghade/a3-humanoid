// pick_demo: pick up a bottle and put it back - the robot FINDS it with its own eyes.
// C++ port of pick_demo.py (same steps, same options). Run from the project folder:
//   ./cpp/build/pick_demo --target blue_bottle
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <string>
#include <thread>

#include "a3/arm.hpp"
#include "a3/config.hpp"
#include "a3/detector.hpp"
#include "a3/render.hpp"
#include "a3/robot.hpp"
#include "a3/safety.hpp"
#include "a3/scene.hpp"
#include "a3/spatial.hpp"

using namespace a3;
static const char* DOC = R"(Pick up the red bottle and put it back - the robot FINDS it with its own eyes.

The robot:  1. scans with its camera until it sees the bottle   (Day 10 detection)
            2. turns the picture position into a room position  (Days 11-12: depth + frames)
            3. walks to a spot in front of it, faces the table
            4. reaches and grasps it (magic grasp)
            5. lifts it, holds it, puts it back, walks home

Usage:
  pick_demo                           # red bottle, found with the camera
  pick_demo --target blue_bottle
  pick_demo --hand right              # choose the hand (default: auto)
  pick_demo --start-yaw 120           # start facing away: the robot must search
  pick_demo --no-vision               # ask the simulator instead
  pick_demo --speedup 2               # faster

Keys in the MuJoCo window: Space = pause (e-stop), R = resume.
Grasping is the plan's "magic grasp": the A3 model has no finger joints.
)";
static const double DT = 0.02, STAND_BACK = 0.45, SIDE_OFFSET = 0.20, GRASP_DIST = 0.02, LIFT = 0.15;
static const double ARM_STEP = 0.004, BOTTLE_RADIUS = 0.035, MIN_HEIGHT = 0.3, MAX_HEIGHT = 1.8;
static const int HEAD_SCAN[3] = {0, 40, -40};
static const char* EYES_WINDOW = "A3 EYES - what the robot sees";
struct Quit {};

struct App {
  std::string target = "red_bottle", hand = "auto", target_words;
  double start_yaw = 0.0, speedup = 1.0;
  bool no_vision = false, no_viewer = false;
  Config cfg;
  mjModel* model = nullptr;
  std::unique_ptr<KinematicSimRobot> robot;
  std::unique_ptr<SafetySupervisor> safe;
  std::unique_ptr<Graphics> gfx;
  std::unique_ptr<DepthCamera> camera;
  std::unique_ptr<ObjectDetector> detector;
  bool viewer = false;
  long frame = 0;
  double t0 = 0;

  void say(const std::string& text) {
    const Pose2D& p = robot->pose;
    std::printf("[%6.1fs] %-52s robot at x=%+.2f y=%+.2f facing %+.0f deg\n", monotonic_seconds() - t0, text.c_str(),
                p.x, p.y, facing_deg(p.yaw));
    std::fflush(stdout);
  }
  void handle_keys() {
    int key;
    while (gfx && gfx->pop_key(key)) {
      if (key == GLFW_KEY_SPACE) safe->emergency_stop();
      else if (key == GLFW_KEY_R) safe->reset_emergency_stop();
    }
  }
  // Take a picture (+ depth), run the detector, show it.
  std::pair<std::vector<Detection>, cv::Mat> look(const std::string& note = "") {
    if (!camera) return {{}, cv::Mat()};
    cv::Mat bgr, depth;
    camera->capture(robot->d, bgr, depth);
    auto found = detector->detect(bgr);
    if (viewer) {
      cv::Mat shown = bgr.clone();
      draw(shown, found, target_words);
      if (!note.empty()) cv::putText(shown, note, {10, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 2, cv::LINE_AA);
      cv::imshow(EYES_WINDOW, shown);
      cv::waitKey(1);
    }
    return {found, depth};
  }
  // Advance one step; while paused, wait here.
  void tick(MagicGrasp* grasp = nullptr) {
    while (safe->estopped) {
      if (viewer) {
        if (!gfx->is_running()) throw Quit();
        gfx->sync(robot->d, "PAUSED (emergency stop) - press R to resume");
        handle_keys();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    robot->step(DT);
    if (grasp) grasp->update();
    if (viewer) {
      gfx->cam.lookat[0] = robot->pose.x + 0.3 * std::cos(robot->pose.yaw);
      gfx->cam.lookat[1] = robot->pose.y + 0.3 * std::sin(robot->pose.yaw);
      gfx->cam.lookat[2] = 0.9;
      gfx->sync(robot->d);
      handle_keys();
      if (!gfx->is_running()) throw Quit();
      if (camera && frame % 5 == 0) look();
      std::this_thread::sleep_for(std::chrono::duration<double>(DT / speedup));
    }
    frame++;
  }
  void turn_to(double yaw, MagicGrasp* grasp = nullptr) {
    while (true) {
      double err = wrap_angle(yaw - robot->pose.yaw);
      if (std::fabs(err) < deg2rad(0.5)) break;
      safe->set_velocity(0.0, 0.0, std::copysign(std::min(2.0, std::max(0.05, 1.5 * std::fabs(err))), err));
      tick(grasp);
    }
    safe->stop();
  }
  void walk_to(double x, double y, MagicGrasp* grasp = nullptr) {
    if (std::hypot(x - robot->pose.x, y - robot->pose.y) < 0.01) return;
    turn_to(std::atan2(y - robot->pose.y, x - robot->pose.x), grasp);
    while (true) {
      double dx = x - robot->pose.x, dy = y - robot->pose.y, remaining = std::hypot(dx, dy);
      if (remaining < 0.005) break;
      double err = wrap_angle(std::atan2(dy, dx) - robot->pose.yaw);
      safe->set_velocity(std::min(0.3, std::max(0.05, remaining)), 0.0, remaining > 0.05 ? 2.0 * err : 0.0);
      tick(grasp);
    }
    safe->stop();
  }
  bool reach(Arm& arm, const Vec3& target, MagicGrasp* grasp = nullptr, double tol = 0.01, int max_steps = 3000) {
    for (int i = 0; i < max_steps; i++) {
      if (safe->estopped) {
        tick(grasp);
        continue;
      }
      if (arm.step_towards(target, ARM_STEP) < tol) return true;
      tick(grasp);
    }
    return false;
  }
  void arm_home(Arm& arm, MagicGrasp* grasp = nullptr) {
    while (arm.step_home(ARM_STEP) > 1e-3) tick(grasp);
  }
  // Scan: head left/right, then turn the body 45 deg at a time, until the target is seen.
  std::optional<Vec3> find_with_eyes() {
    for (int turn = 0; turn < 8; turn++) {
      for (int head_yaw : HEAD_SCAN) {
        robot->set_head_deg(head_yaw, 0);
        for (int k = 0; k < 10; k++) tick();
        auto [found, depth] = look("searching for " + target_words + "...");
        for (const Detection* hit : ObjectDetector::find_all(found, target_words)) {
          int u = hit->centre.x, v = hit->centre.y;
          auto point = camera->locate(robot->d, depth, u, v, 3, BOTTLE_RADIUS);
          if (!point || !(MIN_HEIGHT < (*point)[2] && (*point)[2] < MAX_HEIGHT)) {
            std::string height = point ? fmt("%.2f m", (*point)[2]) : "unknown";
            say("   ignored a '" + target_words + "' look-alike at height " + height + " (not on a table)");
            continue;
          }
          say(fmt("   SEEN at pixel (%d,%d) [%s], head %+d deg", u, v, hit->source.c_str(), head_yaw));
          robot->set_head_deg(0, 0);
          return point;
        }
      }
      say("   not seen here - turning 45 deg");
      turn_to(robot->pose.yaw + deg2rad(45));
    }
    return std::nullopt;
  }
  bool run() {
    MagicGrasp obj(model, robot->d, target);
    Vec3 truth = obj.position(), bottle;
    double sx = robot->pose.x, sy = robot->pose.y;
    if (!camera) {
      bottle = truth;
      say("1. FIND " + target_words + ": asked the simulator");
    } else {
      say("1. FIND " + target_words + " with the camera");
      auto b = find_with_eyes();
      if (!b) {
        say("   " + target_words + " not found anywhere - stopping safely");
        return false;
      }
      bottle = *b;
      double err = std::sqrt(std::pow(bottle[0] - truth[0], 2) + std::pow(bottle[1] - truth[1], 2) + std::pow(bottle[2] - truth[2], 2)) * 100;
      say(fmt("2. LOCATED at x=%.2f y=%+.2f z=%.2f (error vs truth %.1f cm)", bottle[0], bottle[1], bottle[2], err));
    }
    std::string side = hand != "auto" ? hand : (bottle[1] >= 0 ? "left" : "right");
    double stx = bottle[0] - STAND_BACK, sty = bottle[1] - (side == "left" ? SIDE_OFFSET : -SIDE_OFFSET);
    say(fmt("3. NAVIGATE to x=%.2f y=%+.2f", stx, sty));
    walk_to(stx, sty);
    turn_to(0.0);
    robot->animate_arms = false;
    Arm arm(model, robot->d, side);
    Vec3 pre_grasp{bottle[0] - (BOTTLE_RADIUS + 0.10), bottle[1], bottle[2]};
    Vec3 grasp_pt{bottle[0] - (BOTTLE_RADIUS + 0.03), bottle[1], bottle[2]};
    say("4. REACH with the " + side + " hand");
    reach(arm, pre_grasp);
    bool ok = reach(arm, grasp_pt, nullptr, GRASP_DIST);
    Vec3 palm = arm.palm(), op = obj.position();
    if (!ok || std::sqrt(std::pow(palm[0] - op[0], 2) + std::pow(palm[1] - op[1], 2) + std::pow(palm[2] - op[2], 2)) > 0.12) {
      say("   hand is not at the bottle - stopping safely");
      arm_home(arm);
      return false;
    }
    Vec3 home_pos = obj.position();
    obj.attach(&arm);
    say("   GRASPED (magic grasp)");
    reach(arm, {grasp_pt[0], grasp_pt[1], grasp_pt[2] + LIFT}, &obj);
    say(fmt("5. LIFTED %.0f cm, holding for 2 s", LIFT * 100));
    for (int k = 0; k < (int)(2 / DT); k++) tick(&obj);
    reach(arm, grasp_pt, &obj, 0.005);
    obj.release();
    for (int i = 0; i < 3; i++) robot->d->qpos[obj.qadr + i] = home_pos[i];   // exactly where it was
    mj_forward(model, robot->d);
    say("   PLACED back on the table, released");
    reach(arm, pre_grasp);
    arm_home(arm);
    robot->animate_arms = true;
    walk_to(sx, sy);
    turn_to(0.0);
    say("DONE - back at start");
    return true;
  }
};

int main(int argc, char** argv) {
  App app;
  try {
    for (int i = 1; i < argc; i++) {
      std::string s = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error("missing value after " + s);
        return argv[++i];
      };
      if (s == "--target") app.target = next();
      else if (s == "--hand") {
        app.hand = next();
        if (app.hand != "auto" && app.hand != "left" && app.hand != "right")
          throw std::runtime_error("--hand must be auto, left or right");
      } else if (s == "--start-yaw") app.start_yaw = std::stod(next());
      else if (s == "--no-vision") app.no_vision = true;
      else if (s == "--speedup") app.speedup = std::stod(next());
      else if (s == "--no-viewer") app.no_viewer = true;
      else if (s == "-h" || s == "--help") { std::printf("%s", DOC); return 0; }
      else throw std::runtime_error("unrecognized argument: " + s);
    }
    app.target_words = app.target;
    for (auto& c : app.target_words)
      if (c == '_') c = ' ';
    app.cfg = load_config();
    app.model = build_scene(app.cfg.mjcf, true).model;
    app.robot = std::make_unique<KinematicSimRobot>(app.model);
    app.robot->pose.yaw = deg2rad(app.start_yaw);
    app.robot->step(0.0);
    app.safe = std::make_unique<SafetySupervisor>(*app.robot, app.cfg);
    app.safe->stand();
    app.viewer = !app.no_viewer;
    if (app.viewer || !app.no_vision) app.gfx = std::make_unique<Graphics>(app.model, app.viewer, "MuJoCo - A3 pick demo");
    if (!app.no_vision) {
      app.camera = std::make_unique<DepthCamera>(app.model, *app.gfx, CAMERA_NAME);
      app.detector = std::make_unique<ObjectDetector>();
    }
    app.t0 = monotonic_seconds();
    std::printf("%s\n", DOC);
    try {
      if (app.viewer) {
        app.gfx->cam.distance = 2.8;
        app.gfx->cam.azimuth = 150;
        app.gfx->cam.elevation = -20;
        app.gfx->cam.lookat[2] = 0.9;
      }
      app.run();
      if (app.viewer) {
        std::printf("\nFinished. Close the MuJoCo window to exit.\n");
        std::fflush(stdout);
        while (app.gfx->is_running()) {
          app.gfx->sync(app.robot->d);
          cv::waitKey(1);
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
      }
    } catch (const Quit&) {
      app.safe->stop();
      std::printf("\nStopped by user.\n");
    }
    if (app.viewer) cv::destroyAllWindows();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
  return 0;
}
