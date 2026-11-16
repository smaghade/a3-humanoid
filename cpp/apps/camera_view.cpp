// camera_view (Week 2, Days 8-9): see through the robot's eyes. One window, two pictures:
//   LEFT = what the A3's head camera sees, RIGHT = an overview camera following the robot.
// C++ port of camera_view.py. Click the window, then use these keys:
//   W / S = walk forward / back      A / D = turn left / right
//   I / K = look up / down           J / L = look left / right
//   Q = stop      SPACE = pause (e-stop)      R = resume
//   C = save the camera picture to data/snapshots/      X or Esc = exit
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <thread>
#include <vector>

#include "a3/config.hpp"
#include "a3/render.hpp"
#include "a3/robot.hpp"
#include "a3/safety.hpp"
#include "a3/scene.hpp"

using namespace a3;
static const int W = 640, H = 480;
static const double DT = 0.02, STEP_V = 0.1, STEP_W = 0.25, HEAD_STEP = 3.0;
static const char* WINDOW = "A3 camera (left) | overview (right)";

static void label(cv::Mat& img, const std::vector<std::string>& lines, cv::Scalar colour = {255, 255, 255}) {
  for (size_t i = 0; i < lines.size(); i++) {
    cv::Point org(10, 24 + (int)i * 22);
    cv::putText(img, lines[i], org, cv::FONT_HERSHEY_SIMPLEX, 0.55, {0, 0, 0}, 3, cv::LINE_AA);
    cv::putText(img, lines[i], org, cv::FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv::LINE_AA);
  }
}
static double round2(double v) { return std::round(v * 100.0) / 100.0; }

int main() {
  try {
    Config cfg = load_config();
    mjModel* model = build_scene(cfg.mjcf, true).model;
    KinematicSimRobot robot(model);
    SafetySupervisor safe(robot, cfg);
    safe.stand();
    Graphics gfx(model, false);                     // hidden: pictures only
    int cam = camera_id(model, CAMERA_NAME);
    mjvCamera overview;
    mjv_defaultCamera(&overview);
    overview.distance = 3.5;
    overview.azimuth = 210;
    overview.elevation = -20;
    Intrinsics intr = camera_intrinsics(model, W, H);
    save_camera_yaml(intr);
    std::printf("Camera intrinsics saved to config/camera.yaml: fx=%.1f fy=%.1f cx=%.0f cy=%.0f (%dx%d, fovy %.0f deg)\n",
                intr.fx, intr.fy, intr.cx, intr.cy, W, H, intr.fovy_deg);
    double vx = 0, wz = 0;
    std::string message = "ready";
    std::filesystem::create_directories("data/snapshots");
    cv::namedWindow(WINDOW, cv::WINDOW_AUTOSIZE);
    cv::Mat eye, depth, world;
    while (true) {
      double start = monotonic_seconds();
      int key = cv::waitKey(1) & 0xFF;
      if (key != 255) {
        char k = key < 128 ? (char)std::tolower(key) : 0;
        auto [yaw, pitch] = robot.get_head();
        if (key == 27 || k == 'x') break;
        if (safe.estopped && k && std::string("wsad").find(k) != std::string::npos) message = "PAUSED - press R";
        else if (k == 'w') vx += STEP_V;
        else if (k == 's') vx -= STEP_V;
        else if (k == 'a') wz += STEP_W;
        else if (k == 'd') wz -= STEP_W;
        else if (k == 'q') { vx = wz = 0.0; message = "stopped"; }
        else if (k == ' ') { vx = wz = 0.0; safe.emergency_stop(); message = "PAUSED (e-stop)"; }
        else if (k == 'r') { vx = wz = 0.0; safe.reset_emergency_stop(); message = "resumed"; }
        else if (k == 'i') { double v = pitch - HEAD_STEP; robot.set_head(nullptr, &v); }
        else if (k == 'k') { double v = pitch + HEAD_STEP; robot.set_head(nullptr, &v); }
        else if (k == 'j') { double v = yaw + HEAD_STEP; robot.set_head(&v, nullptr); }
        else if (k == 'l') { double v = yaw - HEAD_STEP; robot.set_head(&v, nullptr); }
        else if (k == 'c') {
          std::string name = "data/snapshots/eye_" + now_stamp() + ".png";
          gfx.capture(robot.d, cam, eye, depth, W, H);
          cv::imwrite(name, eye);
          message = "saved " + name;
        }
        vx = round2(vx);
        wz = round2(wz);
      }
      safe.set_velocity(vx, 0.0, wz);
      vx = robot.velocity[0];
      wz = robot.velocity[2];
      robot.step(DT);
      Pose2D p = robot.pose;
      overview.lookat[0] = p.x;
      overview.lookat[1] = p.y;
      overview.lookat[2] = 0.8;
      gfx.capture(robot.d, cam, eye, depth, W, H);
      gfx.render_view(robot.d, overview, world, W, H);
      auto [yaw, pitch] = robot.get_head();
      label(eye, {"ROBOT CAMERA", fmt("head yaw %+.0f  pitch %+.0f deg", yaw, pitch)});
      label(world, {"OVERVIEW", fmt("speed %+.2f m/s  turn %+.2f rad/s", vx, wz),
                    fmt("x=%+.2f y=%+.2f facing %+.0f deg", p.x, p.y, rad2deg(p.yaw)),
                    safe.estopped ? "PAUSED - press R" : message},
            safe.estopped ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 255));
      cv::Mat both;
      cv::hconcat(eye, world, both);
      cv::imshow(WINDOW, both);
      if (cv::getWindowProperty(WINDOW, cv::WND_PROP_VISIBLE) < 1) break;
      std::this_thread::sleep_for(std::chrono::duration<double>(std::max(0.0, DT - (monotonic_seconds() - start))));
    }
    cv::destroyAllWindows();
    std::printf("Closed.\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
  return 0;
}
