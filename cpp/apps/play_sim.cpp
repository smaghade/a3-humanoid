// play_sim: drive the A3 in the MuJoCo viewer through the safety supervisor - with logging and EYES.
// C++ port of play_sim.py (same keys, same log files). Run from the project folder.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <string>
#include <thread>

#include "a3/config.hpp"
#include "a3/detector.hpp"
#include "a3/render.hpp"
#include "a3/robot.hpp"
#include "a3/safety.hpp"
#include "a3/scene.hpp"

using namespace a3;
static const char* DOC = R"(Drive the A3 in the MuJoCo viewer through the safety supervisor - with logging and EYES.

Same robot and keys as before, plus a second window "A3 EYES" showing what the
robot's head camera sees.

Keys work in the MuJoCo window, the A3 EYES window and the terminal:
  Arrows (or W/S/A/D) = walk forward/back, turn left/right
  Enter (or Q)  = stop
  Space         = PAUSE (emergency stop)
  R             = RESUME, then use the arrows again
  I / K         = look up / down        J / L = look left / right   (EYES window or terminal)
  C             = save what the robot sees to data/snapshots/        (EYES window or terminal)
  X (terminal)  = exit
In the MuJoCo window: left mouse = rotate the view, right mouse = move, scroll = zoom.

Options:
  play_sim              robot + eyes
  play_sim --objects    also put a table, bottles and a cup in front of it
  play_sim --no-eyes    without the camera window
  play_sim --objects --detect [--target "blue bottle"]
)";
static const double STEP_V = 0.1, STEP_W = 0.25, DT = 0.02, HEAD_STEP = 3.0;
static const int EYE_W = 640, EYE_H = 480, EYE_EVERY = 3, LOG_EVERY = 5;
static const char* EYES_WINDOW = "A3 EYES - what the robot sees";
static const std::map<int, char> WINDOW_KEYS = {{GLFW_KEY_UP, 'w'},    {GLFW_KEY_DOWN, 's'}, {GLFW_KEY_LEFT, 'a'},
                                                {GLFW_KEY_RIGHT, 'd'}, {GLFW_KEY_ENTER, 'q'}, {GLFW_KEY_SPACE, ' '},
                                                {GLFW_KEY_R, 'r'},     {GLFW_KEY_HOME, 'r'}};
static const std::map<char, std::string> EVENT_NAMES = {
    {'w', "FORWARD +"}, {'s', "BACK / SLOWER"}, {'a', "TURN LEFT"}, {'d', "TURN RIGHT"}, {'q', "STOP"},
    {' ', "PAUSE (E-STOP)"}, {'r', "RESUME"}, {'x', "EXIT"}, {'i', "LOOK UP"}, {'k', "LOOK DOWN"},
    {'j', "LOOK LEFT"}, {'l', "LOOK RIGHT"}, {'c', "SNAPSHOT"}};
static const std::set<char> MOVE_KEYS = {'w', 's', 'a', 'd'};
static double round2(double v) { return std::round(v * 100.0) / 100.0; }

int main(int argc, char** argv) {
  bool objects = false, no_eyes = false, detect = false;
  std::string target = "red bottle";
  for (int i = 1; i < argc; i++) {
    std::string s = argv[i];
    if (s == "--objects") objects = true;
    else if (s == "--no-eyes") no_eyes = true;
    else if (s == "--detect") detect = true;
    else if (s == "--target" && i + 1 < argc) target = argv[++i];
    else if (s == "-h" || s == "--help") { std::printf("%s", DOC); return 0; }
    else { std::fprintf(stderr, "unrecognized argument: %s\n", s.c_str()); return 1; }
  }
  try {
    Config cfg = load_config();
    mjModel* model = build_scene(cfg.mjcf, objects).model;
    KinematicSimRobot robot(model);
    SafetySupervisor safe(robot, cfg);
    safe.stand();
    double vx = 0, wz = 0;
    std::string last = "-", seen;
    bool exit_now = false;
    Graphics gfx(model, true, "MuJoCo - A3 (play_sim)");
    std::unique_ptr<ObjectDetector> detector;
    int cam = camera_id(model, CAMERA_NAME);
    if (!no_eyes) {
      save_camera_yaml(camera_intrinsics(model, EYE_W, EYE_H));
      std::filesystem::create_directories("data/snapshots");
      if (detect) detector = std::make_unique<ObjectDetector>();
    }
    std::filesystem::create_directories("logs");
    std::string stamp = now_stamp();
    std::string csv_path = "logs/run_" + stamp + ".csv", events_path = "logs/run_" + stamp + "_events.txt";
    std::ofstream csv(csv_path), events(events_path);
    csv << "time_s,x_m,y_m,facing_deg,speed_mps,turn_radps,status\r\n";
    double t_start = monotonic_seconds();
    cv::Mat last_eye_image;
    auto log_event = [&](const std::string& text) {
      Pose2D p = robot.pose;
      events << fmt("%7.2fs  %-15s at x=%+.2f y=%+.2f facing %+4.0f deg | speed %+.2f turn %+.2f",
                    monotonic_seconds() - t_start, text.c_str(), p.x, p.y, facing_deg(p.yaw), vx, wz) << "\n";
      events.flush();
    };
    auto handle = [&](char key) {
      char k = (char)std::tolower((unsigned char)key);
      if (safe.estopped && MOVE_KEYS.count(k)) {
        last = "PAUSED - press R";
        return;
      }
      auto it = EVENT_NAMES.find(k);
      last = it != EVENT_NAMES.end() ? it->second : std::string("'") + key + "'";
      auto [yaw, pitch] = robot.get_head();
      if (k == 'w') vx += STEP_V;
      else if (k == 's') vx -= STEP_V;
      else if (k == 'a') wz += STEP_W;
      else if (k == 'd') wz -= STEP_W;
      else if (k == 'q') { vx = wz = 0.0; safe.stop(); }
      else if (k == ' ') { vx = wz = 0.0; safe.emergency_stop(); }
      else if (k == 'r') { vx = wz = 0.0; safe.reset_emergency_stop(); }
      else if (k == 'i') { double v = pitch - HEAD_STEP; robot.set_head(nullptr, &v); }
      else if (k == 'k') { double v = pitch + HEAD_STEP; robot.set_head(nullptr, &v); }
      else if (k == 'j') { double v = yaw + HEAD_STEP; robot.set_head(&v, nullptr); }
      else if (k == 'l') { double v = yaw - HEAD_STEP; robot.set_head(&v, nullptr); }
      else if (k == 'c' && !last_eye_image.empty()) {
        cv::imwrite("data/snapshots/eye_" + now_stamp() + ".png", last_eye_image);
        last = "SAVED PICTURE";
      } else if (k == 'x') exit_now = true;
      vx = round2(vx);
      wz = round2(wz);
      safe.set_velocity(vx, 0.0, wz);
      vx = robot.velocity[0];
      wz = robot.velocity[2];
      if (it != EVENT_NAMES.end()) log_event(it->second);
    };
    std::printf("%s\n", DOC);
    std::printf("Logging to: %s\n            %s\n\n", events_path.c_str(), csv_path.c_str());
    log_event("START");
    TerminalKeys term;
    gfx.cam.distance = 3.0;
    long frame = 0;
    std::deque<char> keys;
    while (gfx.is_running() && !exit_now) {
      double start = monotonic_seconds();
      char c;
      if (term.read(c)) keys.push_back(c);
      int wk;
      while (gfx.pop_key(wk)) {
        auto it = WINDOW_KEYS.find(wk);
        if (it != WINDOW_KEYS.end()) keys.push_back(it->second);
      }
      while (!keys.empty()) {
        handle(keys.front());
        keys.pop_front();
      }
      safe.set_velocity(vx, 0.0, wz);
      vx = robot.velocity[0];
      wz = robot.velocity[2];
      robot.step(DT);
      gfx.cam.lookat[0] = robot.pose.x;
      gfx.cam.lookat[1] = robot.pose.y;
      gfx.cam.lookat[2] = robot.height * 0.6;
      gfx.sync(robot.d);
      if (!no_eyes && frame % EYE_EVERY == 0) {
        cv::Mat depth;
        gfx.capture(robot.d, cam, last_eye_image, depth, EYE_W, EYE_H);
        cv::Mat shown = last_eye_image.clone();
        if (detector) {
          auto found = detector->detect(last_eye_image);
          draw(shown, found, target);
          const Detection* hit = ObjectDetector::find(found, target);
          seen = hit ? fmt("%s at pixel (%d, %d)", target.c_str(), hit->centre.x, hit->centre.y) : target + ": not seen";
        }
        auto [yaw, pitch] = robot.get_head();
        cv::putText(shown, fmt("head yaw %+.0f pitch %+.0f deg", yaw, pitch), {10, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    {255, 255, 255}, 2, cv::LINE_AA);
        cv::imshow(EYES_WINDOW, shown);
        int eye_key = cv::waitKey(1) & 0xFF;
        if (eye_key != 255 && eye_key < 128) keys.push_back((char)eye_key);
      }
      Pose2D p = robot.pose;
      std::string status = safe.estopped ? "PAUSED (E-STOP)" : "ok";
      if (frame % LOG_EVERY == 0)
        csv << fmt("%.2f,%.3f,%.3f,%.1f,%.2f,%.2f,%s\r\n", monotonic_seconds() - t_start, p.x, p.y, facing_deg(p.yaw), vx, wz,
                   safe.estopped ? "E-STOP" : "ok");  // python csv: CRLF
      frame++;
      std::printf("\rlast: %-17s | speed %+.2f m/s turn %+.2f rad/s | x=%+.2f y=%+.2f facing %+4.0f deg | %-15s %-34s",
                  last.c_str(), vx, wz, p.x, p.y, facing_deg(p.yaw), status.c_str(), seen.c_str());
      std::fflush(stdout);
      std::this_thread::sleep_for(std::chrono::duration<double>(std::max(0.0, DT - (monotonic_seconds() - start))));
    }
    log_event("END");
    if (!no_eyes) cv::destroyAllWindows();
    std::printf("\nStopped. Logs saved:\n  %s\n  %s\n", events_path.c_str(), csv_path.c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
  return 0;
}
