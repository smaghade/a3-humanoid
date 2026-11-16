// mission_sim: the A3 walks a route written in a mission file (missions/demo_route.yaml).
// C++ port of mission_sim.py (same steps, same logs). Run from the project folder:
//   ./cpp/build/mission_sim --no-viewer          or   ./cpp/build/mission_sim --speedup 5
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "a3/config.hpp"
#include "a3/render.hpp"
#include "a3/robot.hpp"
#include "a3/safety.hpp"

using namespace a3;
static const double DT = 0.02, POS_TOL = 0.001, YAW_TOL = 0.1 * PI / 180.0;
static const int LOG_EVERY = 5;
struct Quit {};
struct EStop {};

static double wrap(double a) {                     // like Python: (a + pi) % (2 pi) - pi
  double r = std::fmod(a + PI, 2 * PI);
  if (r < 0) r += 2 * PI;
  return r - PI;
}
static double deg(double a) { return rad2deg(wrap(a)); }

struct Step {
  std::string action, text;                        // e.g. "forward", "10"
  double value;
};

class MissionRunner {
 public:
  MissionRunner(KinematicSimRobot& r, SafetySupervisor& s, Graphics* v, double speedup, const std::string& prefix)
      : robot(r), safe(s), viewer(v), speedup_(speedup), heading(r.pose.yaw), csv(prefix + ".csv"), events(prefix + "_events.txt") {
    csv << "time_s,x_m,y_m,facing_deg,speed_mps,turn_radps,status\r\n";
  }
  void event(const std::string& text) {
    Pose2D p = robot.pose;
    std::string line = fmt("%7.2fs  %-26s at x=%+6.2f y=%+6.2f facing %+4.0f deg", sim_time, text.c_str(), p.x, p.y, deg(p.yaw));
    events << line << "\n";
    events.flush();
    std::printf("\r%s%30s\n", line.c_str(), "");
    std::fflush(stdout);
  }
  void tick() {                                    // advance one step: move, log, draw
    if (safe.estopped) throw EStop();
    robot.step(DT);
    sim_time += DT;
    Pose2D p = robot.pose;
    double vx = robot.velocity[0], wz = robot.velocity[2];
    if (frame % LOG_EVERY == 0)
      csv << fmt("%.2f,%.3f,%.3f,%.1f,%.2f,%.2f,%s\r\n", sim_time, p.x, p.y, deg(p.yaw), vx, wz, safe.estopped ? "E-STOP" : "ok");
    frame++;
    if (viewer) {
      if (!viewer->is_running()) throw Quit();
      viewer->cam.lookat[0] = p.x;
      viewer->cam.lookat[1] = p.y;
      viewer->cam.lookat[2] = robot.height * 0.5;
      viewer->sync(robot.d);
      std::this_thread::sleep_for(std::chrono::duration<double>(DT / speedup_));
      if (frame % 10 == 0) {
        std::printf("\r  %-28s | speed %+.2f turn %+.2f | x=%+6.2f y=%+6.2f facing %+4.0f deg   ", current.c_str(), vx, wz, p.x,
                    p.y, deg(p.yaw));
        std::fflush(stdout);
      }
    }
  }
  void walk(double distance, int vx_sign, int vy_sign, double speed) {
    double x0 = robot.pose.x, y0 = robot.pose.y;
    while (true) {
      double remaining = distance - std::hypot(robot.pose.x - x0, robot.pose.y - y0);
      if (remaining <= POS_TOL) break;
      double v = std::min({speed, std::max(0.05, remaining), remaining / DT});
      double steer = 2.0 * (heading - robot.pose.yaw);           // hold the intended direction
      safe.set_velocity(vx_sign * v, vy_sign * v, steer);
      tick();
    }
    safe.stop();
  }
  void turn(double angle_deg) {
    heading += deg2rad(angle_deg);
    while (true) {
      double err = heading - robot.pose.yaw;
      if (std::fabs(err) <= YAW_TOL) break;
      safe.set_velocity(0.0, 0.0, std::copysign(std::min(2.0, std::max(0.05, 1.5 * std::fabs(err))), err));
      tick();
    }
    safe.stop();
  }
  void wait(double seconds) {
    safe.stop();
    for (int i = 0; i < (int)std::lround(seconds / DT); i++) tick();
  }
  void close() {
    Pose2D p = robot.pose;
    csv << fmt("%.2f,%.3f,%.3f,%.1f,0.00,0.00,ok\r\n", sim_time, p.x, p.y, deg(p.yaw));
    csv.close();
    events.close();
  }
  KinematicSimRobot& robot;
  SafetySupervisor& safe;
  Graphics* viewer;
  double sim_time = 0.0;
  long frame = 0;
  std::string current = "-";

 private:
  double speedup_, heading;
  std::ofstream csv, events;
};

static const std::map<std::string, std::pair<int, int>> DIRECTIONS = {
    {"forward", {1, 0}}, {"backward", {-1, 0}}, {"side_left", {0, 1}}, {"side_right", {0, -1}}};

int main(int argc, char** argv) {
  std::string mission_path = "missions/demo_route.yaml";
  double speedup = 1.0;
  bool no_viewer = false;
  for (int i = 1; i < argc; i++) {
    std::string s = argv[i];
    if (s == "--speedup" && i + 1 < argc) speedup = std::stod(argv[++i]);
    else if (s == "--no-viewer") no_viewer = true;
    else if (s[0] != '-') mission_path = s;
    else { std::fprintf(stderr, "unrecognized argument: %s\n", s.c_str()); return 1; }
  }
  try {
    Config cfg = load_config();
    YAML::Node mission = YAML::LoadFile(mission_path);
    double speed = mission["speed"] ? mission["speed"].as<double>() : 0.3;
    std::vector<Step> steps;
    const std::vector<std::string> allowed = {"forward", "backward", "side_left", "side_right", "turn_left", "turn_right", "wait", "stop"};
    for (const auto& st : mission["steps"]) {
      auto kv = st.begin();
      std::string action = kv->first.as<std::string>(), text = kv->second.as<std::string>();
      if (std::find(allowed.begin(), allowed.end(), action) == allowed.end()) {
        std::string list;
        for (const auto& a : allowed) list += (list.empty() ? "" : ", ") + a;
        std::fprintf(stderr, "Unknown step '%s' in %s. Allowed: %s\n", action.c_str(), mission_path.c_str(), list.c_str());
        return 1;
      }
      if (text == "true") text = "True";
      steps.push_back({action, text, action == "stop" ? 0.0 : kv->second.as<double>()});
    }
    char err[1000] = "";
    mjModel* m = mj_loadXML(cfg.mjcf.c_str(), nullptr, err, sizeof(err));
    if (!m) throw std::runtime_error("Cannot load " + cfg.mjcf + ": " + err);
    KinematicSimRobot robot(m);
    SafetySupervisor safe(robot, cfg);
    safe.stand();
    std::filesystem::create_directories("logs");
    std::string prefix = "logs/run_" + now_stamp();
    double ex = 0, ey = 0, eyaw = 0;                // where the robot SHOULD end up
    for (const auto& s : steps) {
      auto it = DIRECTIONS.find(s.action);
      if (it != DIRECTIONS.end()) {
        double dx = it->second.first, dy = it->second.second;
        ex += s.value * (dx * std::cos(eyaw) - dy * std::sin(eyaw));
        ey += s.value * (dx * std::sin(eyaw) + dy * std::cos(eyaw));
      } else if (s.action == "turn_left") eyaw += deg2rad(s.value);
      else if (s.action == "turn_right") eyaw -= deg2rad(s.value);
    }
    std::printf("Mission: %s  (%zu steps, speed %g m/s)\n", mission_path.c_str(), steps.size(), speed);
    std::printf("Expected end: x=%+.2f y=%+.2f facing %+.0f deg\n\n", ex, ey, deg(eyaw));
    std::unique_ptr<Graphics> viewer;
    if (!no_viewer) {
      viewer = std::make_unique<Graphics>(m, true, "MuJoCo - A3 mission");
      viewer->cam.distance = 8.0;
      viewer->cam.elevation = -45;
    }
    MissionRunner runner(robot, safe, viewer.get(), speedup, prefix);
    runner.event("MISSION START");
    try {
      for (size_t i = 0; i < steps.size(); i++) {
        const Step& s = steps[i];
        runner.current = fmt("step %zu/%zu: %s %s", i + 1, steps.size(), s.action.c_str(), s.text.c_str());
        runner.event("START " + s.action + " " + s.text);
        auto dir = DIRECTIONS.find(s.action);
        if (dir != DIRECTIONS.end()) runner.walk(s.value, dir->second.first, dir->second.second, speed);
        else if (s.action == "turn_left") runner.turn(+s.value);
        else if (s.action == "turn_right") runner.turn(-s.value);
        else if (s.action == "wait") runner.wait(s.value);
        else safe.stop();
        runner.event("DONE  " + s.action + " " + s.text);
      }
      runner.event("MISSION COMPLETE");
    } catch (const Quit&) {
      runner.event("MISSION ABORTED (stopped by user)");
    } catch (const EStop&) {
      runner.event("MISSION ABORTED: emergency stop active");
    }
    safe.stop();
    runner.close();
    Pose2D p = robot.pose;
    std::printf("\nExpected end: x=%+.2f y=%+.2f facing %+.0f deg\n", ex, ey, deg(eyaw));
    std::printf("Actual end:   x=%+.2f y=%+.2f facing %+.0f deg\n", p.x, p.y, deg(p.yaw));
    std::printf("Position error: %.1f cm   |   time: %.1f s\n", std::hypot(p.x - ex, p.y - ey) * 100, runner.sim_time);
    std::printf("Logs: %s_events.txt  and  %s.csv\n", prefix.c_str(), prefix.c_str());
    if (viewer && viewer->is_running()) {
      std::printf("\nMission finished. Close the MuJoCo window to exit.\n");
      std::fflush(stdout);
      while (viewer->is_running()) {
        viewer->sync(robot.d);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
  return 0;
}
