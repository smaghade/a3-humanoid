// task_demo: give the robot an instruction in plain words; it plans and carries it out.
// C++ port of task_demo.py - same behaviour, same options. Run from the project folder:
//   ./cpp/build/task_demo --say "pick the red bottle from the table and keep it on the empty table"
//   ./cpp/build/task_demo --course --speedup 5 --say "..."      (85 m city course)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "a3/arm.hpp"
#include "a3/config.hpp"
#include "a3/course.hpp"
#include "a3/detector.hpp"
#include "a3/grid_planner.hpp"
#include "a3/obstacles.hpp"
#include "a3/render.hpp"
#include "a3/robot.hpp"
#include "a3/safety.hpp"
#include "a3/scene.hpp"
#include "a3/spatial.hpp"
#include "a3/tracker.hpp"

using namespace a3;

static const char* DOC = R"(Give the robot an instruction in plain words; it plans and carries it out.

Example instructions:
  "pick the red bottle from the table and keep it on the empty table"
  "move the blue bottle to table 2"
  "put the red bottle back on table 1"
  "bring the yellow bottle to the other table"

What happens (a first version of plan Days 15-16):
  1. PLANNER  : the sentence -> a list of actions from a fixed menu (checked)
  2. EXECUTOR : runs each action, reports SUCCESS / FAILURE + reason, stops safely on failure
  Actions: GO_TO_AREA, FIND, APPROACH, GRASP, LIFT, GO_TO_TABLE, PLACE, RETURN

Usage:
  task_demo --say "pick the red bottle and keep it on the empty table"
  task_demo                 # asks you for instructions, one after another
  task_demo --no-vision     # simulator gives object positions
  task_demo --speedup 3 | --no-viewer

Field mode: task_demo --start-distance 70 --stones 15 --speedup 5 --say "..."
City course (85 m): task_demo --course --speedup 5 --say "..."

Tables are known places (the robot's map): table_1 has the bottles, table_2 is empty (grey).
Keys in the MuJoCo window: Space = pause (e-stop), R = resume.
)";

static const double DT = 0.02, STAND_BACK = 0.45, SIDE_OFFSET = 0.20, GRASP_DIST = 0.02, LIFT = 0.15;
static const double ARM_STEP = 0.004, BOTTLE_RADIUS = 0.035, BOTTLE_Z = TABLE_HEIGHT + 0.12;
static const int HEAD_SCAN[3] = {0, 40, -40};
static const double MIN_HEIGHT = 0.3, MAX_HEIGHT = 1.8;
static const double SPOT_SHIFTS[5] = {0.0, -0.15, 0.15, -0.30, 0.30};
static const char* EYES_WINDOW = "A3 EYES - what the robot sees";
static const char* MAP_WINDOW = "A3 MAP - what the robot knows";
static const std::vector<std::string> COLOURS = {"red", "blue", "yellow"};
static const double STAGING_BACK = 1.5, FAR = 3.0, CLEARANCE = 1.2;

struct Quit {};                                     // thrown when the window is closed

struct Args {
  std::optional<std::string> say;
  bool no_vision = false, no_viewer = false, course = false;
  double speedup = 1.0, start_distance = 0.0;
  std::optional<int> stones;
  unsigned seed = 3;
};

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; i++) {
    std::string s = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value after " + s);
      return argv[++i];
    };
    if (s == "--say") a.say = next();
    else if (s == "--no-vision") a.no_vision = true;
    else if (s == "--no-viewer") a.no_viewer = true;
    else if (s == "--course") a.course = true;
    else if (s == "--speedup") a.speedup = std::stod(next());
    else if (s == "--start-distance") a.start_distance = std::stod(next());
    else if (s == "--stones") a.stones = std::stoi(next());
    else if (s == "--seed") a.seed = (unsigned)std::stoul(next());
    else if (s == "-h" || s == "--help") { std::printf("%s", DOC); std::exit(0); }
    else throw std::runtime_error("unrecognized argument: " + s);
  }
  if (a.course && a.start_distance <= 0) a.start_distance = 85.0;
  return a;
}

// =====================================================================================
// PLANNER: sentence -> checked list of actions
// =====================================================================================
using Step = std::pair<std::string, std::string>;
static const std::set<std::string> ALLOWED = {"GO_TO_AREA", "FIND", "APPROACH", "GRASP", "LIFT",
                                              "GO_TO_TABLE", "PLACE", "RETURN"};
static const std::vector<std::pair<std::string, std::vector<std::string>>> TABLE_WORDS = {
    {"table_2", {"table 2", "table two", "second table", "other table", "empty table", "new table", "grey table",
                 "gray table", "another table"}},
    {"table_1", {"table 1", "table one", "first table", "same table", "original table", "back"}},
};
static std::string pretty(std::string s) {
  std::replace(s.begin(), s.end(), '_', ' ');
  return s;
}
static bool contains(const std::string& t, const std::string& w) { return t.find(w) != std::string::npos; }

static std::pair<std::optional<std::vector<Step>>, std::string> make_plan(const std::string& sentence) {
  std::string t = " " + sentence + " ";
  std::transform(t.begin(), t.end(), t.begin(), ::tolower);
  std::vector<std::string> colours;
  for (const auto& c : COLOURS)
    if (contains(t, " " + c + " ") || contains(t, " " + c + "-")) colours.push_back(c);
  if (contains(t, "cup") && !contains(t, "bottle"))
    return {std::nullopt, "Sorry, I can only move bottles for now (red, blue or yellow)."};
  if (colours.size() != 1) return {std::nullopt, "Which bottle? Please say exactly one colour: red, blue or yellow."};
  std::string target = colours[0] + "_bottle", dest;
  for (const auto& [table, words] : TABLE_WORDS) {           // table_2 words are checked first
    for (const auto& w : words)
      if (contains(t, w)) { dest = table; break; }
    if (!dest.empty()) break;
  }
  if (dest.empty())
    for (const char* w : {" keep ", " put ", " place ", " move ", " bring ", " take "})
      if (contains(t, w)) { dest = "table_2"; break; }   // "move it" with no table named
  if (dest.empty())
    return {std::nullopt, "Where should I put it? Say for example 'on table 2' or 'on the empty table'."};
  std::vector<Step> plan = {{"GO_TO_AREA", target}, {"FIND", target}, {"APPROACH", target}, {"GRASP", target},
                            {"LIFT", target},      {"GO_TO_TABLE", dest}, {"PLACE", dest}, {"RETURN", "start"}};
  for (const auto& s : plan)
    if (!ALLOWED.count(s.first)) throw std::logic_error("action not in the menu");   // only the fixed menu
  return {plan, "Plan: move the " + colours[0] + " bottle to " + pretty(dest) + ", then come back."};
}

// field mode: random stones and boxes between the start and the tables
static std::vector<Obstacle> make_stones(double start_x, int n, unsigned seed) {
  std::mt19937 rng(seed);
  auto U = [&](double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(rng); };
  auto N = [&](double mu, double sd) { return std::normal_distribution<double>(mu, sd)(rng); };
  const Table& t1 = TABLES.at("table_1");
  double end_x = t1.cx - t1.hx - STAGING_BACK - 1.5;
  std::vector<Obstacle> stones;
  for (int k = 0; k < n; k++) {
    double x = U(start_x + 4.0, end_x), y = (k % 3) ? N(0.0, 0.8) : U(-0.2, 0.2);
    float grey = (float)U(0.30, 0.55);
    std::array<float, 4> rgba{grey + (float)U(0, 0.06), grey, grey - (float)U(0, 0.06), 1};
    if (k % 4 == 3) {
      double half = U(0.15, 0.30);
      stones.push_back({"box_" + std::to_string(k), "box", {x, y, half}, {half, half * U(0.8, 1.5), half}, rgba});
    } else {
      double r = U(0.15, 0.40);
      stones.push_back({"stone_" + std::to_string(k), "stone", {x, y, r * 0.5}, {r, r * U(0.7, 1.2), r * 0.6}, rgba});
    }
  }
  return stones;
}

static double dist3(const Vec3& a, const Vec3& b) {
  return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
}

// =====================================================================================
// The program state (the Python version keeps these as module-level variables)
// =====================================================================================
struct Thing {                                      // a moving thing the robot knows about
  Vec2 pos, vel;
  std::string label;
};

class TaskDemo {
 public:
  explicit TaskDemo(const Args& a);
  void run();

 private:
  void say(const std::string& text);
  void handle_keys();
  std::pair<std::vector<Detection>, cv::Mat> look(const std::string& note = "");
  void tick();
  void turn_to(double yaw);
  void walk_to(double x, double y);
  bool reach(Arm& arm, const Vec3& target, double tol = 0.01, const Vec3* tool = nullptr, int max_steps = 3000);
  void arm_home(Arm& arm);
  std::optional<std::string> table_of(const Vec3& p) const;
  std::optional<std::string> table_of2(Vec2 p) const { return table_of({p[0], p[1], 0}); }
  std::pair<Vec2, double> stand_spot(const Vec3& point, const std::string& table, const std::string& side) const;
  std::vector<Vec3> object_positions(const MagicGrasp* skip) const;
  std::optional<Vec3> free_spot(const std::string& table, const std::string& target = "");
  std::vector<ObstacleObj> sense();
  void draw_map(double half = 7.0, int ppm = 40);
  bool on_known_obstacle(Vec2 pos, double radius = 0.4) const;
  std::vector<Thing> moving_things() const;
  struct Gap { double gap; std::string label; double dist; };
  Gap closest_gap(Vec2 here, double heading, double speed) const;
  void look_both_ways();
  std::pair<bool, std::string> navigate(Vec2 goal, double final_tol = 0.05);
  std::pair<Vec2, double> staging_point(const std::string& table) const;
  std::pair<bool, std::string> act(const std::string& action, const std::string& arg);
  std::pair<bool, std::string> act_go_to_area(const std::string& target);
  std::pair<bool, std::string> act_find(const std::string& target);
  std::pair<bool, std::string> act_approach(const std::string& target);
  std::pair<bool, std::string> act_grasp(const std::string& target);
  std::pair<bool, std::string> act_lift();
  std::pair<bool, std::string> act_go_to_table(const std::string& table);
  std::pair<bool, std::string> act_place(const std::string& table);
  std::pair<bool, std::string> act_return();
  void recover();
  bool execute(const std::vector<Step>& plan);
  void run_instructions();

  Args args_;
  Config cfg_;
  double start_x_ = 0;
  std::unique_ptr<Course> course_;
  std::vector<Obstacle> stones_;
  mjModel* model_ = nullptr;
  std::unique_ptr<KinematicSimRobot> robot_;
  std::unique_ptr<SafetySupervisor> safe_;
  Pose2D start_;
  std::unique_ptr<Graphics> gfx_;
  std::unique_ptr<DepthCamera> camera_;
  std::unique_ptr<ObjectDetector> detector_;
  std::unique_ptr<OccupancyGrid> grid_;
  std::unique_ptr<ObstacleDetector> obstacle_eye_;
  std::vector<ObstacleObj> seen_obstacles_;
  Tracker tracker_;
  double sim_t_ = 0.0;
  std::set<int> spotted_ids_;
  int ev_waits_ = 0, ev_crossings_ = 0, ev_gave_way_ = 0;
  bool nav_active_ = false;
  std::optional<Path> nav_path_;
  std::optional<Vec2> nav_goal_;
  std::string nav_status_;
  bool viewer_ = false;
  long frame_ = 0;
  double t0_ = 0;
  std::map<std::string, std::unique_ptr<MagicGrasp>> grasps_;
  MagicGrasp* holding_grasp_ = nullptr;
  std::unique_ptr<Arm> holding_arm_;
  std::map<std::string, Vec3> home_;
  std::map<std::string, std::string> last_table_;
  std::optional<Vec3> mem_object_, mem_spot_;          // what the robot learned during the task
  std::string mem_side_;
  Vec3 mem_face_{1, 0, 0};
};

TaskDemo::TaskDemo(const Args& a) : args_(a), cfg_(load_config()) {
  const Table& t1 = TABLES.at("table_1");
  start_x_ = args_.start_distance > 0 ? t1.cx - t1.hx - args_.start_distance : 0.0;
  int n_stones = args_.stones ? *args_.stones : (args_.start_distance > 5 ? (int)(args_.start_distance / 5) : 0);
  if (args_.course) {
    course_ = std::make_unique<Course>(start_x_, n_stones, args_.seed);
    stones_ = course_->stones;
    Scene s = build_scene(cfg_.mjcf, true);
    course_->add_to_spec(s.spec);
    mj_deleteModel(s.model);
    model_ = compile_scene(s.spec);
  } else {
    if (n_stones) stones_ = make_stones(start_x_, n_stones, args_.seed);
    model_ = build_scene(cfg_.mjcf, true, 70.0, 30.0, stones_).model;
  }
  model_->vis.map.znear = 0.02 / model_->stat.extent;   // camera sees from 2 cm, even in a big scene
  robot_ = std::make_unique<KinematicSimRobot>(model_);
  robot_->pose.x = start_x_;
  if (course_) {
    course_->bind(model_, robot_->d);
    robot_->ground = [this](double x, double y) { return course_->ground(x, y); };
  }
  robot_->step(0.0);
  safe_ = std::make_unique<SafetySupervisor>(*robot_, cfg_);
  safe_->stand();
  start_ = robot_->pose;
  viewer_ = !args_.no_viewer;
  if (viewer_ || !args_.no_vision) gfx_ = std::make_unique<Graphics>(model_, viewer_, "MuJoCo - A3 task demo");
  if (!args_.no_vision) {
    camera_ = std::make_unique<DepthCamera>(model_, *gfx_, CAMERA_NAME);
    detector_ = std::make_unique<ObjectDetector>();
  }
  grid_ = std::make_unique<OccupancyGrid>(Vec2{std::min(start_x_, 0.0) - 5, 5.0}, Vec2{-6.0, 6.0});
  if (camera_) obstacle_eye_ = std::make_unique<ObstacleDetector>(*camera_);
  if (!camera_)                                   // no camera: give the robot the true map
    for (const auto& ob : stones_) {
      double cx = ob.pos[0], cy = ob.pos[1], r = std::max(ob.size[0], ob.size[1]);
      std::vector<Vec3> pts;
      for (double dx = -r; dx < r + 0.01; dx += 0.05)
        for (double dy = -r; dy < r + 0.01; dy += 0.05) pts.push_back({cx + dx, cy + dy, 0.1});
      grid_->add_points(pts);
    }
  t0_ = monotonic_seconds();
  for (const auto& name : OBJECTS)
    if (name.size() > 6 && name.compare(name.size() - 6, 6, "bottle") == 0) {
      grasps_[name] = std::make_unique<MagicGrasp>(model_, robot_->d, name);
      home_[name] = grasps_[name]->position();          // where each bottle started
      last_table_[name] = "table_1";                    // where the robot last saw/put it
    }
}

void TaskDemo::say(const std::string& text) {
  const Pose2D& p = robot_->pose;
  std::printf("[%6.1fs] %-54s robot at x=%+.2f y=%+.2f facing %+.0f deg\n", monotonic_seconds() - t0_, text.c_str(),
              p.x, p.y, facing_deg(p.yaw));
  std::fflush(stdout);
}

void TaskDemo::handle_keys() {
  int key;
  while (gfx_ && gfx_->pop_key(key)) {
    if (key == GLFW_KEY_SPACE) safe_->emergency_stop();
    else if (key == GLFW_KEY_R) safe_->reset_emergency_stop();
  }
}

std::pair<std::vector<Detection>, cv::Mat> TaskDemo::look(const std::string& note) {
  if (!camera_) return {{}, cv::Mat()};
  cv::Mat bgr, depth;
  camera_->capture(robot_->d, bgr, depth);
  auto found = detector_->detect(bgr);
  if (viewer_) {
    cv::Mat shown = bgr.clone();
    draw(shown, found);
    if (!note.empty())
      cv::putText(shown, note, {10, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 2, cv::LINE_AA);
    cv::imshow(EYES_WINDOW, shown);
    cv::waitKey(1);
  }
  return {found, depth};
}

void TaskDemo::tick() {
  while (safe_->estopped) {                         // paused: keep the window alive
    if (viewer_) {
      if (!gfx_->is_running()) throw Quit();
      gfx_->sync(robot_->d, "PAUSED (emergency stop) - press R to resume");
      handle_keys();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  sim_t_ += DT;
  if (course_) course_->update(DT, {robot_->pose.x, robot_->pose.y});
  robot_->step(DT);
  if (holding_grasp_) holding_grasp_->update();
  if (viewer_) {
    gfx_->cam.lookat[0] = robot_->pose.x + 0.3 * std::cos(robot_->pose.yaw);
    gfx_->cam.lookat[1] = robot_->pose.y + 0.3 * std::sin(robot_->pose.yaw);
    gfx_->cam.lookat[2] = 0.9;
    gfx_->sync(robot_->d);
    handle_keys();
    if (!gfx_->is_running()) throw Quit();
    if (camera_ && frame_ % 5 == 0 && !nav_active_) look();
    std::this_thread::sleep_for(std::chrono::duration<double>(DT / args_.speedup));
  }
  frame_++;
}

void TaskDemo::turn_to(double yaw) {
  while (true) {
    double err = wrap_angle(yaw - robot_->pose.yaw);
    if (std::fabs(err) < deg2rad(0.5)) break;
    safe_->set_velocity(0.0, 0.0, std::copysign(std::min(2.0, std::max(0.05, 1.5 * std::fabs(err))), err));
    tick();
  }
  safe_->stop();
}

void TaskDemo::walk_to(double x, double y) {
  if (std::hypot(x - robot_->pose.x, y - robot_->pose.y) < 0.01) return;
  turn_to(std::atan2(y - robot_->pose.y, x - robot_->pose.x));
  while (true) {
    double dx = x - robot_->pose.x, dy = y - robot_->pose.y, remaining = std::hypot(dx, dy);
    if (remaining < 0.005) break;
    double err = wrap_angle(std::atan2(dy, dx) - robot_->pose.yaw);
    safe_->set_velocity(std::min(0.3, std::max(0.05, remaining)), 0.0, remaining > 0.05 ? 2.0 * err : 0.0);
    tick();
  }
  safe_->stop();
}

bool TaskDemo::reach(Arm& arm, const Vec3& target, double tol, const Vec3* tool, int max_steps) {
  for (int i = 0; i < max_steps; i++) {
    if (safe_->estopped) {
      tick();
      continue;
    }
    double dist = tool ? arm.step_towards(target, ARM_STEP, 0.05, *tool) : arm.step_towards(target, ARM_STEP);
    if (dist < tol) return true;
    tick();
  }
  return false;
}

void TaskDemo::arm_home(Arm& arm) {
  while (arm.step_home(ARM_STEP) > 1e-3) tick();
}

std::optional<std::string> TaskDemo::table_of(const Vec3& p) const {
  for (const auto& [name, t] : TABLES)
    if (std::fabs(p[0] - t.cx) <= t.hx + 0.05 && std::fabs(p[1] - t.cy) <= t.hy + 0.05) return name;
  return std::nullopt;
}

std::pair<Vec2, double> TaskDemo::stand_spot(const Vec3& point, const std::string& table, const std::string& side) const {
  const Table& t = TABLES.at(table);
  double lx = -t.fy, ly = t.fx;                    // robot's left when facing the table
  double shift = side == "left" ? -SIDE_OFFSET : SIDE_OFFSET;
  return {{point[0] - STAND_BACK * t.fx + shift * lx, point[1] - STAND_BACK * t.fy + shift * ly}, std::atan2(t.fy, t.fx)};
}

std::vector<Vec3> TaskDemo::object_positions(const MagicGrasp* skip) const {
  std::vector<Vec3> out;
  for (const auto& name : OBJECTS) {
    if (skip && name == skip->name) continue;
    int b = body_id(model_, name);
    out.push_back({robot_->d->xpos[3 * b], robot_->d->xpos[3 * b + 1], robot_->d->xpos[3 * b + 2]});
  }
  return out;
}

std::optional<Vec3> TaskDemo::free_spot(const std::string& table, const std::string& target) {
  auto others = object_positions(holding_grasp_);
  auto clear_of_all = [&](double x, double y) {
    for (const auto& o : others)
      if (std::hypot(x - o[0], y - o[1]) <= 0.12) return false;
    return true;
  };
  if (!target.empty() && table_of(home_[target]) == table) {      // its ORIGINAL spot, if free
    const Vec3& h = home_[target];
    if (clear_of_all(h[0], h[1])) return h;
  }
  const Table& t = TABLES.at(table);
  double depth = t.fx != 0 ? t.hx : t.hy;
  double ax = -t.fy, ay = t.fx;
  double bx = t.cx - (depth - 0.20) * t.fx, by = t.cy - (depth - 0.20) * t.fy;
  for (double s : SPOT_SHIFTS) {
    double px = bx + s * ax, py = by + s * ay;
    if (clear_of_all(px, py)) return Vec3{px, py, BOTTLE_Z};
  }
  return std::nullopt;
}

// =====================================================================================
// NAVIGATION with obstacle recognition (Day 17)
// =====================================================================================
std::vector<ObstacleObj> TaskDemo::sense() {
  std::vector<ObstacleObj> objects;
  if (!obstacle_eye_) return objects;
  cv::Mat bgr, depth;
  camera_->capture(robot_->d, bgr, depth);
  Vec2 here{robot_->pose.x, robot_->pose.y};
  if (!course_) {                                   // flat floor, nothing moves
    auto pts = obstacle_eye_->detect(robot_->d, depth, here, objects);
    grid_->add_points(pts);
  } else {
    obstacle_eye_->detect(robot_->d, depth, here, objects, [this](double x, double y) { return course_->ground(x, y); },
                          25.0);
    tracker_.update(objects, sim_t_);
    for (auto& o : objects) {
      auto& tr = o.track;
      o.label = tr->label();
      o.moving = tr->moving();
      o.speed = tr->speed();
      if (o.near && tr->is_static()) grid_->add_points(o.points);   // only still things go on the map
      if (tr->moving() && !spotted_ids_.count(tr->id) && o.distance < 20 && !table_of2(tr->pos) &&
          !on_known_obstacle(tr->pos)) {
        spotted_ids_.insert(tr->id);
        say(fmt("   SPOTTED moving %s: %.1f m/s, %.1f m away", tr->label().c_str(), tr->speed(), o.distance));
      }
    }
  }
  for (const auto& o : objects) {                   // log each NEW still obstacle once
    if (o.distance > 4.0 || o.moving || (course_ && !o.track->is_static())) continue;
    bool fresh = true;
    for (const auto& k : seen_obstacles_)
      if (std::hypot(o.centre[0] - k.centre[0], o.centre[1] - k.centre[1]) <= 0.7) { fresh = false; break; }
    if (!fresh) continue;
    seen_obstacles_.push_back(o);
    say(fmt("   RECOGNISED %s #%zu: %.2f m high, seen %.1f m away at x=%.1f y=%+.1f", o.kind.c_str(),
            seen_obstacles_.size(), o.height, o.distance, o.centre[0], o.centre[1]));
  }
  if (viewer_) {
    draw_obstacles(bgr, objects);
    cv::putText(bgr, fmt("obstacles recognised: %zu   moving things seen: %zu", seen_obstacles_.size(), spotted_ids_.size()),
                {10, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {255, 255, 255}, 2, cv::LINE_AA);
    cv::imshow(EYES_WINDOW, bgr);
    draw_map();
    cv::waitKey(1);
  }
  return objects;
}

void TaskDemo::draw_map(double half, int ppm) {
  int size = (int)(2 * half * ppm);
  cv::Mat img(size, size, CV_8UC3, cv::Scalar(235, 235, 235));
  double rx = robot_->pose.x, ry = robot_->pose.y;
  auto to_px = [&](double x, double y) { return cv::Point((int)((y - ry + half) * ppm), (int)((rx - x + half) * ppm)); };
  if (course_) {                                    // known terrain from the map
    struct Band { double x0, x1; cv::Scalar colour; const char* name; };
    Band bands[3] = {{ROAD_X0, ROAD_X1, {90, 90, 90}, "ROAD"},
                     {STEPS.start, STEPS_END, {150, 170, 190}, "STEPS"},
                     {HILL.xc - 2 * HILL.sigma, HILL.xc + 2 * HILL.sigma, {170, 205, 170}, "HILL"}};
    for (const auto& b : bands) {
      int a = to_px(b.x1, ry - half).y, bb = to_px(b.x0, ry - half).y;
      if (bb >= 0 && a <= size) {
        cv::rectangle(img, {0, std::max(a, 0)}, {size, std::min(bb, size)}, b.colour, -1);
        cv::putText(img, b.name, {size - 70, std::max(a, 0) + 18}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {40, 40, 40}, 1);
      }
    }
  }
  auto [i0, j0] = grid_->cell(rx - half, ry - half);
  auto [i1, j1] = grid_->cell(rx + half, ry + half);
  for (int i = std::max(i0, 0); i < std::min(i1 + 1, grid_->nx); i++)
    for (int j = std::max(j0, 0); j < std::min(j1 + 1, grid_->ny); j++)
      if (grid_->blocked_at(i, j)) {
        Vec2 c = grid_->centre(i, j);
        cv::Scalar col = grid_->occ_at(i, j) ? cv::Scalar(40, 40, 40) : cv::Scalar(190, 190, 205);
        cv::Point p = to_px(c[0], c[1]);
        cv::rectangle(img, {p.x - 2, p.y - 2}, {p.x + 2, p.y + 2}, col, -1);
      }
  if (nav_path_ && !nav_path_->empty()) {
    std::vector<cv::Point> pts{to_px(rx, ry)};
    for (const auto& q : *nav_path_) pts.push_back(to_px(q[0], q[1]));
    cv::polylines(img, pts, false, {0, 160, 0}, 2);
  }
  if (nav_goal_) cv::circle(img, to_px((*nav_goal_)[0], (*nav_goal_)[1]), 7, {0, 0, 220}, -1);
  for (size_t k = 0; k < seen_obstacles_.size(); k++) {
    cv::Point p = to_px(seen_obstacles_[k].centre[0], seen_obstacles_[k].centre[1]);
    cv::putText(img, "#" + std::to_string(k + 1), {p.x + 6, p.y}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {0, 120, 255}, 1, cv::LINE_AA);
  }
  for (const auto& th : moving_things()) {          // moving things with their direction
    cv::Point p = to_px(th.pos[0], th.pos[1]);
    cv::circle(img, p, 9, {255, 0, 255}, 2);
    cv::Point e = to_px(th.pos[0] + th.vel[0] * 1.5, th.pos[1] + th.vel[1] * 1.5);
    cv::arrowedLine(img, p, e, {255, 0, 255}, 2, 8, 0, 0.3);
    cv::putText(img, th.label, {p.x + 10, p.y - 6}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {160, 0, 160}, 1, cv::LINE_AA);
  }
  cv::Point rp = to_px(rx, ry);
  cv::circle(img, rp, 8, {200, 80, 0}, -1);
  cv::line(img, rp, to_px(rx + 0.5 * std::cos(robot_->pose.yaw), ry + 0.5 * std::sin(robot_->pose.yaw)), {200, 80, 0}, 2);
  std::string lines[3] = {fmt("robot x=%+.1f y=%+.1f  status: %s", rx, ry, nav_status_.c_str()),
                          "dark = obstacle, light = safety margin, green = path",
                          "pink = moving thing + direction, red = goal"};
  for (int k = 0; k < 3; k++)
    cv::putText(img, lines[k], {8, 18 + 18 * k}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {40, 40, 40}, 1, cv::LINE_AA);
  cv::imshow(MAP_WINDOW, img);
}

bool TaskDemo::on_known_obstacle(Vec2 pos, double radius) const {
  auto [i, j] = grid_->cell(pos[0], pos[1]);
  int r = (int)(radius / grid_->res) + 1;
  int ia = std::max(i - r, 0), ib = std::min(i + r + 1, grid_->nx), ja = std::max(j - r, 0), jb = std::min(j + r + 1, grid_->ny);
  for (int a = ia; a < ib; a++)
    for (int b = ja; b < jb; b++)
      if (grid_->occ_at(a, b)) return true;
  return false;
}

std::vector<Thing> TaskDemo::moving_things() const {
  std::vector<Thing> out;
  if (obstacle_eye_) {
    for (const auto& tr : tracker_.moving()) {
      if (table_of2(tr->pos) || on_known_obstacle(tr->pos)) continue;   // a table/stone seen from a new angle
      out.push_back({tr->pos, tr->vel, tr->label()});
    }
    return out;
  }
  if (course_)                                      // no camera: true positions
    for (const auto& ms : course_->visible_movers())
      if (std::hypot(ms.pos[0] - robot_->pose.x, ms.pos[1] - robot_->pose.y) < 20)
        out.push_back({ms.pos, ms.vel, ms.mover->kind});
  return out;
}

TaskDemo::Gap TaskDemo::closest_gap(Vec2 here, double heading, double speed) const {
  // If the robot walks at `speed` for the next 4 s, how close does the nearest moving thing come?
  // Only things that get CLOSER than they are now count.
  auto things = moving_things();
  if (things.empty()) return {1e9, "", 0};
  double fx = std::cos(heading) * speed, fy = std::sin(heading) * speed;
  Gap best{1e18, "", 0};
  for (const auto& th : things) {
    double gmin = 1e18, g0 = 0;
    for (int k = 0; k <= 16; k++) {
      double t = 0.25 * k;
      double g = std::hypot(th.pos[0] + th.vel[0] * t - (here[0] + fx * t), th.pos[1] + th.vel[1] * t - (here[1] + fy * t));
      if (k == 0) g0 = g;
      gmin = std::min(gmin, g);
    }
    double worst = gmin < g0 - 0.05 ? gmin : 1e9;
    if (worst < best.gap) best = {worst, th.label, g0};
  }
  return best;
}

void TaskDemo::look_both_ways() {
  safe_->stop();
  nav_status_ = "at the road: looking left and right";
  say("   at the road - stopping to look left and right");
  while (true) {
    for (double yaw : {60.0, -60.0, 0.0}) {
      robot_->set_head_deg(yaw, -10);               // look a little up: see further
      for (int k = 0; k < (int)(1.2 / DT); k++) {
        if (k % 5 == 0) sense();
        tick();
      }
    }
    double hy = robot_->pose.y;
    std::optional<std::pair<std::string, double>> nearest;
    for (const auto& th : moving_things()) {
      bool coming = ROAD_X0 - 1 <= th.pos[0] && th.pos[0] <= ROAD_X1 + 1 && (hy - th.pos[1]) * th.vel[1] > 0 &&
                    std::fabs(th.pos[1] - hy) < 25;
      if (coming && (!nearest || std::fabs(th.pos[1] - hy) < nearest->second))
        nearest = std::make_pair(th.label, std::fabs(th.pos[1] - hy));
    }
    if (!nearest) break;
    say(fmt("   %s coming, %.0f m away - waiting for it to pass", nearest->first.c_str(), nearest->second));
    ev_waits_++;
    for (int k = 0; k < (int)(2.0 / DT); k++) {
      if (k % 5 == 0) sense();
      tick();
    }
  }
  robot_->set_head_deg(0, 0);
  ev_crossings_++;
  say("   road is clear - crossing on the zebra crossing");
}

std::pair<bool, std::string> TaskDemo::navigate(Vec2 goal, double final_tol) {
  nav_active_ = true;
  nav_goal_ = goal;
  nav_path_.reset();
  nav_status_ = "walking";
  int replans = 0;
  long steps = 0;
  std::optional<Vec2> last_logged;
  double last_plan_t = -1e9, clear_for = 0.0;
  double cmd[2] = {0.0, 0.0};                       // current forward speed, turn speed
  bool waiting = false, crossing_ok = false;
  auto drive = [&](double vx, double wz) {          // smooth speed changes
    cmd[0] += std::max(-0.6 * DT, std::min(0.6 * DT, vx - cmd[0]));
    cmd[1] += std::max(-2.0 * DT, std::min(2.0 * DT, wz - cmd[1]));
    safe_->set_velocity(cmd[0], 0.0, cmd[1]);
  };
  auto finish = [&]() {
    safe_->stop();
    nav_active_ = false;
    nav_path_.reset();
    nav_status_ = "";
  };
  try {
    while (true) {
      if (steps % 5 == 0) sense();
      Vec2 here{robot_->pose.x, robot_->pose.y};
      if (std::hypot(goal[0] - here[0], goal[1] - here[1]) < final_tol) break;
      if (!nav_path_ || (steps % 5 == 0 && sim_t_ - last_plan_t > 1.0 && grid_->path_blocked(*nav_path_, here))) {
        auto path = grid_->astar(here, goal);
        last_plan_t = sim_t_;
        if (!path) {
          finish();
          return {false, "no free path to the goal"};
        }
        nav_path_ = path->size() > 1 ? Path(path->begin() + 1, path->end()) : Path{goal};
        nav_path_->back() = goal;
        replans++;
        if (replans > 1 && (!last_logged || std::hypot(here[0] - (*last_logged)[0], here[1] - (*last_logged)[1]) > 1.0)) {
          say("   new obstacle on the path - planning a way around it");
          last_logged = here;
        }
      }
      Vec2 wp = nav_path_->front();
      double d = std::hypot(wp[0] - here[0], wp[1] - here[1]);
      if (d < (nav_path_->size() == 1 ? final_tol : 0.3)) {
        nav_path_->erase(nav_path_->begin());
        if (nav_path_->empty()) break;
        continue;
      }
      double heading = std::atan2(wp[1] - here[1], wp[0] - here[0]);
      // ---- road: stop at the edge, look both ways, then cross ----
      if (course_) {
        double ahead_x = here[0] + 0.8 * std::cos(heading);
        if (!course_->on_road(here[0]) && course_->on_road(ahead_x) && !crossing_ok) {
          cmd[0] = cmd[1] = 0.0;
          look_both_ways();
          crossing_ok = true;
          nav_status_ = "crossing the road";
          continue;
        }
        if (crossing_ok && !course_->on_road(here[0]) && !course_->on_road(ahead_x)) {
          crossing_ok = false;
          nav_status_ = "walking";
        }
      }
      // ---- give way to moving things: walk on if safe, otherwise wait (or step back) ----
      double nominal = std::min(0.3, std::max(0.05, nav_path_->size() == 1 ? d : 1.0));
      Gap go = closest_gap(here, robot_->pose.yaw, nominal);
      if (waiting) {                                // only walk on when clearly safe
        if (go.gap >= CLEARANCE + 0.3) clear_for += DT;
        else clear_for = 0.0;
        if (clear_for < 0.5) {
          Gap stay = closest_gap(here, robot_->pose.yaw, 0.0);
          if (stay.gap < 0.9 && closest_gap(here, robot_->pose.yaw, -0.2).gap > stay.gap) {
            safe_->set_velocity(-0.2, 0.0, 0.0);    // too close even standing: back off
            cmd[0] = -0.2;
            cmd[1] = 0.0;
          } else {
            safe_->stop();
            cmd[0] = cmd[1] = 0.0;
          }
          tick();
          steps++;
          continue;
        }
        waiting = false;
        say("   way is clear - walking on");
        nav_status_ = crossing_ok ? "crossing the road" : "walking";
      } else if (go.gap < CLEARANCE) {
        Gap stay = closest_gap(here, robot_->pose.yaw, 0.0);
        if (stay.gap >= go.gap) {                   // standing still is safer: wait
          say(fmt("   WAITING: %s %.1f m away is crossing the way", go.label.c_str(), go.dist));
          ev_gave_way_++;
          nav_status_ = "waiting for " + go.label;
          waiting = true;
          clear_for = 0.0;
          safe_->stop();
          cmd[0] = cmd[1] = 0.0;
          tick();
          steps++;
          continue;
        }
        // walking on is safer than standing (already in its way): keep going
      }
      // ---- follow the path smoothly ----
      double err = wrap_angle(heading - robot_->pose.yaw);
      if (std::fabs(err) > 1.0) drive(0.0, std::copysign(0.5, err));     // big turn: turn on the spot
      else drive(nominal * std::max(0.0, std::cos(err)), 2.0 * err);
      tick();
      steps++;
    }
  } catch (...) {
    finish();
    throw;
  }
  finish();
  return {true, fmt("arrived (%d replans)", replans ? replans - 1 : 0)};
}

std::pair<Vec2, double> TaskDemo::staging_point(const std::string& table) const {
  const Table& t = TABLES.at(table);
  double depth = t.fx != 0 ? t.hx : t.hy;
  return {{t.cx - (depth + STAGING_BACK) * t.fx, t.cy - (depth + STAGING_BACK) * t.fy}, std::atan2(t.fy, t.fx)};
}

// =====================================================================================
// EXECUTOR: each action returns (ok, reason)
// =====================================================================================
std::pair<bool, std::string> TaskDemo::act_go_to_area(const std::string& target) {
  std::string table = last_table_[target];
  auto [s, yaw] = staging_point(table);
  double dist = std::hypot(s[0] - robot_->pose.x, s[1] - robot_->pose.y);
  if (dist < FAR) return {true, "already near " + pretty(table)};
  say(fmt("   walking %.1f m to %s (map starts empty; obstacles are added as they are seen)", dist, pretty(table).c_str()));
  auto [ok, reason] = navigate(s);
  if (ok) turn_to(yaw);
  return {ok, reason + fmt("; %zu obstacles recognised so far", seen_obstacles_.size())};
}

std::pair<bool, std::string> TaskDemo::act_find(const std::string& target) {
  std::string words = pretty(target);
  Vec3 truth = grasps_[target]->position();
  if (!camera_) {
    mem_object_ = truth;
    return {true, "position from the simulator"};
  }
  for (int turn = 0; turn < 8; turn++) {
    for (int head_yaw : HEAD_SCAN) {
      robot_->set_head_deg(head_yaw, 0);
      for (int k = 0; k < 10; k++) tick();
      auto [found, depth] = look("searching for " + words + "...");
      for (const Detection* hit : ObjectDetector::find_all(found, words)) {
        auto p = camera_->locate(robot_->d, depth, hit->centre.x, hit->centre.y, 3, BOTTLE_RADIUS);
        if (!p || !(MIN_HEIGHT < (*p)[2] && (*p)[2] < MAX_HEIGHT)) continue;
        robot_->set_head_deg(0, 0);
        mem_object_ = *p;
        return {true, fmt("seen at x=%.2f y=%+.2f (error %.1f cm)", (*p)[0], (*p)[1], dist3(*p, truth) * 100)};
      }
    }
    turn_to(robot_->pose.yaw + deg2rad(45));
  }
  robot_->set_head_deg(0, 0);
  return {false, words + " not seen anywhere"};
}

std::pair<bool, std::string> TaskDemo::act_approach(const std::string& target) {
  Vec3 p = *mem_object_;
  auto table = table_of(p);
  if (!table) return {false, "the object is not on a known table"};
  const Table& t = TABLES.at(*table);
  double rel = (p[0] - t.cx) * (-t.fy) + (p[1] - t.cy) * t.fx;   // on the robot's left or right?
  std::string side = rel >= 0 ? "left" : "right";
  auto [xy, yaw] = stand_spot(p, *table, side);
  walk_to(xy[0], xy[1]);
  turn_to(yaw);
  mem_side_ = side;
  mem_face_ = {t.fx, t.fy, 0.0};
  last_table_[target] = *table;
  return {true, "at " + pretty(*table) + ", using the " + side + " hand"};
}

std::pair<bool, std::string> TaskDemo::act_grasp(const std::string& target) {
  Vec3 p = *mem_object_, f = mem_face_;
  robot_->animate_arms = false;
  auto arm = std::make_unique<Arm>(model_, robot_->d, mem_side_);
  auto back = [&](double k) { return Vec3{p[0] - k * f[0], p[1] - k * f[1], p[2] - k * f[2]}; };
  reach(*arm, back(BOTTLE_RADIUS + 0.10));
  bool ok = reach(*arm, back(BOTTLE_RADIUS + 0.03), GRASP_DIST);
  MagicGrasp* g = grasps_[target].get();
  if (!ok || dist3(arm->palm(), g->position()) > 0.12) {
    arm_home(*arm);
    robot_->animate_arms = true;
    return {false, "hand did not reach the bottle"};
  }
  g->attach(arm.get());
  holding_grasp_ = g;
  holding_arm_ = std::move(arm);
  return {true, "grasped (magic grasp)"};
}

std::pair<bool, std::string> TaskDemo::act_lift() {
  Vec3 up = holding_grasp_->position();
  up[2] += LIFT;
  Vec3 tool = holding_grasp_->local;
  reach(*holding_arm_, up, 0.01, &tool);
  return {true, fmt("lifted %.0f cm", LIFT * 100)};
}

std::pair<bool, std::string> TaskDemo::act_go_to_table(const std::string& table) {
  auto spot = free_spot(table, holding_grasp_ ? holding_grasp_->name : "");
  if (!spot) return {false, "no free place on " + pretty(table)};
  auto [xy, yaw] = stand_spot(*spot, table, mem_side_);
  walk_to(xy[0], xy[1]);
  turn_to(yaw);
  mem_spot_ = spot;
  const Table& t = TABLES.at(table);
  mem_face_ = {t.fx, t.fy, 0.0};
  return {true, fmt("at %s, place spot x=%.2f y=%+.2f", pretty(table).c_str(), (*spot)[0], (*spot)[1])};
}

std::pair<bool, std::string> TaskDemo::act_place(const std::string& table) {
  MagicGrasp* g = holding_grasp_;
  Arm& arm = *holding_arm_;
  Vec3 spot = *mem_spot_, tool = g->local;
  reach(arm, {spot[0], spot[1], spot[2] + 0.10}, 0.01, &tool);      // above the spot
  reach(arm, spot, 0.005, &tool);                                    // lower it
  g->release(spot);                                                  // stand it upright exactly there
  last_table_[g->name] = table;
  holding_grasp_ = nullptr;
  Vec3 palm = arm.palm();
  reach(arm, {palm[0] - 0.10 * mem_face_[0], palm[1] - 0.10 * mem_face_[1], palm[2] - 0.10 * mem_face_[2]});   // pull back
  arm_home(arm);
  holding_arm_.reset();
  robot_->animate_arms = true;
  return {true, fmt("placed on %s at x=%.2f y=%+.2f", pretty(table).c_str(), spot[0], spot[1])};
}

std::pair<bool, std::string> TaskDemo::act_return() {
  if (std::hypot(start_.x - robot_->pose.x, start_.y - robot_->pose.y) > FAR) {
    auto [ok, reason] = navigate({start_.x, start_.y});
    if (!ok) return {false, reason};
  } else {
    walk_to(start_.x, start_.y);
  }
  turn_to(start_.yaw);
  return {true, "back at the start position"};
}

std::pair<bool, std::string> TaskDemo::act(const std::string& action, const std::string& arg) {
  if (action == "GO_TO_AREA") return act_go_to_area(arg);
  if (action == "FIND") return act_find(arg);
  if (action == "APPROACH") return act_approach(arg);
  if (action == "GRASP") return act_grasp(arg);
  if (action == "LIFT") return act_lift();
  if (action == "GO_TO_TABLE") return act_go_to_table(arg);
  if (action == "PLACE") return act_place(arg);
  return act_return();
}

void TaskDemo::recover() {
  // after a failure: put down anything held (where the arm is), arm home, go back
  if (holding_grasp_) {
    Vec3 p = holding_grasp_->position();
    if (table_of(p)) holding_grasp_->release(Vec3{p[0], p[1], BOTTLE_Z});
    else holding_grasp_->release();
    holding_grasp_ = nullptr;
  }
  if (holding_arm_) {
    arm_home(*holding_arm_);
    holding_arm_.reset();
  }
  robot_->animate_arms = true;
  act_return();
}

bool TaskDemo::execute(const std::vector<Step>& plan) {
  mem_object_.reset();
  mem_spot_.reset();
  mem_side_.clear();
  for (size_t i = 0; i < plan.size(); i++) {
    const auto& [action, arg] = plan[i];
    say(fmt("[%zu/%zu] %s %s", i + 1, plan.size(), action.c_str(), arg.c_str()));
    auto [ok, reason] = act(action, arg);
    say(std::string("      ") + (ok ? "SUCCESS" : "FAILURE") + ": " + reason);
    if (!ok) {
      say("Stopping safely and going back to the start.");
      recover();
      return false;
    }
  }
  say("TASK COMPLETE");
  if (!stones_.empty()) {
    int found = 0, near = 0;
    for (const auto& ob : stones_) {
      for (const auto& o : seen_obstacles_)
        if (std::hypot(ob.pos[0] - o.centre[0], ob.pos[1] - o.centre[1]) < 0.8) { found++; break; }
      if (std::fabs(ob.pos[1]) < 2.5) near++;
    }
    std::printf("\nObstacle report: recognised %zu obstacles on the way; %d of %zu placed stones/boxes were recognised "
                "(%d of them lay within 2.5 m of the direct line).\n", seen_obstacles_.size(), found, stones_.size(), near);
  }
  if (course_) {
    std::printf("Moving things: %zu sightings by the camera; the robot gave way %d times, crossed the road %d times "
                "and waited for traffic %d times.\n", spotted_ids_.size(), ev_gave_way_, ev_crossings_, ev_waits_);
    std::printf("Closest each moving thing came to the robot (centre to centre):\n");
    for (const auto& mv : course_->movers) std::printf("   %-11s %5.1f m\n", mv.name.c_str(), course_->min_dist[mv.name]);
  }
  std::fflush(stdout);
  return true;
}

void TaskDemo::run_instructions() {
  bool once = args_.say.has_value(), given = false;
  while (true) {
    std::string sentence;
    if (once) {
      if (given) return;
      sentence = *args_.say;
      given = true;
    } else {
      std::printf("\nInstruction (or 'quit'): ");
      std::fflush(stdout);
      if (!std::getline(std::cin, sentence)) return;
    }
    std::string low = sentence;
    low.erase(0, low.find_first_not_of(" \t"));
    low.erase(low.find_last_not_of(" \t") + 1);
    std::transform(low.begin(), low.end(), low.begin(), ::tolower);
    if (low == "quit" || low == "exit" || low == "q") return;
    auto [plan, message] = make_plan(sentence);
    std::printf("\n\"%s\"\n%s\n", sentence.c_str(), message.c_str());
    if (plan) {
      for (size_t i = 0; i < plan->size(); i++)
        std::printf("   %zu. %s %s\n", i + 1, (*plan)[i].first.c_str(), (*plan)[i].second.c_str());
      execute(*plan);
    }
    if (once) return;
  }
}

void TaskDemo::run() {
  std::printf("%s\n", DOC);
  try {
    if (viewer_) {
      gfx_->cam.distance = 3.5;
      gfx_->cam.azimuth = 200;
      gfx_->cam.elevation = -25;
      gfx_->cam.lookat[0] = robot_->pose.x;
      gfx_->cam.lookat[2] = 0.9;
      gfx_->sync(robot_->d);
    }
    run_instructions();
    if (viewer_) {
      std::printf("\nFinished. Close the MuJoCo window to exit.\n");
      std::fflush(stdout);
      while (gfx_->is_running()) {
        gfx_->sync(robot_->d);
        cv::waitKey(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }
  } catch (const Quit&) {
    safe_->stop();
    std::printf("\nStopped by user.\n");
  }
  if (viewer_) cv::destroyAllWindows();
}

int main(int argc, char** argv) {
  try {
    Args a = parse_args(argc, argv);
    TaskDemo demo(a);
    demo.run();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
  return 0;
}
