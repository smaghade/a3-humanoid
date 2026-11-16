// Day 7: keyboard -> SafetySupervisor -> RobotInterface (MockRobot for now), C++ version of teleop.py.
// Publishes the allowed velocity on /cmd_vel so rosbag2 can record it.
//   cd ros2_ws && colcon build --packages-select a3_teleop_cpp && source install/setup.bash
//   cd .. && ros2 run a3_teleop_cpp teleop          (run from the project folder: reads config/robot.yaml)
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include "a3/config.hpp"
#include "a3/robot.hpp"
#include "a3/safety.hpp"

using namespace a3;
static const char* HELP = R"(
  W / S  = faster forward / backward      A / D = turn left / right
  Q      = stop                           SPACE = EMERGENCY STOP
  R      = reset emergency stop           X     = exit
)";
static const double STEP_V = 0.1, STEP_W = 0.25, DT = 0.1;

// MockRobot::step and the helpers live in robot.cpp/util.cpp, which also need MuJoCo; the small
// pieces teleop uses are defined here so this node does not need the MuJoCo library.
void MockRobot::step(double dt) {
  double c = std::cos(pose.yaw), s = std::sin(pose.yaw);
  pose.x += (velocity[0] * c - velocity[1] * s) * dt;
  pose.y += (velocity[0] * s + velocity[1] * c) * dt;
  pose.yaw += velocity[2] * dt;
}
double a3::monotonic_seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool get_key(double timeout, char& c) {
  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(STDIN_FILENO, &fds);
  timeval tv{0, (suseconds_t)(timeout * 1e6)};
  if (select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv) <= 0) return false;
  return read(STDIN_FILENO, &c, 1) == 1;
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("teleop");
  auto pub = node->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
  MockRobot robot;                                   // later: SimulationRobot / PhysicalRobot
  SafetySupervisor safe(robot, load_config("config/robot.yaml"));
  safe.stand();
  double vx = 0, wz = 0;
  std::printf("%s\n", HELP);
  termios old_settings, t;
  tcgetattr(STDIN_FILENO, &old_settings);
  t = old_settings;
  t.c_lflag &= ~(ICANON | ECHO);                     // read keys without pressing Enter
  tcsetattr(STDIN_FILENO, TCSANOW, &t);
  while (rclcpp::ok()) {
    char key;
    if (get_key(DT, key)) {
      char k = (char)std::tolower((unsigned char)key);
      if (k == 'w') vx += STEP_V;
      else if (k == 's') vx -= STEP_V;
      else if (k == 'a') wz += STEP_W;
      else if (k == 'd') wz -= STEP_W;
      else if (k == 'q') { vx = wz = 0.0; safe.stop(); }
      else if (key == ' ') { vx = wz = 0.0; safe.emergency_stop(); }
      else if (k == 'r') safe.reset_emergency_stop();
      else if (k == 'x') break;
    }
    safe.set_velocity(vx, 0.0, wz);                  // sent every loop, so the timeout stays fresh
    vx = robot.velocity[0];                          // what safety actually allowed
    wz = robot.velocity[2];
    robot.step(DT);
    geometry_msgs::msg::Twist msg;
    msg.linear.x = vx;
    msg.angular.z = wz;
    pub->publish(msg);                               // so rosbag2 can record it
    Pose2D p = robot.pose;
    std::printf("\rspeed %+.2f m/s  turn %+.2f rad/s | x=%+.2f y=%+.2f facing %+5.0f deg | %s      ", vx, wz, p.x, p.y,
                p.yaw * 180.0 / 3.14159265358979323846, safe.estopped ? "EMERGENCY STOP" : "ok");
    std::fflush(stdout);
  }
  tcsetattr(STDIN_FILENO, TCSADRAIN, &old_settings);
  safe.stop();
  rclcpp::shutdown();
  std::printf("\nTeleop closed. Robot stopped.\n");
  return 0;
}
