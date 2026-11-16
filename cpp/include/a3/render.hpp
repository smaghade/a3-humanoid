// Graphics: the 3D viewer window (like Python's mujoco.viewer) and off-screen camera pictures
// (like Python's mujoco.Renderer). One OpenGL context is used for both.
#pragma once
#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include <deque>
#include <opencv2/core.hpp>
#include <string>

namespace a3 {
class Graphics {
 public:
  // visible=false creates a hidden window (needed for camera pictures without a viewer;
  // on a computer without a screen run the program with:  xvfb-run -a ./program ...)
  Graphics(const mjModel* m, bool visible, const std::string& title = "MuJoCo - A3", int w = 1280, int h = 800);
  ~Graphics();
  Graphics(const Graphics&) = delete;
  Graphics& operator=(const Graphics&) = delete;

  bool is_running() const;                       // false once the window is closed
  void sync(const mjData* d, const std::string& overlay = "");   // draw the 3D view, handle events
  // Camera picture: RGB (CV_8UC3, BGR order like OpenCV) + depth in metres (CV_32F).
  void capture(const mjData* d, int cam_id, cv::Mat& bgr, cv::Mat& depth, int width = 640, int height = 480);
  // Picture from any free camera (e.g. an overview camera following the robot), BGR.
  void render_view(const mjData* d, mjvCamera& view, cv::Mat& bgr, int width = 640, int height = 480);
  bool pop_key(int& key);                        // keys pressed in the 3D window (GLFW key codes)

  mjvCamera cam;                                 // the viewer's free camera (lookat, distance, ...)
  mjvOption opt;

 private:
  static void on_key(GLFWwindow* w, int key, int sc, int act, int mods);
  static void on_button(GLFWwindow* w, int b, int act, int mods);
  static void on_move(GLFWwindow* w, double x, double y);
  static void on_scroll(GLFWwindow* w, double dx, double dy);
  const mjModel* m_;
  GLFWwindow* win_ = nullptr;
  bool visible_;
  mjvScene scn_;
  mjrContext con_;
  std::deque<int> keys_;
  bool left_ = false, right_ = false, middle_ = false;
  double lastx_ = 0, lasty_ = 0;
};

// Keys typed in the terminal (W/A/S/D...), read without pressing Enter.
class TerminalKeys {
 public:
  TerminalKeys();
  ~TerminalKeys();
  bool read(char& c);                            // non-blocking
  bool active() const { return active_; }
 private:
  bool active_ = false;
  char saved_[64];                               // saved terminal settings (struct termios)
};
}  // namespace a3
