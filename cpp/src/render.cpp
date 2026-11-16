#include "a3/render.hpp"
#include <cstring>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

namespace a3 {
static int g_glfw_users = 0;

Graphics::Graphics(const mjModel* m, bool visible, const std::string& title, int w, int h)
    : m_(m), visible_(visible) {
  if (g_glfw_users++ == 0 && !glfwInit())
    throw std::runtime_error("Could not start the graphics (GLFW). Is there a screen? "
                             "Without one, run with:  xvfb-run -a <program> ...");
  glfwWindowHint(GLFW_VISIBLE, visible ? GLFW_TRUE : GLFW_FALSE);
  win_ = glfwCreateWindow(visible ? w : 64, visible ? h : 64, title.c_str(), nullptr, nullptr);
  if (!win_)
    throw std::runtime_error("Could not open an OpenGL window. Without a screen, run with:  xvfb-run -a ...");
  glfwMakeContextCurrent(win_);
  glfwSwapInterval(0);
  mjv_defaultCamera(&cam);
  mjv_defaultOption(&opt);
  mjv_defaultScene(&scn_);
  mjr_defaultContext(&con_);
  mjv_makeScene(m, &scn_, 10000);
  // make sure the off-screen buffer is big enough for 640x480 camera pictures
  mjModel* mm = const_cast<mjModel*>(m);
  if (mm->vis.global.offwidth < 640) mm->vis.global.offwidth = 640;
  if (mm->vis.global.offheight < 480) mm->vis.global.offheight = 480;
  mjr_makeContext(m, &con_, mjFONTSCALE_150);
  con_.readDepthMap = mjDEPTH_ZEROFAR;            // reversed-Z depth: precise, same as Python's Renderer
  glfwSetWindowUserPointer(win_, this);
  if (visible) {
    glfwSetKeyCallback(win_, on_key);
    glfwSetMouseButtonCallback(win_, on_button);
    glfwSetCursorPosCallback(win_, on_move);
    glfwSetScrollCallback(win_, on_scroll);
  }
}

Graphics::~Graphics() {
  mjv_freeScene(&scn_);
  mjr_freeContext(&con_);
  if (win_) glfwDestroyWindow(win_);
  if (--g_glfw_users == 0) glfwTerminate();
}

bool Graphics::is_running() const { return !glfwWindowShouldClose(win_); }

void Graphics::sync(const mjData* d, const std::string& overlay) {
  glfwMakeContextCurrent(win_);
  if (!visible_) {
    glfwPollEvents();
    return;
  }
  mjrRect vp = {0, 0, 0, 0};
  glfwGetFramebufferSize(win_, &vp.width, &vp.height);
  mjr_setBuffer(mjFB_WINDOW, &con_);
  mjv_updateScene(m_, const_cast<mjData*>(d), &opt, nullptr, &cam, mjCAT_ALL, &scn_);
  mjr_render(vp, &scn_, &con_);
  if (!overlay.empty()) mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, vp, overlay.c_str(), nullptr, &con_);
  glfwSwapBuffers(win_);
  glfwPollEvents();
}

void Graphics::capture(const mjData* d, int cam_id, cv::Mat& bgr, cv::Mat& depth, int width, int height) {
  glfwMakeContextCurrent(win_);
  mjvCamera fixed;
  mjv_defaultCamera(&fixed);
  fixed.type = mjCAMERA_FIXED;
  fixed.fixedcamid = cam_id;
  mjrRect vp = {0, 0, width, height};
  mjr_setBuffer(mjFB_OFFSCREEN, &con_);
  mjv_updateScene(m_, const_cast<mjData*>(d), &opt, nullptr, &fixed, mjCAT_ALL, &scn_);
  std::vector<unsigned char> rgb(3 * width * height);
  std::vector<float> zbuf(width * height);
  mjr_render(vp, &scn_, &con_);                   // pass 1: colour picture
  mjr_readPixels(rgb.data(), nullptr, vp, &con_);
  mjtByte seg = scn_.flags[mjRND_SEGMENT], idc = scn_.flags[mjRND_IDCOLOR];
  scn_.flags[mjRND_SEGMENT] = 1;                  // pass 2: depth, rendered in segmentation mode
  scn_.flags[mjRND_IDCOLOR] = 1;                  // (more accurate, same as Python's Renderer)
  mjr_render(vp, &scn_, &con_);
  mjr_readPixels(nullptr, zbuf.data(), vp, &con_);
  scn_.flags[mjRND_SEGMENT] = seg;
  scn_.flags[mjRND_IDCOLOR] = idc;
  mjr_setBuffer(mjFB_WINDOW, &con_);
  // OpenGL rows start at the bottom: flip, convert RGB->BGR, and turn the depth buffer into metres
  cv::Mat rgb_img(height, width, CV_8UC3, rgb.data());
  cv::Mat z_img(height, width, CV_32F, zbuf.data());
  cv::Mat flipped;
  cv::flip(rgb_img, flipped, 0);
  cv::cvtColor(flipped, bgr, cv::COLOR_RGB2BGR);
  // MuJoCo renders depth in "reversed Z" mode: undo the OpenGL projection exactly like
  // Python's mujoco.Renderer (coefficients in float32, conversion in float64)
  float zfar = (float)(m_->vis.map.zfar * m_->stat.extent), znear = (float)(m_->vis.map.znear * m_->stat.extent);
  float c_coef = -(zfar + znear) / (zfar - znear);
  float d_coef = -(2.0f * zfar * znear) / (zfar - znear);
  c_coef = -0.5f * c_coef - 0.5f;
  d_coef = -0.5f * d_coef;
  cv::flip(z_img, depth, 0);
  depth = depth.clone();
  for (int r = 0; r < height; r++) {
    float* p = depth.ptr<float>(r);
    for (int c = 0; c < width; c++) p[c] = (float)((double)d_coef / ((double)p[c] + (double)c_coef));
  }
}

void Graphics::render_view(const mjData* d, mjvCamera& view, cv::Mat& bgr, int width, int height) {
  glfwMakeContextCurrent(win_);
  mjrRect vp = {0, 0, width, height};
  mjr_setBuffer(mjFB_OFFSCREEN, &con_);
  mjv_updateScene(m_, const_cast<mjData*>(d), &opt, nullptr, &view, mjCAT_ALL, &scn_);
  mjr_render(vp, &scn_, &con_);
  std::vector<unsigned char> rgb(3 * width * height);
  mjr_readPixels(rgb.data(), nullptr, vp, &con_);
  mjr_setBuffer(mjFB_WINDOW, &con_);
  cv::Mat img(height, width, CV_8UC3, rgb.data()), flipped;
  cv::flip(img, flipped, 0);
  cv::cvtColor(flipped, bgr, cv::COLOR_RGB2BGR);
}

bool Graphics::pop_key(int& key) {
  if (keys_.empty()) return false;
  key = keys_.front();
  keys_.pop_front();
  return true;
}

void Graphics::on_key(GLFWwindow* w, int key, int, int act, int) {
  if (act != GLFW_PRESS) return;
  static_cast<Graphics*>(glfwGetWindowUserPointer(w))->keys_.push_back(key);
}
void Graphics::on_button(GLFWwindow* w, int, int, int) {
  auto* g = static_cast<Graphics*>(glfwGetWindowUserPointer(w));
  g->left_ = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
  g->right_ = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
  g->middle_ = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
  glfwGetCursorPos(w, &g->lastx_, &g->lasty_);
}
void Graphics::on_move(GLFWwindow* w, double x, double y) {
  auto* g = static_cast<Graphics*>(glfwGetWindowUserPointer(w));
  if (!g->left_ && !g->right_ && !g->middle_) return;
  double dx = x - g->lastx_, dy = y - g->lasty_;
  g->lastx_ = x;
  g->lasty_ = y;
  int width, height;
  glfwGetWindowSize(w, &width, &height);
  bool shift = glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
  mjtMouse action = g->right_ ? (shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V)
                    : g->left_ ? (shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V) : mjMOUSE_ZOOM;
  mjv_moveCamera(g->m_, action, dx / height, dy / height, &g->cam);
}
void Graphics::on_scroll(GLFWwindow* w, double, double dy) {
  auto* g = static_cast<Graphics*>(glfwGetWindowUserPointer(w));
  mjv_moveCamera(g->m_, mjMOUSE_ZOOM, 0, -0.05 * dy, &g->cam);
}
}  // namespace a3
