#include "a3/util.hpp"
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <stdexcept>

namespace a3 {
static int must(int id, const std::string& kind, const std::string& name) {
  if (id < 0) throw std::runtime_error("MuJoCo model has no " + kind + " named '" + name + "'");
  return id;
}
int body_id(const mjModel* m, const std::string& n) { return must(mj_name2id(m, mjOBJ_BODY, n.c_str()), "body", n); }
int joint_id(const mjModel* m, const std::string& n) { return must(mj_name2id(m, mjOBJ_JOINT, n.c_str()), "joint", n); }
int camera_id(const mjModel* m, const std::string& n) { return must(mj_name2id(m, mjOBJ_CAMERA, n.c_str()), "camera", n); }
int joint_qadr(const mjModel* m, const std::string& n) { return m->jnt_qposadr[joint_id(m, n)]; }

double monotonic_seconds() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}
std::string now_stamp() {
  std::time_t t = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&t));
  return buf;
}
std::string fmt(const char* f, ...) {
  char buf[4096];
  va_list ap;
  va_start(ap, f);
  std::vsnprintf(buf, sizeof(buf), f, ap);
  va_end(ap);
  return buf;
}
}  // namespace a3
