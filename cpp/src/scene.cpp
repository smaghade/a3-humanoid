#include "a3/scene.hpp"
#include <cmath>
#include <fstream>
#include <stdexcept>
#include "a3/util.hpp"

namespace a3 {
const char* HEAD_BODY = "head_pitch_Link";
const char* CAMERA_NAME = "head_camera";
const std::vector<std::string> OBJECTS = {"red_bottle", "blue_bottle", "yellow_bottle", "cup"};
const std::map<std::string, Table> TABLES = {
    {"table_1", {1.20, 0.00, 0.35, 0.60, 1.0, 0.0}},   // with the bottles
    {"table_2", {0.00, 1.60, 0.60, 0.35, 0.0, 1.0}},   // empty, on the left
};
static const double CAMERA_POS[3] = {0.09, 0.0, -0.02};   // at the stereo-camera shape on the head

mjsBody* add_body(mjSpec* spec, mjsBody* parent, const std::string& name, const std::array<double, 3>& pos) {
  mjsBody* b = mjs_addBody(parent ? parent : mjs_findBody(spec, "world"), nullptr);
  if (!name.empty()) mjs_setName(b->element, name.c_str());
  for (int i = 0; i < 3; i++) b->pos[i] = pos[i];
  return b;
}
mjsGeom* add_geom(mjsBody* body, mjtGeom type, const std::array<double, 3>& size,
                  const std::array<float, 4>& rgba, const std::array<double, 3>& pos, const double* quat) {
  mjsGeom* g = mjs_addGeom(body, nullptr);
  g->type = type;
  for (int i = 0; i < 3; i++) { g->size[i] = size[i]; g->pos[i] = pos[i]; }
  for (int i = 0; i < 4; i++) g->rgba[i] = rgba[i];
  if (quat) for (int i = 0; i < 4; i++) g->quat[i] = quat[i];
  return g;
}

static void camera_quat(double tilt_down_deg, double quat[4]) {
  // camera looking forward from the head, image-up = up, tilted down
  const double base[9] = {0, 0, -1, -1, 0, 0, 0, 1, 0};
  double t = deg2rad(tilt_down_deg);
  const double tilt[9] = {std::cos(t), 0, std::sin(t), 0, 1, 0, -std::sin(t), 0, std::cos(t)};
  double rot[9];
  mju_mulMatMat(rot, tilt, base, 3, 3, 3);
  mju_mat2Quat(quat, rot);
}

mjModel* compile_scene(mjSpec* spec) {
  mjModel* m = mj_compile(spec, nullptr);
  if (!m) throw std::runtime_error(std::string("MuJoCo could not build the scene: ") + mjs_getError(spec));
  return m;
}

static void add_object(mjSpec* spec, const std::string& name, std::array<double, 3> pos, double radius,
                       double half_height, std::array<float, 4> rgba) {
  mjsBody* b = add_body(spec, nullptr, name, pos);
  mjs_addFreeJoint(b);
  mjsGeom* g = add_geom(b, mjGEOM_CYLINDER, {radius, half_height, 0}, rgba);
  g->mass = 0.3;
}

Scene build_scene(const std::string& mjcf_path, bool with_objects, double camera_fovy, double camera_tilt_deg,
                  const std::vector<Obstacle>& obstacles) {
  char err[1000] = "";
  Scene s;
  s.spec = mj_parseXML(mjcf_path.c_str(), nullptr, err, sizeof(err));
  if (!s.spec) throw std::runtime_error("Cannot load " + mjcf_path + ": " + err);
  mjsBody* head = mjs_findBody(s.spec, HEAD_BODY);
  if (!head) throw std::runtime_error(std::string("The robot model has no body ") + HEAD_BODY);
  mjsCamera* cam = mjs_addCamera(head, nullptr);
  mjs_setName(cam->element, CAMERA_NAME);
  for (int i = 0; i < 3; i++) cam->pos[i] = CAMERA_POS[i];
  camera_quat(camera_tilt_deg, cam->quat);
  cam->fovy = camera_fovy;
  if (!with_objects) {
    s.model = compile_scene(s.spec);
    return s;
  }
  const Table& t2 = TABLES.at("table_2");                       // the second, empty table
  add_geom(add_body(s.spec, nullptr, "table_2", {t2.cx, t2.cy, TABLE_HEIGHT / 2}), mjGEOM_BOX,
           {t2.hx, t2.hy, TABLE_HEIGHT / 2}, {0.45f, 0.45f, 0.50f, 1});
  add_geom(add_body(s.spec, nullptr, "table", {1.2, 0.0, 0.375}), mjGEOM_BOX, {0.35, 0.6, 0.375},
           {0.55f, 0.38f, 0.22f, 1});
  add_object(s.spec, "red_bottle", {1.05, 0.18, 0.87}, 0.035, 0.12, {0.85f, 0.1f, 0.1f, 1});
  add_object(s.spec, "blue_bottle", {1.05, -0.18, 0.87}, 0.035, 0.12, {0.1f, 0.25f, 0.85f, 1});
  add_object(s.spec, "yellow_bottle", {1.05, 0.38, 0.87}, 0.035, 0.12, {0.95f, 0.85f, 0.1f, 1});
  add_object(s.spec, "cup", {1.0, 0.0, 0.80}, 0.045, 0.05, {0.95f, 0.95f, 0.95f, 1});
  for (const auto& ob : obstacles)                               // stones / boxes on the floor
    add_geom(add_body(s.spec, nullptr, ob.name, ob.pos), ob.shape == "stone" ? mjGEOM_ELLIPSOID : mjGEOM_BOX,
             ob.size, ob.rgba);
  s.model = compile_scene(s.spec);
  // keep the objects on the table in the robot's saved standing pose (keyframe)
  bool has_keys = false;
  for (mjsElement* el = mjs_firstElement(s.spec, mjOBJ_KEY); el; el = mjs_nextElement(s.spec, el)) {
    has_keys = true;
    mjsKey* key = mjs_asKey(el);
    std::vector<double> qpos = *key->qpos;
    qpos.resize(s.model->nq, 0.0);
    for (const auto& name : OBJECTS) {
      int adr = s.model->jnt_qposadr[s.model->body_jntadr[body_id(s.model, name)]];
      for (int i = 0; i < 7; i++) qpos[adr + i] = s.model->qpos0[adr + i];
    }
    mjs_setDouble(key->qpos, qpos.data(), (int)qpos.size());
  }
  if (has_keys) {
    mj_deleteModel(s.model);
    s.model = compile_scene(s.spec);
  }
  return s;
}

Intrinsics camera_intrinsics(const mjModel* m, int width, int height, const std::string& cam) {
  int id = camera_id(m, cam);
  double fovy = deg2rad(m->cam_fovy[id]);
  double fy = (height / 2.0) / std::tan(fovy / 2);
  return {fy, fy, width / 2.0, height / 2.0, m->cam_fovy[id], width, height};
}
void save_camera_yaml(const Intrinsics& in, const std::string& path) {
  std::ofstream f(path);
  f << "head_camera:\n"
    << "  fx: " << in.fx << "\n  fy: " << in.fy << "\n  cx: " << in.cx << "\n  cy: " << in.cy << "\n"
    << "  width: " << in.width << "\n  height: " << in.height << "\n  fovy_deg: " << in.fovy_deg << "\n";
}
}  // namespace a3
