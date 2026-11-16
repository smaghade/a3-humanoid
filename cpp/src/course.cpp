#include "a3/course.hpp"
#include <algorithm>
#include <cmath>

namespace a3 {
const HillSpec HILL;
const StepsSpec STEPS;
const double STEPS_END = STEPS.start + 2 * STEPS.n * STEPS.depth + STEPS.platform;
static const double LANE_1 = -43.25, LANE_2 = -39.75;
static const std::vector<std::pair<double, double>> STONE_ZONES = {
    {-81, -74}, {-70.5, -64.5}, {-36, -31}, {-24.3, -21.6}, {-18.4, -11.6}, {-8.4, -4}};
static const double ROT_X90[4] = {0.7071068, 0.7071068, 0, 0};   // wheels: cylinder axis sideways

struct Part {
  mjtGeom type;
  std::array<double, 3> size, pos;
  bool wheel;
  std::array<float, 4> rgba;
};
static std::vector<Part> shape(const std::string& kind) {
  std::vector<Part> p;
  const std::array<float, 4> tyre{0.1f, 0.1f, 0.1f, 1};
  if (kind == "car") {
    p = {{mjGEOM_BOX, {1.9, 0.85, 0.35}, {0, 0, 0.62}, false, {0.30f, 0.35f, 0.50f, 1}},
         {mjGEOM_BOX, {1.0, 0.75, 0.30}, {-0.2, 0, 1.27}, false, {0.26f, 0.30f, 0.42f, 1}}};
    for (int sx : {-1, 1})
      for (int sy : {-1, 1}) p.push_back({mjGEOM_CYLINDER, {0.33, 0.12, 0}, {sx * 1.25, sy * 0.86, 0.33}, true, tyre});
  } else if (kind == "motorcycle") {
    p = {{mjGEOM_BOX, {0.8, 0.15, 0.2}, {0, 0, 0.62}, false, {0.50f, 0.32f, 0.30f, 1}},
         {mjGEOM_CYLINDER, {0.32, 0.07, 0}, {0.75, 0, 0.32}, true, tyre},
         {mjGEOM_CYLINDER, {0.32, 0.07, 0}, {-0.75, 0, 0.32}, true, tyre},
         {mjGEOM_CAPSULE, {0.18, 0.33, 0}, {-0.1, 0, 1.2}, false, {0.35f, 0.33f, 0.30f, 1}},
         {mjGEOM_SPHERE, {0.15, 0, 0}, {-0.05, 0, 1.75}, false, {0.25f, 0.25f, 0.28f, 1}}};
  } else if (kind == "bicycle") {
    p = {{mjGEOM_BOX, {0.6, 0.03, 0.03}, {0, 0, 0.7}, false, {0.40f, 0.40f, 0.45f, 1}},
         {mjGEOM_CYLINDER, {0.33, 0.02, 0}, {0.55, 0, 0.33}, true, tyre},
         {mjGEOM_CYLINDER, {0.33, 0.02, 0}, {-0.55, 0, 0.33}, true, tyre},
         {mjGEOM_CAPSULE, {0.17, 0.33, 0}, {-0.05, 0, 1.15}, false, {0.45f, 0.42f, 0.36f, 1}},
         {mjGEOM_SPHERE, {0.12, 0, 0}, {0.0, 0, 1.65}, false, {0.55f, 0.47f, 0.40f, 1}}};
  } else if (kind == "person") {
    p = {{mjGEOM_CAPSULE, {0.08, 0.25, 0}, {0, 0.1, 0.33}, false, {0.30f, 0.30f, 0.35f, 1}},
         {mjGEOM_CAPSULE, {0.08, 0.25, 0}, {0, -0.1, 0.33}, false, {0.30f, 0.30f, 0.35f, 1}},
         {mjGEOM_CAPSULE, {0.2, 0.32, 0}, {0, 0, 1.12}, false, {0.48f, 0.42f, 0.36f, 1}},
         {mjGEOM_SPHERE, {0.12, 0, 0}, {0, 0, 1.75}, false, {0.60f, 0.50f, 0.42f, 1}}};
  } else if (kind == "dog") {
    p = {{mjGEOM_ELLIPSOID, {0.40, 0.15, 0.16}, {0, 0, 0.45}, false, {0.45f, 0.36f, 0.26f, 1}},
         {mjGEOM_SPHERE, {0.13, 0, 0}, {0.45, 0, 0.62}, false, {0.42f, 0.34f, 0.25f, 1}}};
    for (int sx : {-1, 1})
      for (int sy : {-1, 1})
        p.push_back({mjGEOM_CAPSULE, {0.05, 0.14, 0}, {sx * 0.28, sy * 0.1, 0.18}, false, {0.40f, 0.32f, 0.24f, 1}});
  } else if (kind == "cat") {
    p = {{mjGEOM_ELLIPSOID, {0.22, 0.08, 0.09}, {0, 0, 0.25}, false, {0.42f, 0.42f, 0.40f, 1}},
         {mjGEOM_SPHERE, {0.08, 0, 0}, {0.25, 0, 0.35}, false, {0.40f, 0.40f, 0.38f, 1}}};
    for (int sx : {-1, 1})
      for (int sy : {-1, 1})
        p.push_back({mjGEOM_CAPSULE, {0.03, 0.08, 0}, {sx * 0.15, sy * 0.05, 0.1}, false, {0.38f, 0.38f, 0.36f, 1}});
  }
  return p;
}
static bool is_vehicle(const std::string& k) { return k == "car" || k == "motorcycle" || k == "bicycle"; }

Course::Course(double sx, int n_stones, unsigned seed) : start_x(sx), rng_(seed) {
  auto mk = [](std::string n, std::string k, double x, double lo, double hi, double sp, double y0, int dir,
               bool loop = false, double gap = 6.0) {
    Mover m;
    m.name = n; m.kind = k; m.x = x; m.y_min = lo; m.y_max = hi; m.speed = sp; m.y = y0; m.dir = dir;
    m.loop = loop; m.gap = gap;
    return m;
  };
  movers = {mk("person_1", "person", -72.0, -5, 5, 1.0, -5.0, +1),
            mk("dog", "dog", -52.0, -4, 4, 1.5, 4.0, -1),
            mk("car", "car", LANE_1, -32, 32, 3.0, -12.0, +1, true, 6),
            mk("motorcycle", "motorcycle", LANE_1, -32, 32, 4.5, -30.0, +1, true, 9),
            mk("bicycle", "bicycle", LANE_2, -32, 32, 2.0, 20.0, -1, true, 5),
            mk("person_2", "person", -20.0, -4, 4, 0.8, -4.0, +1),
            mk("cat", "cat", -10.0, -3, 3, 0.6, 3.0, -1)};
  make_stones(n_stones);
  for (const auto& m : movers) min_dist[m.name] = 1e9;
}

double Course::ground(double x, double y) const {
  double h = 0.0;
  if (std::fabs(x - HILL.xc) <= HILL.half_x && std::fabs(y) <= HILL.half_y)
    h = HILL.height * std::exp(-0.5 * std::pow((x - HILL.xc) / HILL.sigma, 2));
  for (int k = 1; k <= STEPS.n; k++) {
    double lo = STEPS.start + (k - 1) * STEPS.depth, hi = STEPS_END - (k - 1) * STEPS.depth;
    if (x >= lo && x <= hi && std::fabs(y) <= STEPS.half_y) h = std::max(h, k * STEPS.rise);
  }
  if (on_road(x)) h = std::max(h, 0.01);
  return h;
}

void Course::make_stones(int n) {
  std::vector<std::pair<double, double>> zones;
  for (const auto& z : STONE_ZONES)
    if (z.first > start_x + 3) zones.push_back(z);
  if (zones.empty()) return;
  for (int k = 0; k < n; k++) {
    auto [lo, hi] = zones[k % zones.size()];
    double x = uniform(lo, hi);
    double y = (k % 3) ? normal(0.0, 0.8) : uniform(-0.2, 0.2);
    double g = ground(x, y);
    float grey = (float)uniform(0.30, 0.55);
    std::array<float, 4> rgba{grey + (float)uniform(0, 0.06), grey, grey - (float)uniform(0, 0.06), 1};
    Obstacle o;
    if (k % 4 == 3) {
      double half = uniform(0.15, 0.30);
      o = {"box_" + std::to_string(k), "box", {x, y, g + half}, {half, half * uniform(0.8, 1.5), half}, rgba};
    } else {
      double r = uniform(0.18, 0.40);
      o = {"stone_" + std::to_string(k), "stone", {x, y, g + r * 0.5}, {r, r * uniform(0.7, 1.2), r * 0.6}, rgba};
    }
    stones.push_back(o);
  }
}

void Course::add_to_spec(mjSpec* spec) const {
  // hill: a smooth height field
  const int ncol = 160, nrow = 40;
  std::vector<float> data;
  for (int r = 0; r < nrow; r++)
    for (int c = 0; c < ncol; c++) {
      double x = -HILL.half_x + 2 * HILL.half_x * c / (ncol - 1);
      data.push_back((float)std::exp(-0.5 * std::pow(x / HILL.sigma, 2)));
    }
  mjsHField* hf = mjs_addHField(spec);
  mjs_setName(hf->element, "hill");
  hf->size[0] = HILL.half_x; hf->size[1] = HILL.half_y; hf->size[2] = HILL.height; hf->size[3] = 0.05;
  hf->nrow = nrow;
  hf->ncol = ncol;
  mjs_setFloat(hf->userdata, data.data(), (int)data.size());
  mjsGeom* hg = add_geom(add_body(spec, nullptr, "hill", {HILL.xc, 0, 0}), mjGEOM_HFIELD, {0, 0, 0},
                         {0.42f, 0.48f, 0.36f, 1});
  mjs_setString(hg->hfieldname, "hill");
  // steps: stacked slabs (up, platform, down)
  for (int k = 1; k <= STEPS.n; k++) {
    double lo = STEPS.start + (k - 1) * STEPS.depth, hi = STEPS_END - (k - 1) * STEPS.depth;
    float c = 0.03f * k;
    add_geom(add_body(spec, nullptr, "step_" + std::to_string(k), {(lo + hi) / 2, 0, (k - 0.5) * STEPS.rise}),
             mjGEOM_BOX, {(hi - lo) / 2, STEPS.half_y, STEPS.rise / 2}, {0.62f - c, 0.60f - c, 0.56f - c, 1});
  }
  // road, zebra crossing, lane line
  double mid = (ROAD_X0 + ROAD_X1) / 2;
  mjsBody* road = add_body(spec, nullptr, "road", {mid, 0, 0.005});
  add_geom(road, mjGEOM_BOX, {(ROAD_X1 - ROAD_X0) / 2, 40, 0.005}, {0.18f, 0.18f, 0.20f, 1});
  for (int i = 0; i < 7; i++)
    add_geom(road, mjGEOM_BOX, {0.25, CROSSING_HALF_WIDTH, 0.002}, {0.9f, 0.9f, 0.9f, 1},
             {ROAD_X0 + 0.5 + i * 1.0 - mid, 0, 0.006});
  for (double yy = -38; yy < 39; yy += 4.0)
    if (std::fabs(yy) > CROSSING_HALF_WIDTH + 1)
      add_geom(road, mjGEOM_BOX, {0.06, 1.0, 0.002}, {0.9f, 0.9f, 0.9f, 1}, {0, yy, 0.006});
  for (const auto& ob : stones)
    add_geom(add_body(spec, nullptr, ob.name, ob.pos), ob.shape == "stone" ? mjGEOM_ELLIPSOID : mjGEOM_BOX,
             ob.size, ob.rgba);
  for (const auto& mv : movers) {                 // moving things
    mjsBody* b = add_body(spec, nullptr, mv.name, {mv.x, mv.y, 0});
    mjs_addFreeJoint(b);
    for (const auto& p : shape(mv.kind)) add_geom(b, p.type, p.size, p.rgba, p.pos, p.wheel ? ROT_X90 : nullptr);
  }
}

void Course::bind(const mjModel* m, mjData* d) {
  m_ = m;
  d_ = d;
  for (auto& mv : movers) mv.qadr = m->jnt_qposadr[m->body_jntadr[body_id(m, mv.name)]];
  place_all();
}

void Course::update(double dt, Vec2 robot) {
  double rx = robot[0], ry = robot[1];
  bool robot_on_road = on_road(rx);
  for (auto& mv : movers) {
    mv.stopped_for_robot = false;
    if (mv.hidden) {                              // vehicle waiting off-screen to come again
      mv.wait -= dt;
      if (mv.wait <= 0) {
        mv.hidden = false;
        mv.y = mv.dir > 0 ? mv.y_min : mv.y_max;
      }
      continue;
    }
    if (mv.pause > 0) {                           // resting at the end of its walk
      mv.pause -= dt;
      min_dist[mv.name] = std::min(min_dist[mv.name], std::hypot(mv.x - rx, mv.y - ry));
      continue;
    }
    double ahead = (ry - mv.y) * mv.dir;          // how far the robot is ahead of it
    bool blocked;
    if (is_vehicle(mv.kind)) {
      blocked = robot_on_road && std::fabs(rx - mv.x) < 2.5 && 0 < ahead && ahead < 9.0;
      for (const auto& o : movers)                // keep distance to a vehicle in front
        if (&o != &mv && is_vehicle(o.kind) && !o.hidden && std::fabs(o.x - mv.x) < 0.5 &&
            0 < (o.y - mv.y) * mv.dir && (o.y - mv.y) * mv.dir < 7.0)
          blocked = true;
    } else {                                      // people and animals only stop at the last moment
      blocked = std::fabs(rx - mv.x) < 0.7 && 0 < ahead && ahead < 0.9;
    }
    if (blocked)
      mv.stopped_for_robot = robot_on_road || !is_vehicle(mv.kind);
    else
      mv.y += mv.dir * mv.speed * dt;
    if (mv.loop && (mv.y > mv.y_max || mv.y < mv.y_min)) {
      mv.hidden = true;
      mv.wait = mv.gap;
    } else if (!mv.loop && (mv.y > mv.y_max || mv.y < mv.y_min)) {
      mv.y = std::min(std::max(mv.y, mv.y_min), mv.y_max);
      mv.dir *= -1;
      mv.pause = uniform(3.0, 7.0);               // rest a moment, then walk back
    }
    min_dist[mv.name] = std::min(min_dist[mv.name], mv.hidden ? 1e9 : std::hypot(mv.x - rx, mv.y - ry));
  }
  place_all();
}

void Course::place_all() {
  double* q = d_->qpos;
  for (const auto& mv : movers) {
    double* p = q + mv.qadr;
    if (mv.hidden) {
      double v[7] = {mv.x, 200.0 * mv.dir, -5.0, 1, 0, 0, 0};
      std::copy(v, v + 7, p);
      continue;
    }
    double yaw = mv.dir > 0 ? PI / 2 : -PI / 2;
    double v[7] = {mv.x, mv.y, ground(mv.x, mv.y), std::cos(yaw / 2), 0, 0, std::sin(yaw / 2)};
    std::copy(v, v + 7, p);
  }
}

std::vector<MoverState> Course::visible_movers() const {
  std::vector<MoverState> out;
  for (const auto& mv : movers)
    if (!mv.hidden) {
      double v = (mv.stopped_for_robot || mv.pause > 0) ? 0.0 : mv.speed * mv.dir;
      out.push_back({&mv, {mv.x, mv.y}, {0.0, v}});
    }
  return out;
}
}  // namespace a3
