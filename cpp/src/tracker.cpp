#include "a3/tracker.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace a3 {
static const double WINDOW = 1.5;

Track::Track(int tid, const ObstacleObj& o, double t)
    : id(tid), pos(o.centre), height(o.height), size(o.size), last(t) {
  history.push_back({t, pos});
}
void Track::add(const ObstacleObj& o, double t) {
  pos = o.centre;
  history.push_back({t, pos});
  while (!history.empty() && t - history.front().first > WINDOW) history.pop_front();
  double t0 = history.front().first;
  Vec2 p0 = history.front().second;
  double sp = t - t0;
  vel = sp > 0.2 ? Vec2{(pos[0] - p0[0]) / sp, (pos[1] - p0[1]) / sp} : Vec2{0, 0};
  height = std::max(height * 0.8, o.height);
  size = std::max(size * 0.8, o.size);
  age += 1;
  last = t;
  if (moving()) was_moving = true;
}
double Track::travel() const {
  const Vec2& p0 = history.front().second;
  return std::hypot(pos[0] - p0[0], pos[1] - p0[1]);
}
bool Track::moving() const {
  double s = speed();
  return age >= 4 && span() >= 0.5 && travel() >= 0.4 && 0.3 <= s && s <= 8.0;
}
std::string Track::label() const {
  if (!moving() && !was_moving) return height < 0.6 ? "stone" : "obstacle";
  if (height > 1.2 && size > 2.5) return "vehicle (car?)";
  if (height > 1.2 && size > 1.1) return "rider (bike?)";
  if (height > 1.2) return "person?";
  return "animal?";
}

void Tracker::update(std::vector<ObstacleObj>& objects, double t) {
  std::vector<std::shared_ptr<Track>> free = tracks;
  std::vector<size_t> order(objects.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return objects[a].distance < objects[b].distance; });
  for (size_t idx : order) {
    ObstacleObj& obj = objects[idx];
    std::shared_ptr<Track> best;
    double best_d = gate_;
    for (const auto& tr : free) {
      double gx = tr->pos[0] + tr->vel[0] * (t - tr->last), gy = tr->pos[1] + tr->vel[1] * (t - tr->last);
      double d = std::hypot(gx - obj.centre[0], gy - obj.centre[1]);
      if (d < best_d) { best = tr; best_d = d; }
    }
    if (!best) {
      best = std::make_shared<Track>(next_id_++, obj, t);
      tracks.push_back(best);
    } else {
      free.erase(std::find(free.begin(), free.end(), best));
      best->add(obj, t);
    }
    obj.track = best;
  }
  tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [&](const std::shared_ptr<Track>& tr) { return t - tr->last > forget_; }),
               tracks.end());
}
std::vector<std::shared_ptr<Track>> Tracker::moving() const {
  std::vector<std::shared_ptr<Track>> out;
  for (const auto& tr : tracks)
    if (tr->moving()) out.push_back(tr);
  return out;
}
}  // namespace a3
