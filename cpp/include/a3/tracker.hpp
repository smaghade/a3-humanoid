// Following objects from picture to picture to find out which ones MOVE (speed measured over
// the last 1.5 s). moving: >= 0.4 m travelled over >= 0.5 s at 0.3-8 m/s; static: never moving.
#pragma once
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "a3/obstacles.hpp"

namespace a3 {
struct Track {
  int id;
  Vec2 pos, vel{0, 0};
  double height, size;
  int age = 1;
  double last;
  std::deque<std::pair<double, Vec2>> history;
  bool was_moving = false;
  Track(int tid, const ObstacleObj& o, double t);
  void add(const ObstacleObj& o, double t);
  double speed() const { return std::hypot(vel[0], vel[1]); }
  double travel() const;
  double span() const { return last - history.front().first; }
  bool moving() const;
  bool is_static() const { return age >= 3 && travel() < 0.25 && !was_moving; }
  std::string label() const;
};

class Tracker {
 public:
  Tracker(double gate = 1.5, double forget = 1.0) : gate_(gate), forget_(forget) {}
  void update(std::vector<ObstacleObj>& objects, double t);     // sets obj.track for every object
  std::vector<std::shared_ptr<Track>> moving() const;
  std::vector<std::shared_ptr<Track>> tracks;

 private:
  double gate_, forget_;
  int next_id_ = 1;
};
}  // namespace a3
