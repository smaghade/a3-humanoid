// Day 17 navigation: a 2D map of the floor (occupancy grid, 10 cm cells) + A* path planning.
// Obstacles are inflated by the robot's radius (0.45 m) for planning; a path is only replaced
// when a NEW obstacle comes within 0.3 m of it.
#pragma once
#include <optional>
#include <utility>
#include <vector>
#include "a3/util.hpp"

namespace a3 {
using Path = std::vector<Vec2>;

class OccupancyGrid {
 public:
  OccupancyGrid(Vec2 x_range, Vec2 y_range, double res = 0.1, double inflate = 0.45);
  std::pair<int, int> cell(double x, double y) const;
  Vec2 centre(int i, int j) const;
  bool inside(int i, int j) const { return 0 <= i && i < nx && 0 <= j && j < ny; }
  std::vector<std::pair<int, int>> add_points(const std::vector<Vec3>& pts);
  bool path_blocked(const Path& path, std::optional<Vec2> start = std::nullopt) const;
  std::optional<Path> astar(Vec2 start, Vec2 goal) const;

  bool occ_at(int i, int j) const { return occ[i * ny + j]; }
  bool blocked_at(int i, int j) const { return blocked[i * ny + j]; }
  double x0, y0, res;
  int nx, ny;
  std::vector<char> occ, blocked, tight;     // seen obstacle / + robot radius / + 0.3 m

 private:
  std::vector<char> with_escape(Vec2 start, double radius = 0.8) const;
  bool free_line(Vec2 a, Vec2 b, const std::vector<char>& grid) const;
  Path smooth(const Path& pts, const std::vector<char>& grid) const;
  std::vector<std::pair<int, int>> disk_, disk_tight_;
};
}  // namespace a3
