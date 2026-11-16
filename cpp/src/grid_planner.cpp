#include "a3/grid_planner.hpp"
#include <cmath>
#include <map>
#include <queue>
#include <set>
#include <tuple>

namespace a3 {
OccupancyGrid::OccupancyGrid(Vec2 xr, Vec2 yr, double r, double inflate) : x0(xr[0]), y0(yr[0]), res(r) {
  nx = (int)std::ceil((xr[1] - xr[0]) / res);
  ny = (int)std::ceil((yr[1] - yr[0]) / res);
  occ.assign(nx * ny, 0);
  blocked.assign(nx * ny, 0);
  tight.assign(nx * ny, 0);
  int ri = (int)std::ceil(inflate / res), rt = (int)std::ceil(0.3 / res);
  for (int i = -ri; i <= ri; i++)
    for (int j = -ri; j <= ri; j++)
      if (i * i + j * j <= ri * ri) disk_.push_back({i, j});
  for (int i = -rt; i <= rt; i++)
    for (int j = -rt; j <= rt; j++)
      if (i * i + j * j <= rt * rt) disk_tight_.push_back({i, j});
}
std::pair<int, int> OccupancyGrid::cell(double x, double y) const {
  return {(int)((x - x0) / res), (int)((y - y0) / res)};    // truncation, like Python int()
}
Vec2 OccupancyGrid::centre(int i, int j) const { return {x0 + (i + 0.5) * res, y0 + (j + 0.5) * res}; }

std::vector<std::pair<int, int>> OccupancyGrid::add_points(const std::vector<Vec3>& pts) {
  std::set<std::pair<int, int>> cells;
  for (const auto& p : pts) {
    int i = (int)((p[0] - x0) / res), j = (int)((p[1] - y0) / res);
    if (i >= 0 && i < nx && j >= 0 && j < ny) cells.insert({i, j});
  }
  std::vector<std::pair<int, int>> fresh;
  for (auto [i, j] : cells) {
    if (occ[i * ny + j]) continue;
    occ[i * ny + j] = 1;
    fresh.push_back({i, j});
    for (auto [di, dj] : disk_)
      if (inside(i + di, j + dj)) blocked[(i + di) * ny + j + dj] = 1;
    for (auto [di, dj] : disk_tight_)
      if (inside(i + di, j + dj)) tight[(i + di) * ny + j + dj] = 1;
  }
  return fresh;
}

bool OccupancyGrid::free_line(Vec2 a, Vec2 b, const std::vector<char>& g) const {
  int n = (int)(std::hypot(b[0] - a[0], b[1] - a[1]) / (res * 0.5)) + 1;
  for (int k = 0; k < n; k++) {
    double t = n == 1 ? 0.0 : (double)k / (n - 1);              // numpy.linspace(0, 1, n)
    auto [i, j] = cell(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]));
    if (g[i * ny + j]) return false;
  }
  return true;
}

bool OccupancyGrid::path_blocked(const Path& path, std::optional<Vec2> start) const {
  Path pts;
  if (start) pts.push_back(*start);
  pts.insert(pts.end(), path.begin(), path.end());
  std::vector<char> t = tight;
  if (start) {
    auto [i, j] = cell((*start)[0], (*start)[1]);
    t[i * ny + j] = 0;
  }
  for (size_t k = 0; k + 1 < pts.size(); k++)
    if (!free_line(pts[k], pts[k + 1], t)) return true;
  if (pts.size() == 1) {
    auto [i, j] = cell(pts[0][0], pts[0][1]);
    return t[i * ny + j];
  }
  return false;
}

std::vector<char> OccupancyGrid::with_escape(Vec2 start, double radius) const {
  // safety-margin cells (not real obstacles) near the start are free, so the robot can walk
  // away from a table it is standing next to
  std::vector<char> b = blocked;
  auto [si, sj] = cell(start[0], start[1]);
  int r = (int)(radius / res);
  for (int i = std::max(si - r, 0); i < std::min(si + r + 1, nx); i++)
    for (int j = std::max(sj - r, 0); j < std::min(sj + r + 1, ny); j++)
      if ((i - si) * (i - si) + (j - sj) * (j - sj) <= r * r && !occ[i * ny + j]) b[i * ny + j] = 0;
  return b;
}

Path OccupancyGrid::smooth(const Path& pts, const std::vector<char>& g) const {
  Path out{pts[0]};
  size_t i = 0;
  while (i < pts.size() - 1) {
    size_t j = pts.size() - 1;
    while (j > i + 1 && !free_line(pts[i], pts[j], g)) j--;
    out.push_back(pts[j]);
    i = j;
  }
  return out;
}

std::optional<Path> OccupancyGrid::astar(Vec2 start, Vec2 goal) const {
  auto s = cell(start[0], start[1]), g = cell(goal[0], goal[1]);
  if (!inside(s.first, s.second) || !inside(g.first, g.second)) return std::nullopt;
  std::vector<char> bl = with_escape(start);
  bl[s.first * ny + s.second] = 0;
  if (bl[g.first * ny + g.second]) return std::nullopt;
  const int mv[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
  const double w[8] = {1, 1, 1, 1, 1.414, 1.414, 1.414, 1.414};
  auto h = [&](int i, int j) { return std::hypot(i - g.first, j - g.second); };
  // same ordering as Python's heapq of (f, cost, (i, j)) tuples
  using Item = std::tuple<double, double, int, int>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> open;
  std::vector<double> cost(nx * ny, 1e18);
  std::vector<int> came(nx * ny, -1);
  open.push({h(s.first, s.second), 0.0, s.first, s.second});
  cost[s.first * ny + s.second] = 0.0;
  while (!open.empty()) {
    auto [f, c, ci, cj] = open.top();
    open.pop();
    if (ci == g.first && cj == g.second) {
      Path cells;
      int cur = ci * ny + cj;
      cells.push_back(centre(ci, cj));
      while (came[cur] >= 0) {
        cur = came[cur];
        cells.push_back(centre(cur / ny, cur % ny));
      }
      Path fwd(cells.rbegin(), cells.rend());
      return smooth(fwd, bl);
    }
    if (c > cost[ci * ny + cj]) continue;
    for (int k = 0; k < 8; k++) {
      int ni = ci + mv[k][0], nj = cj + mv[k][1];
      if (!inside(ni, nj) || bl[ni * ny + nj]) continue;
      double nc = c + w[k];
      if (nc < cost[ni * ny + nj]) {
        cost[ni * ny + nj] = nc;
        came[ni * ny + nj] = ci * ny + cj;
        open.push({nc + h(ni, nj), nc, ni, nj});
      }
    }
  }
  return std::nullopt;
}
}  // namespace a3
