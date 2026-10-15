"""Day 17 navigation: a 2D map of the floor (occupancy grid) + A* path planning.

The map starts EMPTY. While walking, obstacle points seen by the camera are written
into it. Each obstacle is 'inflated' by the robot's radius, so a path that avoids the
inflated cells keeps the robot's body clear. A* finds the shortest free path; the robot
replans whenever a new obstacle blocks the current path.
"""
import heapq
import math

import numpy as np


class OccupancyGrid:
    def __init__(self, x_range, y_range, res=0.1, inflate=0.45):
        self.x0, self.y0, self.res = x_range[0], y_range[0], res
        self.nx = int(math.ceil((x_range[1] - x_range[0]) / res))
        self.ny = int(math.ceil((y_range[1] - y_range[0]) / res))
        self.occ = np.zeros((self.nx, self.ny), bool)       # seen obstacle
        self.blocked = np.zeros((self.nx, self.ny), bool)   # obstacle + robot radius (for planning)
        self.tight = np.zeros((self.nx, self.ny), bool)     # obstacle + smaller margin (for "is my
                                                            # path still OK?" - avoids endless replanning)
        r = int(math.ceil(inflate / res))
        self.disk = [(i, j) for i in range(-r, r + 1) for j in range(-r, r + 1) if i * i + j * j <= r * r]
        rt = int(math.ceil(0.3 / res))
        self.disk_tight = [(i, j) for i in range(-rt, rt + 1) for j in range(-rt, rt + 1) if i * i + j * j <= rt * rt]

    def cell(self, x, y):
        return int((x - self.x0) / self.res), int((y - self.y0) / self.res)

    def centre(self, i, j):
        return self.x0 + (i + 0.5) * self.res, self.y0 + (j + 0.5) * self.res

    def inside(self, i, j):
        return 0 <= i < self.nx and 0 <= j < self.ny

    def add_points(self, pts):
        """Mark obstacle points. Returns the list of NEW obstacle cells."""
        if len(pts) == 0:
            return []
        ij = ((pts[:, :2] - [self.x0, self.y0]) / self.res).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.nx) & (ij[:, 1] >= 0) & (ij[:, 1] < self.ny)
        new = []
        for i, j in {tuple(c) for c in ij[ok]}:
            if not self.occ[i, j]:
                self.occ[i, j] = True
                new.append((i, j))
                for di, dj in self.disk:
                    if self.inside(i + di, j + dj):
                        self.blocked[i + di, j + dj] = True
                for di, dj in self.disk_tight:
                    if self.inside(i + di, j + dj):
                        self.tight[i + di, j + dj] = True
        return new

    def path_blocked(self, path, start=None):
        """True if any straight piece of the path (from `start`, through every point) comes
        within 0.3 m of a real obstacle. (Planning keeps 0.45 m; this smaller check means a path
        is only replaced when a NEW obstacle really gets in the way.)"""
        pts = ([tuple(start)] if start is not None else []) + [tuple(p) for p in path]
        tight = self.tight.copy()
        if start is not None:
            tight[self.cell(*start)] = False
        return any(not self._free_line(a, b, tight) for a, b in zip(pts, pts[1:])) or \
            (len(pts) == 1 and tight[self.cell(*pts[0])])

    def _with_escape(self, start, radius=0.8):
        """Copy of the blocked map where safety-margin cells (not real obstacles) near the
        start are free, so the robot can walk away from a table it is standing next to."""
        blocked = self.blocked.copy()
        si, sj = self.cell(*start)
        r = int(radius / self.res)
        i0, i1 = max(si - r, 0), min(si + r + 1, self.nx)
        j0, j1 = max(sj - r, 0), min(sj + r + 1, self.ny)
        ii, jj = np.mgrid[i0:i1, j0:j1]
        near = (ii - si) ** 2 + (jj - sj) ** 2 <= r * r
        sub = blocked[i0:i1, j0:j1]
        sub[near & ~self.occ[i0:i1, j0:j1]] = False
        return blocked

    def astar(self, start, goal):
        """Shortest free path from start to goal (room coordinates). None if impossible."""
        s, g = self.cell(*start), self.cell(*goal)
        if not (self.inside(*s) and self.inside(*g)):
            return None
        blocked = self._with_escape(start)
        blocked[s] = False                                   # the robot's own cell is free
        if blocked[g]:
            return None
        moves = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        h = lambda c: math.hypot(c[0] - g[0], c[1] - g[1])
        open_set = [(h(s), 0.0, s)]
        came, cost = {}, {s: 0.0}
        while open_set:
            _, c, cur = heapq.heappop(open_set)
            if cur == g:
                cells = [cur]
                while cur in came:
                    cur = came[cur]
                    cells.append(cur)
                return self._smooth([self.centre(*c) for c in reversed(cells)], blocked)
            if c > cost.get(cur, 1e18):
                continue
            for di, dj, w in moves:
                nb = (cur[0] + di, cur[1] + dj)
                if not self.inside(*nb) or blocked[nb]:
                    continue
                nc = c + w
                if nc < cost.get(nb, 1e18):
                    cost[nb], came[nb] = nc, cur
                    heapq.heappush(open_set, (nc + h(nb), nc, nb))
        return None

    def _free_line(self, a, b, blocked):
        n = int(math.hypot(b[0] - a[0], b[1] - a[1]) / (self.res * 0.5)) + 1
        for t in np.linspace(0, 1, n):
            if blocked[self.cell(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))]:
                return False
        return True

    def _smooth(self, pts, blocked):
        """Remove unnecessary corners: keep only points needed for straight free lines."""
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self._free_line(pts[i], pts[j], blocked):
                j -= 1
            out.append(pts[j])
            i = j
        return out
