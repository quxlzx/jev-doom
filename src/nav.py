"""Map-based navigation for deadly_corridor.

Builds a walkability grid from the sector line geometry ViZDoom exposes,
flood-fills from the player to find the farthest reachable cell (the corridor
runs to the exit), then A* routes the agent there. The brain gets a real
compass: "next waypoint at bearing +30, exit 2400 units away".
"""
from __future__ import annotations
import heapq
import math

CELL = 24  # grid resolution in map units


class NavMap:
    def __init__(self, sectors):
        xs, ys = [], []
        segs = []
        for s in sectors:
            for ln in s.lines:
                if not ln.is_blocking:
                    continue
                x1, y1 = ln.x1, ln.y1
                x2, y2 = ln.x2, ln.y2
                segs.append((x1, y1, x2, y2))
                xs += [x1, x2]
                ys += [y1, y2]
        self.x0, self.y0 = min(xs), min(ys)
        self.w = int((max(xs) - self.x0) / CELL) + 2
        self.h = int((max(ys) - self.y0) / CELL) + 2
        self.wall = [[False] * self.w for _ in range(self.h)]
        for x1, y1, x2, y2 in segs:
            self._raster(x1, y1, x2, y2)

    def _raster(self, x1, y1, x2, y2):
        d = math.hypot(x2 - x1, y2 - y1)
        n = max(1, int(d / (CELL / 3)))
        for i in range(n + 1):
            t = i / n
            cx = int(((x1 + (x2 - x1) * t) - self.x0) / CELL)
            cy = int(((y1 + (y2 - y1) * t) - self.y0) / CELL)
            if 0 <= cx < self.w and 0 <= cy < self.h:
                self.wall[cy][cx] = True

    def cell(self, x, y):
        return (int((x - self.x0) / CELL), int((y - self.y0) / CELL))

    def world(self, cx, cy):
        return (self.x0 + (cx + 0.5) * CELL, self.y0 + (cy + 0.5) * CELL)

    def free(self, c):
        cx, cy = c
        return 0 <= cx < self.w and 0 <= cy < self.h and not self.wall[cy][cx]

    def _clear_disk(self, c, r=1):
        # nudge out of a wall cell: open up a small disk around it
        cx, cy = c
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < self.w and 0 <= y < self.h:
                    self.wall[y][x] = False

    def reachable_farthest(self, start):
        """BFS from start; return the reachable free cell farthest by path distance."""
        if not self.free(start):
            self._clear_disk(start)
        dist = {start: 0}
        q = [start]
        best, bestd = start, -1
        while q:
            nq = []
            for (cx, cy) in q:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (cx + dx, cy + dy)
                    if self.free(n) and n not in dist:
                        dist[n] = dist[(cx, cy)] + 1
                        nq.append(n)
            q = nq
        for c, d in dist.items():
            if d > bestd:
                best, bestd = c, d
        return best, bestd, dist

    def astar(self, a, b):
        if not (self.free(a) and self.free(b)):
            return None
        openh = [(0, a)]
        came, gs = {a: None}, {a: 0}
        while openh:
            _, cur = heapq.heappop(openh)
            if cur == b:
                path = []
                while cur:
                    path.append(cur)
                    cur = came[cur]
                return path[::-1]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                n = (cur[0] + dx, cur[1] + dy)
                if not self.free(n):
                    continue
                if dx and dy and (not self.free((cur[0] + dx, cur[1])) or not self.free((cur[0], cur[1] + dy))):
                    continue
                ng = gs[cur] + (1.41 if dx and dy else 1)
                if n in gs and ng >= gs[n]:
                    continue
                came[n], gs[n] = cur, ng
                heapq.heappush(openh, (ng + abs(n[0] - b[0]) + abs(n[1] - b[1]), n))
        return None

    def los(self, a, b):
        """Bresenham grid line-of-sight between two cells."""
        (x0, y0), (x1, y1) = a, b
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        err = dx - dy
        x, y = x0, y0
        while (x, y) != (x1, y1):
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
            if not self.free((x, y)):
                return False
        return True

    def next_waypoint(self, pos, angle_deg, path):
        """Farthest path node with grid line-of-sight; walk corner-to-corner."""
        cur = self.cell(*pos)
        best = None
        for c in path[1:]:
            if math.hypot(*[self.world(*c)[i] - pos[i] for i in (0, 1)]) < 40:
                continue
            if self.los(cur, c):
                best = c
            else:
                break
        c = best if best is not None else path[min(1, len(path) - 1)]
        wx, wy = self.world(*c)
        b = math.degrees(math.atan2(wy - pos[1], wx - pos[0])) - angle_deg
        while b > 180:
            b -= 360
        while b < -180:
            b += 360
        return (wx, wy), b
