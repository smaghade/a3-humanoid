"""Following objects from picture to picture to find out which ones MOVE.

Each object seen by the depth camera is matched to the closest object from the previous
pictures (within 1.5 m). Its speed is measured over the last 1.5 seconds (not just between two
pictures), because a still stone seen from a moving robot seems to 'jump' a little as its
visible side changes.
  - moving : travelled at least 0.4 m over at least 0.5 s, at 0.3 - 8 m/s
             -> person, animal or vehicle
  - static : seen at least 3 times, hardly moved, never moving -> goes on the map
The type of a moving object is guessed from its size (not from an AI model):
  tall and long -> vehicle, tall and medium -> rider, tall and slim -> person, low -> animal.
"""
from collections import deque

import numpy as np

WINDOW = 1.5          # seconds of history used to measure speed


class Track:
    def __init__(self, tid, obj, t):
        self.id = tid
        self.pos = np.array(obj["centre"], float)
        self.vel = np.zeros(2)
        self.height, self.size, self.age, self.last = obj["height"], obj["size"], 1, t
        self.history = deque([(t, self.pos.copy())])
        self.was_moving = False

    def add(self, obj, t):
        self.pos = np.array(obj["centre"], float)
        self.history.append((t, self.pos.copy()))
        while self.history and t - self.history[0][0] > WINDOW:
            self.history.popleft()
        t0, p0 = self.history[0]
        span = t - t0
        travel = self.pos - p0
        self.vel = travel / span if span > 0.2 else np.zeros(2)
        self.height = max(self.height * 0.8, obj["height"])
        self.size = max(self.size * 0.8, obj["size"])
        self.age += 1
        self.last = t
        if self.moving:
            self.was_moving = True

    @property
    def speed(self):
        return float(np.hypot(*self.vel))

    @property
    def travel(self):
        return float(np.hypot(*(self.pos - self.history[0][1])))

    @property
    def span(self):
        return self.last - self.history[0][0]

    @property
    def moving(self):
        return (self.age >= 4 and self.span >= 0.5 and self.travel >= 0.4
                and 0.3 <= self.speed <= 8.0)

    @property
    def static(self):
        """Seen several times, hardly moved, and never seen moving (a person who stops is still
        a person, not a stone)."""
        return self.age >= 3 and self.travel < 0.25 and not self.was_moving

    @property
    def label(self):
        if not self.moving and not self.was_moving:
            return "stone" if self.height < 0.6 else "obstacle"
        if self.height > 1.2 and self.size > 2.5:
            return "vehicle (car?)"
        if self.height > 1.2 and self.size > 1.1:
            return "rider (bike?)"
        if self.height > 1.2:
            return "person?"
        return "animal?"


class Tracker:
    def __init__(self, gate=1.5, forget=1.0):
        self.gate, self.forget = gate, forget
        self.tracks, self.next_id = [], 1

    def update(self, objects, t):
        """Match objects to tracks; sets obj['track'] for every object."""
        free = list(self.tracks)
        for obj in sorted(objects, key=lambda o: o["distance"]):
            best, best_d = None, self.gate
            for tr in free:
                guess = tr.pos + tr.vel * (t - tr.last)
                d = float(np.hypot(*(guess - obj["centre"])))
                if d < best_d:
                    best, best_d = tr, d
            if best is None:
                best = Track(self.next_id, obj, t)
                self.next_id += 1
                self.tracks.append(best)
            else:
                free.remove(best)
                best.add(obj, t)
            obj["track"] = best
        self.tracks = [tr for tr in self.tracks if t - tr.last <= self.forget]

    def moving(self):
        return [tr for tr in self.tracks if tr.moving]
