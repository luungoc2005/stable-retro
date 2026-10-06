"""
RAM-based "teacher" driver for NFS Carbon (GBA), used to bootstrap the vision policy.

The expert follows the racing line driven by the AI opponents (recorded once per race
state while the player idles) with a pure-pursuit controller plus handbrake-tap drifts.
It reads car position / heading from RAM, so it is only used to generate training labels
(DAgger); the trained agent itself drives from pixels.

RAM (offsets from each car's distance field D; player D=0x202c6b6, opponents
0x202d1fa / 0x202d6fa / 0x202dbfa):
  D+0x3e <i4 world x (16.16)   D+0x46 <i4 world z (16.16)
Player heading (rotation matrix row, 4096 = 1.0): fx 0x202c764, fz 0x202c76c

    python -m stable_retro.examples.nfs_expert --record         # record lines for all states
    python -m stable_retro.examples.nfs_expert --drive STATE     # let the expert race
"""
import argparse
import math
import multiprocessing as mp
import os

import numpy as np

import stable_retro as retro
from stable_retro.examples.nfs_env import (
    GAME, LINES_FILE, MIRROR, NFSRaceEnv, add_pose_vars, game_dir, load_lines, load_states_meta,
)


def record_line(state, max_frames=30000):
    """Path of the opponent that drives furthest while the player stays on the grid."""
    env = retro.make(GAME, state, render_mode=None, use_restricted_actions=retro.Actions.ALL)
    env.reset()
    data = env.unwrapped.data
    add_pose_vars(data)
    noop = np.zeros(len(env.unwrapped.buttons), np.int8)
    tracks = {i: [] for i in (1, 2, 3)}
    for t in range(max_frames):
        _, _, term, _, _ = env.step(noop)
        if t % 4 == 0:
            for i in (1, 2, 3):
                tracks[i].append((data.lookup_value(f"car{i}_x") / 65536, data.lookup_value(f"car{i}_z") / 65536))
        if term:
            break
    env.close()

    def length(i):
        return np.hypot(*np.diff(np.array(tracks[i]), axis=0).T).sum()

    line = np.array(tracks[max((1, 2, 3), key=length)])
    keep = [0]
    for k in range(1, len(line)):
        if np.hypot(*(line[k] - line[keep[-1]])) > 0.5:
            keep.append(k)
    return line[keep].astype(np.float32)


def _record(state):
    return state, record_line(state)


def record_all(states=None, workers=8):
    lines = load_lines()
    todo = [s for s in (states or sorted(load_states_meta())) if s not in lines]
    with mp.get_context("spawn").Pool(workers) as pool:
        for state, line in pool.imap_unordered(_record, todo):
            lines[state] = line
            print(f"{state}: {len(line)} points")
    np.savez_compressed(os.path.join(game_dir(), LINES_FILE), **lines)
    return lines


class PursuitExpert:
    """Pure pursuit on a recorded racing line. Returns RACE_COMBOS action indices."""

    def __init__(self, line, look=10, steer_th=0.12, drift_th=0.45, drift_cool=4,
                 corner_k=None, preview=40):
        """corner_k: if set, limit speed to corner_k / sqrt(curvature ahead) (mph), for fast cars."""
        self.corner_k = corner_k
        self.preview = preview
        self.line = line
        self.look = look
        self.steer_th = steer_th
        self.drift_th = drift_th
        self.drift_cool = drift_cool
        self.closed = np.hypot(*(line[0] - line[-1])) < 30
        self.k = None
        self.cool = 0

    def reset(self):
        self.k = None
        self.cool = 0

    def act(self, data, speed):
        p = np.array([data.lookup_value("car0_x"), data.lookup_value("car0_z")]) / 65536
        f = np.array([data.lookup_value("fwd_x"), data.lookup_value("fwd_z")]) / 4096
        n = len(self.line)
        if self.k is None:
            # global search, preferring line points whose direction matches the car heading
            d = np.hypot(*(self.line - p).T)
            seg = np.roll(self.line, -1, axis=0) - self.line
            align = (seg @ f) / (np.linalg.norm(seg, axis=1) + 1e-6)
            self.k = int(np.argmin(d - 5.0 * align))
        rng = self.k + np.arange(-10, 80)
        idx = rng % n if self.closed else np.clip(rng, 0, n - 1)
        dist = np.hypot(*(self.line[idx] - p).T)
        self.k = int(idx[np.argmin(dist)])
        ahead = self.k + self.look + int(speed / 12)
        target = self.line[ahead % n if self.closed else min(ahead, n - 1)]
        v = target - p
        ang = math.atan2(f[0] * v[1] - f[1] * v[0], f[0] * v[0] + f[1] * v[1])
        self.ang = ang
        self.cool -= 1
        if self.corner_k is not None and speed > 30:
            # heading change of the line over the next `preview` points ~ how sharp the next corner is
            pts = self.line[(self.k + np.arange(0, self.preview, 4)) % n] if self.closed else \
                self.line[np.clip(self.k + np.arange(0, self.preview, 4), 0, n - 1)]
            d = np.diff(pts, axis=0)
            hd = np.arctan2(d[:, 1], d[:, 0])
            turn = np.abs(np.angle(np.exp(1j * np.diff(hd)))).max() if len(hd) > 1 else 0.0
            v_max = self.corner_k / math.sqrt(turn + 1e-3)
            if speed > v_max + 10:
                return 7 if abs(ang) < self.steer_th else (8 if ang > 0 else 9)  # brake (+steer)
            if speed > v_max:
                return 10 if abs(ang) < self.steer_th else (5 if ang > 0 else 6)  # lift off
        if abs(ang) < self.steer_th:
            return 0  # A
        left = ang > 0
        if abs(ang) > self.drift_th and speed > 40 and self.cool <= 0:
            self.cool = self.drift_cool
            return 3 if left else 4  # handbrake tap to start a drift
        return 1 if left else 2


class RacingExpert:
    """Pure pursuit + speed planning (brake before corners) + opponent avoidance.

    The racing line's curvature gives a corner speed limit at every point,
    v_corner = grip / sqrt(curvature); the allowed speed now is the minimum over the
    upcoming line of sqrt(v_corner^2 + decel * distance) (i.e. brake early enough).
    Steering looks ahead a speed-dependent distance along the line, shifted sideways
    to go around opponents that are just ahead.
    """

    DEFAULTS = dict(look0=6.0, look_v=0.06, steer_th=0.12, grip=60.0, decel=40.0, v_cap=200.0,
                    horizon=60.0, brake_margin=12.0, drift_th=9.0, drift_cool=4,
                    avoid_dist=12.0, avoid_width=3.0, avoid_gain=0.0, steer_off=None)

    def __init__(self, line, **params):
        self.p = dict(self.DEFAULTS, **params)
        self.line = line
        self.closed = np.hypot(*(line[0] - line[-1])) < 30
        seg = np.diff(line, axis=0, append=line[:1] if self.closed else line[-1:] * 2 - line[-2:-1])
        ds = np.hypot(*seg.T) + 1e-6
        self.s = np.concatenate([[0.0], np.cumsum(ds)[:-1]])
        self.length = float(ds.sum())
        head = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
        # curvature: heading change over a ~6 unit window, per unit length
        w = 6
        dh = np.abs(np.roll(head, -w) - np.roll(head, w)) if self.closed else \
            np.abs(head[np.clip(np.arange(len(head)) + w, 0, len(head) - 1)] - head[np.clip(np.arange(len(head)) - w, 0, len(head) - 1)])
        dh = np.minimum(dh, 2 * np.pi - dh)
        span = np.maximum(2 * w * ds.mean(), 1e-3)
        self.curv = np.nan_to_num(dh / span)
        self.v_corner = np.minimum(self.p["v_cap"], self.p["grip"] / np.sqrt(self.curv + 1e-4))
        self.k = None
        self.cool = 0
        self.ang = 0.0
        self.steering = 0  # -1 right, 0 none, 1 left (held steering keeps its ramped-up angle)

    def reset(self):
        self.k = None
        self.cool = 0

    def _idx(self, k):
        n = len(self.line)
        return k % n if self.closed else np.clip(k, 0, n - 1)

    def _ahead(self, dist):
        """indices of line points within `dist` units ahead of k."""
        n = len(self.line)
        j = np.arange(self.k, self.k + n if self.closed else n)
        j = self._idx(j)
        rel = (self.s[j] - self.s[self.k]) % self.length if self.closed else self.s[j] - self.s[self.k]
        m = rel <= dist
        return j[m], rel[m]

    def act(self, data, speed):
        P = self.p
        p = np.array([data.lookup_value("car0_x"), data.lookup_value("car0_z")]) / 65536
        f = np.array([data.lookup_value("fwd_x"), data.lookup_value("fwd_z")]) / 4096
        if self.k is None:
            d = np.hypot(*(self.line - p).T)
            seg = np.roll(self.line, -1, axis=0) - self.line
            align = (seg @ f) / (np.linalg.norm(seg, axis=1) + 1e-6)
            self.k = int(np.argmin(d - 5.0 * align))
        idx = self._idx(self.k + np.arange(-10, 80))
        self.k = int(idx[np.argmin(np.hypot(*(self.line[idx] - p).T))])

        # steering target: speed-dependent distance ahead along the line
        j, rel = self._ahead(P["look0"] + P["look_v"] * speed)
        target = self.line[j[-1]].copy()
        if P["avoid_gain"] > 0:
            side = np.array([-f[1], f[0]])
            for i in (1, 2, 3):
                q = np.array([data.lookup_value(f"car{i}_x"), data.lookup_value(f"car{i}_z")]) / 65536
                r = q - p
                fwd, lat = r @ f, r @ side
                if 0 < fwd < P["avoid_dist"] and abs(lat) < P["avoid_width"]:
                    target = target - side * np.sign(lat or 1) * P["avoid_gain"] * (P["avoid_width"] - abs(lat))
        v = target - p
        ang = math.atan2(f[0] * v[1] - f[1] * v[0], f[0] * v[0] + f[1] * v[1])
        self.ang = ang
        left = ang > 0
        if P["steer_off"] is None:
            steer = 0 if abs(ang) < P["steer_th"] else (1 if left else 2)
        else:
            # hysteresis: the game ramps the steering angle while a direction is held and resets it
            # to zero on release, so keep holding through the corner instead of toggling
            want = 1 if ang > 0 else -1
            if self.steering == 0 and abs(ang) > P["steer_th"]:
                self.steering = want
            elif self.steering != 0 and (self.steering != want or abs(ang) < P["steer_off"]):
                self.steering = want if abs(ang) > P["steer_th"] else 0
            steer = {0: 0, 1: 1, -1: 2}[self.steering]
            left = self.steering > 0

        # speed planning: brake early enough for the corners ahead
        j, rel = self._ahead(P["horizon"])
        v_allow = float(np.min(np.sqrt(self.v_corner[j] ** 2 + P["decel"] * rel)))
        self.v_allow = v_allow
        self.cool -= 1
        if speed > v_allow + P["brake_margin"]:
            return {0: 7, 1: 8, 2: 9}[steer]  # brake
        if speed > v_allow:
            return {0: 10, 1: 5, 2: 6}[steer]  # lift off
        if steer and abs(ang) > P["drift_th"] and speed > 40 and self.cool <= 0:
            self.cool = P["drift_cool"]
            return 3 if left else 4
        return steer


# RacingExpert parameters tuned for pace vs. the AI on 12 training races (validated on 12 others)
TUNED = dict(look0=7.454, look_v=0.07, steer_th=0.171, grip=32.062, decel=49.358, horizon=108.23,
             brake_margin=18.155, avoid_gain=0.733, avoid_width=2.934)


def expert_for(env, lines):
    """Expert for the env's current state; labels are mirrored if the episode is."""
    return RacingExpert(lines[env.state_name], **TUNED)


def expert_label(expert, env):
    a = expert.act(env.data, env._last_speed)
    return MIRROR[a] if env.mirror else a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record", action="store_true")
    p.add_argument("--drive", default=None)
    args = p.parse_args()
    if args.record:
        record_all()
    if args.drive:
        lines = load_lines()
        env = NFSRaceEnv([args.drive], sticky_prob=0.0)
        add_pose_vars(env.data)
        env.reset()
        ex = expert_for(env, lines)
        while True:
            _, _, term, trunc, info = env.step(expert_label(ex, env))
            if term or trunc:
                break
        print(info, env._steps)


if __name__ == "__main__":
    main()
