"""
Need for Speed Carbon: Own the City (GBA) racing environment.

Builds a gymnasium env around the retro integration that rewards actual race
progress (read from RAM) instead of raw speed, so the agent learns to follow the
track, overtake and finish races on any track/difficulty.

RAM map (EWRAM offsets as used by retro `|u1`/`<u2` variables):
  0x202c6b6  <u2  player distance along the track (starts at 0 or 65535, +~50/sec at race pace)
  0x202c6cc  |u1  player race position (1..4)
  0x202c6e8  <u2  player laps remaining * 256
  0x3005a84  |d1  wrong-way flag (the red X icon), `reverse` in data.json
Note the distance counter is a signed odometer: it also increases when driving the
wrong way round the track, so it only counts as progress while the wrong-way flag is off.
  0x30056e0  |u1  event type (0 circuit, 1 elimination, 2 hunter, 3 sprint)
Race settings (written by the menu, see states_meta.json):
  0x2000bc8 direction, 0x2000bcc laps, 0x2000bd0 difficulty (0 easy..2 hard),
  0x2000bd4 opponents, 0x2000bd8 traffic, 0x2000bfc catch-up

States are named `<Event>.<Track>.<Fwd|Rev>.<Easy|Normal|Hard>` and generated
from Quick Play with catch-up off. Tracks in HELDOUT_TRACKS are never used for
training so generalization can be measured on unseen tracks.
"""
import json
import os
import random

import cv2
import gymnasium as gym
import numpy as np

import stable_retro as retro
from stable_retro.examples.discretizer import Discretizer
from stable_retro.examples.wrappers import NeedForSpeedDiscretizer

GAME = "NeedForSpeedCarbon-GBA-v0"
HELDOUT_TRACKS = {
    # circuit / elimination tracks
    "Parkside", "MountainSpeedzone", "Crossover",
    # sprint tracks
    "ParkZone", "DoubleSwitch", "CrossHighway", "MountainSide",
}
DIFFICULTIES = ["Easy", "Normal", "Hard"]
LINES_FILE = "racing_lines.npz"
# odometer units per world unit, so line progress is on the same scale as the "dist" counter
LINE_SCALE = 19.0
CARS = [0x202C6B6, 0x202D1FA, 0x202D6FA, 0x202DBFA]  # distance fields of player, opponents


def add_pose_vars(data):
    """World position of every car (D+0x3e / D+0x46, 16.16) and the player's heading."""
    for i, D in enumerate(CARS):
        data.set_variable(f"car{i}_x", {"address": D + 0x3E, "type": "<i4"})
        data.set_variable(f"car{i}_z", {"address": D + 0x46, "type": "<i4"})
    data.set_variable("fwd_x", {"address": 0x202C764, "type": "<i4"})
    data.set_variable("fwd_z", {"address": 0x202C76C, "type": "<i4"})


def load_lines():
    path = os.path.join(game_dir(), LINES_FILE)
    if not os.path.exists(path):
        return {}
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


class LineTracker:
    """Measures progress as arc length along a recorded racing line (AI path)."""

    def __init__(self, line):
        self.line = line
        seg = np.hypot(*np.diff(line, axis=0).T)
        self.closed = np.hypot(*(line[0] - line[-1])) < 30
        if self.closed:
            seg = np.append(seg, np.hypot(*(line[0] - line[-1])))
        self.s = np.concatenate([[0.0], np.cumsum(seg)])[: len(line)]
        self.length = float(seg.sum())
        self.k = None
        self.off = 0.0

    def locate(self, p, f=None):
        d = np.hypot(*(self.line - p).T)
        if f is not None:
            seg = np.roll(self.line, -1, axis=0) - self.line
            d = d - 5.0 * (seg @ f) / (np.linalg.norm(seg, axis=1) + 1e-6)
        self.k = int(np.argmin(d))

    def update(self, p):
        """Advance to the nearest line point; returns arc-length gained (negative if backwards)."""
        n = len(self.line)
        rng = self.k + np.arange(-30, 80)
        idx = rng % n if self.closed else np.clip(rng, 0, n - 1)
        d = np.hypot(*(self.line[idx] - p).T)
        j = int(np.argmin(d))
        self.off = float(d[j])
        if self.off > 40:  # lost (e.g. respawned elsewhere): re-locate, no progress
            self.locate(p)
            return 0.0
        k_new = int(idx[j])
        ds = self.s[k_new] - self.s[self.k]
        if self.closed and abs(ds) > self.length / 2:
            ds -= np.sign(ds) * self.length
        self.k = k_new
        return float(ds)

# Left/right symmetric driving actions (R = handbrake). MIRROR maps each action to its
# mirror image so an episode can be played horizontally flipped (data augmentation).
RACE_COMBOS = [
    ["A"],                    # 0 accelerate
    ["A", "LEFT"],            # 1
    ["A", "RIGHT"],           # 2
    ["A", "R", "LEFT"],       # 3 handbrake turn
    ["A", "R", "RIGHT"],      # 4
    ["LEFT"],                 # 5 coast + steer
    ["RIGHT"],                # 6
    ["B"],                    # 7 brake / reverse
    ["B", "LEFT"],            # 8
    ["B", "RIGHT"],           # 9
    [],                       # 10 coast
]
MIRROR = [0, 2, 1, 4, 3, 6, 5, 7, 9, 8, 10]


def game_dir():
    return os.path.dirname(retro.data.get_file_path(GAME, "rom.sha"))


def load_states_meta():
    with open(os.path.join(game_dir(), "states_meta.json")) as f:
        return json.load(f)


def list_states(split="train", difficulties=None, events=None):
    """Return state names for a split: 'train', 'test' (held-out tracks) or 'all'."""
    meta = load_states_meta()
    out = []
    for name, m in meta.items():
        heldout = m["track"] in HELDOUT_TRACKS
        if split == "train" and heldout:
            continue
        if split == "test" and not heldout:
            continue
        if difficulties is not None and DIFFICULTIES[m["difficulty"]] not in difficulties:
            continue
        if events is not None and m["event"] not in events:
            continue
        out.append(name)
    return sorted(out)


class NFSRaceEnv(gym.Env):
    """Frame-skipped, grayscale, frame-stacked NFS race env with a progress reward.

    reward per agent step:
      + progress_scale * (distance gained)           dense "drive forward along the track" signal
      + overtake_bonus * (positions gained)          encourages passing opponents
      - collision_penalty if speed collapses          discourages hitting walls/traffic
      terminal: finish_bonus[final position]         win the race
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        states,
        frame_skip=4,
        sticky_prob=0.25,
        frame_stack=4,
        obs_size=(80, 120),
        max_steps=4500,
        stuck_steps=250,
        progress_scale=0.02,
        overtake_bonus=1.0,
        collision_penalty=0.5,
        finish_bonus=(10.0, 4.0, 0.0, -4.0),
        lap_bonus=5.0,
        progress="line",
        augment=False,
        mirror_prob=0.0,
        actions="race",
        restart_prob=0.0,
        snapshot_every=100,
        pool_size=150,
        render_mode=None,
        seed=None,
    ):
        self.states = list(states)
        assert self.states, "no states given"
        self.frame_skip = frame_skip
        self.sticky_prob = sticky_prob
        self.frame_stack = frame_stack
        self.obs_size = obs_size
        self.max_steps = max_steps
        self.stuck_steps = stuck_steps
        self.progress_scale = progress_scale
        self.overtake_bonus = overtake_bonus
        self.collision_penalty = collision_penalty
        self.finish_bonus = finish_bonus
        self.lap_bonus = lap_bonus
        self.augment = augment
        self.mirror_prob = mirror_prob
        self.mirror = False
        self.restart_prob = restart_prob
        self.snapshot_every = snapshot_every
        self.pool_size = pool_size
        self.pool = []  # mid-race snapshots: (state_name, emulator state, dist, pos)
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        env = retro.make(GAME, self.states[0], render_mode=render_mode)
        self.retro_env = env
        # actions="legacy" keeps the 13-action NeedForSpeedDiscretizer used by older runs
        self.disc = NeedForSpeedDiscretizer(env) if actions == "legacy" else Discretizer(env, RACE_COMBOS)
        self.mirror_map = MIRROR if actions != "legacy" else list(range(self.disc.action_space.n))
        self.action_space = self.disc.action_space
        self.observation_space = gym.spaces.Box(0, 255, (frame_stack, *obs_size), np.uint8)
        data = env.unwrapped.data
        data.set_variable("dist", {"address": 0x202C6B6, "type": "<u2"})
        data.set_variable("pos", {"address": 0x202C6CC, "type": "|u1"})
        data.set_variable("event", {"address": 0x30056E0, "type": "|u1"})
        data.set_variable("laps_left", {"address": 0x202C6E8, "type": "<u2"})
        self.data = data
        self.meta = load_states_meta()
        add_pose_vars(data)
        self.lines = load_lines() if progress == "line" else {}
        self.tracker = None

    # --- helpers -----------------------------------------------------------
    def _proc(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (self.obs_size[1], self.obs_size[0]), interpolation=cv2.INTER_AREA)
        if self.mirror:
            g = g[:, ::-1]
        if self.augment:
            g = np.clip(g.astype(np.float32) * self._gain + self._bias, 0, 255).astype(np.uint8)
        return g

    def _pose(self):
        return np.array([self.data.lookup_value("car0_x"), self.data.lookup_value("car0_z")]) / 65536

    def _heading(self):
        return np.array([self.data.lookup_value("fwd_x"), self.data.lookup_value("fwd_z")]) / 4096

    def _speed(self, info):
        return info.get("speed1", 0) * 100 + info.get("speed2", 0) * 10 + info.get("speed3", 0)

    def _obs(self):
        return np.stack(self._frames, 0)

    # --- gym API -----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        state = (options or {}).get("state")
        snap = None
        if state is None:
            if self.pool and self.rng.random() < self.restart_prob:
                snap = self.pool[self.rng.integers(len(self.pool))]
                state = snap[0]
            else:
                state = self.states[self.rng.integers(len(self.states))]
        self.state_name = state
        self.retro_env.unwrapped.load_state(state)
        frame, _ = self.retro_env.reset()
        if snap is not None:
            # continue a previous race from the middle (better coverage of whole tracks)
            em = self.retro_env.unwrapped.em
            em.set_state(snap[1])
            self.data.update_ram()
            frame = em.get_screen()
        self.mirror = self.rng.random() < self.mirror_prob
        if self.augment:
            self._gain = self.rng.uniform(0.75, 1.25)
            self._bias = self.rng.uniform(-20, 20)
        self._last_dist = self.data.lookup_value("dist")
        self.tracker = None
        if self.state_name in self.lines:
            self.tracker = LineTracker(self.lines[self.state_name])
            self.tracker.locate(self._pose(), self._heading())
        self._last_pos = self.data.lookup_value("pos")
        self._laps_left = self.data.lookup_value("laps_left") >> 8
        self._wrong_way_steps = 0
        self._pos_candidate, self._pos_count = self._last_pos, 0
        self._last_speed = 0
        self._last_action = 0
        self._steps = 0
        self._best_dist = 0
        self._progress = 0
        self._since_best = 0
        g = self._proc(frame)
        self._frames = [g] * self.frame_stack
        return self._obs(), self._info(race_over=False)

    def _info(self, race_over):
        m = self.meta.get(self.state_name, {})
        return {
            "state": self.state_name,
            "track": m.get("track", ""),
            "event": m.get("event", ""),
            "difficulty": m.get("difficulty", -1),
            "position": self._last_pos,
            "progress": self._progress,
            "race_over": race_over,
            "mirror": self.mirror,
            "wrong_way_steps": getattr(self, "_wrong_way_steps", 0),
        }

    def step(self, action):
        action = int(action)
        if self.mirror:
            action = self.mirror_map[action]
        reward = 0.0
        terminated = truncated = False
        race_over = False
        wrong_way = False
        frame = None
        for i in range(self.frame_skip):
            a = action
            if i == 0 and self.sticky_prob > 0 and self.rng.random() < self.sticky_prob:
                a = self._last_action
            frame, _, term, trunc, info = self.retro_env.step(self.disc.action(a))
            if term or info.get("screen", 1) != 1:
                race_over = True
                terminated = True
                break
            dist = self.data.lookup_value("dist")
            delta = ((dist - self._last_dist + 32768) % 65536) - 32768
            self._last_dist = dist
            if abs(delta) > 400:  # counter reset/glitch (e.g. respawn), ignore
                delta = 0
            if info.get("reverse", 0):
                wrong_way = True
            if self.tracker is not None:
                # progress along the racing line: weaving, wide lines and wrong turns don't pay
                delta = LINE_SCALE * self.tracker.update(self._pose())
            elif wrong_way:
                # odometer fallback: it still goes up when driving the wrong way, so penalize
                delta = -abs(delta)
            self._progress += delta
            reward += self.progress_scale * delta
            laps_left = self.data.lookup_value("laps_left") >> 8
            if laps_left < self._laps_left:
                reward += self.lap_bonus
            self._laps_left = laps_left
            # the position byte glitches for single frames, only count stable changes
            pos = self.data.lookup_value("pos")
            if pos == self._pos_candidate:
                self._pos_count += 1
            else:
                self._pos_candidate, self._pos_count = pos, 1
            if 1 <= pos <= 4 and pos != self._last_pos and self._pos_count >= 3:
                if self._progress > 50:  # grid order before the start is meaningless
                    reward += self.overtake_bonus * (self._last_pos - pos)
                self._last_pos = pos
        self._last_action = action

        if not race_over:
            speed = self._speed(info)
            if self._last_speed - speed > 25:  # sudden stop: crashed into something
                reward -= self.collision_penalty
            self._last_speed = speed
            self._frames.pop(0)
            self._frames.append(self._proc(frame))
        else:
            reward += self.finish_bonus[min(max(self._last_pos, 1), 4) - 1]

        self._steps += 1
        self._wrong_way_steps += int(wrong_way)
        if (self.restart_prob > 0 and not race_over and self._steps % self.snapshot_every == 0
                and self._last_speed > 20):
            item = (self.state_name, self.retro_env.unwrapped.em.get_state())
            if len(self.pool) < self.pool_size:
                self.pool.append(item)
            else:
                self.pool[self.rng.integers(self.pool_size)] = item
        if self._progress > self._best_dist + 20:
            self._best_dist = self._progress
            self._since_best = 0
        else:
            self._since_best += 1
        if not terminated:
            if self._since_best >= self.stuck_steps:
                truncated = True
                reward -= 5.0
            elif self._steps >= self.max_steps:
                truncated = True
        return self._obs(), reward, terminated, truncated, self._info(race_over)

    def render(self):
        return self.retro_env.render()

    def get_frame(self):
        return self.retro_env.unwrapped.em.get_screen()

    def close(self):
        self.retro_env.close()
