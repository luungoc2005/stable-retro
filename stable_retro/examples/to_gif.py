"""
Train an agent using Proximal Policy Optimization from Stable Baselines 3
"""

import argparse

import gymnasium as gym
import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
    VecVideoRecorder,
)
from retro.examples.wrappers import StreetFighterFlipEnvWrapper
import retro


class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=False,
                )
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info


def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, render_mode="rgb_array", **kwargs)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def wrap_deepmind_retro(env):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
    """
    env = WarpFrame(env)
    # env = ClipRewardEnv(env)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Airstriker-Genesis")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--video", action='store_true')
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    def make_env():
        env = make_retro(game=args.game, state=args.state, scenario=args.scenario)
        if args.game == "StreetFighterIISpecialChampionEdition-Genesis":
            env = StreetFighterFlipEnvWrapper(env)
        env = wrap_deepmind_retro(env)
        return env

    venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env] * 9, start_method="spawn"), n_stack=4))
    tb_log_name = f"ppo-{args.game}"

    print(f"Loading from {tb_log_name}.zip")
    model = PPO.load(tb_log_name + ".zip", device="mps", env=venv, print_system_info=True)

    total_frames = 0
    max_frames = 10000
    if not args.video:
        # save images
        import imageio
        images = []
        obs = venv.reset()
        img = venv.render(mode="rgb_array")
        for _ in range(max_frames):
            images.append(img)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _ ,_ = venv.step(action)
            img = venv.render(mode="rgb_array")
            total_frames += 1
            if np.all(done):
                break

        imageio.mimsave(f"gifs/{tb_log_name}.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], duration=total_frames // 29)
    else:
        venv = VecVideoRecorder(venv, "videos/", record_video_trigger=lambda x: x == 0, video_length=total_frames, name_prefix=tb_log_name)
        venv.reset()
        for _ in range(max_frames + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done ,_ = venv.step(action)
            if np.all(done):
                break
        venv.close()

    print(f'Total frames rendered: {total_frames}')


if __name__ == "__main__":
    main()
