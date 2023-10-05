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
)
from retro.examples.wrappers import StreetFighterFlipEnvWrapper, StochasticFrameSkip

import retro


def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs)
    if game == "StreetFighterIISpecialChampionEdition-Genesis":
        env = StreetFighterFlipEnvWrapper(env)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def wrap_deepmind_retro(env):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
    """
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Airstriker-Genesis")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    def make_env(render_mode="human"):
        env = make_retro(game=args.game, state=args.state, scenario=args.scenario, render_mode=render_mode)
        env = wrap_deepmind_retro(env)
        return env

    subproc_env = SubprocVecEnv([make_env] * 8, start_method="spawn")
    venv = VecTransposeImage(VecFrameStack(subproc_env, n_stack=4))
    tb_log_name = f"ppo-{args.game}"

    def on_finish(model):
        model.save(tb_log_name)

        # save images
        import imageio
        images = []

        new_venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([lambda: make_env(render_mode="rgb_array")] * 9, start_method="spawn"), n_stack=4))

        obs = new_venv.reset()
        img = new_venv.render(mode="rgb_array")
        total_frames = 1000
        for _ in range(total_frames):
            images.append(img)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _ ,_ = new_venv.step(action)
            img = new_venv.render(mode="rgb_array")

        imageio.mimsave(f"gifs/{tb_log_name}.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], duration=total_frames // 29)

    model = PPO(
        policy="CnnPolicy",
        env=venv,
        learning_rate=lambda f: f * 2.5e-4,
        n_steps=256,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="tb_logs",
        device="mps"
    )
    try:
        model.learn(
            total_timesteps=10_000_000,
            log_interval=1,
            progress_bar=True,
            tb_log_name=tb_log_name
        )
    except KeyboardInterrupt:
        on_finish(model)
        raise

    on_finish(model)


if __name__ == "__main__":
    main()
