"""
Train an agent using Proximal Policy Optimization from Stable Baselines 3
"""

import argparse

import gymnasium as gym
from gymnasium.core import Env
import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from stable_baselines3.common.monitor import Monitor
from retro.examples.wrappers import (
    StreetFighterFlipEnvWrapper, 
    StochasticFrameSkip, 
    ActionBias,
    GAME_WRAPPERS,
)
import torch.nn as nn
from retro.examples.impala_cnn import ImpalaCNN
import retro


DEFAULT_HYPERPARAMS = {
    "policy": "CnnPolicy",
    "learning_rate": 2.5e-4,
    "n_steps": 256,
    "batch_size": 32,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.1,
    "ent_coef": 0.01,
    "verbose": 1,
    "device": "mps",
    "tensorboard_log": "tb_logs",
}
CUSTOM_HYPERPARAMS = {
    "StreetFighterIISpecialChampionEdition-Genesis": {
        "learning_rate": 1e-4,
        "n_steps": 64,
        "batch_size": 512,
        "n_epochs": 5,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.005,
        "max_grad_norm": 0.9,
        "vf_coef": 0.85,
    }
}

def make_retro(*, game, state=None, max_episode_steps=0, action_bias='', frame_skip=True, discrete=False, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs)

    if game in GAME_WRAPPERS:
        for _wrapper in GAME_WRAPPERS[game]:
            env = _wrapper(env)

    if action_bias != '':
        action_bias_list = []
        if ',' in action_bias:
            action_bias_list = action_bias.split(',')
        else:
            action_bias_list = action_bias.split(' ')
        action_bias_list = [float(item.strip()) for item in action_bias_list]
        action_meaning = env.get_action_meaning([1 if item > 0 else 0 for item in action_bias_list])
        if len(action_meaning) > 0:
            import warnings
            warnings.warn(f"Action bias on: {action_meaning}")
        env = ActionBias(env, action_bias_list)
    if frame_skip:
        env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps > 0:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def wrap_deepmind_retro(env):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
    """
    env = Monitor(env)
    env = WarpFrame(env)
    # env = ClipRewardEnv(env)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Airstriker-Genesis")
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--discrete", action='store_true')
    parser.add_argument("--cnn", default='nature', choices=['nature', 'impala'])
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--action-bias", default='0 0 0 0 0 0 0 0 0 0 0 0')
    parser.add_argument("--no-frame-skip", action='store_true')
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()
    print(args)

    def make_env(render_mode="human"):
        env = make_retro(
            game=args.game, 
            state=args.state, 
            scenario=args.scenario, 
            action_bias=args.action_bias, 
            frame_skip=not args.no_frame_skip, 
            discrete=args.discrete,
            render_mode=render_mode
        )
        env = wrap_deepmind_retro(env)
        return env

    subproc_env = SubprocVecEnv([make_env] * 8, start_method="spawn")
    venv = VecTransposeImage(VecFrameStack(subproc_env, n_stack=4))
    tb_log_name = f"ppo-{args.game}"
    if args.discrete:
        tb_log_name += "-discrete"

    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    if args.game in CUSTOM_HYPERPARAMS:
        print("Using custom hparams")
        kwargs.update(CUSTOM_HYPERPARAMS[args.game])
    if args.cnn == 'impala':
        tb_log_name += '-impala'
        kwargs['policy_kwargs'] = {
            'features_extractor_class': ImpalaCNN
        }

    kwargs["env"] = venv
    # Create the RL model.
    model = PPO(**kwargs)
    if args.resume:
        import os
        model_path = tb_log_name + ".zip"
        if os.path.exists(model_path):
            model = PPO.load(model_path, venv, print_system_info=True)

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


    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    if args.game in CUSTOM_HYPERPARAMS:
        print("Using custom hparams")
        hparams = CUSTOM_HYPERPARAMS[args.game]
        activation_fn = hparams["activation_fn"]
        del hparams["activation_fn"]
        kwargs.update(hparams)
        activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]
        kwargs["policy_kwargs"] = dict(
            activation_fn=activation_fn,
        )

    kwargs["env"] = venv
    # Create the RL model.
    model = PPO(**kwargs)
    if args.resume:
        import os
        model_path = tb_log_name + ".zip"
        if os.path.exists(model_path):
            model = PPO.load(model_path, venv, print_system_info=True)

    try:
        model.learn(
            total_timesteps=25_000_000,
            log_interval=1,
            progress_bar=True,
            tb_log_name=tb_log_name,
        )
    except KeyboardInterrupt:
        on_finish(model)
        raise

    on_finish(model)


if __name__ == "__main__":
    main()
