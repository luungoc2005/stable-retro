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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from retro.examples.wrappers import StreetFighterFlipEnvWrapper, StochasticFrameSkip
from typing import Dict, Any
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import optuna
import retro
import torch.nn as nn

N_TRIALS = 100
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(5e5)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3


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
    env = Monitor(env) # before everything else
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env


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
    "verbose": 0,
    "device": "mps"
}

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

    def make_subproc_env():
        subproc_env = SubprocVecEnv([make_env] * 8, start_method="spawn")
        venv = VecTransposeImage(VecFrameStack(subproc_env, n_stack=4))
        return venv
    tb_log_name = f"ppo-{args.game}"


    def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
        n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
        # lr_schedule = "constant"
        # Uncomment to enable learning rate schedule
        # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
        ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
        clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
        n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
        gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
        max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
        vf_coef = trial.suggest_float("vf_coef", 0, 1)
        activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
        ortho_init = False
        # ortho_init = trial.suggest_categorical('ortho_init', [False, True])

        # Display true values.
        trial.set_user_attr("gamma_", gamma)
        trial.set_user_attr("gae_lambda_", gae_lambda)
        trial.set_user_attr("n_steps", n_steps)
        activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

        return {
            "n_steps": n_steps,
            "batch_size": batch_size,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "clip_range": clip_range,
            "n_epochs": n_epochs,
            "gae_lambda": gae_lambda,
            "max_grad_norm": max_grad_norm,
            "vf_coef": vf_coef,
            # "sde_sample_freq": sde_sample_freq,
            "policy_kwargs": dict(
                # log_std_init=log_std_init,
                activation_fn=activation_fn,
                ortho_init=ortho_init,
            ),
        }

    venv = make_subproc_env()

    def objective(trial: optuna.Trial) -> float:
        kwargs = DEFAULT_HYPERPARAMS.copy()
        # Sample hyperparameters.
        kwargs.update(sample_ppo_params(trial))
        kwargs["env"] = venv
        # Create the RL model.
        model = PPO(**kwargs)
        # Create env used for evaluation.
        # eval_env = Monitor(make_subproc_env())

        nan_encountered = False
        try:
            model.learn(N_TIMESTEPS)
        except AssertionError as e:
            # Sometimes, random hyperparams can generate NaN.
            print(e)
            nan_encountered = True
        # finally:
        #     # Free memory.
        #     model.env.close()
        #     eval_env.close()

        # Tell the optimizer that the trial failed.
        if nan_encountered:
            return float("nan")

        mean_rw, std_rw = evaluate_policy(model, venv, N_EVAL_EPISODES)
        return mean_rw
    
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)
    
    journal_file = tb_log_name + ".log"
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(journal_file),
    )
    lock_obj = optuna.storages.JournalFileOpenLock(journal_file)

    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(journal_file, lock_obj=lock_obj),
    )

    study = optuna.create_study(
        storage=storage,
        sampler=sampler, 
        pruner=pruner, 
        direction="maximize",
        study_name=tb_log_name,
        load_if_exists=True
    )
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=60000)
    except KeyboardInterrupt:
        pass
    
    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
