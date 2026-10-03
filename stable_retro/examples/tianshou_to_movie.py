import argparse
import json
import os
import stable_retro as retro
import torch
import numpy as np
from tianshou.env import ShmemVectorEnv
from tianshou.algorithm.modelfree.c51 import C51Policy
from stable_retro.examples.sf_rainbow_tianshou import make_env, RainbowNet
from stable_retro.examples.tianshou_patch import load_policy_weights
from stable_retro.examples.discretizer import Discretizer
from tianshou.data import Batch
from stable_baselines3.common.vec_env import (
    VecVideoRecorder,
    DummyVecEnv,
)

class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="StreetFighterIISpecialChampionEdition-Genesis")
    parser.add_argument("--path", type=str, default='')
    parser.add_argument("--state", type=str, default=retro.State.DEFAULT)

    args = parser.parse_args()

    with open(os.path.join(args.path, 'args.json'), 'r') as fp:
        saved_args = DotDict(json.load(fp))
        saved_args.update({"state": args.state})
        print(f'Loaded args: {saved_args}')

    def _make_env():
        return make_env(saved_args, render_mode="rgb_array")

    env = _make_env()
    observation_space, action_space = env.observation_space, env.action_space
    model = RainbowNet(observation_space, action_space, use_impala=saved_args.impala).to(saved_args.device)
    policy = C51Policy(
        model=model,
        action_space=action_space,
        v_min=saved_args.v_min,
        v_max=saved_args.v_max,
    ).to(saved_args.device)
    env.close()

    load_policy_weights(policy, os.path.join(args.path, 'policy.pth'), saved_args.device)
    policy.eval()
    
    vec_env = DummyVecEnv([_make_env])
    # find discretizer
    discretizer_env = vec_env.envs[0]
    while not isinstance(discretizer_env, Discretizer):
        discretizer_env = discretizer_env.env

    max_frames = 10000
    vec_env = VecVideoRecorder(vec_env, "videos/", record_video_trigger=lambda x: x == 0, video_length=max_frames, name_prefix=f"tianshou_{args.game}")

    obs = vec_env.reset()
    total_frames = 0
    for i in range(max_frames + 1):
        action = policy(Batch(obs=np.array(obs), info=None)).act
        action_array = discretizer_env._decode_discrete_action[int(action[0])]
        action_meaning = vec_env.envs[0].unwrapped.get_action_meaning([1 if item > 0 else 0 for item in action_array])
        print(f"\rStep {i}: {action_meaning}\033[K", end="")
        obs, _, done ,info = vec_env.step(action)
        total_frames += 1
        if np.all(done):
            break
    vec_env.close()

if __name__ == '__main__':
    main()