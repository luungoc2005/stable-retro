"""
Watch/evaluate a trained NeedForSpeed Rainbow DQN agent.

Modes:
  --render     : Watch the agent play in a window (default)
  --record     : Record a video of the agent playing
  --stats      : Run N episodes and print performance statistics

Usage examples:
    # Watch live
    python -m stable_retro.examples.nfs_watch_agent --path tb_logs_tianshou/rainbow-NeedForSpeedCarbon-GBA

    # Record video
    python -m stable_retro.examples.nfs_watch_agent --path tb_logs_tianshou/rainbow-NeedForSpeedCarbon-GBA --record

    # Run 10 episodes and print stats
    python -m stable_retro.examples.nfs_watch_agent --path tb_logs_tianshou/rainbow-NeedForSpeedCarbon-GBA --stats --episodes 10
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
import stable_retro as retro

from tianshou.data import Batch
from tianshou.algorithm.modelfree.c51 import C51Policy
from stable_retro.examples.nfs_rainbow_tianshou import make_env, RainbowNet
from stable_retro.examples.tianshou_patch import load_policy_weights
from stable_retro.examples.discretizer import Discretizer


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


def load_policy(path, device='mps'):
    args_path = os.path.join(path, 'args.json')
    policy_path = os.path.join(path, 'policy.pth')

    if not os.path.exists(args_path):
        print(f"ERROR: args.json not found at {args_path}")
        sys.exit(1)
    if not os.path.exists(policy_path):
        print(f"ERROR: policy.pth not found at {policy_path}")
        sys.exit(1)

    with open(args_path, 'r') as fp:
        saved_args = DotDict(json.load(fp))

    saved_args['device'] = device
    print(f"Loaded args: {saved_args}")

    env = make_env(saved_args, render_mode="rgb_array")
    observation_space, action_space = env.observation_space, env.action_space
    print(f"Observation: {observation_space.shape}, Actions: {action_space.n}")

    use_impala = saved_args.get('impala', False)
    model = RainbowNet(observation_space, action_space, use_impala=use_impala).to(device)
    policy = C51Policy(
        model=model,
        action_space=action_space,
        v_min=saved_args.v_min,
        v_max=saved_args.v_max,
        eps_inference=saved_args.get('eps_test', 0.005),
    ).to(device)

    load_policy_weights(policy, policy_path, device)
    policy.eval()

    env.close()
    return policy, saved_args


def get_discretizer(env):
    """Walk the wrapper chain to find the Discretizer."""
    e = env
    while e is not None:
        if isinstance(e, Discretizer):
            return e
        e = getattr(e, 'env', None)
    return None


def run_episode(policy, args, render=False, render_mode=None):
    """Run one episode and return (total_reward, steps, info_history)."""
    if render_mode is None:
        render_mode = "human" if render else "rgb_array"
    env = make_env(args, render_mode=render_mode)
    discretizer = get_discretizer(env)

    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    frames = []

    while not done:
        obs_batch = np.array([obs])
        with torch.no_grad():
            action = policy(Batch(obs=obs_batch, info=None)).act
        action_int = int(action[0])

        if discretizer and render:
            action_array = discretizer._decode_discrete_action[action_int]
            meaning = env.unwrapped.get_action_meaning(
                [1 if item > 0 else 0 for item in action_array]
            )
            print(f"\rStep {steps}: reward={total_reward:.3f} action={meaning}\033[K", end="")

        obs, reward, terminated, truncated, info = env.step(action_int)
        total_reward += reward
        steps += 1
        done = terminated or truncated

        if render_mode == "rgb_array":
            frames.append(env.render())

    if render:
        print()  # newline after \r

    env.close()
    return total_reward, steps, frames


def watch_live(policy, args, episodes=1):
    """Watch the agent play live."""
    for ep in range(episodes):
        print(f"\n=== Episode {ep + 1}/{episodes} ===")
        reward, steps, _ = run_episode(policy, args, render=True, render_mode="human")
        print(f"Episode {ep + 1}: reward={reward:.3f}, steps={steps}")


def record_video(policy, args, output_dir="videos", episodes=1):
    """Record agent gameplay to video."""
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
    except ImportError:
        print("ERROR: stable_baselines3 required for video recording.")
        print("Install with: pip install stable-baselines3")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    def _make_env():
        return make_env(args, render_mode="rgb_array")

    vec_env = DummyVecEnv([_make_env])
    discretizer = get_discretizer(vec_env.envs[0])

    max_frames = args.get('max_episode_steps', 4500) + 100
    vec_env = VecVideoRecorder(
        vec_env, output_dir,
        record_video_trigger=lambda x: x == 0,
        video_length=max_frames,
        name_prefix=f"nfs_{args.game}"
    )

    obs = vec_env.reset()
    total_reward = 0.0
    for i in range(max_frames):
        action = policy(Batch(obs=np.array(obs), info=None)).act
        if discretizer:
            action_array = discretizer._decode_discrete_action[int(action[0])]
            meaning = vec_env.envs[0].unwrapped.get_action_meaning(
                [1 if item > 0 else 0 for item in action_array]
            )
            print(f"\rRecording step {i}: {meaning}\033[K", end="")
        obs, rew, done, info = vec_env.step(action)
        total_reward += rew[0]
        if np.all(done):
            break

    print(f"\nTotal reward: {total_reward:.3f}")
    print(f"Video saved to {output_dir}/")
    vec_env.close()


def run_stats(policy, args, episodes=10):
    """Run multiple episodes and print performance statistics."""
    rewards = []
    lengths = []

    for ep in range(episodes):
        reward, steps, _ = run_episode(policy, args, render=False)
        rewards.append(reward)
        lengths.append(steps)
        print(f"  Episode {ep + 1}/{episodes}: reward={reward:.3f}, steps={steps}")

    rewards = np.array(rewards)
    lengths = np.array(lengths)

    print(f"\n{'=' * 50}")
    print(f"Performance over {episodes} episodes:")
    print(f"  Reward:  mean={rewards.mean():.3f}  std={rewards.std():.3f}  "
          f"min={rewards.min():.3f}  max={rewards.max():.3f}")
    print(f"  Steps:   mean={lengths.mean():.1f}  std={lengths.std():.1f}  "
          f"min={lengths.min()}  max={lengths.max()}")
    print(f"{'=' * 50}")

    return rewards, lengths


def main():
    parser = argparse.ArgumentParser(description="Watch/evaluate trained NFS agent")
    parser.add_argument("--path", type=str, required=True,
                        help="Path to checkpoint directory (containing policy.pth and args.json)")
    parser.add_argument("--device", type=str, default='mps')
    parser.add_argument("--state", type=str, default=None,
                        help="Override game state (e.g., 3LapsNormalDifficulty.state)")
    parser.add_argument("--render", action='store_true', default=True,
                        help="Watch agent play live (default)")
    parser.add_argument("--record", action='store_true',
                        help="Record video instead of live rendering")
    parser.add_argument("--stats", action='store_true',
                        help="Run multiple episodes and print statistics")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run")
    parser.add_argument("--video-dir", type=str, default="videos",
                        help="Directory to save videos")
    args = parser.parse_args()

    policy, saved_args = load_policy(args.path, device=args.device)

    if args.state:
        saved_args['state'] = args.state

    if args.stats:
        run_stats(policy, saved_args, episodes=args.episodes)
    elif args.record:
        record_video(policy, saved_args, output_dir=args.video_dir, episodes=args.episodes)
    else:
        watch_live(policy, saved_args, episodes=args.episodes)


if __name__ == '__main__':
    main()
