"""
"Self-driving" demo: let a trained NFS agent drive any race state live in a window
(or record it to a video), printing race position and progress as it goes.

Usage:
    python -m stable_retro.examples.nfs_drive --checkpoint tb_logs_tianshou/ppo-nfs-v3/policy.pth
    python -m stable_retro.examples.nfs_drive --checkpoint ... --state Circuit.Parkside.Fwd.Hard
    python -m stable_retro.examples.nfs_drive --checkpoint ... --split test --races 5 --video drive.mp4
"""
import argparse
import random
import time

import numpy as np
import torch

from stable_retro.examples.nfs_env import NFSRaceEnv, list_states
from stable_retro.examples.nfs_eval import load_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--state", default=None, help="race state name (default: random held-out track)")
    p.add_argument("--split", default="test", choices=["train", "test", "all"])
    p.add_argument("--races", type=int, default=1)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--video", default=None, help="write an mp4 instead of opening a window")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier for the window")
    args = p.parse_args()

    policy = load_policy(args.checkpoint, "cpu")
    states = [args.state] if args.state else list_states(args.split)
    env = NFSRaceEnv(states, sticky_prob=0.0, actions=policy.actions,
                     render_mode=None if args.video else "human")
    frames = []
    for race in range(args.races):
        state = args.state or random.choice(states)
        obs, info = env.reset(options={"state": state})
        print(f"race {race + 1}: {state}")
        t0 = time.time()
        steps = 0
        while True:
            with torch.no_grad():
                logits, _ = policy.actor(obs[None])
            a = int(logits.argmax(-1)) if args.deterministic else int(
                torch.distributions.Categorical(logits=logits).sample())
            obs, r, term, trunc, info = env.step(a)
            steps += 1
            if args.video:
                frames.append(env.get_frame())
            else:
                # emulator runs 4 frames per step at 60 fps
                delay = steps * 4 / 60 / args.speed - (time.time() - t0)
                if delay > 0:
                    time.sleep(delay)
            if steps % 150 == 0:
                print(f"  t={steps * 4 / 60:5.1f}s position={info['position']} progress={info['progress']}")
            if term or trunc:
                break
        result = f"finished in position {info['position']}" if info["race_over"] else "did not finish (stuck/timeout)"
        print(f"  -> {result}")
    if args.video:
        import imageio
        imageio.mimsave(args.video, frames[::2], fps=30, macro_block_size=1)
        print("wrote", args.video)
    env.close()


if __name__ == "__main__":
    main()
