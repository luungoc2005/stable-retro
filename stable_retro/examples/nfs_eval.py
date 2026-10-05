"""
Evaluate a PPO agent trained by nfs_ppo_tianshou on NFS Carbon (GBA) races.

Reports, per state and aggregated by split/difficulty/event:
  finished  - the race ended with the player still racing (not stuck / timed out)
  position  - final race position (1 = win)
  win       - finished in 1st place

Usage:
    python -m stable_retro.examples.nfs_eval --checkpoint tb_logs_tianshou/ppo-nfs/policy.pth
    python -m stable_retro.examples.nfs_eval --checkpoint ... --split train --episodes 1
    python -m stable_retro.examples.nfs_eval --checkpoint ... --states Circuit.Parkside.Fwd.Hard --video out.mp4
"""
import argparse
import collections
import json
import multiprocessing as mp
import os

import numpy as np
import torch

from stable_retro.examples.nfs_env import DIFFICULTIES, NFSRaceEnv, list_states


def load_policy(checkpoint, device):
    from stable_retro.examples import nfs_ppo_tianshou as M

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    targs = argparse.Namespace(**ckpt["args"])
    targs.device = device
    actions = getattr(targs, "actions", "legacy")
    probe = NFSRaceEnv(list_states("all")[:1], actions=actions)
    algorithm, policy = M.build(targs, probe.observation_space, probe.action_space)
    probe.close()
    algorithm.load_state_dict(ckpt["algorithm_state_dict"])
    policy.eval()
    policy.actions = actions
    return policy


def run_episode(env, policy, state, deterministic, device, seed, frames=None):
    obs, info = env.reset(seed=seed, options={"state": state})
    total = 0.0
    steps = 0
    while True:
        with torch.no_grad():
            logits, _ = policy.actor(obs[None])
            if deterministic:
                a = int(logits.argmax(-1).item())
            else:
                a = int(torch.distributions.Categorical(logits=logits).sample().item())
        obs, r, term, trunc, info = env.step(a)
        if frames is not None:
            frames.append(env.get_frame())
        total += r
        steps += 1
        if term or trunc:
            break
    return {
        "state": state,
        "event": info["event"],
        "track": info["track"],
        "difficulty": DIFFICULTIES[info["difficulty"]],
        "finished": bool(info["race_over"]),
        "position": int(info["position"]) if info["race_over"] else 5,
        "progress": int(info["progress"]),
        "return": total,
        "steps": steps,
    }


def _worker(job):
    checkpoint, states, episodes, deterministic, seed = job
    torch.set_num_threads(1)
    policy = load_policy(checkpoint, "cpu")
    env = NFSRaceEnv(states, sticky_prob=0.0, seed=seed, actions=policy.actions)
    out = []
    for s in states:
        for e in range(episodes):
            out.append(run_episode(env, policy, s, deterministic, "cpu", seed + e))
    env.close()
    return out


def summarize(results):
    def agg(rows):
        n = len(rows)
        return {
            "n": n,
            "finish_rate": np.mean([r["finished"] for r in rows]),
            "win_rate": np.mean([r["finished"] and r["position"] == 1 for r in rows]),
            "podium_rate": np.mean([r["finished"] and r["position"] <= 2 for r in rows]),
            "mean_position": np.mean([r["position"] for r in rows]),
        }

    groups = collections.defaultdict(list)
    for r in results:
        groups["ALL"].append(r)
        groups[f"difficulty={r['difficulty']}"].append(r)
        groups[f"event={r['event']}"].append(r)
    print(f"{'group':28s} {'n':>4s} {'finish':>7s} {'win':>6s} {'top2':>6s} {'meanpos':>8s}")
    for k in sorted(groups):
        a = agg(groups[k])
        print(f"{k:28s} {a['n']:4d} {a['finish_rate']:7.2f} {a['win_rate']:6.2f} {a['podium_rate']:6.2f} {a['mean_position']:8.2f}")
    return {k: agg(v) for k, v in groups.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "test", "all"])
    p.add_argument("--states", nargs="*", default=None)
    p.add_argument("--difficulties", nargs="*", default=None)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--video", default=None, help="record the first state's episode to this mp4")
    p.add_argument("--out", default=None, help="write per-episode results json here")
    args = p.parse_args()

    states = args.states or list_states(args.split, difficulties=args.difficulties)

    if args.video:
        import imageio

        policy = load_policy(args.checkpoint, "cpu")
        env = NFSRaceEnv(states, sticky_prob=0.0, actions=policy.actions)
        frames = []
        res = run_episode(env, policy, states[0], args.deterministic, "cpu", 0, frames)
        imageio.mimsave(args.video, frames[::2], fps=30, macro_block_size=1)
        print(res)
        return

    chunks = [states[i::args.workers] for i in range(args.workers)]
    jobs = [(args.checkpoint, c, args.episodes, args.deterministic, 100 * i) for i, c in enumerate(chunks) if c]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        results = [r for rows in pool.map(_worker, jobs) for r in rows]
    for r in sorted(results, key=lambda r: r["state"]):
        print(f"{r['state']:45s} finished={r['finished']!s:5s} pos={r['position']} progress={r['progress']:6d} steps={r['steps']}")
    summary = summarize(results)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"results": results, "summary": summary}, f, indent=1, default=float)


if __name__ == "__main__":
    main()
