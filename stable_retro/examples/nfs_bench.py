"""
Quick pace benchmark: agent progress relative to the leading AI car.

For each state, drives up to --steps agent steps and reports
pace = player progress / leader progress, both measured along the recorded racing line
(1.0 = as fast as the race leader).
A continuous metric that moves long before wins do.

    python -m stable_retro.examples.nfs_bench --checkpoint <pth> [--split test] [--difficulties Hard]
"""
import argparse
import multiprocessing as mp

import numpy as np
import torch

from stable_retro.examples.nfs_env import LINE_SCALE, LineTracker, NFSRaceEnv, list_states
from stable_retro.examples.nfs_eval import load_policy

def _run(job):
    checkpoint, states, steps, deterministic = job
    torch.set_num_threads(1)
    policy = load_policy(checkpoint, "cpu")
    env = NFSRaceEnv(states, sticky_prob=0.0, actions=policy.actions, seed=0)
    out = []
    for s in states:
        obs, info = env.reset(options={"state": s})
        # opponents measured along the same racing line as the player
        trackers = []
        for i in (1, 2, 3):
            t = LineTracker(env.lines[s])
            t.locate(np.array([env.data.lookup_value(f"car{i}_x"), env.data.lookup_value(f"car{i}_z")]) / 65536)
            trackers.append(t)
        opp = [0.0, 0.0, 0.0]
        lead = 0
        for n in range(steps):
            with torch.no_grad():
                logits, _ = policy.actor(obs[None])
            a = int(logits.argmax(-1)) if deterministic else int(
                torch.distributions.Categorical(logits=logits).sample())
            obs, r, term, trunc, info = env.step(a)
            if not info["race_over"]:
                for j, t in enumerate(trackers):
                    p = np.array([env.data.lookup_value(f"car{j + 1}_x"), env.data.lookup_value(f"car{j + 1}_z")]) / 65536
                    opp[j] += LINE_SCALE * t.update(p)
                lead = max(lead, max(opp))
            if term or trunc:
                break
        out.append((s, info["progress"], lead, n + 1, info["race_over"], info["position"],
                    info.get("wrong_way_steps", 0) / (n + 1)))
    env.close()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--difficulties", nargs="*", default=["Hard"])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    states = list_states(args.split, difficulties=args.difficulties)
    chunks = [states[i::args.workers] for i in range(args.workers)]
    jobs = [(args.checkpoint, c, args.steps, args.deterministic) for c in chunks if c]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        rows = [r for rr in pool.map(_run, jobs) for r in rr]
    paces = []
    for s, prog, lead, n, over, pos, ww in sorted(rows):
        pace = prog / max(lead, 1)
        paces.append(pace)
        print(f"{s:42s} pace={pace:5.2f} progress={prog:7.0f} leader={lead:7.0f} steps={n} pos={pos} wrongway={ww:.2f}")
    print(f"MEAN PACE {np.mean(paces):.3f}  median {np.median(paces):.3f}")


if __name__ == "__main__":
    main()
