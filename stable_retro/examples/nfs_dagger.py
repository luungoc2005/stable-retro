"""
DAgger: teach the pixel-based policy to drive by imitating the RAM-based expert
(nfs_expert.PursuitExpert), then fine-tune it with PPO (nfs_ppo_tianshou --init).

Each iteration rolls out a mixture of expert (prob beta) and current student on the
training tracks, labels every visited frame with the expert's action, aggregates the
data and retrains the student with cross-entropy. Only training tracks are used.

    python -m stable_retro.examples.nfs_dagger --out tb_logs_tianshou/dagger
    python -m stable_retro.examples.nfs_ppo_tianshou --init tb_logs_tianshou/dagger/policy.pth
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from stable_retro.examples.nfs_env import NFSRaceEnv, list_states


def ppo_args(device, **kw):
    """Arguments for nfs_ppo_tianshou.build, so the student can be used directly by PPO/eval."""
    a = dict(
        encoder="nature", channels=[16, 32, 32], device=device, lr=2.5e-4, lr_decay=False,
        gamma=0.995, gae_lambda=0.95, max_grad_norm=0.5, vf_coef=0.5, ent_coef=0.005,
        eps_clip=0.2, batch_size=512, epoch=200, step_per_epoch=200_000, step_per_collect=4096,
        actions="race",
    )
    a.update(kw)
    return argparse.Namespace(**a)


def build_student(device):
    from stable_retro.examples import nfs_ppo_tianshou as M

    probe = NFSRaceEnv(list_states("train")[:1])
    args = ppo_args(device)
    algorithm, policy = M.build(args, probe.observation_space, probe.action_space)
    probe.close()
    return algorithm, policy, args


def _collect(job):
    from stable_retro.examples.nfs_expert import add_pose_vars, expert_for, expert_label, load_lines

    states, n_steps, beta, weights, seed = job
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    student = None
    if weights is not None and beta < 1.0:
        _, policy, _ = build_student("cpu")
        policy.actor.load_state_dict(torch.load(weights, map_location="cpu"))
        policy.actor.eval()
        student = policy.actor
    lines = load_lines()
    env = NFSRaceEnv(states, sticky_prob=0.0, mirror_prob=0.5, augment=True, seed=seed)
    add_pose_vars(env.data)
    obs_buf = np.zeros((n_steps, *env.observation_space.shape), np.uint8)
    act_buf = np.zeros(n_steps, np.int64)
    stats = []
    obs, _ = env.reset()
    expert = expert_for(env, lines)
    for t in range(n_steps):
        label = expert_label(expert, env)
        obs_buf[t] = obs
        act_buf[t] = label
        if student is None or rng.random() < beta:
            a = label
        else:
            with torch.no_grad():
                logits, _ = student(obs[None])
            a = int(torch.distributions.Categorical(logits=logits).sample())
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            stats.append((info["race_over"], info["position"], info["progress"]))
            obs, _ = env.reset()
            expert = expert_for(env, lines)
    env.close()
    return obs_buf, act_buf, stats


def train(policy, obs, acts, epochs, device, lr, batch=256):
    actor = policy.actor
    actor.train()
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    n = len(obs)
    weights = torch.tensor(np.bincount(acts, minlength=11) + 100.0, dtype=torch.float32)
    weights = (weights.sum() / weights) ** 0.5  # mild rebalancing toward rare (turning) actions
    weights = (weights / weights.mean()).to(device)
    for ep in range(epochs):
        perm = np.random.permutation(n)
        tot = correct = 0
        loss_sum = 0.0
        for i in range(0, n - batch + 1, batch):
            idx = np.sort(perm[i:i + batch])
            x = obs[idx]
            y = torch.as_tensor(acts[idx], device=device)
            logits, _ = actor(x)
            loss = F.cross_entropy(logits, y, weight=weights)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss) * len(idx)
            correct += int((logits.argmax(-1) == y).sum())
            tot += len(idx)
        print(f"    epoch {ep}: loss {loss_sum / tot:.3f} acc {correct / tot:.3f}", flush=True)
    actor.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tb_logs_tianshou/dagger")
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--steps-per-iter", type=int, default=40_000)
    p.add_argument("--betas", type=float, nargs="*", default=[1.0, 0.6, 0.4, 0.2, 0.1, 0.0])
    p.add_argument("--max-data", type=int, default=200_000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    states = list_states("train")
    algorithm, policy, pargs = build_student(args.device)
    weights_path = os.path.join(args.out, "actor.pth")
    data_obs = data_act = None
    log = []
    for it in range(args.iters):
        beta = args.betas[min(it, len(args.betas) - 1)]
        t0 = time.time()
        per = args.steps_per_iter // args.workers
        jobs = [(states, per, beta, weights_path if it > 0 else None, 1000 * it + w) for w in range(args.workers)]
        with mp.get_context("spawn").Pool(args.workers) as pool:
            results = pool.map(_collect, jobs)
        obs = np.concatenate([r[0] for r in results])
        acts = np.concatenate([r[1] for r in results])
        stats = [s for r in results for s in r[2]]
        if data_obs is None:
            data_obs, data_act = obs, acts
        else:
            data_obs = np.concatenate([data_obs, obs])[-args.max_data:]
            data_act = np.concatenate([data_act, acts])[-args.max_data:]
        fin = [s for s in stats if s[0]]
        info = {
            "iter": it, "beta": beta, "episodes": len(stats), "finished": len(fin),
            "wins": sum(1 for s in fin if s[1] == 1), "mean_pos_finished": float(np.mean([s[1] for s in fin])) if fin else None,
            "collect_s": round(time.time() - t0), "data": len(data_obs),
            "action_hist": np.bincount(acts, minlength=11).tolist(),
        }
        print(json.dumps(info), flush=True)
        log.append(info)
        train(policy, data_obs, data_act, args.epochs, args.device, args.lr)
        torch.save(policy.actor.state_dict(), weights_path)
        torch.save({"algorithm_state_dict": algorithm.state_dict(), "args": vars(pargs)},
                   os.path.join(args.out, "policy.pth"))
        with open(os.path.join(args.out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)


if __name__ == "__main__":
    main()
