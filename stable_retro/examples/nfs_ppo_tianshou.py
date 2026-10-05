"""
PPO (tianshou 2.x) for Need for Speed Carbon (GBA), trained for generalization.

The agent is trained on many tracks / event types (circuit, elimination, sprint) and all
three AI difficulties, and is evaluated during training on tracks it never sees
(nfs_env.HELDOUT_TRACKS). The reward is race progress read from RAM (see nfs_env.py).

Usage:
    python -m stable_retro.examples.nfs_ppo_tianshou                   # train
    python -m stable_retro.examples.nfs_ppo_tianshou --resume <logdir> # continue training
    python -m stable_retro.examples.nfs_eval --checkpoint <logdir>/policy.pth   # evaluate
"""
import argparse
import json
import os
import pprint

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.algorithm.modelfree.reinforce import DiscreteActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory, LRSchedulerFactoryLinear
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env import ShmemVectorEnv
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils import TensorboardLogger

import stable_retro.examples.tianshou_patch  # noqa: F401  (float64 -> float32 for MPS)
from stable_retro.examples.impala_cnn import ConvSequence
from stable_retro.examples.nfs_env import NFSRaceEnv, list_states


# Categorical argument validation costs ~2ms per collector step
torch.distributions.Distribution.set_default_validate_args(False)


class ImpalaEncoder(nn.Module):
    def __init__(self, c, h, w, output_dim=256, channels=(16, 32, 32)):
        super().__init__()
        shape = (c, h, w)
        layers = []
        for out_channels in channels:
            seq = ConvSequence(shape, out_channels)
            shape = seq.get_output_shape()
            layers.append(seq)
        layers += [nn.Flatten(), nn.ReLU(), nn.Linear(int(np.prod(shape)), output_dim), nn.ReLU()]
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, obs):
        p = next(self.parameters())
        x = torch.as_tensor(obs, device=p.device, dtype=torch.float32) / 255.0
        return self.net(x)


class NatureEncoder(nn.Module):
    """DQN 'Nature' CNN: ~10x cheaper than IMPALA on MPS, used by default."""

    def __init__(self, c, h, w, output_dim=512):
        super().__init__()
        conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = conv(torch.zeros(1, c, h, w)).shape[1]
        self.net = nn.Sequential(conv, nn.Linear(n, output_dim), nn.ReLU())
        for m in self.net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.zeros_(m.bias)
        self.output_dim = output_dim

    def forward(self, obs):
        p = next(self.parameters())
        x = torch.as_tensor(obs, device=p.device, dtype=torch.float32) / 255.0
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, encoder, n_actions):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.output_dim, n_actions)
        nn.init.orthogonal_(self.head.weight, 0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs, state=None, info=None):
        return self.head(self.encoder(obs)), state


class Critic(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.output_dim, 1)
        nn.init.orthogonal_(self.head.weight, 1.0)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs, **kwargs):
        return self.head(self.encoder(obs))


def build(args, observation_space, action_space):
    c, h, w = observation_space.shape
    if args.encoder == "impala":
        encoder = ImpalaEncoder(c, h, w, channels=tuple(args.channels)).to(args.device)
    else:
        encoder = NatureEncoder(c, h, w).to(args.device)
    actor = Actor(encoder, action_space.n).to(args.device)
    critic = Critic(encoder).to(args.device)
    policy = DiscreteActorPolicy(
        actor=actor,
        action_space=action_space,
        deterministic_eval=False,
    )
    optim = AdamOptimizerFactory(lr=args.lr, eps=1e-5)
    if args.lr_decay:
        optim.with_lr_scheduler_factory(LRSchedulerFactoryLinear(
            max_epochs=args.epoch,
            epoch_num_steps=args.step_per_epoch,
            collection_step_num_env_steps=args.step_per_collect,
        ))
    algorithm = PPO(
        policy=policy,
        critic=critic,
        optim=optim,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        eps_clip=args.eps_clip,
        value_clip=True,
        advantage_normalization=True,
        return_scaling=True,
        max_batchsize=args.batch_size,
    ).to(args.device)
    return algorithm, policy


def make_env_fn(states, train, seed, args):
    def _f():
        return NFSRaceEnv(
            states,
            sticky_prob=args.sticky if train else 0.0,
            augment=train and args.augment,
            mirror_prob=args.mirror if train else 0.0,
            actions=args.actions,
            restart_prob=args.restart_prob if train else 0.0,
            stuck_steps=args.stuck_steps,
            max_steps=args.max_steps,
            seed=seed,
        )
    return _f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default="tb_logs_tianshou/ppo-nfs")
    p.add_argument("--resume", default=None, help="log dir (or policy file) to continue from")
    p.add_argument("--init", default=None, help="start a new run from the weights in this .pth file")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--training-num", type=int, default=16)
    p.add_argument("--test-num", type=int, default=8)
    p.add_argument("--epoch", type=int, default=200)
    p.add_argument("--step-per-epoch", type=int, default=200_000)
    p.add_argument("--step-per-collect", type=int, default=4096)
    p.add_argument("--repeat-per-collect", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--lr-decay", action="store_true")
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--eps-clip", type=float, default=0.2)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--max-steps", type=int, default=4500)
    p.add_argument("--encoder", choices=["nature", "impala"], default="nature")
    p.add_argument("--channels", type=int, nargs=3, default=[16, 32, 32])
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--actions", choices=["race", "legacy"], default="race")
    p.add_argument("--mirror", type=float, default=0.5, help="probability to play a training episode mirrored")
    p.add_argument("--restart-prob", type=float, default=0.5,
                   help="probability to start a training episode from a mid-race snapshot")
    p.add_argument("--stuck-steps", type=int, default=250,
                   help="end an episode after this many steps without progress")
    p.add_argument("--sticky", type=float, default=0.1)
    p.add_argument("--train-filter", default=None, help="only train on states whose name contains this")
    p.add_argument("--test-filter", default=None, help="only test on states whose name contains this")
    p.add_argument("--difficulties", nargs="*", default=None, help="restrict training difficulties")
    args = p.parse_args()

    train_states = list_states("train", difficulties=args.difficulties)
    test_states = list_states("test")
    if args.train_filter:
        train_states = [s for s in list_states("all") if args.train_filter in s]
    if args.test_filter:
        test_states = [s for s in list_states("all") if args.test_filter in s]
    print(f"{len(train_states)} train states, {len(test_states)} held-out test states")

    logdir = args.logdir
    if args.resume is None:
        i = 0
        while os.path.exists(logdir):
            i += 1
            logdir = f"{args.logdir}_{i}"
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print("logging to", logdir)

    train_envs = ShmemVectorEnv([make_env_fn(train_states, True, 1000 + i, args) for i in range(args.training_num)])
    test_envs = ShmemVectorEnv([make_env_fn(test_states, False, 2000 + i, args) for i in range(args.test_num)])
    probe = NFSRaceEnv(train_states[:1], actions=args.actions)
    algorithm, policy = build(args, probe.observation_space, probe.action_space)
    probe.close()

    if args.resume:
        path = args.resume if args.resume.endswith(".pth") else os.path.join(args.resume, "checkpoint.pth")
        ckpt = torch.load(path, map_location=args.device, weights_only=False)
        algorithm.load_state_dict(ckpt.get("algorithm_state_dict", ckpt))
        print("resumed from", path)

    if args.init:
        ckpt = torch.load(args.init, map_location=args.device, weights_only=False)
        full = ckpt.get("algorithm_state_dict", ckpt)
        try:
            # same architecture: take weights and optimizer state (a fresh Adam destabilizes training)
            algorithm.load_state_dict(dict(full))
            print("initialized weights + optimizer from", args.init)
        except (RuntimeError, ValueError, KeyError):
            own = algorithm.state_dict()
            sd = {k: v for k, v in full.items()
                  if k in own and hasattr(v, "shape") and own[k].shape == v.shape}
            # weights only; mismatched heads stay freshly initialized
            nn.Module.load_state_dict(algorithm, sd, strict=False)
            print(f"initialized {len(sd)}/{len(own)} tensors from", args.init)

    buffer = VectorReplayBuffer(args.step_per_collect, len(train_envs))
    train_collector = Collector[CollectStats](algorithm, train_envs, buffer, exploration_noise=False)
    test_collector = Collector[CollectStats](algorithm, test_envs)

    writer = SummaryWriter(logdir)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer, training_interval=args.step_per_collect, update_interval=args.step_per_collect, save_interval=1)

    def save_best_fn(algo):
        torch.save({"algorithm_state_dict": algo.state_dict(), "args": vars(args)}, os.path.join(logdir, "policy.pth"))

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        path = os.path.join(logdir, "checkpoint.pth")
        torch.save({"algorithm_state_dict": algorithm.state_dict(), "args": vars(args),
                    "epoch": epoch, "env_step": env_step}, path)
        return path

    result = algorithm.run_training(
        OnPolicyTrainerParams(
            training_collector=train_collector,
            test_collector=test_collector,
            max_epochs=args.epoch,
            epoch_num_steps=args.step_per_epoch,
            update_step_num_repetitions=args.repeat_per_collect,
            test_step_num_episodes=args.test_num,
            batch_size=args.batch_size,
            collection_step_num_env_steps=args.step_per_collect,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            logger=logger,
            resume_from_log=args.resume is not None and not args.resume.endswith(".pth"),
            test_in_training=False,
        )
    )
    pprint.pprint(result)


if __name__ == "__main__":
    main()
