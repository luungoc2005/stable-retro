import tianshou_patch as _
import torch
import torch.nn as nn
import numpy as np
import retro
import argparse
import os
import pprint
import random
from tianshou.utils.net.discrete import NoisyLinear
from gymnasium import spaces
from typing import Any, List, Union, cast
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.monitor import Monitor
from retro.examples.wrappers import (
    StochasticFrameSkip,
    ActionBias,
    FrameStack,
    ScaledFloatFrame,
    WarpFrame,
    GAME_WRAPPERS,
    GAME_STATES,
)
from retro.examples.impala_cnn import ConvSequence
from torch.utils.tensorboard import SummaryWriter

from tianshou.env import ShmemVectorEnv
from tianshou.data import Batch, Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer, to_numpy
from tianshou.policy import RainbowPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
import json

GAME_NAME = "StreetFighterIISpecialChampionEdition-Genesis"

def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Network
class VisEncoder(nn.Module):
    def __init__(self, c, h, w, output_dim):
        super().__init__()
        cnn_encoder = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            nn.LeakyReLU(inplace=True),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.LeakyReLU(inplace=True),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.output_dim = int(np.prod(cnn_encoder(torch.zeros(1, c, h, w)).shape[1:]))
        self.net = nn.Sequential(
            cnn_encoder,
            layer_init(nn.Linear(self.output_dim, output_dim)),
            nn.LeakyReLU(inplace=True),
        )

        self._next_param = self.net.parameters().__next__()
        
    def forward(self, obs) -> torch.Tensor:
        obs = torch.as_tensor(obs, dtype=self._next_param.dtype, device=self._next_param.device)
        return self.net(obs)

class VisEncoderImpala(nn.Module):
    def __init__(self, c, h, w, output_dim):
        super().__init__()
        
        shape = (c, h, w)
        
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)

        conv_seqs += [
            nn.Flatten(),
            nn.LeakyReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=output_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=output_dim, out_features=output_dim),
        ]
        self.network = nn.Sequential(*conv_seqs)

        self._next_param = self.network.parameters().__next__()

    def forward(self, obs) -> torch.Tensor:
        obs = torch.as_tensor(obs, dtype=self._next_param.dtype, device=self._next_param.device)
        return self.network(obs)

class RainbowNet(nn.Module):
    def __init__(self, observation_space: spaces.Space, action_space: spaces.MultiBinary, num_atoms: int = 51, is_noisy: bool = True, use_impala=False):
        super().__init__()
        c, h, w = observation_space.shape[:3]
        self.action_num = action_space.n
        self.num_atoms = num_atoms
        hidden_dim = 512

        if use_impala:
            self.encoder = VisEncoderImpala(c, h, w, hidden_dim)
        else:
            self.encoder = VisEncoder(c, h, w, hidden_dim)

        def linear(x, y):
            if is_noisy:
                return NoisyLinear(x, y, .5)
            return nn.Linear(x, y)
        
        self.Q = nn.Sequential(
            linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            linear(hidden_dim, self.action_num * self.num_atoms),
        )

        self.V = nn.Sequential(
            linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            linear(hidden_dim, self.num_atoms),
        )

        self.output_dim = self.action_num * self.num_atoms

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        x = self.encoder(obs)
        
        q = self.Q(x)
        q = q.view(-1, self.action_num, self.num_atoms)

        v = self.V(x)
        v = v.view(-1, 1, self.num_atoms)

        logits: torch.Tensor = q - q.mean(dim=1, keepdim=True) + v

        probs = logits.softmax(dim=2)
        return probs, state

def make_retro(*, game, state=None, max_episode_steps=0, action_bias='', frame_skip=True, **kwargs):
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


def wrap_deepmind_retro(env, grayscale=True):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
    """
    env = Monitor(env)
    env = WarpFrame(env, grayscale=grayscale)
    env = ScaledFloatFrame(env)
    env = FrameStack(env, 4)
    # env = ClipRewardEnv(env)
    return env

def make_env(args, render_mode="human"):
    if args.game in GAME_STATES:
        all_states = GAME_STATES[args.game]
        state = random.choice(all_states)
        print(f"Starting new env with state={state}")
    else:
        state = args.state
    env = make_retro(
        game=args.game, 
        state=state, 
        scenario=args.scenario, 
        action_bias=args.action_bias, 
        frame_skip=not args.no_frame_skip,
        max_episode_steps=args.max_episode_steps,
        render_mode=render_mode,
    )
    env = wrap_deepmind_retro(env, grayscale=not args.no_grayscale)
    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default=GAME_NAME)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--impala", action='store_true', default=False)
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--action-bias", default='0 0 0 0 0 0 0 0 0 0 0 0')
    parser.add_argument("--no-frame-skip", action='store_true')
    parser.add_argument("--no-grayscale", action='store_true', default=False)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=100000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--training-num", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--beta-final", type=float, default=1.0)
    parser.add_argument("--beta-anneal-step", type=int, default=5000000)
    parser.add_argument("--eps-test", type=float, default=0.005)
    parser.add_argument("--eps-train", type=float, default=1.0)
    parser.add_argument("--eps-train-final", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--v-min", type=float, default=-10.0)
    parser.add_argument("--v-max", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=0.0000625)
    parser.add_argument("--device", type=str, default='mps')
    args = parser.parse_args()
    print(args)

    def _make_env():
        return make_env(args)

    dummy_env = _make_env()
    observation_space, action_space = dummy_env.observation_space, dummy_env.action_space
    model = RainbowNet(observation_space, action_space, use_impala=args.impala).to(args.device)
    torch.compile(model, backend="aot_eager")
    dummy_env.close()
    del dummy_env

    venv = ShmemVectorEnv([_make_env] * args.training_num)
    tb_log_name = f"ppo-{args.game}"
    if args.no_frame_skip:
        tb_log_name += "-NoSkip"
    if args.no_grayscale:
        tb_log_name += "-rgb"
    if args.impala:
        tb_log_name += "-impala"
    tb_log_path = f"tb_logs_tianshou/{tb_log_name}"
    increment = -1
    def get_final_path():
        return (tb_log_path + f"_{increment}") if increment > -1 else tb_log_path
    while os.path.exists(get_final_path()):
        increment += 1
    tb_log_path = get_final_path()
    print(f"Saving experiment into {tb_log_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    policy = RainbowPolicy(
        model=model,
        optim=optimizer,
        action_space=action_space,
        discount_factor=args.gamma,
        v_min=args.v_min,
        v_max=args.v_max,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
    ).to(args.device)

    stack_num = 4
    if args.no_grayscale:
        stack_num *= 3
    buffer = PrioritizedVectorReplayBuffer(
        args.buffer_size,
        buffer_num=args.training_num,
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=stack_num,
        alpha=args.alpha,
        beta=args.beta,
    )

    train_collector = Collector(policy, venv, buffer, exploration_noise=True)
    test_collector = Collector(policy, venv, exploration_noise=True)

    train_collector.collect(n_step=args.batch_size * args.training_num)

    writer = SummaryWriter(tb_log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(tb_log_path, "policy.pth"))
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        with open(os.path.join(tb_log_path, 'args.json'), 'w') as fp:
            json.dump(args.__dict__, fp, indent=2, default=default)

    def stop_fn(mean_rewards: float) -> bool:
        return False
    
    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})
        if env_step <= args.beta_anneal_step:
            beta = args.beta - env_step / args.beta_anneal_step * (args.beta - args.beta_final)
        else:
            beta = args.beta_final
        buffer.set_beta(beta)
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/beta": beta})

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.training_num,
        batch_size=args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=args.update_per_step,
        test_in_train=False,
    ).run()

    pprint.pprint(result)

    # watch agent's performance
    def watch():
        print("Setup test envs ...")
        policy.eval()
        policy.set_eps(args.eps_test)

        print("Testing agent ...")
        test_collector.reset()
        result = test_collector.collect(n_episode=args.training_num, render=True)

        pprint.pprint(result)
        rew = result["rews"].mean()
        print(f"Mean reward (over {result['n/ep']} episodes): {rew}")
    
    watch()

if __name__ == '__main__':
    main()