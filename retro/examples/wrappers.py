import numpy as np
import gymnasium as gym
import random
from collections import deque
from retro.examples.discretizer import Discretizer
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
import cv2

INITIAL_KEY = '_INITIAL'

class StreetFighter2Discretizer(Discretizer):
    """
    Use Street Fighter 2
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """
    def __init__(self, env):
        super().__init__(env=env, combos=[[], 
                                         ['UP'],                                     
                                         ['DOWN'],               
                                         ['LEFT'], 
                                         ['UP', 'LEFT'],
                                         ['DOWN', 'LEFT'],
                                         ['RIGHT'], 
                                         ['UP', 'RIGHT'], 
                                         ['DOWN', 'RIGHT'],
                                         ['B'],
                                         ['B', 'DOWN'],
                                         ['B', 'LEFT'],
                                         ['B', 'RIGHT'],
                                         ['A'],
                                         ['A', 'DOWN'],
                                         ['A', 'LEFT'],
                                         ['A', 'RIGHT'],
                                         ['C'],
                                         ['DOWN', 'C'],
                                         ['LEFT', 'C'],
                                         ['RIGHT', 'C'],
                                         ['Y'],
                                         ['DOWN', 'Y'],
                                         ['LEFT', 'Y'],
                                         ['DOWN', 'LEFT', 'Y'],
                                         ['RIGHT', 'Y'],
                                         ['X'],
                                         ['DOWN', 'X'],
                                         ['LEFT', 'X'],
                                         ['DOWN', 'LEFT', 'X'],
                                         ['RIGHT', 'X'],
                                         ['DOWN', 'RIGHT', 'X'],
                                         ['Z'],
                                         ['DOWN', 'Z'],
                                         ['LEFT', 'Z'],
                                         ['DOWN', 'LEFT', 'Z'],
                                         ['RIGHT', 'Z'],
                                         ['DOWN', 'RIGHT', 'Z']])

class SuperHangOnDiscretizer(Discretizer):
    def __init__(self, env):
        super().__init__(env=env, combos=[
            ['B'],
            ['B', 'LEFT'],
            ['B', 'RIGHT'],
            ['A'],
            ['A', 'LEFT'],
            ['A', 'RIGHT'],
            ['C'],
            ['C', 'LEFT'],
            ['C', 'RIGHT']
        ])

class SuperHangOnStageSaver(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.saved_states = {}

    def reset(self, **kwargs):
        self.stage = 0
        self.prev_time = 0
        if INITIAL_KEY not in self.saved_states and self.unwrapped.initial_state:
            self.saved_states[INITIAL_KEY] = self.unwrapped.initial_state
        if len(self.saved_states) > 0:
            picked_state = random.choice(list(self.saved_states.keys()))
            self.unwrapped.initial_state = self.saved_states[picked_state]
        
        return self.env.reset(**kwargs)
    
    def add_state(self, statename):
        if statename not in self.saved_states:
            self.saved_states[statename] = self.unwrapped.get_state()
            print(f'Saved new state {statename}')

    def step(self, action):
        ob, rew, terminated, truncated, info = self.env.step(action)

        if 'time' in info:
            curr_time = info['time']
            if self.prev_time > 0 and curr_time > self.prev_time: # timer increased
                self.stage += 1
                save_key = f'state_{self.stage}'
                self.add_state(save_key)
            self.prev_time = info['time']

        return ob, rew, terminated, truncated, info 

class NeedForSpeedDiscretizer(Discretizer):
    def __init__(self, env):
        super().__init__(env=env, combos=[
            ['A'],
            ['A', 'LEFT'],
            ['A', 'RIGHT'],
            ['B'],
            ['B', 'LEFT'],
            ['B', 'RIGHT'],
            ['A', 'L'],
            ['A', 'R'],
        ])

class StreetFighterFlipEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.flipped = False
        self.left_idx, self.right_idx = env.buttons.index("LEFT"), env.buttons.index("RIGHT")

    def step(self, action):
        if self.flipped:
            # flip left and right keys
            action[self.left_idx], action[self.right_idx] = action[self.right_idx], action[self.left_idx]

        _obs, reward, done, truncated, info = self.env.step(action)

        obs = _obs
        if 'agent_x' in info and info['agent_x'] > info['enemy_x']: # flipped
            self.flipped = True
            obs = np.flip(_obs, axis=1)
            # # debugging
            # self.env.img = obs
        else:
            self.flipped = False

        return obs, reward, done, truncated, info


class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=False,
                )
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew / self.n, terminated, truncated, info
    

class ActionBias(gym.Wrapper):
    def __init__(self, env, bias: list):
        super().__init__(env)
        self.bias = bias
        self.rng = np.random.RandomState()

    def step(self, ac):
        additional_reward = 0
        for i, bias_value in enumerate(self.bias):
            # if bias_value > 0 and self.rng.random() < bias_value:
            #     ac[i] = 1
            if bias_value != 0 and ac[i] > 0:
                additional_reward += bias_value
        ob, rew, terminated, truncated, info = self.env.step(ac)

        return ob, rew + additional_reward, terminated, truncated, info 



def _parse_reset_result(reset_result):
    contains_info = (
        isinstance(reset_result, tuple)
        and len(reset_result) == 2
        and isinstance(reset_result[1], dict)
    )
    if contains_info:
        return reset_result[0], reset_result[1], contains_info
    return reset_result, {}, contains_info


class FrameStack(gym.Wrapper):
    """Stack n_frames last frames.

    :param gym.Env env: the environment to wrap.
    :param int n_frames: the number of frames to stack.
    """

    def __init__(self, env, n_frames):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        obs_space = env.observation_space.shape
        if len(obs_space) == 2:
            shape = (n_frames, *obs_space)
        else:
            shape = (n_frames * obs_space[0], *obs_space[1:])
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=shape,
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info, return_info = _parse_reset_result(self.env.reset(**kwargs))
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return (self._get_ob(), info) if return_info else self._get_ob()

    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            new_step_api = False
        else:
            obs, reward, term, trunc, info = step_result
            new_step_api = True
        self.frames.append(obs)
        if new_step_api:
            return self._get_ob(), reward, term, trunc, info
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        # the original wrapper use `LazyFrames` but since we use np buffer,
        # it has no effect
        obs = np.stack(self.frames, axis=0)
        if len(obs.shape) == 4:
            obs = np.reshape(obs, (obs.shape[0] * obs.shape[1], *obs.shape[2:]))
        return obs
    

class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize observations to 0~1.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        low = np.min(env.observation_space.low)
        high = np.max(env.observation_space.high)
        self.bias = low
        self.scale = high - low
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        return (observation - self.bias) / self.scale

class WarpFrame(gym.ObservationWrapper):
    """Warp frames to 84x84 as done in the Nature paper and later work.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env, grayscale=True):
        super().__init__(env)
        self.size = 96
        self.grayscale = grayscale
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=(self.size, self.size) if grayscale else (3, self.size, self.size),
            dtype=env.observation_space.dtype,
        )

    def observation(self, frame):
        """Returns the current observation from a frame."""
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if len(frame.shape) == 3:
            return np.transpose(frame, (2, 0, 1))
        return frame

GAME_WRAPPERS = {
    'NeedForSpeedCarbon-GBA': [NeedForSpeedDiscretizer, ClipRewardEnv],
    'StreetFighterIISpecialChampionEdition-Genesis': [StreetFighter2Discretizer, StreetFighterFlipEnvWrapper],
    'SuperHangOn-Genesis': [SuperHangOnDiscretizer, SuperHangOnStageSaver],
}

GAME_STATES = {
    'NeedForSpeedCarbon-GBA': [
        '3LapsHardDifficulty.state',
        '3LapsNormalDifficulty.state',
        '3LapsNormalDifficulty2.state',
        '3LapsNormalDifficulty3.state',
        '3LapsNormalDifficulty4.state',
    ]
}