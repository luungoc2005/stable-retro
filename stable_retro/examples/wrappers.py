import numpy as np
import gymnasium as gym

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
