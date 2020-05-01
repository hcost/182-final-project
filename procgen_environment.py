# procgen_environment.py
# Copyright (c) 2020 Daniel Grimshaw (danielgrimshaw@berkeley.edu)
#

import gym
import numpy as np
from gym import spaces
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step


class ProcgenEnvironment(py_environment.PyEnvironment):
  def __init__(self, env_name, **kwargs):
    super().__init__()

    self.env_name = env_name
    self.kwargs = kwargs
    self._reset()

    self._n_actions = self._game.action_space.n
    self._info = None
    self._done = False

    self._act_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=self._n_actions - 1,
                                                 name='action')
    self._obs_spec = array_spec.BoundedArraySpec(shape=(64, 64, 3), dtype=np.float32, minimum=0, maximum=1,
                                                 name='observation')

  def action_spec(self):
    return self._act_spec

  def observation_spec(self):
    return self._obs_spec

  def _reset(self):
    self._game = ProcessFrame64(gym.make(self.env_name, **self.kwargs))
    self._info = None
    self._done = False
    obs = self._game.reset()
    return time_step.restart(obs)

  def _step(self, action):
    if self._done:
      return self._reset()
    obs, rew, self._done, self._info = self._game.step(action)

    if not self._done:
      return time_step.transition(obs, rew)

    return time_step.termination(obs, rew)

  def render(self, mode='rgb_array'):
    return self._game.render(mode=mode)

  def get_info(self):
    return self._info


def _process_frame64(frame):
  img = np.reshape(frame, [64, 64, 3]).astype(np.float32)
  x_t = img / 255.0
  return x_t.astype(np.float32)


class ProcessFrame64(gym.Wrapper):
  def __init__(self, env=None):
    super().__init__(env)
    self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    return _process_frame64(obs), reward, done, info

  def reset(self):
    return _process_frame64(self.env.reset())

