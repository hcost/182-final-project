import gym
import stable_baselines as sb
import numpy as np
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import ACKTR
import stable_baselines.common as sbc
from stable_baselines.gail import generate_expert_traj
from PIL import Image

env = gym.make(id='procgen:procgen-fruitbot-v0', num_levels=100, distribution_mode='easy')
episodes = 1


actions = [0]



# Here the expert is a random agent
# but it can be any python function, e.g. a PID controller
def dummy_expert(_obs):
	env.render()
	# action = input("Enter input (0-2 left, 3-5 none, 6-8: right, 9: key, 10+: none): ")
	inp = input("Enter input (wasd): ")
	mapping = {'w': 9, 'a': 0, 's': 3, 'd': 6}
	try:
		action = mapping[inp]
		actions[0] = action
	except:
		action = actions[0]
	return action
# Data will be saved in a numpy archive named `expert_cartpole.npz`
# when using something different than an RL expert,
# you must pass the environment object explicitly

welcome_msg = \
" Hello! Welcome to expert trajection creation.\n \
The controls are w: shoot keys, a: left, s: nop, d: right\n \
You must press enter after each command. Hold enter to repeatedly enter last used command."

print('-'*50)
print(welcome_msg)
print('-'*50)

generate_expert_traj(dummy_expert, 'i_am_the_expert', env, n_episodes=episodes)
