import gym
import stable_baselines as sb
import numpy as np
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import ACKTR
import stable_baselines.common as sbc
from stable_baselines.gail import ExpertDataset
import os


root_dir = os.getcwd()
model_name = root_dir + '/models/'+ 'pretrained_ppo2'
use_vec_envs = True
number_of_vec_envs = 4
framestack = 3 #NOTE: cannot use framstacking and pretraining together
use_pretraining = True
expert_dir = root_dir + '/' + 'i_am_the_expert.npz'
traj_limitation = -1 #-1 uses whole dataset
pretraining_epochs = 10
training_iterations = 1000
policy = 'CnnPolicy'
seed = 0
log_dir = root_dir + '/' + 'stats/pretrain_ppo2'
vid_dir = root_dir + '/' + 'videos/pretrain_ppo2'
vid_freq = 250
verbose = 1
num_levels = 100


print("Initializing")
def make_env(rank, seed=seed):
	def func():
		env = gym.make(id='procgen:procgen-fruitbot-v0', num_levels=num_levels, distribution_mode='easy')
		env.seed(rank+seed)
		return env
	set_global_seeds(seed)
	return func

vid_trigger = lambda step: (step % vid_freq == 0)

print("Setting Up Vectorized Environment")

def make_vec(num):
	return SubprocVecEnv([make_env(i) for i in range(num)], start_method='fork')


if use_vec_envs:
	env = make_vec(number_of_vec_envs)
	env = sbc.vec_env.VecNormalize(env)
	# env = sbc.vec_env.VecFrameStack(env, framestack)
	env = sbc.vec_env.VecVideoRecorder(env, vid_dir, vid_trigger)
else:
	env = gym.make(id='procgen:procgen-fruitbot-v0', num_levels=num_levels, distribution_mode='easy')


model = sb.PPO2(policy, env, verbose=verbose, tensorboard_log=log_dir)

if use_pretraining:
	print("Pretraining")
	dataset = ExpertDataset(expert_path=expert_dir, traj_limitation=traj_limitation, verbose=verbose)
	model.pretrain(dataset, n_epochs=pretraining_epochs)

print("Training")
model.learn(training_iterations)


model.save(model_name)