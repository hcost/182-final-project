from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image


import sys, os
import contextlib

import tensorflow as tf
import gym
from gym import wrappers

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.networks import categorical_q_network

tf.compat.v1.enable_v2_behavior()



#Hyperparamaters
params = {
			'model_name' : 'cat_model',
			'env' : 'procgen:procgen-fruitbot-v0', #procgen:procgen-fruitbot-v0
			'distribution' : 'easy',
			'atari' : False, #should really say "not procgen"
			'video' : True,
			'apply_visual_wrappers' : True, #same as above
			'frames' : 2, #number of frames to use with visual wrappers
			'categorical' : True,
			'num_iter' : int(1e6),
			'initial_collect_steps' : 15000,
			'collect_steps_per_iter' : 1,
			'replay_buffer_size' : 1000000,
			'batch_size' : 256,
			'lr' : 1e-3,
			'fc_layer_params' : (128, 64),
			'log_interval' : 5000,
			'num_eval_episodes' : 10,
			'eval_interval' : 10000,
			'record_percent' : 1/10,
			'gamma' : 0.99,
			'num_atoms' : 51,
			'min_q_value' : -5,
			'max_q_value' : 10,
			'n_step_update' : 2
}

params['record_freq'] = int(params['num_iter'] * params['record_percent'])



def make_tf_env(video=False):
	if params['atari']:
		py_env = gym.make(params['env'])
	else:
		py_env = gym.make(params['env'], distribution_mode=params['distribution'])
	if params["apply_visual_wrappers"]:
		py_env = wrappers.gray_scale_observation.GrayScaleObservation(py_env)
		py_env = wrappers.FrameStack(py_env, params['frames'])
	env = suite_gym.wrap_env(py_env)
	env = tf_py_environment.TFPyEnvironment(env)
	if video:
		return env, py_env
	return env

train_env, eval_env = make_tf_env(), make_tf_env()

observation_spec = train_env.observation_spec()
action_spec = train_env.action_spec()


#set up q_net and agent
if params['categorical']:
	q_net = categorical_q_network.CategoricalQNetwork(
									observation_spec,
									action_spec,
									num_atoms=params['num_atoms'],
									fc_layer_params=params['fc_layer_params'])

	optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params['lr'])

	train_step_counter = tf.compat.v2.Variable(0)
	agent = categorical_dqn_agent.CategoricalDqnAgent(
									train_env.time_step_spec(),
									action_spec,
									categorical_q_network=q_net,
									optimizer=optimizer,
									min_q_value=params['min_q_value'],
									max_q_value=params['max_q_value'],
									n_step_update=params['n_step_update'],
									td_errors_loss_fn=common.element_wise_squared_loss,
									gamma=params['gamma'],
									train_step_counter=train_step_counter
	)
else:
	q_net = q_network.QNetwork(
						observation_spec,
						action_spec,
						fc_layer_params=params['fc_layer_params'])
	train_step_counter = tf.Variable(0)
	agent = dqn_agent.DqnAgent(
						train_env.time_step_spec(),
						action_spec,
						q_network = q_net,
						optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=params['lr']),
						td_errors_loss_fn=common.element_wise_squared_loss,
						train_step_counter=train_step_counter)

agent.initialize()
collect_policy = greedy_policy.GreedyPolicy(agent.policy)
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), action_spec)

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
											data_spec=agent.collect_data_spec,
											batch_size=train_env.batch_size,
											max_length=params['replay_buffer_size'])


print("Hey! This is the updated version 4.0!")

def collect_data(env, policy, buffer, steps=100):
	for i in range(steps):
		collect_step(env, policy, buffer)

def collect_step(env, policy, buffer):
	time_step = env.current_time_step()
	action_step = policy.action(time_step)
	next_time_step = env.step(action_step.action)
	traj = trajectory.from_transition(time_step, action_step, next_time_step)
	buffer.add_batch(traj)
	if next_time_step.is_last() and not params['atari']:
		env = make_tf_env()
		env.reset()

collect_data(train_env, random_policy, replay_buffer, steps=params['initial_collect_steps'])

def compute_avg_return(environment, policy, num_episodes=5):
	total_return = 0.0

	for i in range(num_episodes):
		if params['atari']:
			time_step = environment.reset()
		else:
			environment = make_tf_env()
			time_step = environment.reset()

		curr_return = 0.0

		while not time_step.is_last():
			action_step = policy.action(time_step)
			time_step = environment.step(action_step.action)
			curr_return += time_step.reward
		total_return += curr_return

	avg_return = total_return/num_episodes
	return avg_return.numpy()[0]


agent.train = common.function(agent.train)

agent.train_step_counter.assign(0)

dataset = replay_buffer.as_dataset(
							num_parallel_calls=3,
							sample_batch_size=params['batch_size'],
							num_steps=params['n_step_update']+1).prefetch(3)
iterator = iter(dataset)


avg_return = compute_avg_return(eval_env, agent.policy, params['num_eval_episodes'])

returns = [avg_return]


def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 3, length = 100, fill = '|', printEnd = "\r"):
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
	# Print New Line on Complete
	if iteration == total:
		print()

def create_video(policy, filename, num_episodes=5, fps=30):
    filename += '.mp4'
    filename = 'videos/'+filename
    with imageio.get_writer(filename, fps=fps) as video:
        with open(os.devnull, "w") as f, contextlib.redirect_stderr(f):
            for _ in range(num_episodes):
                environment, py_env = make_tf_env(video=True)
                time_step = environment.reset()
                while not time_step.is_last():
                    py_env.reset()
                    action_step = policy.action(time_step)
                    time_step = eval_env.step(action_step.action)
                    py_env.step(int(action_step.action))
                    video.append_data(py_env.render(mode='rgb_array'))

def make_graphs(returns):
	plt.plot(range(0,params['eval_interval']*len(returns), params['eval_interval']),returns)
	plt.ylabel('Average Return')
	plt.xlabel('Iterations')
	plt.savefig(params['model_name']+'_avg_return.png')


#Training
print('Beginning Training')
def train(start=0, returns=[]):
	try:
		for i in range(start, params['num_iter']):
			printProgressBar(i, params['num_iter'])
			for _ in range(params['collect_steps_per_iter']):
				collect_step(train_env, collect_policy, replay_buffer)

			exp, info = next(iterator)
			train_loss = agent.train(exp).loss
			step = agent.train_step_counter.numpy()

			if step % params['log_interval'] == 0:
				print("\nStep: {}, Loss: {}\n".format(step, train_loss))
			if step % params['eval_interval'] == 0:
				avg_return = compute_avg_return(eval_env, agent.policy, params['num_eval_episodes'])
				print('-'*20)
				print("\nStep: {}, Current Return: {}, Previous Return: {}\n".format(step, avg_return, returns[-1]))
				print('-'*20)
				returns.append(avg_return)
				make_graphs(returns)
				if params['video'] and step % params['record_freq'] == 0:
					create_video(agent.policy, params['model_name']+'_step_'+str(step), params['num_eval_episodes'])
		return returns
	except KeyboardInterrupt:
		return returns
	except:
		print("Unexpected Error; Trying to Resume Training")
		return train(start=i, returns=returns)


create_video(agent.policy, params['model_name']+'pretraining', params['num_eval_episodes'])
returns = train(returns=returns)
make_graphs(returns)
create_video(agent.policy, params['model_name']+'post-training', params['num_eval_episodes'])
