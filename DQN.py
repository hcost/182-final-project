from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay

import tensorflow as tf
import gym

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

tf.compat.v1.enable_v2_behavior()

#Hyperparamaters
params = {
            'env' : 'procgen:procgen-fruitbot-v0',
            'distribution' : 'easy',
            'atari' : False,
            'num_iter' : 1000000,
            'initial_collect_steps' : 10000,
            'collect_steps_per_iter' : 1,
            'replay_buffer_size' : 1000000,
            'batch_size' : 256,
            'lr' : 3e-4,
            'fc_layer_params' : (256, 128),
            'log_interval' : 200,
            'num_eval_episodes' : 10,
            'eval_interval' : 1000
}

def make_tf_env():
    if params['atari']:
        py_env = gym.make(params['env'])
    else:
        py_env = gym.make(params['env'], distribution_mode=params['distribution'])
    env = suite_gym.wrap_env(py_env)
    env = tf_py_environment.TFPyEnvironment(env)
    return env, py_env

train_env, train_py_env = make_tf_env()
eval_env, eval_py_env = make_tf_env()

observation_spec = train_env.observation_spec()
action_spec = train_env.action_spec()


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

eval_policy = greedy_policy.GreedyPolicy(agent.policy)

collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), action_spec)

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                                            data_spec=agent.collect_data_spec,
                                            batch_size=train_env.batch_size,
                                            max_length=params['replay_buffer_size'])

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
        env, _ = make_tf_env()
        env.reset()

collect_data(train_env, random_policy, replay_buffer, steps=params['initial_collect_steps'])



def compute_avg_return(environment, policy, num_episodes=5):
    total_return = 0.0

    for i in range(num_episodes):
        if params['atari']:
            time_step = environment.reset()
        else:
            environment, _ = make_tf_env()
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
                            num_steps=2)
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

#Training
print('Beginning Training')

for i in range(params['num_iter']):
    printProgressBar(i, params['num_iter'])
    for _ in range(params['collect_steps_per_iter']):
        collect_step(train_env, agent.collect_policy, replay_buffer)

    exp, info = next(iterator)
    train_loss = agent.train(exp).loss
    step = agent.train_step_counter.numpy()

    if step % params['log_interval'] == 0:
        print("Step: {}, Loss: {}".format(step, train_loss))
    if step % params['eval_interval'] == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, params['num_eval_episodes'])
        print("Step: {}, Current Return: {}, Previous Return: {}".format(step, avg_return, returns[-1]))
        returns.append(avg_return)
        create_video(agent.policy, 'step_'+str(step), params['num_eval_episodes'])

iterations = range(0, params['num_iter'] + 1, params['eval_interval'])
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=250)
plt.savefig('loss.png')

def create_video(policy, filename, num_episodes=5, fps=30):
    filename += '.mp4'
    filename = 'videos/'+filename
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            if params['atari']:
                time_step = environment.reset()
            else:
                environment, eval_py_env = make_tf_env()
                time_step = environment.reset()
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = eval_env.step(action_step.action)
            video.append_data(eval_py_env.render())
