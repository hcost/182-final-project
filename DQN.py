import base64
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import gym as gym
import tf_agents

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import parallel_py_environment as ppenv
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

"""
A great deal of this code comes from the DQN tutorial using the tf_agents library. I have made adjustments where 
necessary, but credit must be attributed to: https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
That being said, this still does not work very well yet.

The adjustments I have made include initialization of a double dqn instead of just a regular dqn, and use of the 
tf_agents drivers for collecting data and adding to the replay buffer. More adjustments will be made as soon as we
get the code working and are actually able to start tweaking it.
"""

### HYPERPARAMETERS ###
parallel_train = 4
learning_rate = 0.0001
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

fc_layer_params = (100,)
batch_size = 128
num_iterations = 20000
collect_per_iteration = 1

eval_interval = 1000
num_eval_episodes = 10
log_interval = 200

env_names = ["procgen:procgen-coinrun-v0",
             'procgen:procgen-starpilot-v0',
             'procgen:procgen-caveflyer-v0',
             'procgen:procgen-dodgeball-v0',
             'procgen:procgen-fruitbot-v0',
             'procgen:procgen-chaser-v0',
             'procgen:procgen-miner-v0',
             'procgen:procgen-jumper-v0',
             'procgen:procgen-leaper-v0',
             'procgen:procgen-maze-v0',
             'procgen:procgen-bigfish-v0',
             'procgen:procgen-heist-v0',
             'procgen:procgen-climber-v0',
             'procgen:procgen-plunder-v0',
             'procgen:procgen-ninja-v0',
             'procgen:procgen-bossfight-v0']

env_name = 'procgen:procgen-coinrun-v0'
env = gym.make(env_name)
obs = env.reset()

train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
train_step_counter = tf.Variable(0)

q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

agent = dqn_agent.DdqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    epsilon_greedy=0.15,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_huber_loss,
    train_step_counter=train_step_counter)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

buffer = tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=100000
)

dataset = buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

metric = tf_metrics.AverageReturnMetric()

num_episodes = tf_metrics.NumberOfEpisodes()
env_steps = tf_metrics.EnvironmentSteps()

observers = [buffer.add_batch]
d = dynamic_episode_driver.DynamicEpisodeDriver(eval_env, collect_policy, observers=observers, num_episodes=1)
d.run()
"""
print('final_time_step', final_time_step)
print('Number of Steps: ', env_steps.result().numpy())
print('Number of Episodes: ', num_episodes.result().numpy())
print('Average Return: ', metric.result().numpy())
"""
returns = []
dataset = buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

iterator = iter(dataset)

agent.train_step_counter.assign(0)
for _ in range(num_iterations):
    experience, _ = next(iterator)
    loss = agent.train(experience=experience).loss
    step = agent.train_step_counter.numpy()

    # collect_step(train_env, collect_policy, buffer)
    for _ in range(collect_per_iteration):
        time_step = train_env.current_time_step()
        action_step = agent.collect_policy.action(time_step)
        next_time_step = train_env.step(action_step.action)
        buffer.add_batch(trajectory.from_transition(time_step, action_step, next_time_step))

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, loss))

    if step % eval_interval == 0:
        observers = [metric]
        drive2 = dynamic_episode_driver.DynamicEpisodeDriver(eval_env, agent.policy, observers=observers,
                                                             num_episodes=10)
        drive2.run()

        avg_return = metric.result().numpy()
        # avg_return = compute_avg_return(eval_env, agent.policy)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)