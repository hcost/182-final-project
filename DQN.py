import base64
import imageio
import matplotlib
import matplotlib.pyplot as plt
import procgen

import numpy as np
import PIL.Image
import pyvirtualdisplay
import gym as gym
import tf_agents
import os

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
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
Some code adapted from TensorFlow tf_agents DQN tutorial with CartPole as an example. 
Capable of implementing a DQN, DDQN, or Categorical DQN. 
"""

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


class DQN:
    def __init__(self,
                 learning_rate=1e-3,
                 batch_size=128,
                 fc_layer_params=(256, 11),
                 env_name=env_names[0],
                 num_iterations=20000,
                 DDQN=False,
                 categorical=False,
                 epsilon=0.1,
                 temp=None,
                 gamma=0.99,
                 tut=1,
                 tup=1000,
                 initial_collect_steps=1000,
                 start_level=0,
                 num_levels=500,
                 difficulty='easy',
                 model_dir='/content/drive/My Drive/Colab Notebooks/final/DQN_eval_policies'):

        self.learning_rate = learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        self.fc_layer_params = fc_layer_params
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.initial_collect_steps = initial_collect_steps
        self.env_name = env_name
        self.model_dir = model_dir
        train_py_env = gym.make(self.env_name, start_level=start_level, num_levels=num_levels,
                                distribution_mode=difficulty)
        tpe = suite_gym.wrap_env(train_py_env)
        eval_py_env = gym.make(self.env_name, start_level=start_level, num_levels=num_levels,
                               distribution_mode=difficulty)
        epe = suite_gym.wrap_env(train_py_env)

        self.train_env = tf_py_environment.TFPyEnvironment(tpe)
        self.eval_env = tf_py_environment.TFPyEnvironment(epe)

        if DDQN:
            init_agent = dqn_agent.DdqnAgent
        elif categorical:
            init_agent = categorical_dqn_agent.CategoricalDqnAgent
        else:
            init_agent = dqn_agent.DqnAgent

        obs = self.train_env.reset()

        if not categorical:
            self.q_net = q_network.QNetwork(
                self.train_env.observation_spec(),
                self.train_env.action_spec(),
                fc_layer_params=fc_layer_params)
        else:
            self.q_net = categorical_q_network.CategoricalQNetwork(
                self.train_env.observation_spec(),
                self.train_env.action_spec(),
                num_atoms=51,
                fc_layer_params=fc_layer_params)

        self.batch_size = batch_size

        self.agent = init_agent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            self.q_net,
            epsilon_greedy=epsilon,
            boltzmann_temperature=temp,
            optimizer=self.optimizer,
            target_update_tau=tut,
            gamma=gamma,
            target_update_period=tup,
            td_errors_loss_fn=common.element_wise_huber_loss,
            train_step_counter=tf.Variable(0))

        self.returns = []
        self.losses = []

    def train_eval(self, num_iterations=20000, log_interval=200, eval_interval=1000):
        replay_buffer = tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=100000
        )

        self.eval_policy = self.agent.policy

        observers = [replay_buffer.add_batch]
        random_policy = random_tf_policy.RandomTFPolicy(self.train_env.time_step_spec(), self.train_env.action_spec())
        initial_collect = random_policy
        initial_collect_op = dynamic_step_driver.DynamicStepDriver(self.train_env,
                                                                   initial_collect,
                                                                   observers=observers,
                                                                   num_steps=1000).run()

        collect_policy = self.agent.collect_policy
        collect_op = dynamic_step_driver.DynamicStepDriver(self.train_env,
                                                           collect_policy,
                                                           observers=observers,
                                                           num_steps=2).run()

        dataset = replay_buffer.as_dataset(num_parallel_calls=3,
                                           sample_batch_size=self.batch_size,
                                           num_steps=2).prefetch(3)
        iterator = iter(dataset)
        self.agent.train_step_counter.assign(0)
        self.agent.initialize()

        for _ in range(num_iterations):
            for _ in range(1):
                time_step = self.train_env.current_time_step()
                action_step = self.agent.collect_policy.action(time_step)
                next_time_step = self.train_env.step(action_step.action)
                replay_buffer.add_batch(trajectory.from_transition(time_step, action_step, next_time_step))

            experience, _ = next(iterator)
            loss = self.agent.train(experience).loss
            step = self.agent.train_step_counter.numpy()
            self.losses.append(loss)
            self.train_env.reset()

            if step % log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, loss))

            if step % eval_interval == 0:
                metric = tf_metrics.AverageReturnMetric()
                observers = [metric]
                drive2 = dynamic_step_driver.DynamicStepDriver(self.eval_env, self.agent.policy, observers=observers,
                                                               num_steps=eval_interval)
                drive2.run()
                self.eval_env.reset()
                avg_return = metric.result().numpy()
                self.returns.append(avg_return)
                if step % 1000 == 0:
                    policy_saver.PolicySaver(self.eval_policy).save(self.model_dir)
                    print('Average Return = {}'.format(avg_return))


def main():
    iters = 50000
    eval_int = 500
    log = 250

    # model = DQN(env_name='CartPole-v0', temp=None, tut=1, tup=1000, epsilon=0.2)
    # model = DQN(env_name='CartPole-v0', temp=0.4, tut=1, tup=1000, epsilon=None)
    model = DQN(env_name=env_names[4],
                learning_rate=0.0006,
                temp=None,
                tut=1,
                tup=100,
                epsilon=0.1,
                gamma=0.9999,
                categorical=True)
    model.train_eval(num_iterations=iters, log_interval=log, eval_interval=eval_int)

if __name__ == "__main__":
    main()
