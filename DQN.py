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
from procgen import ProcgenEnv

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
                 batch_size=64,
                 fc_layer_params=(100,),
                 env_name=env_names[0],
                 num_iterations=20000,
                 DDQN=False,
                 epsilon=0.1,
                 initial_collect_steps=1000):
        self.learning_rate = learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        self.fc_layer_params = fc_layer_params
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.initial_collect_steps = initial_collect_steps

        self.env_name = env_name
        train_py_env = ProcgenEnv(num_envs=1, env_name=self.env_name)
        eval_py_env = ProcgenEnv(num_envs=1, env_name=self.env_name)
        self.train_env = tf_py_environment.TFPyEnvironment(train_py_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

        root_dir = os.path.expanduser('/content/drive/My Drive/Colab Notebooks/final')
        self.train_dir = os.path.join(root_dir, 'train')

        if DDQN:
            init_agent = dqn_agent.DdqnAgent
        else:
            init_agent = dqn_agent.DqnAgent

        self.q_net = q_network.QNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            fc_layer_params=fc_layer_params)

        self.batch_size = batch_size

        self.agent = init_agent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            q_network=self.q_net,
            epsilon_greedy=None,
            boltzmann_temperature=0.4,
            optimizer=self.optimizer,
            target_update_tau=1,
            target_update_period=1000,
            td_errors_loss_fn=common.element_wise_huber_loss,
            train_step_counter=tf.Variable(0))

        self.returns = []

    def train_eval(self, num_iterations=20000, log_interval=200, eval_interval=1000):
        global_step = tf.compat.v1.train.get_or_create_global_step()

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

        # iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        iterator = iter(dataset)

        self.agent.train = common.function(self.agent.train)
        self.agent.train_step_counter.assign(0)

        experience, _ = iterator.get_next()
        # train_op = common.function(self.agent.train)(experience=experience)

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

            if step % log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, loss))

            if step % eval_interval == 0:
                metric = tf_metrics.AverageReturnMetric()
                observers = [metric]
                drive2 = dynamic_step_driver.DynamicStepDriver(self.eval_env, self.agent.policy, observers=observers,
                                                               num_steps=eval_interval)
                drive2.run()

                # avg_return = compute_avg_return(self.eval_env, self.eval_policy)
                avg_return = metric.result().numpy()
                print('Average Return = {}'.format(avg_return))
                self.returns.append(avg_return)


def main():
    iters = 20000
    eval_int = 1000
    log = 200

    model = DQN(env_name='CartPole-v0')
    model.train_eval(num_iterations=iters, log_interval=log, eval_interval=eval_int)

    iterations = range(0, iters, eval_int)
    plt.plot(iterations, model.returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.ylim(top=250)


if __name__ == "__main__":
    main()
