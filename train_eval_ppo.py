# train_eval_ppo.py
# Copyright (c) 2020 Daniel Grimshaw (danielgrimshaw@berkeley.edu)
#

import os
import time
import typing
from datetime import datetime

import tensorflow as tf
from absl import app
from absl import logging
from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics as tfm
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.networks import actor_distribution_rnn_network as adrn
from tf_agents.networks import value_rnn_network as vrn

import procgen_environment
import util


def create_ppo_networks(tf_env: tf_py_environment.TFPyEnvironment) -> \
    typing.Tuple[adrn.ActorDistributionRnnNetwork,
                 vrn.ValueRnnNetwork]:
  actor_net = adrn.ActorDistributionRnnNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    conv_layer_params=[(16, 8, 4), (32, 4, 2)],
    input_fc_layer_params=(256,),
    lstm_size=(256,),
    output_fc_layer_params=(128,),
    activation_fn=tf.nn.elu
  )
  value_net = vrn.ValueRnnNetwork(tf_env.observation_spec(),
                                  conv_layer_params=[(16, 8, 4), (32, 4, 2)],
                                  input_fc_layer_params=(256,),
                                  lstm_size=(256,),
                                  output_fc_layer_params=(128,),
                                  activation_fn=tf.nn.elu)
  return actor_net, value_net


def train_eval(root_dir,
               env_name='procgen:procgen-fruitbot-v0',
               num_levels=0,
               num_environment_steps=50000,
               collect_episodes_per_iter=10,
               n_parallel_envs=1,
               replay_buffer_capacity=100000,
               num_epochs=50,
               learning_rate=5e-4,
               num_eval_episodes=40,
               eval_interval=500,
               num_video_episodes=20,
               video_interval=10000,
               checkpoint_interval=500,
               log_interval=50,
               summary_interval=50,
               summary_flush_secs=1,
               use_tf_funcs=True):
  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')
  saved_model_dir = os.path.join(root_dir, 'policy_models')

  for dir_name in (root_dir, train_dir, eval_dir, saved_model_dir):
    if not os.path.exists(dir_name):
      os.makedirs(dir_name)

  train_summary_writer = tf.summary.create_file_writer(train_dir,
                                                       flush_millis=1000 * summary_flush_secs)
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.summary.create_file_writer(eval_dir,
                                                      flush_millis=1000 * summary_flush_secs)
  eval_metrics = [tfm.AverageReturnMetric(buffer_size=replay_buffer_capacity),
                  tfm.AverageEpisodeLengthMetric(buffer_size=replay_buffer_capacity)]
  global_step = tf.compat.v1.train.get_or_create_global_step()

  with tf.summary.record_if(lambda: tf.math.equal(global_step % summary_interval, 0)):
    start = time.time()
    eval_py_env = procgen_environment.ProcgenEnvironment(env_name, distribution_mode='easy', num_levels=num_levels)

    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    tf_env = tf_py_environment.TFPyEnvironment(
      parallel_py_environment.ParallelPyEnvironment(
        [lambda: procgen_environment.ProcgenEnvironment(env_name, distribution_mode='easy',
                                                        num_levels=num_levels)] * n_parallel_envs)
      if n_parallel_envs != 1 else procgen_environment.ProcgenEnvironment(env_name, distribution_mode='easy',
                                                                          num_levels=num_levels))
    logging.info('Took %s to start environments', time.time() - start)
    optim = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)

    actor_net, value_net = create_ppo_networks(tf_env)

    agent = ppo_agent.PPOAgent(tf_env.time_step_spec(),
                               tf_env.action_spec(),
                               optim,
                               actor_net=actor_net,
                               value_net=value_net,
                               num_epochs=num_epochs,
                               gradient_clipping=0.5,
                               entropy_regularization=1e-2,
                               importance_ratio_clipping=0.2,
                               use_gae=True,
                               use_td_lambda_return=True)
    agent.initialize()

    environment_step_metric = tfm.EnvironmentSteps()
    step_metrics = [tfm.NumberOfEpisodes(), environment_step_metric]
    train_metrics = step_metrics + [tfm.AverageReturnMetric(batch_size=n_parallel_envs),
                                    tfm.AverageEpisodeLengthMetric(batch_size=n_parallel_envs)]

    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      agent.collect_data_spec,
      batch_size=n_parallel_envs,
      max_length=replay_buffer_capacity
    )

    train_checkpointer = common.Checkpointer(ckpt_dir=train_dir,
                                             agent=agent,
                                             global_step=global_step,
                                             metrics=metric_utils.MetricsGroup(train_metrics,
                                                                               'train_metrics'))
    policy_checkpointer = common.Checkpointer(ckpt_dir=os.path.join(train_dir, 'policy'),
                                              policy=eval_policy,
                                              global_step=global_step)
    saved_model = policy_saver.PolicySaver(eval_policy, train_step=global_step)
    train_checkpointer.initialize_or_restore()

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
      tf_env,
      collect_policy,
      observers=[replay_buffer.add_batch] + train_metrics,
      num_episodes=collect_episodes_per_iter
    )

    def _step():
      trajectories = replay_buffer.gather_all()
      return agent.train(experience=trajectories)

    def _eval():
      logging.debug('eval')
      logging.debug('\tComputer metrics')
      logging.debug('\tusing tf_funcs? %s', use_tf_funcs)
      metric_utils.eager_compute(eval_metrics,
                                 eval_tf_env,
                                 eval_policy,
                                 num_eval_episodes,
                                 global_step,
                                 eval_summary_writer,
                                 'Metrics',
                                 use_function=use_tf_funcs)

    def _video():
      logging.debug('\tCreating video')
      util.create_video(eval_py_env,
                        eval_tf_env,
                        eval_policy,
                        num_episodes=num_video_episodes,
                        video_filename=os.path.join(eval_dir, 'video_%d.mp4' % global_step_val))

    if use_tf_funcs:
      collect_driver.run = common.function(collect_driver.run, autograph=True)
      agent.train = common.function(agent.train, autograph=True)
      _step = common.function(_step)

    collect_time = 0
    train_time = 0
    timed_at_step = global_step.numpy()

    while environment_step_metric.result() < num_environment_steps:
      start_time = time.time()
      collect_driver.run()
      collect_time += time.time() - start_time

      start_time = time.time()
      total_loss, _ = _step()
      replay_buffer.clear()
      train_time += time.time() - start_time

      for train_metric in train_metrics:
        train_metric.tf_summaries(train_step=global_step, step_metrics=step_metrics)

      global_step_val = global_step.numpy()

      if global_step_val % log_interval == 0:
        logging.info('step %d: loss %f', global_step_val, total_loss)
        steps_per_sec = (global_step_val - timed_at_step) / (collect_time + train_time)
        logging.info('%.3f steps per second', steps_per_sec)
        logging.info('collect time: %f; train_time: %f', collect_time, train_time)

        with tf.summary.record_if(True):
          tf.summary.scalar(name='global_steps_per_sec', data=steps_per_sec, step=global_step)

        timed_at_step = global_step_val
        collect_time = 0
        train_time = 0

      if global_step_val % eval_interval == 0:
        _eval()

      if global_step_val % video_interval == 0:
        _video()

      if global_step_val % checkpoint_interval == 0:
        train_checkpointer.save(global_step=global_step_val)
        policy_checkpointer.save(global_step=global_step_val)
        saved_model.save(os.path.join(saved_model_dir, 'policy_' + str(global_step_val).zfill(9)))

    # final eval
    _eval()


def main(_):
  logging.set_verbosity(logging.INFO)
  train_eval(root_dir='ppo_fruitbot_%s' % datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


if __name__ == '__main__':
  app.run(main)
