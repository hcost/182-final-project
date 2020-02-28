import imageio

from absl import logging
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.policies import tf_py_policy


def create_video(py_env: py_environment.PyEnvironment,
                 tf_env: tf_environment.TFEnvironment,
                 policy: tf_py_policy.TFPyPolicy,
                 num_episodes=10,
                 max_episode_length=60*30,
                 video_filename='eval_video.mp4'):
  logging.info('Generating video %s', video_filename)
  py_env.reset()
  with imageio.get_writer(video_filename, fps=60) as vid:
    for episode in range(num_episodes):
      logging.info('\tEpisode %s of %s', episode + 1, num_episodes)

      frames = 0
      time_step = tf_env.reset()
      py_env.reset()
      state = policy.get_initial_state(tf_env.batch_size)

      vid.append_data(py_env.render(mode='rgb_array'))
      while not time_step.is_last() and frames < max_episode_length:
        if frames % 60 == 0:
          logging.info('Frame %s of %s', frames, max_episode_length)
        policy_step = policy.action(time_step, state)
        state = policy_step.state
        time_step = tf_env.step(policy_step.action)
        py_env.step(policy_step.action)
        vid.append_data(py_env.render(mode='rgb_array'))
        frames += 1
      py_env.close()
  logging.info('Finished rendering video %s', video_filename)

