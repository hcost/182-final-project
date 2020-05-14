import tensorflow as tf
from baselines.ppo2 import ppo2
from baselines.deepq import deepq
from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize,
)
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines import logger
from mpi4py import MPI
import argparse
import os

def main():
    learning_rate = 5e-4
    ent_coef = .01
    gamma = .999
    lam = .95
    nsteps = 256
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    timesteps_per_proc = 1_000_000
    use_vf_clipping = True

    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--distribution_mode', type=str, default='hard', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--num_video_steps', type=int, default=20000)
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'dqn'])

    args = parser.parse_args()

    num_envs = 64 if args.algorithm == 'ppo' else 1
    fname = args.algorithm + '_fruitbot_final'

    test_worker_interval = args.test_worker_interval

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False

    if test_worker_interval > 0:
        is_test_worker = comm.Get_rank() % test_worker_interval == (test_worker_interval - 1)

    mpi_rank_weight = 0 if is_test_worker else 1
    num_levels = 0 if is_test_worker else args.num_levels

    log_comm = comm.Split(1 if is_test_worker else 0, 0)
    format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
    logger.configure(dir=fname, format_strs=format_strs)

    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=num_envs, env_name=args.env_name, num_levels=num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=os.path.join(fname, 'logs'), keep_buf=100,
    )

    venv = VecNormalize(venv=venv, ob=False)
    venv = VecVideoRecorder(venv=venv,
                            directory=os.path.join(fname, 'videos'),
                            record_video_trigger=lambda step: step % args.num_video_steps == 0,
                            video_length=60 * 30)

    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    conv_fn = lambda x: build_impala_cnn(x, depths=[16, 32, 32], emb_size=256)

    logger.info("training")
    if args.algorithm == 'ppo':
        ppo2.learn(
            env=venv,
            network=conv_fn,
            total_timesteps=timesteps_per_proc,
            save_interval=100,
            nsteps=nsteps,
            nminibatches=nminibatches,
            lam=lam,
            gamma=gamma,
            noptepochs=ppo_epochs,
            log_interval=1,
            ent_coef=ent_coef,
            mpi_rank_weight=mpi_rank_weight,
            clip_vf=use_vf_clipping,
            comm=comm,
            lr=learning_rate,
            cliprange=clip_range,
            update_fn=None,
            init_fn=None,
            vf_coef=0.5,
            max_grad_norm=0.5,
            load_path=os.path.join(fname, os.path.join('checkpoints', '01500'))
        )
    elif args.algorithm == 'dqn':
        deepq.learn(
            env=venv,
            network=conv_fn,
            lr=learning_rate,
            total_timesteps=timesteps_per_proc,
            checkpoint_freq=500,
            checkpoint_path=fname,
            print_freq=1000,
        )


if __name__ == '__main__':
    main()
