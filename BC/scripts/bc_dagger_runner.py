import pickle
import os
import time
import gymnasium as gym
import numpy as np
import torch

from BC.infrastructure.logger import Logger
from BC.infrastructure.replay_buffer import ReplayBuffer
from BC.policies.MLP_policy import MLPPolicySL
from BC.policies.loaded_gaussian_policy import LoadedGaussianPolicy
from BC.infrastructure import pytorch_util as ptu
from BC.infrastructure import sample_utils

MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40

MJ_ENV_NAMES = ['Ant-v4', 'HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4']

def run_training_loop(params):
    logger = Logger(params['logdir'])

    seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    ptu.init_gpu(use_gpu=not params['no_gpu'], gpu_id=params['which_gpu'])

    log_video = True
    log_metrics = True


    env = gym.make(params['env_name'], render_mode='rgb_array')
    env.reset(seed=seed)

    params['ep_len'] = params['ep_len'] or env.spec.max_episode_steps
    MAX_VIDEO_LEN = params['ep_len']

    assert isinstance(env.observation_space, gym.spaces.Box), "This script only supports environments with continuous state spaces."
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    if 'model' in dir(env):
        fps = 1/env.model.opt.timestep
    else:
        fps = env.env.metadata['render_fps']
    
    actor = MLPPolicySL(
        obs_dim=ob_dim,
        ac_dim=ac_dim,
        n_layers=params['n_layers'],
        size=params['size'],
        learning_rate=params['learning_rate'],
    )

    replay_buffer = ReplayBuffer(max_size=params['max_replay_buffer_size'])

    print('Loading expert policy from...', params['expert_policy_file'])
    expert_policy = LoadedGaussianPolicy(params['expert_policy_file'])
    expert_policy.to(ptu.device)
    print('Expert policy loaded.')

    total_envsteps = 0
    start_time = time.time()


    for itr in range(params['n_iter']):
        print("\n\n********** Iteration %i ************"%itr)

        log_video = ((itr % params['video_log_freq'] == 0) or (itr == params['n_iter']-1)) and (params['video_log_freq'] != -1)
        log_metrics = ((itr % params['scalar_log_freq'] == 0) or (itr == params['n_iter']-1)) and (params['scalar_log_freq'] != -1)

        print("\nCollecting data to be used for training...")
        if itr == 0:
            paths = pickle.load(open(params['expert_data'], 'rb'))
            envsteps_this_batch = 0

            print('\nCollecting initial video data...')
            eval_video_paths = sample_utils.sample_n_trajectories(env, actor, MAX_NVIDEO, MAX_VIDEO_LEN, render=True)
            if eval_video_paths:
                logger.log_paths_as_videos(eval_video_paths, 0, max_videos_to_save=MAX_NVIDEO, fps=fps, video_title='eval_rollouts')

            print("\nCollecting initial eval metrics...")
            eval_paths, _ = sample_utils.sample_trajectories(env, actor, params['eval_batch_size'], params['ep_len'], render=False)
            logs = sample_utils.compute_metrics(paths, eval_paths)
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            
            for key, value in logs.items():
                print('\t{}: {}'.format(key, value))
                logger.log_scalar(value, key, 0)

            print('Done logging...\n\n')

            logger.flush()
        
        else:
            assert params['do_dagger']

            paths, envsteps_this_batch = sample_utils.sample_trajectories(env, actor, params['batch_size'], params['ep_len'], render=False)

            if params['do_dagger']:
                print("\nRelabelling data using the expert policy...")
                for i in range(len(paths)):
                    obs = paths[i]['observation']
                    new_actions = expert_policy.get_action(obs)
                    paths[i]['action'] = new_actions

        total_envsteps += envsteps_this_batch

        # add data to replay buffer
        replay_buffer.add_rollouts(paths)

        print('\nTraining agent...')
        training_logs = []
        for _ in range(params['num_agent_train_steps_per_iter']):
            indices = np.random.choice(len(replay_buffer), size=params['train_batch_size'], replace=False)
            ob_batch, ac_batch = replay_buffer.obs[indices], replay_buffer.acs[indices]
            training_log = actor.update(ob_batch, ac_batch)
            training_logs.append(training_log)

        print('\nLogging training statistics...')


        # log videos
        if log_video:
            print('\nCollecting video data...')
            eval_video_paths = sample_utils.sample_n_trajectories(env, actor, MAX_NVIDEO, MAX_VIDEO_LEN, render=True)
            if eval_video_paths:
                logger.log_paths_as_videos(eval_video_paths, itr+1, max_videos_to_save=MAX_NVIDEO, fps=fps, video_title='eval_rollouts')
        

        # log metrics
        if log_metrics:
            print("\nCollecting data for eval metrics...")
            eval_paths, eval_envsteps_this_batch = sample_utils.sample_trajectories(env, actor, params['eval_batch_size'], params['ep_len'], render=False)
            logs = sample_utils.compute_metrics(paths, eval_paths)
            logs.update(training_logs[-1]) # training_logs[-1] is the last training log
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs["Train_AverageReturn"]
            
            for key, value in logs.items():
                print('\t{}: {}'.format(key, value))
                logger.log_scalar(value, key, itr+1)

            print('Done logging...\n\n')

            logger.flush()
        
        if params['save_params']:
            print("\nSaving agent...")
            actor.save('{}/policy_itr_{}.pt'.format(params['logdir'], itr))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', '-epf', type=str, required=True)
    parser.add_argument('--expert_data', '-ed', type=str, required=True)
    parser.add_argument('--env_name', '-env', help=f'choices: {", ".join(MJ_ENV_NAMES)}', type=str, required=True)
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)
    parser.add_argument('--do_dagger', action='store_true')
    parser.add_argument('--ep_len', type=int)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)
    parser.add_argument('--n_iter', '-n', type=int, default=1)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--eval_batch_size', '-eb', type=int, default=1000)
    parser.add_argument('--train_batch_size', '-tb', type=int, default=100)

    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)

    parser.add_argument('--video_log_freq', type=int, default=5)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    params = vars(args)

    if args.do_dagger:
        logdir_prefix = 'dagger_'
        assert args.n_iter > 1, "DAgger requires multiple iterations"  
    else:
        logdir_prefix = 'behavior_cloning_'
        assert args.n_iter == 1, "Without DAgger, only one iteration is allowed"

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    run_training_loop(params)

if __name__ == "__main__":
    main()


