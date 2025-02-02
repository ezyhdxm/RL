from collections import OrderedDict
import cv2
import numpy as np
import time
from BC.infrastructure import pytorch_util as ptu

def sample_trajectory(env, policy, max_path_length, render=False):
    """
    Collects samples until we have exactly one batch worth of data
    """
    ob, _ = env.reset()
    cur_path_length = 0
    obs, acs, rewards, next_obs, terminals, img_obs = [], [], [], [], [], []
    while True:
        if render:
            if hasattr(env, 'sim'):
                img = env.sim.render(camera_name='track', height=500, width=500)[::-1]
            else:
                img = env.render()
            img_obs.append(cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC))
        
        ac = policy.get_action(ob).squeeze()
        next_ob, reward, done, _, _ = env.step(ac)

        cur_path_length += 1

        is_terminated = 1 if done or cur_path_length >= max_path_length else 0

        obs.append(ob)
        acs.append(ac)
        rewards.append(reward)
        next_obs.append(next_ob)
        terminals.append(is_terminated)

        
        if is_terminated:
            break

        ob = next_ob

    return dict(
        observation=np.array(obs, dtype=np.float32),
        action=np.array(acs, dtype=np.float32),
        reward=np.array(rewards, dtype=np.float32),
        next_observation=np.array(next_obs, dtype=np.float32),
        terminal=np.array(terminals, dtype=np.float32),
        image_obs=np.array(img_obs, dtype=np.uint8),
    )

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False):
    """
    Sample multiple trajectories using the current policy.
    """
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)
        timesteps_this_batch += len(path['reward'])
    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False):
    """
    Sample multiple trajectories using the current policy.
    """
    paths = []
    for _ in range(ntraj):
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)
    return paths

def convert_listofrollouts(rollouts, concat_rew=True):
    """
    Take a list of rollouts (dictionaries) and return separate arrays,
    where each array is a concatenation of that array from the rollouts.
    """
    obs = np.concatenate([path["observation"] for path in rollouts])
    acs = np.concatenate([path["action"] for path in rollouts])
    rewards = np.concatenate([path["reward"] for path in rollouts])
    next_obs = np.concatenate([path["next_observation"] for path in rollouts])
    terminals = np.concatenate([path["terminal"] for path in rollouts])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in rollouts])
    else:
        rewards = [path["reward"] for path in rollouts]
    return obs, acs, rewards, next_obs, terminals


def compute_metrics(paths, eval_paths):
    train_results = [path["reward"].sum() for path in paths]
    eval_results = [path["reward"].sum() for path in eval_paths]

    train_ep_lens = [len(path["reward"]) for path in paths]
    eval_ep_lens = [len(path["reward"]) for path in eval_paths]

    logs = OrderedDict()
    logs["Eval_AverageReturn"] = np.mean(eval_results)
    logs["Eval_StdReturn"] = np.std(eval_results)
    logs["Eval_MaxReturn"] = np.max(eval_results)
    logs["Eval_MinReturn"] = np.min(eval_results)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    logs["Train_AverageReturn"] = np.mean(train_results)
    logs["Train_StdReturn"] = np.std(train_results)
    logs["Train_MaxReturn"] = np.max(train_results)
    logs["Train_MinReturn"] = np.min(train_results)
    logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

    return logs