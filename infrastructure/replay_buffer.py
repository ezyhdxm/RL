from BC.infrastructure.sample_utils import *
import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size=int(1e6)):
        self.max_size = max_size # keep track of the most recent max_size transitions
        self.paths = []
        self.obs, self.acs, self.rews, self.next_obs, self.terminals = None, None, None, None, None
    
    def __len__(self):
        if self.obs is not None:
            return len(self.obs)
        return 0
    
    def add_rollouts(self, paths, concat_rew=True):
        for path in paths:
            self.paths.append(path)

        obs, acs, rews, next_obs, terminals = convert_listofrollouts(paths, concat_rew) # convert list of rollouts to separate arrays

        if self.obs is None:
            self.obs = obs[-self.max_size:]
            self.acs = acs[-self.max_size:]
            self.rews = rews[-self.max_size:]
            self.next_obs = next_obs[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, obs])[-self.max_size:]
            self.acs = np.concatenate([self.acs, acs])[-self.max_size:]
            self.next_obs = np.concatenate([self.next_obs, next_obs])[-self.max_size:]
            self.terminals = np.concatenate([self.terminals, terminals])[-self.max_size:]
            if concat_rew:
                self.rews = np.concatenate([self.rews, rews])[-self.max_size:]
            else:
                if isinstance(rews, list):
                    self.rews += rews
                else:
                    self.rews.append(rews)
                self.rews = self.rews[-self.max_size:]
            

    