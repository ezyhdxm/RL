import itertools
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import distributions
from infrastructure import pytorch_util as ptu
from infrastructure.pytorch_util import MLP
import numpy as np
from typing import Dict, Any

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, ac_dim, discrete, n_layers, size, learning_rate=1e-4, training=True, nn_baseline=False, **kwargs):
        super().__init__(**kwargs)
        self.obs_dim = obs_dim
        self.ac_dim = ac_dim
        if discrete:
            self.logit_net = MLP(obs_dim, ac_dim, n_layers, size).to(ptu.device)
            self.parameters = self.logit_net.parameters()
        else:
            self.mean_net = MLP(obs_dim, ac_dim, n_layers, size).to(ptu.device)
            self.log_std = nn.Parameter(torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device))
            self.parameters = itertools.chain([self.log_std], self.mean_net.parameters())

        self.training = training
        self.nn_baseline = nn_baseline
        self.optimizer = optim.Adam(itertools.chain([self.log_std], self.mean_net.parameters()), 
                                    lr=learning_rate)
        self.discrete = discrete
    
    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None, :]
        observation = ptu.from_numpy(observation.astype(np.float32))
        ac = self(observation).sample()
        return ptu.to_numpy(ac)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.discrete:
            return distributions.Categorical(logits=self.logit_net(obs))
        else:
            mean = self.mean_net(obs)
            std = torch.exp(self.log_std)
            return distributions.Normal(mean, std)

    def update(self, obs: np.ndarray, acs: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError
    
    def save(self, filepath: str):
        torch.save(self.state_dict(), filepath)

class MLPPolicyPG(MLPPolicy):
    def update(self, obs: np.ndarray, acs: np.ndarray, advantages: np.ndarray, method: str, clip_param: float) -> Dict[str, Any]:
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        advantages = ptu.from_numpy(advantages)

        self.optimizer.zero_grad()
        
        if method == 'ppo':
            distributions = self(obs)
            log_probs_old = distributions.log_prob(acs).detach()
            log_probs = distributions.log_prob(acs)
            ratio = torch.exp(log_probs - log_probs_old)
            clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
            loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        elif method == 'trpo':
            delta = 0.01
            distributions = self(obs)
            
        else:
            raise NotImplementedError