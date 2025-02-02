import abc
import itertools
from typing import Dict, Any
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch
import numpy as np
from torch import distributions
from BC.infrastructure import pytorch_util as ptu
from BC.policies.base_policies import BasePolicy

class MLP(nn.Module):
    def __init__(self, input_size:int, output_size:int, n_layers:int, hidden_size:int, activation=nn.ReLU):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.layers = nn.ModuleList()
        in_size = input_size
        for _ in range(n_layers):
            self.layers.append(nn.Linear(in_size, hidden_size))
            self.layers.append(activation())
            in_size = hidden_size
        
        self.layers.append(nn.Linear(in_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, obs_dim, ac_dim, n_layers, size, learning_rate=1e-4, training=True, nn_baseline=False, **kwargs):
        super().__init__(**kwargs)
        self.obs_dim = obs_dim
        self.ac_dim = ac_dim
        self.mean_net = MLP(obs_dim, ac_dim, n_layers, size).to(ptu.device)
        self.training = training
        self.nn_baseline = nn_baseline
        self.log_std = nn.Parameter(torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device))
        self.optimizer = optim.Adam(itertools.chain([self.log_std], self.mean_net.parameters()), 
                                    lr=learning_rate)
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None, :]
        observation = ptu.from_numpy(observation.astype(np.float32))
        with torch.no_grad():
            ac = self(observation).sample()
        return ptu.to_numpy(ac)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std)
        return distributions.Normal(mean, std)

    def update(self, obs: np.ndarray, acs: np.ndarray) -> Dict[str, Any]:
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)

        self.optimizer.zero_grad()
        mean = self.mean_net(obs)
        normed_acs = (acs - mean) / torch.exp(self.log_std)
        loss = torch.sum(0.5 * normed_acs.pow(2) + self.log_std[None, :], dim=-1).mean()
        loss.backward()
        self.optimizer.step()

        return {'Training Loss': loss.item()}
    
    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
    
