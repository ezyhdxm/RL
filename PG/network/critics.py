import torch
from torch import nn
from torch import optim
from torch import distributions
from torch.nn import functional as F
from infrastructure import pytorch_util as ptu
from infrastructure.pytorch_util import MLP
import numpy as np
from typing import Dict, Any

class ValueCritic(nn.Module):
    def __init__(self, obs_dim, n_layers, size, learning_rate=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.obs_dim = obs_dim
        self.value_net = MLP(obs_dim, 1, n_layers, size).to(ptu.device)
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_net(obs)

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> Dict[str, Any]:
        obs = ptu.from_numpy(obs)
        q_values = ptu.from_numpy(q_values)
        self.optimizer.zero_grad()
        val_preds = self(obs)
        loss = F.mse_loss(val_preds, q_values)
        loss.backward()
        self.optimizer.step()
        return {'Baseline Loss': loss.item()}
    
    def save(self, filepath):
        torch.save(self.state_dict(), filepath)