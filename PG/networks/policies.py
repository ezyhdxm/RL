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
        self.optimizer = optim.Adam(self.parameters, lr=learning_rate)
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
        
        elif method=="trpo":
            delta = 0.01
            distribution = self(obs)
            if self.discrete:
                distribution_detached = torch.distributions.Categorical(logits=distribution.logits.detach())
            else:
                detached_loc = distribution.loc.detach()
                detached_covariance_matrix = distribution.covariance_matrix.detach()
                distribution_detached = torch.distributions.MultivariateNormal(detached_loc, detached_covariance_matrix)
            probs = torch.exp(distribution.log_prob(actions))
            loss = surrogate_loss(probs, probs.detach(), advantages).mean()
            #kl_reg = KL(distribution, distribution)
            #params = list(self.parameters)
            if self.discrete:
                params = list(self.logits_net.parameters())
            else:
                params = list(itertools.chain([self.logstd], self.mean_net.parameters()))
            grad_loss = torch.autograd.grad(loss, params, retain_graph=True)
            grad_loss = torch.cat([grad.view(-1) for grad in grad_loss])
            #grad_kl = flat_grad(kl_reg, params, create_graph=True)
            kl = torch.distributions.kl.kl_divergence(distribution, distribution_detached).mean()
            grad_kl = torch.autograd.grad(kl, params, create_graph=True)
            grad_kl = torch.cat([grad.view(-1) for grad in grad_kl])

            #def HVP(v):
            #    kl_v = torch.dot(grad_kl, v)
            #    kl_hessian_vector = torch.autograd.grad(kl_v, params)
            #    return torch.cat([h.view(-1) for h in kl_hessian_vector]).detach()

            def HVP(v):
                kl_v = torch.dot(grad_kl, v)
                #print(kl_v)
                return flat_grad(kl_v, params, retain_graph=True)
            
            search_direction = conjugate_gradient(HVP, grad_loss)
            #print(search_direction)
            max_length = torch.sqrt(2 * delta / (search_direction @ HVP(search_direction)))
            max_step = max_length * search_direction

            def apply_update(grad_flattened):
                n = 0
                for p in params:
                    numel = p.numel()
                    g = grad_flattened[n:n + numel].view(p.shape)
                    p.data += g
                    n += numel

            def line_search(step):
                apply_update(step)
                with torch.no_grad():
                    distribution_new = self(obs)
                    probs_new = torch.exp(distribution_new.log_prob(actions))
                    loss_new = surrogate_loss(probs_new, probs, advantages)
                    kl_new = torch.distributions.kl.kl_divergence(distribution, distribution_new).mean()
                #print(loss_new - loss)
                #print(kl_new)
                if (loss_new - loss >= 0) and (kl_new <= delta):
                    return True

                apply_update(-step)
                return False

            i = 0
            while not line_search((0.9**i) * max_step) and i < 10:
                i += 1

        else:
            ll = self(obs).log_prob(actions)
            loss = torch.neg(torch.mean(torch.mul(ll, advantages)))
            loss.backward()
            self.optimizer.step()

        return {"Actor Loss": ptu.to_numpy(loss)}


def conjugate_gradient(A, b, delta=0., max_iters=10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    r_dot_r = torch.dot(r, r)

    i = 0
    while i < max_iters:
        AVP = A(p)
        #print(AVP)
        alpha = r_dot_r / torch.dot(p, AVP)
        x_new = x + alpha*p
        #print(x_new)

        if (x-x_new).norm() <= delta:
            return x_new
        
        i += 1
        r -= alpha * AVP
        r_dot_r_new = torch.dot(r, r) 
        beta = r_dot_r_new / r_dot_r
        p = r + beta * p
        x = x_new
        r_dot_r = r_dot_r_new
    return x

def surrogate_loss(new_probs, old_probs, advantages):
    return (new_probs / old_probs * advantages)

def ppo_surrogate_loss(new_probs, old_probs, advantages, clip_param):
    ratio = new_probs / old_probs
    return (torch.clamp(ratio, 1-clip_param, 1+clip_param) * advantages)

def flat_grad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True
    
    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g