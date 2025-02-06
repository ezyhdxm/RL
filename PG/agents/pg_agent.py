from typing import Optional, Sequence
import numpy as np
import torch

from PG.networks.policies import MLPPolicyPG
from PG.networks.critics import ValueCritic
from infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module): 
    def __init__(
        self, ob_dim: int, ac_dim: int, discrete: bool, n_layers: int, size: int,
        gamma: float, learning_rate: float, use_baseline: bool, use_reward_to_go: bool,
        baseline_learning_rate: Optional[float], baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float], normalize_advantages: bool, method: str, clip_param: float,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(ob_dim, ac_dim, discrete, n_layers, size, learning_rate)

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(ob_dim, n_layers, size, baseline_learning_rate)
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.method = method
        self.clip_param = clip_param

    def update(
        self, obs: Sequence[np.ndarray], acs: Sequence[np.ndarray], rews: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray], method: str, clip_param: float,
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rews)

        # flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.
        obs = np.concatenate(obs, axis=0)
        acs = np.concatenate(acs, axis=0)
        rews = np.concatenate(rews, axis=0)
        terminals = np.concatenate(terminals, axis=0)
        q_values = np.concatenate(q_values, axis=0)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(obs, rews, q_values, terminals)

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # update the PG actor/policy network once using the advantages
        info: dict = self.actor.update(obs, acs, advantages, method, clip_param)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # perform `self.baseline_gradient_steps` updates to the critic/baseline network
            for _ in range(self.baseline_gradient_steps-1):
                self.critic.update(obs, q_values)
            critic_info: dict = self.critic.update(obs, q_values)
            info.update(critic_info)

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""
        # rewards: (B, T) where B is the batch size, T is the length of the trajectory (not necessarily the same for all trajectories)
        
        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point. In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            q_values = [self._discounted_return(reward) for reward in rewards]
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            q_values = [self._discounted_reward_to_go(reward) for reward in rewards]

        return q_values

    def _estimate_advantage(self, obs: np.ndarray, rewards: np.ndarray, q_values: np.ndarray, terminals: np.ndarray) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            advantages = q_values.copy()
        else:
            values = self.critic(ptu.from_numpy(obs))
            values = ptu.to_numpy(values.squeeze())
            assert values.shape == q_values.shape

            if self.gae_lambda == -1 or self.gae_lambda is None: # if GAE is disabled
                advantages = q_values - values
            else: # if GAE is enabled
                batch_size = obs.shape[0]

                # append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    # recursively compute advantage estimates starting from timestep T.
                    # use terminals to handle edge cases. terminals[i] is 1 if the state is the last in its
                    # trajectory, and 0 otherwise.
                    if terminals[i] == True:
                        delta = rewards[i] - values[i]
                    else:
                        delta = rewards[i] + self.gamma*values[i+1] - values[i]
                    advantages[i] = delta + self.gamma*self.gae_lambda*advantages[i+1]

                # remove dummy advantage
                advantages = advantages[:-1]

        # normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            adv_mean = advantages.mean()
            adv_std = advantages.std()+1e-9
            advantages = (advantages - adv_mean) / adv_std

        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        gamma_powers = np.array([self.gamma**t for t in range(len(rewards))])
        discounted_sum = np.dot(np.array(rewards), gamma_powers)
        return [discounted_sum]*len(rewards)


    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        discounted_returns = []
        discounted_sum = 0
        for reward in rewards[::-1]:
            discounted_sum = reward + discounted_sum*self.gamma
            discounted_returns.insert(0, discounted_sum)
        return discounted_returns
