"""
PPO Agent: Policy and Value networks for proposal refinement.

Both heads share a small MLP trunk, then branch into:
  - policy head  → Categorical(NUM_ACTIONS)
  - value  head  → scalar V(s)
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .environment import NUM_ACTIONS


class PPOAgent(nn.Module):
    """
    Parameters
    ----------
    state_dim : int
        Dimensionality of the flattened state vector.
    hidden_dim : int
        Width of the shared MLP trunk.
    n_layers : int
        Number of hidden layers in the trunk.
    """

    def __init__(self, state_dim, hidden_dim=256, n_layers=2):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        self.policy_head = nn.Linear(hidden_dim, NUM_ACTIONS)
        self.value_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01
                                    if m is self.policy_head else 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        """
        Parameters
        ----------
        state : Tensor (B, state_dim) or (state_dim,)

        Returns
        -------
        dist   : Categorical distribution over actions
        value  : Tensor (B, 1) or (1,)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        h = self.trunk(state)
        logits = self.policy_head(h)
        value = self.value_head(h)
        dist = Categorical(logits=logits)
        return dist, value

    def get_action(self, state, deterministic=False):
        """Sample an action and return (action, log_prob, value)."""
        dist, value = self.forward(state)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value.squeeze(-1)

    def evaluate_actions(self, states, actions):
        """
        Re-evaluate stored (state, action) pairs for PPO loss.

        Parameters
        ----------
        states  : Tensor (B, state_dim)
        actions : Tensor (B,) long

        Returns
        -------
        log_probs : (B,)
        values    : (B,)
        entropy   : (B,)
        """
        dist, values = self.forward(states)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy
