"""
PPO Trainer for proposal refinement.

Handles:
  - rollout collection from ProposalRefineEnv
  - GAE advantage estimation
  - PPO clipped surrogate updates
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict

from .environment import ProposalRefineEnv, NUM_ACTIONS
from .state_builder import StateBuilder
from .agent import PPOAgent


class RolloutBuffer:
    """Buffer that stores one episode's transitions + raw state info for MoE recompute."""

    def __init__(self):
        self.states: List[torch.Tensor] = []
        self.raw_states: List[dict] = []
        self.actions: List[int] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[bool] = []

    def store(self, state, action, log_prob, reward, value, done, raw_state=None):
        self.states.append(state)
        self.raw_states.append(raw_state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalised Advantage Estimation → (advantages, returns)."""
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else 0.0
        if isinstance(next_val, torch.Tensor):
            next_val = next_val.item()
        val = values[t].item() if isinstance(values[t], torch.Tensor) else values[t]
        delta = rewards[t] + gamma * next_val * (1 - int(dones[t])) - val
        last_gae = delta + gamma * lam * (1 - int(dones[t])) * last_gae
        advantages[t] = last_gae
    returns = advantages + torch.tensor([
        v.item() if isinstance(v, torch.Tensor) else v for v in values
    ])
    return advantages, returns


class PPOTrainer:
    """
    Parameters
    ----------
    agent : PPOAgent
    state_builder : StateBuilder
    lr : float
    ppo_epochs : int          – number of optimisation passes per batch
    clip_eps : float          – PPO clip range
    vf_coef : float           – value loss coefficient
    ent_coef : float          – entropy bonus coefficient
    gamma, gae_lam : float    – discount and GAE lambda
    max_grad_norm : float
    """

    def __init__(
        self,
        agent: PPOAgent,
        state_builder: StateBuilder,
        lr=3e-4,
        ppo_epochs=4,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        gamma=0.99,
        gae_lam=0.95,
        max_grad_norm=0.5,
        device="cuda",
    ):
        self.agent = agent.to(device)
        self.sb = state_builder
        self.device = device

        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
        self.ppo_epochs = ppo_epochs
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.gamma = gamma
        self.gae_lam = gae_lam
        self.max_grad_norm = max_grad_norm

    # ── rollout ─────────────────────────────────────────────────────
    @torch.no_grad()
    def collect_rollout(self, env: ProposalRefineEnv) -> Dict:
        """Run one full episode, return a RolloutBuffer + episode info."""
        buf = RolloutBuffer()
        raw_state = env.reset()
        done = False

        while not done:
            state_t = self.sb.build(raw_state, device=self.device)
            action, log_prob, value = self.agent.get_action(state_t)
            raw_next, reward, done, info = env.step(action)
            # store raw_state snapshot for MoE gradient recompute
            raw_snap = {k: v for k, v in raw_state.items()} if self.sb.use_moe else None
            buf.store(state_t, action, log_prob.detach(), reward,
                      value.detach(), done, raw_state=raw_snap)
            raw_state = raw_next

        return {
            "buffer": buf,
            "final_tiou": info["tiou"],
            "final_be": info["be"],
            "final_l": info["l"],
            "final_r": info["r"],
            "episode_len": len(buf),
            "total_reward": sum(buf.rewards),
        }

    # ── PPO update ──────────────────────────────────────────────────
    def _rebuild_states_with_grad(self, all_raw_states):
        """Re-run StateBuilder.build with gradients for MoE parameters."""
        states = []
        for raw in all_raw_states:
            s = self.sb.build(raw, device=self.device)
            states.append(s)
        return torch.stack(states)

    def update(self, buffers: List[RolloutBuffer]):
        """Run PPO optimisation on a batch of collected rollouts."""
        all_states, all_raw_states, all_actions, all_old_logp = [], [], [], []
        all_advantages, all_returns = [], []

        for buf in buffers:
            adv, ret = compute_gae(
                buf.rewards, buf.values, buf.dones,
                self.gamma, self.gae_lam,
            )
            all_states.extend(buf.states)
            all_raw_states.extend(buf.raw_states)
            all_actions.extend(buf.actions)
            all_old_logp.extend([lp for lp in buf.log_probs])
            all_advantages.append(adv)
            all_returns.append(ret)

        actions = torch.tensor(all_actions, dtype=torch.long, device=self.device)
        old_logp = torch.stack(all_old_logp).to(self.device)
        advantages = torch.cat(all_advantages).to(self.device)
        returns = torch.cat(all_returns).to(self.device)

        # normalise advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # pre-stack detached states for non-MoE path
        states_detached = torch.stack(all_states).to(self.device)

        # collect all params for grad clipping (agent + MoE if present)
        all_params = list(self.agent.parameters())
        if self.sb.use_moe and self.sb.moe is not None:
            all_params += list(self.sb.moe.parameters())

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        for _ in range(self.ppo_epochs):
            if self.sb.use_moe:
                states = self._rebuild_states_with_grad(all_raw_states)
            else:
                states = states_detached

            new_logp, values, entropy = self.agent.evaluate_actions(states, actions)
            ratio = (new_logp - old_logp).exp()

            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(values, returns)
            entropy_bonus = entropy.mean()

            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
            self.optimizer.step()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy_bonus.item()

        n = self.ppo_epochs
        stats = {k: v / n for k, v in stats.items()}
        return stats
