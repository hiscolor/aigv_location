"""
Role-aware Mixture of Experts (MoE) for PPO state enhancement.

Three specialised experts extract proposal-conditioned evidence from the
backbone feature map, each focusing on a different temporal role:

  - BoundaryExpert   – attends to the left & right transition zones
  - InteriorExpert   – summarises the content *inside* the proposal
  - ContextExpert    – captures the surrounding temporal context

A lightweight router produces soft gating weights conditioned on proposal
geometry, fusing the three expert outputs into a single enhanced state
vector that replaces the naive average-pooled evidence in StateBuilder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Expert(nn.Module):
    """Base 2-layer MLP expert:  C → hidden → out_dim."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class BoundaryExpert(nn.Module):
    """
    Extracts evidence from the two boundary zones of a proposal.

    Given a proposal [l, r] and boundary buffer ratio β, the left boundary
    region is [l - βw, l + βw] and similarly for the right boundary.
    Both regions are average-pooled then concatenated and projected.
    """

    def __init__(self, feat_dim, hidden_dim, out_dim, boundary_ratio=0.15):
        super().__init__()
        self.boundary_ratio = boundary_ratio
        self.proj = _Expert(feat_dim * 2, hidden_dim, out_dim)

    def forward(self, feat_map, l, r, T):
        """
        feat_map : (C, T)
        l, r     : proposal boundaries (float, token coords)
        T        : sequence length
        """
        w = max(r - l, 1.0)
        bw = max(1, int(self.boundary_ratio * w))

        lb_start = int(max(0, round(l) - bw))
        lb_end   = int(min(T, round(l) + bw))
        rb_start = int(max(0, round(r) - bw))
        rb_end   = int(min(T, round(r) + bw))

        e_left  = self._pool(feat_map, lb_start, lb_end)
        e_right = self._pool(feat_map, rb_start, rb_end)
        return self.proj(torch.cat([e_left, e_right], dim=-1))

    @staticmethod
    def _pool(feat, start, end):
        end = max(start + 1, end)
        end = min(end, feat.shape[-1])
        return feat[:, start:end].mean(dim=-1)


class InteriorExpert(nn.Module):
    """Summarises the proposal interior [l, r] via adaptive average pooling."""

    def __init__(self, feat_dim, hidden_dim, out_dim):
        super().__init__()
        self.proj = _Expert(feat_dim, hidden_dim, out_dim)

    def forward(self, feat_map, l, r, T):
        li = int(max(0, round(l)))
        ri = int(min(T, round(r) + 1))
        ri = max(li + 1, ri)
        e_int = feat_map[:, li:ri].mean(dim=-1)
        return self.proj(e_int)


class ContextExpert(nn.Module):
    """
    Captures surrounding context by pooling the regions *outside* the
    proposal within a context window defined by context_ratio.
    """

    def __init__(self, feat_dim, hidden_dim, out_dim, context_ratio=0.5):
        super().__init__()
        self.context_ratio = context_ratio
        self.proj = _Expert(feat_dim * 2, hidden_dim, out_dim)

    def forward(self, feat_map, l, r, T):
        w = max(r - l, 1.0)
        cw = max(2, int(self.context_ratio * w))

        li = int(max(0, round(l)))
        ri = int(min(T, round(r) + 1))

        ctx_l_start = int(max(0, li - cw))
        ctx_r_end   = int(min(T, ri + cw))

        e_ctx_left  = self._pool(feat_map, ctx_l_start, li)
        e_ctx_right = self._pool(feat_map, ri, ctx_r_end)
        return self.proj(torch.cat([e_ctx_left, e_ctx_right], dim=-1))

    @staticmethod
    def _pool(feat, start, end):
        end = max(start + 1, end)
        end = min(end, feat.shape[-1])
        return feat[:, start:end].mean(dim=-1)


class RoleAwareRouter(nn.Module):
    """
    Produces soft gating weights [w_b, w_i, w_c] for the three experts,
    conditioned on the proposal's normalised geometry g = [l/T, r/T, w/T, c/T].
    """

    def __init__(self, geom_dim=4, hidden_dim=32, n_experts=3, temperature=1.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(geom_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_experts),
        )
        self.temperature = temperature

    def forward(self, geom):
        """geom : (4,) or (B, 4)."""
        logits = self.gate(geom)
        return F.softmax(logits / self.temperature, dim=-1)


class RoleAwareMoE(nn.Module):
    """
    Full MoE module that:
      1. Runs each expert on the proposal-conditioned feat_map region
      2. Computes router weights from proposal geometry
      3. Returns the weighted sum as the enhanced evidence vector

    Parameters
    ----------
    feat_dim : int           – backbone feature channel count (C from FPN)
    expert_hidden : int      – hidden dim inside each expert MLP
    expert_out : int         – output dim of each expert (= MoE output dim)
    boundary_ratio : float   – β for BoundaryExpert
    context_ratio : float    – context window ratio for ContextExpert
    router_hidden : int      – hidden dim of the router MLP
    temperature : float      – softmax temperature for router
    """

    def __init__(
        self,
        feat_dim,
        expert_hidden=128,
        expert_out=64,
        boundary_ratio=0.15,
        context_ratio=0.5,
        router_hidden=32,
        temperature=1.0,
    ):
        super().__init__()
        self.boundary_expert = BoundaryExpert(
            feat_dim, expert_hidden, expert_out, boundary_ratio
        )
        self.interior_expert = InteriorExpert(
            feat_dim, expert_hidden, expert_out
        )
        self.context_expert = ContextExpert(
            feat_dim, expert_hidden, expert_out, context_ratio
        )
        self.router = RoleAwareRouter(
            geom_dim=4,
            hidden_dim=router_hidden,
            n_experts=3,
            temperature=temperature,
        )
        self.expert_out = expert_out

    def forward(self, feat_map, l, r, T):
        """
        Parameters
        ----------
        feat_map : Tensor (C, T)
        l, r     : proposal boundaries in token coords
        T        : sequence length

        Returns
        -------
        enhanced : Tensor (expert_out,) — MoE-fused evidence vector
        weights  : Tensor (3,)          — router weights for logging
        """
        e_b = self.boundary_expert(feat_map, l, r, T)
        e_i = self.interior_expert(feat_map, l, r, T)
        e_c = self.context_expert(feat_map, l, r, T)

        w_norm = max(r - l, 1.0)
        geom = torch.tensor(
            [l / T, r / T, w_norm / T, 0.5 * (l + r) / T],
            dtype=feat_map.dtype, device=feat_map.device,
        )
        weights = self.router(geom)  # (3,)

        experts = torch.stack([e_b, e_i, e_c], dim=0)  # (3, expert_out)
        enhanced = (weights.unsqueeze(-1) * experts).sum(dim=0)  # (expert_out,)

        return enhanced, weights
