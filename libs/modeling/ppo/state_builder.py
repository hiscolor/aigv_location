"""
State Builder for PPO Proposal Refinement.

Converts the raw env state dict into a fixed-size tensor that the
policy / value networks can consume.

When use_moe=False (default, ablation-friendly):
  s_t = [ g_t | u_t | v_t | a_{t-1}_onehot | m_t ]
  where v_t (4C) – naive average-pooled evidence

When use_moe=True:
  s_t = [ g_t | u_t | v_moe_t | a_{t-1}_onehot | m_t ]
  where v_moe_t (expert_out) – MoE-fused evidence from 3 role-aware experts

  g_t  (4)   – normalised proposal geometry
  u_t  (4)   – coarse score-map statistics
  a    (A+1) – one-hot previous action (A actions + 1 sentinel for t=0)
  m_t  (1)   – step progress
"""

import torch
import torch.nn.functional as F

from .environment import NUM_ACTIONS


class StateBuilder:
    """
    Call ``build(raw_state)`` to get a 1-D state tensor.

    Parameters
    ----------
    context_ratio : float
        Fraction of proposal width used as left / right context window.
    boundary_ratio : float
        Fraction of proposal width used as boundary buffer.
    moe_module : RoleAwareMoE | None
        If provided and use_moe=True, replaces naive pooled evidence with
        the MoE-fused vector.
    use_moe : bool
        Config switch — when False the MoE module is bypassed even if given.
    """

    def __init__(self, context_ratio=0.25, boundary_ratio=0.15,
                 moe_module=None, use_moe=False):
        self.context_ratio = context_ratio
        self.boundary_ratio = boundary_ratio
        self._n_actions = NUM_ACTIONS + 1
        self.moe = moe_module
        self.use_moe = use_moe and (moe_module is not None)
        self._last_moe_weights = None

    @property
    def last_moe_weights(self):
        """Router weights from the most recent build() call (for logging)."""
        return self._last_moe_weights

    # ── public API ──────────────────────────────────────────────────
    @property
    def state_dim(self):
        """Must be called after at least one ``build`` so C is known."""
        return self._cached_dim

    def build(self, raw: dict, device=None) -> torch.Tensor:
        l, r = raw["l"], raw["r"]
        T = raw["T"]
        step = raw["step"]
        max_steps = raw["max_steps"]
        prev_action = raw["prev_action"]
        feat_map = raw["feat_map"]   # (C, T)
        score_map = raw["score_map"] # (T,)

        C = feat_map.shape[0]

        # 1) geometry  (4,)
        w = max(r - l, 1.0)
        c = (l + r) / 2.0
        g = torch.tensor([l / T, r / T, w / T, c / T], dtype=torch.float32)

        # 2) score statistics  (4,)
        li, ri = int(max(0, round(l))), int(min(T, round(r) + 1))
        li = min(li, T - 1)
        ri = max(ri, li + 1)
        scores = score_map.float()

        mu_in = scores[li:ri].mean() if ri > li else torch.tensor(0.0)

        bw = max(1, int(self.boundary_ratio * w))
        bl = int(max(0, li - bw))
        br = int(min(T, ri + bw))
        bd_scores = torch.cat([scores[bl:li], scores[ri:br]])
        mu_bd = bd_scores.mean() if bd_scores.numel() > 0 else torch.tensor(0.0)

        cw = max(2, int(self.context_ratio * w))
        cl = int(max(0, li - cw))
        cr = int(min(T, ri + cw))
        ctx_scores = torch.cat([scores[cl:li], scores[ri:cr]])
        mu_out = ctx_scores.mean() if ctx_scores.numel() > 0 else torch.tensor(0.0)

        local_scores = scores[li:ri]
        if local_scores.numel() > 1:
            H = local_scores.var()
        else:
            H = torch.tensor(0.0)

        u = torch.stack([mu_in, mu_bd, mu_out, H])

        # 3) local evidence — MoE or naive pooling
        if self.use_moe:
            moe_device = next(self.moe.parameters()).device
            v, moe_w = self.moe(feat_map.float().to(moe_device), l, r, T)
            v = v.cpu()
            self._last_moe_weights = moe_w.detach().cpu()
        else:
            self._last_moe_weights = None
            feat = feat_map.float()
            e_left = self._pool_region(feat, cl, li)
            e_interior = self._pool_region(feat, li, ri)
            e_right = self._pool_region(feat, ri, cr)
            e_global = feat.mean(dim=-1)
            v = torch.cat([e_left, e_interior, e_right, e_global])  # (4C,)

        # 4) previous action one-hot  (NUM_ACTIONS+1,)
        a_oh = torch.zeros(self._n_actions, dtype=torch.float32)
        a_oh[min(prev_action, self._n_actions - 1)] = 1.0

        # 5) step progress  (1,)
        m = torch.tensor([step / max(max_steps, 1)], dtype=torch.float32)

        state = torch.cat([g, u, v, a_oh, m])
        self._cached_dim = state.shape[0]

        if device is not None:
            state = state.to(device)
        return state

    # ── helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _pool_region(feat, start, end):
        """Average-pool feat[:, start:end] → (C,).  Handles empty regions."""
        start = int(max(0, start))
        end = int(max(start + 1, end))
        end = min(end, feat.shape[-1])
        if end <= start:
            return torch.zeros(feat.shape[0], dtype=feat.dtype)
        return feat[:, start:end].mean(dim=-1)

    def compute_state_dim(self, C: int) -> int:
        """Compute state dim analytically given feature channel count C."""
        if self.use_moe:
            evidence_dim = self.moe.expert_out
        else:
            evidence_dim = 4 * C
        return 4 + 4 + evidence_dim + self._n_actions + 1
