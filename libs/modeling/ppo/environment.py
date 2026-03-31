"""
Proposal Refinement Environment for PPO.

The environment wraps a single coarse proposal and allows an agent to
iteratively edit its boundaries via discrete actions.  It is *not* a
gym.Env subclass to avoid an external dependency, but follows the same
reset / step / done protocol.
"""

import torch
import numpy as np


# ── action definitions ──────────────────────────────────────────────
#  0  left_boundary_left     l -= δ
#  1  left_boundary_right    l += δ
#  2  right_boundary_left    r -= δ
#  3  right_boundary_right   r += δ
#  4  shift_left             l -= δ, r -= δ
#  5  shift_right            l += δ, r += δ
#  6  shrink                 l += δ, r -= δ
#  7  expand                 l -= δ, r += δ
#  8  stop                   terminate episode
NUM_ACTIONS = 9
ACTION_NAMES = [
    "left_boundary_left", "left_boundary_right",
    "right_boundary_left", "right_boundary_right",
    "shift_left", "shift_right",
    "shrink", "expand",
    "stop",
]


def _apply_action(l: float, r: float, action: int, delta: float):
    """Return (new_l, new_r, is_stop)."""
    if action == 0:
        return l - delta, r, False
    elif action == 1:
        return l + delta, r, False
    elif action == 2:
        return l, r - delta, False
    elif action == 3:
        return l, r + delta, False
    elif action == 4:
        return l - delta, r - delta, False
    elif action == 5:
        return l + delta, r + delta, False
    elif action == 6:
        return l + delta, r - delta, False
    elif action == 7:
        return l - delta, r + delta, False
    elif action == 8:
        return l, r, True
    else:
        raise ValueError(f"Unknown action {action}")


def temporal_iou(pred_l, pred_r, gt_l, gt_r):
    """Compute temporal IoU between two intervals."""
    inter_l = max(pred_l, gt_l)
    inter_r = min(pred_r, gt_r)
    inter = max(0.0, inter_r - inter_l)
    union = (pred_r - pred_l) + (gt_r - gt_l) - inter
    if union <= 0:
        return 0.0
    return inter / union


def boundary_error(pred_l, pred_r, gt_l, gt_r, T):
    """Normalised sum of absolute boundary errors."""
    return (abs(pred_l - gt_l) + abs(pred_r - gt_r)) / max(T, 1.0)


class ProposalRefineEnv:
    """
    Environment for a single proposal refinement episode.

    Parameters
    ----------
    feat_map : Tensor (C, T)       – backbone feature map (detached, on CPU or GPU)
    score_map : Tensor (T,)        – coarse forgery score per token
    gt_segment : tuple (gt_l, gt_r) – ground-truth interval in *token* coordinates
    init_proposal : tuple (l0, r0) – initial coarse proposal in token coordinates
    max_steps : int                – maximum refinement steps before forced stop
    delta : float                  – step size in token units
    reward_cfg : dict              – λ weights for reward terms
    """

    def __init__(
        self,
        feat_map,
        score_map,
        gt_segment,
        init_proposal,
        max_steps=20,
        delta=1.0,
        reward_cfg=None,
    ):
        self.feat_map = feat_map          # (C, T)
        self.score_map = score_map        # (T,)
        self.T = feat_map.shape[-1]
        self.C = feat_map.shape[0]
        self.gt_l, self.gt_r = gt_segment
        self.init_l, self.init_r = init_proposal
        self.max_steps = max_steps
        self.delta = delta

        rcfg = reward_cfg or {}
        self.lam_iou = rcfg.get("lam_iou", 1.0)
        self.lam_be = rcfg.get("lam_be", 0.5)
        self.lam_step = rcfg.get("lam_step", 0.01)
        self.lam_invalid = rcfg.get("lam_invalid", 0.2)
        self.lam_stop = rcfg.get("lam_stop", 0.5)
        self.stop_iou_thresh = rcfg.get("stop_iou_thresh", 0.5)

        self.reset()

    # ── reset / step ────────────────────────────────────────────────
    def reset(self):
        self.l = float(self.init_l)
        self.r = float(self.init_r)
        self.step_count = 0
        self.done = False
        self.prev_action = NUM_ACTIONS  # sentinel "no previous action"
        self.prev_iou = temporal_iou(self.l, self.r, self.gt_l, self.gt_r)
        self.prev_be = boundary_error(self.l, self.r, self.gt_l, self.gt_r, self.T)
        return self._build_state()

    def step(self, action: int):
        """Execute *action*, return (next_state, reward, done, info)."""
        assert not self.done, "Episode already finished"

        new_l, new_r, is_stop = _apply_action(self.l, self.r, action, self.delta)

        # ── legality projection ─────────────────────────────────────
        invalid = False
        if new_l < 0 or new_r >= self.T or new_l >= new_r:
            invalid = True
            new_l = max(0.0, min(new_l, self.T - 2))
            new_r = max(new_l + 1, min(new_r, self.T - 1))

        self.l, self.r = new_l, new_r
        self.step_count += 1

        cur_iou = temporal_iou(self.l, self.r, self.gt_l, self.gt_r)
        cur_be = boundary_error(self.l, self.r, self.gt_l, self.gt_r, self.T)

        # ── reward ──────────────────────────────────────────────────
        reward = 0.0
        reward += self.lam_iou * (cur_iou - self.prev_iou)
        reward += self.lam_be * (self.prev_be - cur_be)
        reward -= self.lam_step
        if invalid:
            reward -= self.lam_invalid
        if is_stop:
            stop_bonus = max(0.0, cur_iou - self.stop_iou_thresh)
            reward += self.lam_stop * stop_bonus

        self.prev_iou = cur_iou
        self.prev_be = cur_be
        self.prev_action = action

        if is_stop or self.step_count >= self.max_steps:
            self.done = True

        info = {
            "tiou": cur_iou,
            "be": cur_be,
            "invalid": invalid,
            "is_stop": is_stop,
            "l": self.l,
            "r": self.r,
        }
        return self._build_state(), reward, self.done, info

    # ── state construction ──────────────────────────────────────────
    def _build_state(self):
        """
        Returns a dict with raw components so that StateBulder can
        assemble the final vector (keeps env decoupled from net dims).
        """
        return {
            "l": self.l,
            "r": self.r,
            "T": self.T,
            "step": self.step_count,
            "max_steps": self.max_steps,
            "prev_action": self.prev_action,
            "feat_map": self.feat_map,
            "score_map": self.score_map,
        }
