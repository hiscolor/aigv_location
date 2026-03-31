"""
PPO Proposal Refinement Training Script.

Usage:
    python train_ppo.py configs/baseline/lavdf_videomae_ppo.yaml \
        --coarse_ckpt ./ckpt/lavdf_videomae_clean_xxx/model_best.pth.tar \
        --output ppo_run1

Two-stage training:
  1. Load a frozen coarse model (LocPointTransformer)
  2. For each training video, run coarse inference to get Top-K proposals
  3. Train a PPO agent to refine those proposals towards GT
"""

import argparse
import os
import time
import random
from pprint import pprint
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.modeling.ppo import (
    ProposalRefineEnv, PPOAgent, PPOTrainer, StateBuilder, temporal_iou,
    RoleAwareMoE,
)
from libs.utils import fix_random_seed, ANETdetection


def extract_coarse_proposals(model, video_list, topk=5):
    """
    Run frozen coarse model on a single video, return Top-K proposals
    plus the backbone feature map and token-level scores.

    Returns list of dicts, one per video in video_list.
    """
    model.eval()
    with torch.no_grad():
        results = model(video_list)

    outputs = []
    for idx, r in enumerate(results):
        segs = r["segments"]     # (N, 2)  in seconds
        scores = r["scores"]     # (N,)
        labels = r["labels"]     # (N,)

        if segs.shape[0] == 0:
            outputs.append(None)
            continue

        k = min(topk, segs.shape[0])
        topk_idx = scores.argsort(descending=True)[:k]

        outputs.append({
            "segments": segs[topk_idx],
            "scores": scores[topk_idx],
            "labels": labels[topk_idx],
            "video_id": r["video_id"],
        })
    return outputs


def seconds_to_tokens(seg_seconds, fps, feat_stride, num_frames):
    """Convert segment [start_sec, end_sec] to token-grid coordinates."""
    offset = 0.5 * num_frames / feat_stride
    tok = seg_seconds * fps / feat_stride - offset
    return tok


def build_proposal_augmentations(gt_l, gt_r, T, n_aug=4, max_jitter_ratio=0.3):
    """
    Generate augmented initial proposals around GT for PPO training.
    Mixes slight and moderate perturbations.
    """
    proposals = []
    w = gt_r - gt_l
    for _ in range(n_aug):
        jitter = max_jitter_ratio * w
        dl = random.uniform(-jitter, jitter)
        dr = random.uniform(-jitter, jitter)
        pl = max(0.0, gt_l + dl)
        pr = min(float(T - 1), gt_r + dr)
        if pl >= pr:
            pr = min(pl + 1.0, float(T - 1))
        proposals.append((pl, pr))
    return proposals


def get_backbone_features_and_scores(model, video_item, device):
    """
    Run backbone + neck + cls_head on a single video to get:
      - feat_map (C, T)  at the finest FPN level
      - score_map (T,)   sigmoid of cls logits at finest level
    The model must be LocPointTransformer (or compatible).
    """
    inner = model.module if isinstance(model, nn.DataParallel) else model
    inner.eval()

    with torch.no_grad():
        batched_inputs, batched_masks = inner.preprocessing([video_item])
        feats, masks = inner.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = inner.neck(feats, masks)

        cls_logits = inner.cls_head(fpn_feats, fpn_masks)

    finest_feat = fpn_feats[0].squeeze(0).detach().cpu()   # (C, T0)
    finest_cls = cls_logits[0].squeeze(0).detach().cpu()    # (num_cls, T0)
    score_map = finest_cls.sigmoid().squeeze(0)             # (T0,)
    if score_map.dim() > 1:
        score_map = score_map.max(dim=0).values

    return finest_feat, score_map


# ════════════════════════════════════════════════════════════════════
def main(args):
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")

    ppo_cfg = cfg.get("ppo", {})
    pprint(cfg)

    rng = fix_random_seed(cfg["init_rand_seed"], include_cuda=True)

    # ── output folder ───────────────────────────────────────────────
    if not os.path.exists(cfg["output_folder"]):
        os.makedirs(cfg["output_folder"], exist_ok=True)
    cfg_name = os.path.basename(args.config).replace(".yaml", "")
    tag = args.output or time.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_folder = os.path.join(cfg["output_folder"], f"{cfg_name}_{tag}")
    os.makedirs(ckpt_folder, exist_ok=True)

    # ── dataset ─────────────────────────────────────────────────────
    train_dataset = make_dataset(
        cfg["dataset_name"], True, cfg["train_split"], **cfg["dataset"]
    )
    print(f"PPO training set: {len(train_dataset)} videos")

    # ── coarse model (frozen) ───────────────────────────────────────
    device = cfg["devices"][0]
    train_db_vars = train_dataset.get_attributes()
    cfg["model"]["train_cfg"]["head_empty_cls"] = train_db_vars["empty_label_ids"]

    coarse_model = make_meta_arch(cfg["model_name"], **cfg["model"])
    coarse_model = nn.DataParallel(coarse_model, device_ids=cfg["devices"])

    assert os.path.isfile(args.coarse_ckpt), f"Coarse ckpt not found: {args.coarse_ckpt}"
    ckpt = torch.load(args.coarse_ckpt, map_location=lambda s, l: s.cuda(device))
    if "state_dict_ema" in ckpt:
        coarse_model.load_state_dict(ckpt["state_dict_ema"])
        print("Loaded coarse model from EMA weights")
    else:
        coarse_model.load_state_dict(ckpt["state_dict"])
        print("Loaded coarse model weights")
    del ckpt

    for p in coarse_model.parameters():
        p.requires_grad = False
    coarse_model.eval()

    # ── MoE (optional) ──────────────────────────────────────────────
    moe_cfg = ppo_cfg.get("moe", {})
    use_moe = moe_cfg.get("enable", False)
    embd_dim = cfg["model"].get("embd_dim", 256)

    moe_module = None
    if use_moe:
        moe_module = RoleAwareMoE(
            feat_dim=embd_dim,
            expert_hidden=moe_cfg.get("expert_hidden", 128),
            expert_out=moe_cfg.get("expert_out", 64),
            boundary_ratio=moe_cfg.get("boundary_ratio", 0.15),
            context_ratio=moe_cfg.get("context_ratio", 0.5),
            router_hidden=moe_cfg.get("router_hidden", 32),
            temperature=moe_cfg.get("temperature", 1.0),
        )
        print(f"MoE enabled: expert_out={moe_module.expert_out}")

    # ── PPO agent ───────────────────────────────────────────────────
    sb = StateBuilder(
        context_ratio=ppo_cfg.get("context_ratio", 0.25),
        boundary_ratio=ppo_cfg.get("boundary_ratio", 0.15),
        moe_module=moe_module,
        use_moe=use_moe,
    )
    state_dim = sb.compute_state_dim(embd_dim)
    print(f"PPO state_dim = {state_dim}  (embd_dim={embd_dim}, use_moe={use_moe})")

    agent = PPOAgent(
        state_dim=state_dim,
        hidden_dim=ppo_cfg.get("hidden_dim", 256),
        n_layers=ppo_cfg.get("n_layers", 2),
    )

    # collect all trainable parameters (agent + optional MoE)
    all_params = list(agent.parameters())
    if use_moe and moe_module is not None:
        all_params += list(moe_module.parameters())

    trainer = PPOTrainer(
        agent=agent,
        state_builder=sb,
        lr=ppo_cfg.get("lr", 3e-4),
        ppo_epochs=ppo_cfg.get("ppo_epochs", 4),
        clip_eps=ppo_cfg.get("clip_eps", 0.2),
        vf_coef=ppo_cfg.get("vf_coef", 0.5),
        ent_coef=ppo_cfg.get("ent_coef", 0.01),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lam=ppo_cfg.get("gae_lam", 0.95),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        device=device,
    )
    # override optimizer to include MoE params
    if use_moe and moe_module is not None:
        moe_module = moe_module.to(device)
        trainer.optimizer = torch.optim.Adam(all_params, lr=ppo_cfg.get("lr", 3e-4))

    # ── training loop ───────────────────────────────────────────────
    n_epochs = ppo_cfg.get("epochs", 20)
    topk = ppo_cfg.get("topk", 5)
    max_steps = ppo_cfg.get("max_steps", 20)
    delta = ppo_cfg.get("delta", 1.0)
    n_aug = ppo_cfg.get("n_aug", 3)
    batch_size = ppo_cfg.get("batch_episodes", 64)
    reward_cfg = ppo_cfg.get("reward", {})
    print_freq = ppo_cfg.get("print_freq", 50)

    best_avg_tiou = 0.0

    for epoch in range(n_epochs):
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        epoch_stats = defaultdict(list)
        buffers = []
        n_episodes = 0

        for i, idx in enumerate(indices):
            video_item = train_dataset[idx]
            if video_item["segments"] is None:
                continue

            gt_segments = video_item["segments"]  # (N, 2) in token coords
            if gt_segments.shape[0] == 0:
                continue

            feat_map, score_map = get_backbone_features_and_scores(
                coarse_model, video_item, device
            )
            T = feat_map.shape[-1]

            for seg_idx in range(gt_segments.shape[0]):
                gt_l = gt_segments[seg_idx, 0].item()
                gt_r = gt_segments[seg_idx, 1].item()
                if gt_l >= gt_r or gt_r <= 0:
                    continue

                aug_proposals = build_proposal_augmentations(
                    gt_l, gt_r, T, n_aug=n_aug
                )

                for init_l, init_r in aug_proposals:
                    env = ProposalRefineEnv(
                        feat_map=feat_map,
                        score_map=score_map,
                        gt_segment=(gt_l, gt_r),
                        init_proposal=(init_l, init_r),
                        max_steps=max_steps,
                        delta=delta,
                        reward_cfg=reward_cfg,
                    )
                    rollout_info = trainer.collect_rollout(env)
                    buffers.append(rollout_info["buffer"])

                    epoch_stats["tiou_init"].append(
                        temporal_iou(init_l, init_r, gt_l, gt_r)
                    )
                    epoch_stats["tiou_final"].append(rollout_info["final_tiou"])
                    epoch_stats["reward"].append(rollout_info["total_reward"])
                    epoch_stats["ep_len"].append(rollout_info["episode_len"])
                    n_episodes += 1

                    if len(buffers) >= batch_size:
                        update_stats = trainer.update(buffers)
                        for k, v in update_stats.items():
                            epoch_stats[k].append(v)
                        buffers = []

            if (i + 1) % print_freq == 0 and epoch_stats["tiou_final"]:
                avg_init = np.mean(epoch_stats["tiou_init"][-print_freq * n_aug:])
                avg_final = np.mean(epoch_stats["tiou_final"][-print_freq * n_aug:])
                avg_rew = np.mean(epoch_stats["reward"][-print_freq * n_aug:])
                print(
                    f"  [{epoch}][{i+1}/{len(indices)}]  "
                    f"tIoU {avg_init:.3f}→{avg_final:.3f}  "
                    f"reward {avg_rew:.3f}  episodes {n_episodes}"
                )

        if buffers:
            trainer.update(buffers)

        avg_tiou_init = np.mean(epoch_stats["tiou_init"]) if epoch_stats["tiou_init"] else 0
        avg_tiou_final = np.mean(epoch_stats["tiou_final"]) if epoch_stats["tiou_final"] else 0
        avg_reward = np.mean(epoch_stats["reward"]) if epoch_stats["reward"] else 0
        avg_plen = np.mean(epoch_stats["policy_loss"]) if epoch_stats["policy_loss"] else 0
        avg_vlen = np.mean(epoch_stats["value_loss"]) if epoch_stats["value_loss"] else 0

        print(
            f"\n[Epoch {epoch}] episodes={n_episodes}  "
            f"tIoU {avg_tiou_init:.3f}→{avg_tiou_final:.3f}  "
            f"Δ={avg_tiou_final - avg_tiou_init:+.3f}  "
            f"reward={avg_reward:.3f}  "
            f"π_loss={avg_plen:.4f}  V_loss={avg_vlen:.4f}"
        )

        is_best = avg_tiou_final > best_avg_tiou
        if is_best:
            best_avg_tiou = avg_tiou_final
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": agent.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "avg_tiou_final": avg_tiou_final,
            "use_moe": use_moe,
        }
        if use_moe and moe_module is not None:
            save_dict["moe_state_dict"] = moe_module.state_dict()
        torch.save(save_dict, os.path.join(ckpt_folder, f"ppo_epoch_{epoch+1:03d}.pth.tar"))
        if is_best:
            torch.save(save_dict, os.path.join(ckpt_folder, "ppo_best.pth.tar"))
            print(f"  ★ New best avg tIoU: {best_avg_tiou:.4f}")

    print("\nPPO training done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO proposal refiner")
    parser.add_argument("config", type=str, help="path to config yaml")
    parser.add_argument("--coarse_ckpt", type=str, required=True,
                        help="path to trained coarse model checkpoint")
    parser.add_argument("--output", type=str, default="",
                        help="experiment folder name")
    args = parser.parse_args()
    main(args)
