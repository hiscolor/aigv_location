"""
PPO-refined evaluation.

Usage:
    python eval_ppo.py configs/baseline/lavdf_videomae_ppo.yaml \
        --coarse_ckpt ./ckpt/lavdf_videomae_clean_xxx/model_best.pth.tar \
        --ppo_ckpt ./ckpt/lavdf_videomae_ppo_xxx/ppo_best.pth.tar \
        [--topk 10] [--max_steps 20] [--deterministic]

Pipeline:
  1. Load frozen coarse model → produce Top-K proposals per video
  2. Load PPO agent → iteratively refine each proposal
  3. Run standard evaluation with the refined proposals
"""

import argparse
import json
import os
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.modeling.ppo import (
    ProposalRefineEnv, PPOAgent, StateBuilder, temporal_iou, RoleAwareMoE,
)
from libs.utils import fix_random_seed


def get_backbone_features_and_scores(model, video_item, device):
    inner = model.module if isinstance(model, nn.DataParallel) else model
    inner.eval()
    with torch.no_grad():
        batched_inputs, batched_masks = inner.preprocessing([video_item])
        feats, masks = inner.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = inner.neck(feats, masks)
        cls_logits = inner.cls_head(fpn_feats, fpn_masks)

    finest_feat = fpn_feats[0].squeeze(0).detach().cpu()
    finest_cls = cls_logits[0].squeeze(0).detach().cpu()
    score_map = finest_cls.sigmoid().squeeze(0)
    if score_map.dim() > 1:
        score_map = score_map.max(dim=0).values
    return finest_feat, score_map


def refine_proposal(agent, sb, feat_map, score_map, init_l, init_r,
                    max_steps, delta, device, deterministic=True):
    """Run the PPO agent greedily to refine a single proposal."""
    T = feat_map.shape[-1]

    l, r = float(init_l), float(init_r)
    prev_action = 9  # sentinel

    for step in range(max_steps):
        raw = {
            "l": l, "r": r, "T": T,
            "step": step, "max_steps": max_steps,
            "prev_action": prev_action,
            "feat_map": feat_map,
            "score_map": score_map,
        }
        state = sb.build(raw, device=device)

        with torch.no_grad():
            dist, _ = agent(state)
            if deterministic:
                action = dist.probs.argmax(dim=-1).item()
            else:
                action = dist.sample().item()

        if action == 8:  # stop
            break

        # apply action
        from libs.modeling.ppo.environment import _apply_action
        new_l, new_r, _ = _apply_action(l, r, action, delta)
        new_l = max(0.0, min(new_l, T - 2))
        new_r = max(new_l + 1, min(new_r, T - 1))
        l, r = new_l, new_r
        prev_action = action

    return l, r


def seconds_to_tokens(seg_sec, fps, stride, nframes):
    """Inverse of postprocessing: convert seconds → feature-grid tokens."""
    return (seg_sec * fps - 0.5 * nframes) / stride


def tokens_to_seconds(seg_tok, fps, stride, nframes):
    """Same as postprocessing: convert feature-grid tokens → seconds."""
    return (seg_tok * stride + 0.5 * nframes) / fps


def main(args):
    cfg = load_config(args.config)
    ppo_cfg = cfg.get("ppo", {})
    pprint(cfg)

    fix_random_seed(cfg["init_rand_seed"], include_cuda=True)
    device = cfg["devices"][0]

    # ── dataset ─────────────────────────────────────────────────────
    split = cfg["val_split"] if args.split == "val" else cfg["test_split"]
    val_dataset = make_dataset(
        cfg["dataset_name"], False, split, **cfg["dataset"]
    )
    val_loader = make_data_loader(
        val_dataset, False, None, batch_size=1, num_workers=2
    )
    print(f"Eval split={args.split}: {len(val_dataset)} videos")

    # ── coarse model ────────────────────────────────────────────────
    db_vars = val_dataset.get_attributes()
    cfg["model"]["train_cfg"]["head_empty_cls"] = db_vars["empty_label_ids"]

    coarse_model = make_meta_arch(cfg["model_name"], **cfg["model"])
    coarse_model = nn.DataParallel(coarse_model, device_ids=cfg["devices"])

    ckpt = torch.load(args.coarse_ckpt, map_location=lambda s, l: s.cuda(device))
    if "state_dict_ema" in ckpt:
        coarse_model.load_state_dict(ckpt["state_dict_ema"])
    else:
        coarse_model.load_state_dict(ckpt["state_dict"])
    del ckpt
    coarse_model.eval()

    # ── MoE (optional, determined by checkpoint) ─────────────────
    moe_cfg = ppo_cfg.get("moe", {})
    embd_dim = cfg["model"].get("embd_dim", 256)

    ppo_ckpt = torch.load(args.ppo_ckpt, map_location="cpu")
    use_moe = ppo_ckpt.get("use_moe", False)

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
        if "moe_state_dict" in ppo_ckpt:
            moe_module.load_state_dict(ppo_ckpt["moe_state_dict"])
        moe_module = moe_module.to(device)
        moe_module.eval()
        print("MoE module loaded and enabled")

    # ── PPO agent ───────────────────────────────────────────────────
    sb = StateBuilder(
        context_ratio=ppo_cfg.get("context_ratio", 0.25),
        boundary_ratio=ppo_cfg.get("boundary_ratio", 0.15),
        moe_module=moe_module,
        use_moe=use_moe,
    )
    state_dim = sb.compute_state_dim(embd_dim)

    agent = PPOAgent(
        state_dim=state_dim,
        hidden_dim=ppo_cfg.get("hidden_dim", 256),
        n_layers=ppo_cfg.get("n_layers", 2),
    )
    agent.load_state_dict(ppo_ckpt["state_dict"])
    agent = agent.to(device)
    agent.eval()
    print(f"Loaded PPO agent from {args.ppo_ckpt} (use_moe={use_moe})")

    topk = args.topk or ppo_cfg.get("topk", 10)
    max_steps = ppo_cfg.get("max_steps", 20)
    delta = ppo_cfg.get("delta", 1.0)
    deterministic = args.deterministic

    # ── evaluate ────────────────────────────────────────────────────
    all_results = []

    for video_list in val_loader:
        video_item = video_list[0] if isinstance(video_list, list) else video_list

        # coarse inference → proposals in seconds (after postprocessing)
        coarse_model.eval()
        with torch.no_grad():
            results = coarse_model(video_list)

        for r in results:
            segs = r["segments"]       # (N, 2) in seconds
            scores = r["scores"]       # (N,)
            labels = r["labels"]       # (N,)
            vid = r["video_id"]

            if segs.shape[0] == 0:
                all_results.append({
                    "video_id": vid,
                    "segments": segs.cpu().numpy(),
                    "scores": scores.cpu().numpy(),
                    "labels": labels.cpu().numpy(),
                })
                continue

            fps = video_item["fps"]
            stride = video_item["feat_stride"]
            nframes = video_item["feat_num_frames"]

            k = min(topk, segs.shape[0])
            topk_idx = scores.argsort(descending=True)[:k]

            feat_map, score_map = get_backbone_features_and_scores(
                coarse_model, video_item, device
            )
            T = feat_map.shape[-1]

            refined_segs_sec = []
            for ti in topk_idx:
                seg_sec = segs[ti]
                l_tok = seconds_to_tokens(seg_sec[0].item(), fps, stride, nframes)
                r_tok = seconds_to_tokens(seg_sec[1].item(), fps, stride, nframes)
                l_tok = max(0.0, min(l_tok, T - 2))
                r_tok = max(l_tok + 1, min(r_tok, T - 1))

                new_l_tok, new_r_tok = refine_proposal(
                    agent, sb, feat_map, score_map,
                    l_tok, r_tok, max_steps, delta,
                    device, deterministic,
                )

                new_l_sec = tokens_to_seconds(new_l_tok, fps, stride, nframes)
                new_r_sec = tokens_to_seconds(new_r_tok, fps, stride, nframes)
                new_l_sec = max(0.0, min(new_l_sec, video_item["duration"]))
                new_r_sec = max(0.0, min(new_r_sec, video_item["duration"]))
                refined_segs_sec.append([new_l_sec, new_r_sec])

            refined_segs = torch.tensor(refined_segs_sec, dtype=segs.dtype,
                                        device=segs.device)
            remaining_idx = [j for j in range(segs.shape[0])
                            if j not in topk_idx.tolist()]
            if remaining_idx:
                remaining_segs = segs[remaining_idx]
                all_segs = torch.cat([refined_segs, remaining_segs], dim=0)
                all_scores_t = torch.cat([scores[topk_idx],
                                          scores[remaining_idx]], dim=0)
                all_labels_t = torch.cat([labels[topk_idx],
                                          labels[remaining_idx]], dim=0)
            else:
                all_segs = refined_segs
                all_scores_t = scores[topk_idx]
                all_labels_t = labels[topk_idx]

            all_results.append({
                "video_id": vid,
                "segments": all_segs.cpu().numpy(),
                "scores": all_scores_t.cpu().numpy(),
                "labels": all_labels_t.cpu().numpy(),
            })

    # save results
    out_path = os.path.join(
        os.path.dirname(args.ppo_ckpt),
        f"ppo_eval_{args.split}.json"
    )
    save_list = []
    for r in all_results:
        for i in range(len(r["scores"])):
            save_list.append({
                "video_id": r["video_id"],
                "t_start": float(r["segments"][i][0]),
                "t_end": float(r["segments"][i][1]),
                "score": float(r["scores"][i]),
                "label": int(r["labels"][i]),
            })
    with open(out_path, "w") as f:
        json.dump(save_list, f, indent=2)
    print(f"Saved {len(save_list)} predictions to {out_path}")

    # ── compute AP metrics ──────────────────────────────────────────
    if cfg['dataset_name'].lower() in ['lavdf', 'lavdf_videomae']:
        gt_file = cfg['dataset']['json_file']
        tiou_thresholds = np.linspace(0.3, 0.7, 5)
        print(f"Computing AP at tIoU={tiou_thresholds} ...")
        from libs.utils import ANETdetection
        det = ANETdetection(
            ground_truth_filename=gt_file,
            prediction_filename=out_path,
            tiou_thresholds=tiou_thresholds,
            subset='test' if args.split == 'test' else 'dev',
        )
        mAP, _ = det.evaluate()
        for tidx, tiou in enumerate(tiou_thresholds):
            print(f"  tIoU={tiou:.2f}  AP={mAP[tidx]*100:.2f}")
        print(f"  Average mAP = {mAP.mean()*100:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPO refined proposals")
    parser.add_argument("config", type=str)
    parser.add_argument("--coarse_ckpt", type=str, required=True)
    parser.add_argument("--ppo_ckpt", type=str, required=True)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()
    main(args)
