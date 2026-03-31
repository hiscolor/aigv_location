# AIGV Temporal Forgery Localization: Method Documentation

> **VideoMAE Baseline + PPO Proposal Refinement + Role-aware MoE**
>
> 基于 UMMAFormer 代码框架，构建了一套三阶段递进式时序伪造定位方法。
> 本文档详细说明每个模块的设计动机、技术细节、张量流转、训练策略和消融开关。

---

## Table of Contents

- [1. Overall Architecture](#1-overall-architecture)
- [2. Phase 1 — Clean Baseline](#2-phase-1--clean-baseline)
  - [2.1 设计动机](#21-设计动机)
  - [2.2 架构选择](#22-架构选择)
  - [2.3 VideoMAE 特征接入](#23-videomae-特征接入)
  - [2.4 数据流与张量形状](#24-数据流与张量形状)
  - [2.5 配置文件](#25-配置文件)
- [3. Phase 2 — PPO Proposal Refinement](#3-phase-2--ppo-proposal-refinement)
  - [3.1 核心思想](#31-核心思想)
  - [3.2 RL 环境设计](#32-rl-环境设计)
  - [3.3 状态空间 (State)](#33-状态空间-state)
  - [3.4 动作空间 (Action)](#34-动作空间-action)
  - [3.5 奖励函数 (Reward)](#35-奖励函数-reward)
  - [3.6 PPO Agent 网络结构](#36-ppo-agent-网络结构)
  - [3.7 PPO 训练算法](#37-ppo-训练算法)
  - [3.8 坐标系统与转换](#38-坐标系统与转换)
- [4. Phase 3 — Role-aware Mixture of Experts](#4-phase-3--role-aware-mixture-of-experts)
  - [4.1 动机：为什么需要 MoE](#41-动机为什么需要-moe)
  - [4.2 三个角色专家](#42-三个角色专家)
  - [4.3 几何条件路由器 (Router)](#43-几何条件路由器-router)
  - [4.4 MoE 融合过程](#44-moe-融合过程)
  - [4.5 MoE 梯度回传机制](#45-moe-梯度回传机制)
  - [4.6 MoE 消融开关](#46-moe-消融开关)
- [5. File Structure](#5-file-structure)
- [6. Training & Evaluation](#6-training--evaluation)
  - [6.1 Stage 1: Coarse Baseline Training](#61-stage-1-coarse-baseline-training)
  - [6.2 Stage 2: PPO Refinement Training](#62-stage-2-ppo-refinement-training)
  - [6.3 Stage 3: PPO + MoE Training](#63-stage-3-ppo--moe-training)
  - [6.4 Inference & Evaluation](#64-inference--evaluation)
- [7. Ablation Study Design](#7-ablation-study-design)
- [8. Hyperparameter Reference](#8-hyperparameter-reference)

---

## 1. Overall Architecture

整体方法采用 **coarse-to-fine** 的两阶段范式：

```
VideoMAE Features (T, 1408)
        │
        ▼
┌─────────────────────────┐
│  Stage 1: Coarse Model  │  ActionFormer (conv+transformer backbone)
│  LocPointTransformer    │  → multi-scale score maps + boundary offsets
│                         │  → NMS → Top-K coarse proposals
└────────┬────────────────┘
         │  Top-K proposals (seconds)
         │  + backbone feature map (C, T)
         │  + token-level score map (T,)
         ▼
┌─────────────────────────┐
│  Stage 2: PPO Refiner   │  强化学习 agent 逐步调整边界
│  + Role-aware MoE       │  MoE 提供上下文感知的状态增强
│                         │  → 精修后的 proposals
└────────┬────────────────┘
         │
         ▼
    Final Predictions
```

三个阶段形成完整的消融实验链：

| 实验 | Coarse | PPO | MoE | 对应 Config |
|------|:------:|:---:|:---:|------------|
| Baseline | ✅ | ❌ | ❌ | `lavdf_videomae_clean.yaml` |
| + PPO | ✅ | ✅ | ❌ | `lavdf_videomae_ppo.yaml` |
| + PPO + MoE | ✅ | ✅ | ✅ | `lavdf_videomae_ppo_moe.yaml` |

---

## 2. Phase 1 — Clean Baseline

### 2.1 设计动机

UMMAFormer 原始代码包含三个创新模块：
- **TFAA** (Temporal Feature Anomaly Attention) — `DeepInterpolator` in `blocks.py`
- **PCA-FPN** — `ConvHRLRFullResSelfAttTransformerRevised` in `backbones.py`
- **Multimodal Cross-Attention** — `MutilModelTransformerBlock` in `blocks.py`

这些模块与 TSN (rgb+flow) + audio 深度耦合。为了引入 VideoMAE 单流特征并为后续 PPO/MoE 提供干净的实验基线，我们**绕过**而非删除这些模块——直接使用代码中已存在但未被默认启用的 `LocPointTransformer` (即 ActionFormer 原始架构)。

### 2.2 架构选择

```
LocPointTransformer (meta_archs.py)
├── Backbone: ConvTransformerBackbone
│   ├── Embedding: 1D Conv (1408 → 256)
│   ├── Stem: 2 × TransformerBlock (local attention, win=7)
│   ├── Branch: 5 levels, each with 2 × TransformerBlock + 2× downsample
│   └── Output: 6-level feature pyramid [(B, 256, T), ..., (B, 256, T/32)]
├── Neck: FPNIdentity (passthrough, no cross-level fusion)
├── Cls Head: 3-layer 1D Conv → per-token classification scores
├── Reg Head: 3-layer 1D Conv → per-token boundary offsets
└── Post-processing: Soft NMS → final proposals in seconds
```

### 2.3 VideoMAE 特征接入

**特征格式**：预提取的 VideoMAE-Large 特征，存储为 `.npy` 文件：
- 路径: `{feat_folder}/{video_id}.npy`
- Shape: `(T, 1408)` — T 为视频的 token 数量，1408 为 VideoMAE-Large 的隐层维度
- 单流设计：不区分 rgb/flow，无需 `file_prefix`

**Dataset 类**: `LAVDFVideoMAEDataset` (`libs/datasets/lavdf_videomae.py`)

与原始 `LAVDFDataset` 的关键差异：

| 特性 | 原始 (TSN) | VideoMAE |
|------|-----------|---------|
| 输入维度 | 768 (rgb) + 1024 (flow) = 1792 | 1408 |
| 文件结构 | `feat_folder/{rgb,flow}/{id}.npy` | `feat_folder/{id}.npy` |
| 模态 | 双流 (rgb + optical flow) | 单流 |
| 音频 | 可选 BYOLA 特征 | 无 |
| `file_prefix` | `"rgb"` / `"flow"` | `null` |

### 2.4 数据流与张量形状

```python
# Dataset __getitem__ 返回
data_dict = {
    'video_id': str,
    'feats':    Tensor(1408, T'),  # C×T, upsampled to max_seq_len=768
    'segments': Tensor(N, 2),     # GT segments in token coordinates
    'labels':   Tensor(N,),       # all 0 (binary: Fake class)
    'fps':      float,
    'duration': float,
    'feat_stride':     float,
    'feat_num_frames': float,
}

# 坐标转换 (seconds → tokens)
# token_coord = seconds * fps / feat_stride - 0.5 * num_frames / feat_stride

# Model forward (training)
batched_inputs  # (B, 1408, 768)
→ backbone      # [(B, 256, 768), (B, 256, 384), ..., (B, 256, 24)]
→ neck          # same (identity FPN)
→ cls_head      # [(B, 1, 768), (B, 1, 384), ..., (B, 1, 24)]
→ reg_head      # [(B, 2, 768), (B, 2, 384), ..., (B, 2, 24)]
→ losses        # focal_loss + diou_loss

# Model forward (inference)
→ decode offsets → proposals in token coords
→ NMS
→ convert to seconds: seconds = (tokens * stride + 0.5 * nframes) / fps
```

### 2.5 配置文件

`configs/baseline/lavdf_videomae_clean.yaml` 的核心参数:

```yaml
dataset_name: lavdf_videomae
dataset:
  input_dim: 1408          # VideoMAE feature dimension
  max_seq_len: 768         # all features upsampled to this length
  force_upsampling: True   # enable interpolation to fixed length

model_name: LocPointTransformer  # ActionFormer, not UMMAFormer
model:
  backbone_type: convTransformer
  fpn_type: identity       # no FPN fusion, passthrough
  embd_dim: 256            # backbone output channels
  backbone_arch: [2, 2, 5] # [stem_layers, branch_layers, num_levels]
```

---

## 3. Phase 2 — PPO Proposal Refinement

### 3.1 核心思想

Coarse model 的输出虽然能产生合理的 proposals，但边界定位精度有限——尤其在 forgery segment 的起止点附近，由于 token 级别的 stride 和 NMS 后处理，边界往往存在偏差。

我们引入 **PPO (Proximal Policy Optimization)** 将 proposal 边界调整建模为一个**序贯决策问题**：一个 RL agent 观察当前 proposal 的几何信息、局部特征和 score 分布，然后选择一系列离散动作逐步调整 proposal 的左右边界，直到它认为当前边界已足够准确（选择 stop 动作）或达到最大步数。

### 3.2 RL 环境设计

**文件**: `libs/modeling/ppo/environment.py` → `ProposalRefineEnv`

环境包装了单个 coarse proposal，遵循 `reset()` / `step(action)` / `done` 协议：

```
ProposalRefineEnv
├── 输入
│   ├── feat_map:  (C, T)    — backbone 特征图 (frozen, detached)
│   ├── score_map: (T,)      — coarse cls sigmoid scores
│   ├── gt_segment: (gt_l, gt_r)  — GT in token coords (训练时)
│   └── init_proposal: (l₀, r₀)  — 初始 coarse proposal in token coords
├── 参数
│   ├── max_steps: 20        — 最大修正步数
│   ├── delta: 1.0           — 每步移动量 (token 单位)
│   └── reward_cfg: dict     — 奖励权重
└── 输出
    └── (next_state, reward, done, info)
```

环境在每一步中：
1. 接收 agent 选择的动作
2. 修改 proposal 边界 `[l, r]`
3. 做合法性检查（clip 到 `[0, T-1]`，确保 `l < r`）
4. 计算奖励
5. 返回新状态

### 3.3 状态空间 (State)

**文件**: `libs/modeling/ppo/state_builder.py` → `StateBuilder`

状态向量由五部分拼接而成：

```
s_t = [ g_t | u_t | v_t | a_{t-1} | m_t ]
```

| 组件 | 维度 | 含义 |
|------|------|------|
| **g_t** | 4 | 归一化 proposal 几何: `[l/T, r/T, w/T, c/T]` |
| **u_t** | 4 | Score-map 统计量: `[μ_in, μ_bd, μ_out, H]` |
| **v_t** | 4C 或 MoE_out | 局部特征证据 (见下文) |
| **a_{t-1}** | 10 | 上一步动作的 one-hot (9 actions + 1 sentinel) |
| **m_t** | 1 | 步进进度: `step / max_steps` |

各组件的详细计算：

**g_t — Proposal Geometry (4-dim)**
```
l/T  — 左边界归一化位置
r/T  — 右边界归一化位置
w/T  — proposal 宽度归一化 (w = r - l)
c/T  — proposal 中心归一化 (c = (l+r)/2)
```

**u_t — Score Statistics (4-dim)**
```
μ_in   — proposal 内部 [l, r] 的 score 均值
μ_bd   — 边界缓冲区 [l-βw, l] ∪ [r, r+βw] 的 score 均值
μ_out  — 外部上下文 [l-αw, l] ∪ [r, r+αw] 的 score 均值
H      — proposal 内部 score 的方差 (homogeneity)
```
其中 `β = boundary_ratio (0.15)`, `α = context_ratio (0.25)`

**v_t — Local Evidence**

不启用 MoE 时 (4C-dim):
```
v_t = [e_left | e_interior | e_right | e_global]
```
每个分量是对 feat_map 对应区域的 average pooling → (C,)

启用 MoE 时 (expert_out-dim, 默认 64):
```
v_t = MoE(feat_map, l, r, T)  → (expert_out,)
```
由三个角色专家的加权融合给出 (详见 Phase 3)

**总 state_dim:**
- 无 MoE: `4 + 4 + 4×256 + 10 + 1 = 1043`
- 有 MoE: `4 + 4 + 64 + 10 + 1 = 83`

### 3.4 动作空间 (Action)

9 个离散动作，覆盖所有单步边界编辑模式：

| ID | 名称 | 效果 | 类别 |
|----|------|------|------|
| 0 | `left_boundary_left` | `l -= δ` | 单边 |
| 1 | `left_boundary_right` | `l += δ` | 单边 |
| 2 | `right_boundary_left` | `r -= δ` | 单边 |
| 3 | `right_boundary_right` | `r += δ` | 单边 |
| 4 | `shift_left` | `l -= δ, r -= δ` | 整体平移 |
| 5 | `shift_right` | `l += δ, r += δ` | 整体平移 |
| 6 | `shrink` | `l += δ, r -= δ` | 缩放 |
| 7 | `expand` | `l -= δ, r += δ` | 缩放 |
| 8 | `stop` | 终止 episode | 终止 |

其中 `δ = delta` (默认 1.0 token)。

**非法动作处理**: 如果动作导致 `l < 0` 或 `r ≥ T` 或 `l ≥ r`，执行 clip 投影并施加额外的 `lam_invalid` 惩罚。

### 3.5 奖励函数 (Reward)

每步奖励由五项加权求和：

```
r_t = λ_iou · ΔtIoU + λ_be · ΔBE - λ_step - λ_invalid · 𝟙[invalid] + λ_stop · stop_bonus
```

| 项 | 公式 | 默认权重 | 作用 |
|----|------|---------|------|
| **ΔtIoU** | `tIoU_t - tIoU_{t-1}` | λ_iou = 1.0 | 鼓励 IoU 提升 |
| **ΔBE** | `BE_{t-1} - BE_t` | λ_be = 0.5 | 鼓励边界误差减小 |
| **step cost** | 常数 | λ_step = 0.01 | 鼓励尽快终止 |
| **invalid** | `𝟙[动作越界]` | λ_invalid = 0.2 | 惩罚非法动作 |
| **stop bonus** | `max(0, tIoU - θ)` | λ_stop = 0.5, θ = 0.5 | 在高 IoU 时主动停止 |

其中:
- `tIoU = Intersection(pred, gt) / Union(pred, gt)` — temporal IoU
- `BE = (|l - gt_l| + |r - gt_r|) / T` — 归一化边界误差

### 3.6 PPO Agent 网络结构

**文件**: `libs/modeling/ppo/agent.py` → `PPOAgent`

```
                    ┌──────────────────┐
    state (s_t) ──▶│   Shared Trunk   │
                    │  Linear → ReLU   │ × n_layers (default: 2)
                    │  (state_dim→256) │
                    └────────┬─────────┘
                             │ (256,)
                    ┌────────┴────────┐
                    │                 │
              ┌─────▼─────┐   ┌──────▼──────┐
              │Policy Head│   │ Value Head  │
              │Linear(256 │   │Linear(256→1)│
              │   →9)     │   │             │
              └─────┬─────┘   └──────┬──────┘
                    │                │
              Categorical(π)     V(s_t)
```

- **初始化**: Orthogonal init, policy head 用 gain=0.01 (小初始 logits → 接近均匀)
- **推理**: `get_action(state)` → 采样 action, 返回 (action, log_prob, value)
- **评估**: `evaluate_actions(states, actions)` → 重新计算 log_prob, value, entropy (用于 PPO loss)

### 3.7 PPO 训练算法

**文件**: `libs/modeling/ppo/trainer.py` → `PPOTrainer`

训练流程：

```
for each epoch:
    for each training video:
        1. 从 frozen coarse model 提取 feat_map (C, T) + score_map (T,)
        2. 对每个 GT segment:
           - 生成 n_aug 个 jittered initial proposals
           - 对每个 initial proposal 运行 collect_rollout():
             - reset env → 循环: build_state → agent.get_action → env.step
             - 存储 (state, action, log_prob, reward, value, done)
        3. 当 buffer 积累 ≥ batch_episodes 时:
           - compute_gae() → advantages + returns
           - PPO update × ppo_epochs:
             - 重新评估 log_probs, values, entropy
             - L = -min(ratio·A, clip(ratio)·A) + c_v·MSE(V, R) - c_e·H
             - optimizer step
```

**GAE (Generalized Advantage Estimation)**:
```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
A_t = Σ_{k=0}^{T-t-1} (γλ)^k · δ_{t+k}
```

**PPO Clipped Surrogate**:
```
ratio = exp(log_π_new - log_π_old)
L_clip = -min(ratio · A, clip(ratio, 1-ε, 1+ε) · A)
L_total = L_clip + c_v · L_value - c_e · H(π)
```

**Proposal Augmentation (训练时)**:

训练时不依赖 coarse model 的推理质量。对每个 GT segment `[gt_l, gt_r]`，生成 `n_aug` (默认 3) 个 jittered proposals:
```python
jitter = max_jitter_ratio × w  # 默认 0.3
l_init = gt_l + Uniform(-jitter, jitter)
r_init = gt_r + Uniform(-jitter, jitter)
```
这确保 agent 在各种初始偏差下都能学到有效策略。

### 3.8 坐标系统与转换

系统中存在两套坐标：

| 坐标系 | 使用场景 | 范围 |
|--------|---------|------|
| **Token coordinates** | Dataset, PPO 环境, Feature map | `[0, T)` |
| **Seconds** | Coarse model 推理输出, 最终评估 | `[0, duration]` |

转换公式 (与 ActionFormer `postprocessing` 一致):

```python
# tokens → seconds
seconds = (tokens × feat_stride + 0.5 × num_frames) / fps

# seconds → tokens
tokens = (seconds × fps - 0.5 × num_frames) / feat_stride
```

`eval_ppo.py` 在推理时自动完成双向转换：coarse output (秒) → token (给 PPO) → 秒 (给评估)。

---

## 4. Phase 3 — Role-aware Mixture of Experts

### 4.1 动机：为什么需要 MoE

Phase 2 的朴素状态中，局部证据 `v_t` 由四个 average-pooled 向量拼接而成 (`4C = 1024` 维)。这种设计存在两个问题：

1. **维度过高**: 1024 维的原始特征拼接占据了 state 的绝大部分维度，容易导致策略网络过拟合于特征噪声
2. **缺乏角色意识**: 四个区域被等权对待，但实际上 proposal 的不同区域对边界修正的价值截然不同——**边界过渡区**的特征变化是判断边界是否准确的最直接线索，而**内容区**和**上下文区**分别提供 forgery 置信度和背景参照

Role-aware MoE 用三个角色专家替代朴素 pooling，并通过 router 根据 proposal 的几何形态动态分配权重。

### 4.2 三个角色专家

**文件**: `libs/modeling/ppo/moe.py`

```
                  feat_map (C, T)
                       │
          ┌────────────┼────────────┐
          │            │            │
    ┌─────▼─────┐ ┌───▼───┐ ┌─────▼─────┐
    │ Boundary  │ │Interior│ │  Context  │
    │  Expert   │ │ Expert │ │  Expert   │
    └─────┬─────┘ └───┬───┘ └─────┬─────┘
          │(d)        │(d)        │(d)
          │            │            │
          └────────────┼────────────┘
                       │
                  ┌────▼────┐
                  │ Router  │ ← geom [l/T, r/T, w/T, c/T]
                  │ softmax │
                  └────┬────┘
                       │ weights (3,)
                       ▼
              weighted sum → (d,)
```

**BoundaryExpert** — 关注边界过渡区:
```
输入区域: [l - β·w, l + β·w] ∪ [r - β·w, r + β·w]
操作: pool(left_boundary) ∥ pool(right_boundary) → Linear(2C → hidden → d)
直觉: 边界处的特征变化是判断左右边界是否准确的最直接信号
```

**InteriorExpert** — 关注 proposal 内容:
```
输入区域: [l, r]
操作: pool(interior) → Linear(C → hidden → d)
直觉: 内部特征的 forgery 一致性反映 proposal 是否覆盖了完整的伪造片段
```

**ContextExpert** — 关注周围上下文:
```
输入区域: [l - α·w, l] ∪ [r, r + α·w]
操作: pool(left_context) ∥ pool(right_context) → Linear(2C → hidden → d)
直觉: 上下文提供"真实片段看起来是什么样"的参照，帮助 agent 判断边界偏差方向
```

每个 Expert 内部是 2-layer MLP: `in_dim → hidden(128) → out_dim(64)`

### 4.3 几何条件路由器 (Router)

Router 是一个轻量级的 2-layer MLP，输入 proposal 的归一化几何向量，输出三个 expert 的 softmax 权重：

```
geom = [l/T, r/T, w/T, c/T]  (4-dim)
    │
    ▼
Linear(4 → 32) → ReLU → Linear(32 → 3)
    │
    ▼
softmax(logits / τ)  →  [w_boundary, w_interior, w_context]
```

**设计意图**:
- 窄 proposal (w/T 小) → 边界更敏感 → router 倾向提高 BoundaryExpert 权重
- 宽 proposal → 内容更丰富 → InteriorExpert 权重可能更高
- proposal 靠近视频边缘 → 上下文不对称 → ContextExpert 权重调整

温度参数 `τ` (默认 1.0) 控制权重的锐度，τ 越小越接近 hard routing。

### 4.4 MoE 融合过程

```python
e_b = BoundaryExpert(feat_map, l, r, T)   # (expert_out,)
e_i = InteriorExpert(feat_map, l, r, T)   # (expert_out,)
e_c = ContextExpert(feat_map, l, r, T)    # (expert_out,)

weights = Router([l/T, r/T, w/T, c/T])   # (3,)  softmax

enhanced = w[0]·e_b + w[1]·e_i + w[2]·e_c  # (expert_out,)
```

融合结果 `enhanced` 替代原始 `v_t` (4C-dim) 进入 PPO state:
```
s_t = [ g(4) | u(4) | enhanced(64) | a_onehot(10) | m(1) ] = 83-dim
```

对比朴素 pooling 的 1043-dim，MoE 方式降低了 **92% 的状态维度**，同时提供了更有语义的特征表示。

### 4.5 MoE 梯度回传机制

MoE 模块的参数需要通过 PPO loss 进行端到端训练。由于 rollout 收集阶段使用 `@torch.no_grad()` 加速，存储的 state 已经 detach，无法直接为 MoE 提供梯度。

解决方案：**Update 阶段 State 重算**

```
Rollout 阶段 (no_grad):
  存储 state_t (detached) + raw_state_dict (feat_map, l, r, ...)

Update 阶段 (grad enabled):
  if use_moe:
    states = _rebuild_states_with_grad(raw_states)  # 重新执行 MoE forward
  else:
    states = stack(detached_states)  # 直接使用
  
  → agent.evaluate_actions(states, actions)
  → PPO loss → backward  # 梯度流经 MoE 的 experts + router
```

这确保 MoE 的三个 Expert 和 Router 都能收到梯度更新。已验证所有 MoE 参数在 update 后发生变化。

### 4.6 MoE 消融开关

在 YAML config 中设置 `ppo.moe.enable`:

```yaml
ppo:
  moe:
    enable: true    # true → MoE state (83-dim)
                    # false → naive pooled state (1043-dim)
    expert_hidden: 128
    expert_out: 64
    boundary_ratio: 0.15
    context_ratio: 0.5
    router_hidden: 32
    temperature: 1.0
```

当 `enable: false` 时，StateBuilder 跳过 MoE 模块，回退到朴素 4C pooling，且不创建任何 MoE 网络参数。

---

## 5. File Structure

```
UMMAFormer/
├── configs/baseline/
│   ├── lavdf_videomae_clean.yaml       # Phase 1: coarse baseline
│   ├── lavdf_videomae_ppo.yaml         # Phase 2: PPO only (no MoE)
│   └── lavdf_videomae_ppo_moe.yaml     # Phase 3: PPO + MoE
│
├── libs/
│   ├── datasets/
│   │   ├── lavdf_videomae.py           # VideoMAE feature dataset
│   │   └── ...
│   │
│   └── modeling/
│       ├── meta_archs.py               # LocPointTransformer (coarse model)
│       ├── backbones.py                # ConvTransformerBackbone
│       ├── necks.py                    # FPNIdentity
│       │
│       └── ppo/                        # ★ PPO + MoE 模块
│           ├── __init__.py
│           ├── environment.py          # ProposalRefineEnv (RL 环境)
│           ├── state_builder.py        # StateBuilder (状态构造)
│           ├── agent.py                # PPOAgent (策略+价值网络)
│           ├── trainer.py              # PPOTrainer (GAE + PPO update)
│           └── moe.py                  # RoleAwareMoE (三角色专家 + 路由器)
│
├── train.py                            # Phase 1: 训练 coarse model
├── eval.py                             # Phase 1: 评估 coarse model
├── train_ppo.py                        # Phase 2/3: 训练 PPO (+MoE)
└── eval_ppo.py                         # Phase 2/3: 评估 PPO refined results
```

---

## 6. Training & Evaluation

### 6.1 Stage 1: Coarse Baseline Training

```bash
python train.py configs/baseline/lavdf_videomae_clean.yaml \
    --output clean_v1 --eval
```

训练标准 ActionFormer，产出 `model_best.pth.tar`。

### 6.2 Stage 2: PPO Refinement Training

```bash
python train_ppo.py configs/baseline/lavdf_videomae_ppo.yaml \
    --coarse_ckpt ./ckpt/lavdf_videomae_clean_clean_v1/model_best.pth.tar \
    --output ppo_v1
```

- 冻结 coarse model，仅训练 PPO agent
- 每个 epoch 遍历全部训练视频，每个 GT segment 生成 `n_aug=3` 个 episode
- 每 `batch_episodes=64` 个 episode 做一次 PPO update

### 6.3 Stage 3: PPO + MoE Training

```bash
python train_ppo.py configs/baseline/lavdf_videomae_ppo_moe.yaml \
    --coarse_ckpt ./ckpt/lavdf_videomae_clean_clean_v1/model_best.pth.tar \
    --output ppo_moe_v1
```

与 Stage 2 相同的训练流程，但 MoE 被自动启用（`moe.enable: true`），MoE 参数与 PPO agent 联合优化。

### 6.4 Inference & Evaluation

```bash
# PPO-only evaluation
python eval_ppo.py configs/baseline/lavdf_videomae_ppo.yaml \
    --coarse_ckpt ./ckpt/.../model_best.pth.tar \
    --ppo_ckpt ./ckpt/.../ppo_best.pth.tar \
    --split val --deterministic

# PPO + MoE evaluation
python eval_ppo.py configs/baseline/lavdf_videomae_ppo_moe.yaml \
    --coarse_ckpt ./ckpt/.../model_best.pth.tar \
    --ppo_ckpt ./ckpt/.../ppo_best.pth.tar \
    --split test --deterministic
```

推理流程：
1. Coarse model 生成 proposals (秒)
2. 转换为 token 坐标
3. PPO agent 逐步修正 Top-K proposals
4. 转换回秒
5. 合并 refined + remaining proposals
6. 计算 AP@tIoU

---

## 7. Ablation Study Design

通过三份 config 文件，可以直接进行以下消融实验:

| 实验编号 | 描述 | Config | 关键差异 |
|---------|------|--------|---------|
| A1 | Coarse baseline only | `lavdf_videomae_clean.yaml` | 无 PPO |
| A2 | + PPO (naive state) | `lavdf_videomae_ppo.yaml` | `moe.enable: false` (默认) |
| A3 | + PPO + MoE | `lavdf_videomae_ppo_moe.yaml` | `moe.enable: true` |

进一步可调：

| 消融维度 | 修改方式 |
|---------|---------|
| MoE 开/关 | `ppo.moe.enable: true/false` |
| Expert 数量 | 注释掉 `moe.py` 中个别 expert |
| Router 温度 | `ppo.moe.temperature: 0.5 / 1.0 / 2.0` |
| PPO 步数 | `ppo.max_steps: 10 / 20 / 30` |
| 动作步长 | `ppo.delta: 0.5 / 1.0 / 2.0` |
| 奖励权重 | `ppo.reward.lam_*` |
| Top-K | `ppo.topk: 5 / 10 / 20` |

---

## 8. Hyperparameter Reference

### PPO Hyperparameters

| 参数 | 默认值 | Config Key | 说明 |
|------|--------|-----------|------|
| Learning rate | 3e-4 | `ppo.lr` | Adam optimizer |
| PPO epochs | 4 | `ppo.ppo_epochs` | 每批 rollout 的优化遍数 |
| Clip range | 0.2 | `ppo.clip_eps` | PPO surrogate clip ε |
| Value coef | 0.5 | `ppo.vf_coef` | Value loss 权重 |
| Entropy coef | 0.01 | `ppo.ent_coef` | 探索 bonus 权重 |
| GAE γ | 0.99 | `ppo.gamma` | 折扣因子 |
| GAE λ | 0.95 | `ppo.gae_lam` | GAE 平滑因子 |
| Max grad norm | 0.5 | `ppo.max_grad_norm` | 梯度裁剪 |
| Hidden dim | 256 | `ppo.hidden_dim` | Agent trunk 宽度 |
| N layers | 2 | `ppo.n_layers` | Agent trunk 层数 |

### Environment Hyperparameters

| 参数 | 默认值 | Config Key | 说明 |
|------|--------|-----------|------|
| Max steps | 20 | `ppo.max_steps` | 每 episode 最大步数 |
| Delta | 1.0 | `ppo.delta` | 每步移动量 (token) |
| Top-K | 5 | `ppo.topk` | 每视频 refine 的 proposal 数 |
| N augmentation | 3 | `ppo.n_aug` | 每 GT segment 的训练 episode 数 |
| Batch episodes | 64 | `ppo.batch_episodes` | PPO 更新批大小 |

### Reward Hyperparameters

| 参数 | 默认值 | Config Key | 说明 |
|------|--------|-----------|------|
| λ_iou | 1.0 | `ppo.reward.lam_iou` | IoU 提升奖励 |
| λ_be | 0.5 | `ppo.reward.lam_be` | 边界误差减小奖励 |
| λ_step | 0.01 | `ppo.reward.lam_step` | 每步时间惩罚 |
| λ_invalid | 0.2 | `ppo.reward.lam_invalid` | 非法动作惩罚 |
| λ_stop | 0.5 | `ppo.reward.lam_stop` | 主动停止 bonus |
| θ_stop | 0.5 | `ppo.reward.stop_iou_thresh` | Stop bonus 阈值 |

### MoE Hyperparameters

| 参数 | 默认值 | Config Key | 说明 |
|------|--------|-----------|------|
| Enable | true | `ppo.moe.enable` | MoE 总开关 |
| Expert hidden | 128 | `ppo.moe.expert_hidden` | Expert MLP hidden dim |
| Expert out | 64 | `ppo.moe.expert_out` | Expert 输出维度 |
| Boundary ratio β | 0.15 | `ppo.moe.boundary_ratio` | 边界缓冲区比例 |
| Context ratio α | 0.5 | `ppo.moe.context_ratio` | 上下文窗口比例 |
| Router hidden | 32 | `ppo.moe.router_hidden` | Router MLP hidden dim |
| Temperature τ | 1.0 | `ppo.moe.temperature` | Softmax 温度 |
