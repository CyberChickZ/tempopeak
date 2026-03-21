"""TempoPeak multi-hit training & evaluation (T-DEED + flow-guided spatial attention).

新增 (vs 旧版):
  CrossAttnPooling   — 替换 nn.Linear(1024→512), 用 [B,T,196,1024] patch tokens
  GatedBiMamba2      — 替换 BiMamba2 concat 融合, 用 sigmoid gate
  Flow KL loss       — CrossAttn attention weights 对齐 flow 伪标签
  --use_cross_attn   — 开关: True 启用 spatial 模式, False 退化为旧版 Linear
  --spatial_pt       — [N,196,1024] spatial patch tokens 路径
  --flow_pt          — [N,196] Farneback flow magnitude 路径
  --lambda_flow      — KL loss 初始权重 (default=0.5, 前50epoch线性衰减到0)
  --flow_anneal_epochs — flow KL loss 衰减 epoch 数 (default=50)
  gated_bimamba2     — 新增到 --temporal_head choices

训练命令 (主实验 E2):
  python train_multihit.py \\
      --features_pt  datasets/v1/export/0006/00001/00001_ce224.pt \\
      --spatial_pt   datasets/v1/export/0006/00001/00001_spatial.pt \\
      --flow_pt      datasets/v1/export/0006/00001/00001_flow.pt \\
      --hit_json     datasets/v1/export/0006/00001/annot.json \\
      --temporal_head gated_bimamba2 \\
      --use_cross_attn \\
      --segment_len 2700 --epochs 300 --batch_size 4 --lr 3e-4

Ablation E1 (gate only, no CrossAttn):
  python train_multihit.py \\
      --features_pt datasets/v1/export/0006/00001/00001_ce224.pt \\
      --hit_json    datasets/v1/export/0006/00001/annot.json \\
      --temporal_head gated_bimamba2 \\
      --segment_len 2700 --epochs 300 --batch_size 4 --lr 3e-4

Ablation E3 (CrossAttn + BiLSTM):
  python train_multihit.py \\
      --features_pt  datasets/v1/export/0006/00001/00001_ce224.pt \\
      --spatial_pt   datasets/v1/export/0006/00001/00001_spatial.pt \\
      --flow_pt      datasets/v1/export/0006/00001/00001_flow.pt \\
      --hit_json     datasets/v1/export/0006/00001/annot.json \\
      --temporal_head bilstm \\
      --use_cross_attn \\
      --segment_len 2700 --epochs 300 --batch_size 4 --lr 3e-4
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from temporal_heads import HEAD_REGISTRY, TransformerHead
from dataloader_multihit import build_multihit_dataloaders
from eval_multihit import cls_displacement_loss, cls_disp_detect, match_predictions


# ---------------------------------------------------------------------------
# CrossAttnPooling: [B,T,196,1024] + [B,T,196] → [B,T,512]
# ---------------------------------------------------------------------------

class CrossAttnPooling(nn.Module):
    """Flow-guided cross-attention spatial pooling.

    物理含义 (以一帧为例, B=1,T=1):
      patches[0,0]  = [196, 1024]  14×14 patch 语义表示 (DINOv2 frozen)
      flow_mag[0,0] = [196]        每个 patch 的光流幅值 (Farneback, 训练时用)
      output[0,0]   = [512]        运动引导的空间加权汇聚

    Attention 机制:
      Q: [BT, n_heads, 1, d_head]    — 可学习的 "我想提取什么" query
      K: [BT, n_heads, 196, d_head]  — patch 语义 + flow 幅值 concat 后投影
      V: [BT, n_heads, 196, d_head]  — patch 语义 (纯外观, 不含 flow)
      attn: softmax(Q·K^T / sqrt(d_head)) → [BT, n_heads, 1, 196]

    Flow 进入 Key 而不是 Value 的原因:
      Key 决定 "哪里值得关注" → flow 幅值大的 patch 更显著 → 注意力聚焦运动区域
      Value 决定 "取出什么内容" → 纯 DINOv2 语义, 不受 flow 污染

    推理时 flow_mag=None → K 退化为 pure-appearance key, attention 仍然有效
    (训练已经把 "关注运动区域" 学进 k_proj 的权重里了)

    KL 监督:
      attn_weights = attn.mean(dim=1).squeeze(2)  # [BT, 196], 已是概率分布
      flow_pseudo  = softmax(flow_mag)             # [BT, 196], flow 伪标签
      loss_flow    = KL(attn_weights || flow_pseudo) = Σ p·log(p/q)
    """
    out_dim = 512

    def __init__(self, d_in=1024, d_out=512, n_heads=4):
        super().__init__()
        assert d_out % n_heads == 0, f"d_out={d_out} 必须能被 n_heads={n_heads} 整除"
        self.n_heads = n_heads
        self.d_head  = d_out // n_heads
        self.scale   = self.d_head ** -0.5

        # 可学习 query: 每个 head 一个 d_head 维向量
        self.query   = nn.Parameter(torch.randn(1, n_heads, self.d_head) * 0.02)
        # Key: patch_feat [1024] + flow_mag [1] → d_out
        # flow_mag 为 None 时用 0 填充 (推理时)
        self.k_proj  = nn.Linear(d_in + 1, d_out, bias=False)
        # Value: patch_feat [1024] → d_out (不含 flow)
        self.v_proj  = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, patches, flow_mag=None):
        """
        Args:
            patches:  [B, T, 196, 1024]  DINOv2 patch tokens
            flow_mag: [B, T, 196]        光流幅值, 训练时传入, 推理时 None

        Returns:
            out:         [B, T, 512]     空间汇聚后的帧特征
            attn_weights:[B*T, 196]      attention weights (用于 KL loss 监督)
                         推理时也返回, 但调用方不用于计算 loss
        """
        B, T, N, D = patches.shape       # N=196, D=1024
        p = patches.reshape(B * T, N, D)  # [BT, 196, 1024]

        # 构造 flow channel: 训练时用真实幅值, 推理时用零
        if flow_mag is not None:
            f = flow_mag.reshape(B * T, N, 1).float()  # [BT, 196, 1]
        else:
            f = torch.zeros(B * T, N, 1, device=patches.device,
                            dtype=patches.dtype)

        # Keys: motion-aware
        K = self.k_proj(torch.cat([p, f], dim=-1))    # [BT, 196, d_out]
        K = K.reshape(B * T, N, self.n_heads, self.d_head).transpose(1, 2)
        # K: [BT, n_heads, 196, d_head]

        # Values: pure appearance
        V = self.v_proj(p).reshape(B * T, N, self.n_heads, self.d_head).transpose(1, 2)
        # V: [BT, n_heads, 196, d_head]

        # Query: broadcast across batch
        Q = self.query.unsqueeze(0).expand(B * T, -1, -1).unsqueeze(2)
        # Q: [BT, n_heads, 1, d_head]

        attn = (Q @ K.transpose(-2, -1)) * self.scale  # [BT, n_heads, 1, 196]
        attn = attn.softmax(dim=-1)

        # 跨 head 平均后 squeeze: 用于 KL loss
        attn_weights = attn.mean(dim=1).squeeze(1)     # [BT, 196]

        out = (attn @ V).squeeze(2)                    # [BT, n_heads, d_head]
        out = self.out_proj(out.reshape(B * T, -1))    # [BT, d_out]
        return out.reshape(B, T, self.out_dim), attn_weights


# ---------------------------------------------------------------------------
# Flow KL loss helper
# ---------------------------------------------------------------------------

def flow_kl_loss(attn_weights, flow_mag, valid_len=None):
    """CrossAttn attention weights 对齐 flow 幅值伪标签.

    数学:
      flow_pseudo[bt] = softmax(flow_mag[bt])  ← 把 flow 归一化成概率分布
      loss = KL(attn_weights || flow_pseudo)
           = Σ_{patch} attn[patch] · log(attn[patch] / flow_pseudo[patch])

    为什么用 KL 而不是 MSE:
      attn_weights 和 flow_pseudo 都是概率分布 (sum=1)
      KL 惩罚分布形状的差异, MSE 惩罚数值差异 → KL 更适合对齐 attention map

    Args:
        attn_weights: [B*T, 196]  CrossAttnPooling 输出的 attention 分布
        flow_mag:     [B, T, 196] Farneback 幅值
        valid_len:    [B]         只在有效帧上计算 loss (pad 帧排除)

    Returns:
        scalar loss
    """
    B_T, N = attn_weights.shape
    B = flow_mag.shape[0]
    T = B_T // B

    # Flow → 概率分布 (per-frame softmax)
    flow_flat = flow_mag.reshape(B_T, N).float()
    # 数值稳定: flow 幅值可能全零 (第0帧), softmax 退化为均匀分布 → OK
    flow_pseudo = F.softmax(flow_flat, dim=-1)  # [BT, 196]

    # valid mask: 只算有效帧, 排除 padding
    if valid_len is not None:
        t_idx = torch.arange(T, device=attn_weights.device).unsqueeze(0).expand(B, T)
        vl_exp = valid_len.unsqueeze(1).expand(B, T)  # [B, T]
        mask = (t_idx < vl_exp).reshape(B_T)           # [BT] bool
    else:
        mask = torch.ones(B_T, dtype=torch.bool, device=attn_weights.device)

    if mask.sum() == 0:
        return attn_weights.sum() * 0.0

    p = attn_weights[mask]       # [M, 196]
    q = flow_pseudo[mask]        # [M, 196]

    # KL(p||q) = Σ p·log(p/q), 用 F.kl_div(log_q, p) = Σ p·(log_p - log_q)
    # F.kl_div input=log(q), target=p, reduction='batchmean'
    loss = F.kl_div(
        q.clamp(min=1e-8).log(),     # log(q): [M, 196]
        p.clamp(min=1e-8),           # p:      [M, 196]
        reduction="batchmean",
    )
    return loss


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiHitModel(nn.Module):
    """Feature-only model for multi-hit detection.

    两种空间特征模式:
      use_cross_attn=False (默认, 兼容旧版):
        feat_proj = nn.Linear(feat_dim, 512)
        forward(features [B,T,D]) → (cls_logits, disp_logits)

      use_cross_attn=True (新):
        feat_proj = CrossAttnPooling(1024 → 512)
        forward(features, patches, flow_mag) → (cls_logits, disp_logits, attn_weights)
        patches:  [B, T, 196, 1024]
        flow_mag: [B, T, 196] 或 None (推理时)

    出口维度 (fc_cls/fc_disp 输入):
      BiLSTM:        256  (hidden*2)
      MS-TCN:        128
      Mamba2:        256
      BiMamba2:      512  (256 fwd + 256 bwd concat)
      GatedBiMamba2: 256  (gate weighted, 不是 concat)
      Transformer:   512
      Identity:      512
    """

    def __init__(self, temporal_head_name, feat_dim=1024, t_max=2700,
                 use_cross_attn=False):
        super().__init__()
        self.use_cross_attn = use_cross_attn

        if use_cross_attn:
            self.feat_proj = CrossAttnPooling(d_in=feat_dim, d_out=512, n_heads=4)
        elif feat_dim != 512:
            self.feat_proj = nn.Linear(feat_dim, 512)
        else:
            self.feat_proj = nn.Identity()

        # Temporal head
        if temporal_head_name == "transformer":
            self.head = TransformerHead(d_model=512, max_len=t_max)
        else:
            self.head = HEAD_REGISTRY[temporal_head_name]()

        head_out = HEAD_REGISTRY[temporal_head_name].out_dim
        self.fc_cls  = nn.Linear(head_out, 1)
        self.fc_disp = nn.Linear(head_out, 1)

    def forward(self, features, patches=None, flow_mag=None):
        """
        Args:
            features: [B, T, D]             CLS token features (always required)
            patches:  [B, T, 196, 1024]     spatial patch tokens (use_cross_attn only)
            flow_mag: [B, T, 196] or None   flow magnitudes (training only)

        Returns:
            cls_logits:  [B, T]
            disp_logits: [B, T]
            attn_weights:[B*T, 196] or None  (only when use_cross_attn=True)
        """
        attn_weights = None

        if self.use_cross_attn:
            assert patches is not None, \
                "use_cross_attn=True 时 forward() 必须传入 patches"
            x, attn_weights = self.feat_proj(patches, flow_mag)  # [B,T,512], [BT,196]
        else:
            x = self.feat_proj(features)                          # [B, T, 512]

        h = self.head(x)                                          # [B, T, head_out]
        cls_logits  = self.fc_cls(h).squeeze(-1)                  # [B, T]
        disp_logits = self.fc_disp(h).squeeze(-1)                 # [B, T]
        return cls_logits, disp_logits, attn_weights


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device,
                    lambda_disp=0.01, lambda_flow=0.0):
    """Train one epoch.

    Returns:
        (avg_total_loss, avg_cls_loss, avg_disp_loss, avg_flow_loss)
    """
    model.train()
    total_loss = total_cls = total_disp = total_flow = 0.0
    n = 0

    for batch in loader:
        features    = batch["features"].to(device)     # [B, T, D]
        cls_target  = batch["cls_target"].to(device)
        disp_target = batch["disp_target"].to(device)
        disp_mask   = batch["disp_mask"].to(device)
        valid_len   = batch["valid_len"].to(device)

        # spatial 模式: 传入 patches + flow_mag
        patches  = batch.get("patches")
        flow_mag = batch.get("flow_mag")
        if patches  is not None: patches  = patches.to(device)
        if flow_mag is not None: flow_mag = flow_mag.to(device)

        cls_logits, disp_logits, attn_weights = model(
            features, patches=patches, flow_mag=flow_mag)

        # --- 主 loss ---
        loss, cls_l, disp_l = cls_displacement_loss(
            cls_logits, disp_logits,
            cls_target, disp_target, disp_mask, valid_len,
            lambda_disp=lambda_disp)

        # --- Flow KL auxiliary loss ---
        flow_l = 0.0
        if (lambda_flow > 0.0 and attn_weights is not None
                and flow_mag is not None):
            kl = flow_kl_loss(attn_weights, flow_mag, valid_len)
            loss = loss + lambda_flow * kl
            flow_l = kl.item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_cls  += cls_l
        total_disp += disp_l
        total_flow += flow_l
        n += 1

    denom = max(n, 1)
    return (total_loss / denom, total_cls / denom,
            total_disp / denom, total_flow / denom)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, tolerances=(1, 2, 3),
             threshold=0.5, min_distance=10, lambda_disp=0.01):
    """Evaluate: cls+disp detection → P/R/F1 at multiple tolerances."""
    model.eval()

    total_loss = total_cls_loss = total_disp_loss = 0.0
    n_batches = 0
    matched = {k: 0 for k in tolerances}
    total_gt = total_pred = 0

    for batch in loader:
        features    = batch["features"].to(device)
        cls_target  = batch["cls_target"].to(device)
        disp_target = batch["disp_target"].to(device)
        disp_mask   = batch["disp_mask"].to(device)
        valid_len   = batch["valid_len"].to(device)
        hit_indices = batch["hit_indices"]

        patches  = batch.get("patches")
        flow_mag = batch.get("flow_mag")
        if patches  is not None: patches  = patches.to(device)
        if flow_mag is not None: flow_mag = flow_mag.to(device)

        # eval 时 flow_mag=None (不用 KL loss, 不需要真实 flow)
        cls_logits, disp_logits, _ = model(
            features, patches=patches, flow_mag=None)

        loss, cls_l, disp_l = cls_displacement_loss(
            cls_logits, disp_logits,
            cls_target, disp_target, disp_mask, valid_len,
            lambda_disp=lambda_disp)

        total_loss     += loss.item()
        total_cls_loss += cls_l
        total_disp_loss += disp_l
        n_batches += 1

        B = cls_logits.shape[0]
        for b in range(B):
            vl = valid_len[b].item()
            gt = hit_indices[b]

            pred = cls_disp_detect(
                cls_logits[b], disp_logits[b], vl,
                threshold=threshold, min_distance=min_distance)
            total_pred += len(pred)
            total_gt   += len(gt)

            for k in tolerances:
                n_m, _, _ = match_predictions(pred, gt, k)
                matched[k] += n_m

    results = {
        "loss":       total_loss     / max(n_batches, 1),
        "cls_loss":   total_cls_loss / max(n_batches, 1),
        "disp_loss":  total_disp_loss / max(n_batches, 1),
        "total_gt":   total_gt,
        "total_pred": total_pred,
    }
    for k in tolerances:
        p  = matched[k] / total_pred if total_pred > 0 else 0.0
        r  = matched[k] / total_gt   if total_gt   > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        results[f"P@{k}"] = p
        results[f"R@{k}"] = r
        results[f"F1@{k}"] = f1

    return results


# ---------------------------------------------------------------------------
# Inference benchmark
# ---------------------------------------------------------------------------

def inference_benchmark(feat_dim=1024, t_max=2700, device="cuda", n_runs=50):
    heads = ["identity", "bilstm", "mstcn", "transformer",
             "mamba2", "bimamba2", "gated_bimamba2"]
    dummy = torch.randn(1, t_max, feat_dim, device=device)
    results = {}

    for name in heads:
        try:
            model = MultiHitModel(name, feat_dim=feat_dim,
                                  t_max=t_max).to(device)
            model.eval()
            params = sum(p.numel() for p in model.parameters()
                         if p.requires_grad)

            for _ in range(5):
                with torch.no_grad():
                    model(dummy)
            if "cuda" in str(device):
                torch.cuda.synchronize()

            t0 = time.time()
            for _ in range(n_runs):
                with torch.no_grad():
                    model(dummy)
            if "cuda" in str(device):
                torch.cuda.synchronize()

            avg_ms = (time.time() - t0) / n_runs * 1000
            results[name] = (avg_ms, params)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            results[name] = (None, 0, str(e))

    return results


def single_model_timing(model, feat_dim, t_max, device, n_runs=50):
    model.eval()
    dummy = torch.randn(1, t_max, feat_dim, device=device)

    for _ in range(5):
        with torch.no_grad():
            model(dummy)
    if "cuda" in str(device):
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            model(dummy)
    if "cuda" in str(device):
        torch.cuda.synchronize()

    return (time.time() - t0) / n_runs * 1000


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("TempoPeak Multi-Hit Training")

    # Data: full-video mode
    p.add_argument("--features_pt",  type=str, default=None,
                   help="CLS token features [N, D] .pt")
    p.add_argument("--hit_json",     type=str, default=None,
                   help="JSON with HIT array (global frame indices)")
    p.add_argument("--spatial_pt",   type=str, default=None,
                   help="Spatial patch tokens [N, 196, 1024] .pt (CrossAttn mode)")
    p.add_argument("--flow_pt",      type=str, default=None,
                   help="Farneback flow magnitude [N, 196] .pt (flow KL supervision)")

    # Data: clip folder mode
    p.add_argument("--clip_data_dir", type=str, default=None)
    p.add_argument("--backbone",      type=str, default="dinov2",
                   choices=["resnet18", "resnet34", "vit_small", "dinov2"])

    # Model
    p.add_argument("--temporal_head", type=str, default="gated_bimamba2",
                   choices=["identity", "bilstm", "mstcn", "transformer",
                            "mamba2", "bimamba2", "gated_bimamba2"])
    p.add_argument("--use_cross_attn", action="store_true",
                   help="用 CrossAttnPooling 替换 nn.Linear(feat_dim→512)")
    p.add_argument("--feat_dim",       type=int, default=None,
                   help="Feature dim (auto-detected if None)")

    # Training
    p.add_argument("--segment_len",  type=int,   default=2700)
    p.add_argument("--stride",       type=int,   default=None)
    p.add_argument("--radius",       type=int,   default=5)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda_disp",  type=float, default=0.01)
    p.add_argument("--lambda_flow",  type=float, default=0.5,
                   help="Flow KL loss 初始权重, 线性衰减到 0")
    p.add_argument("--flow_anneal_epochs", type=int, default=50,
                   help="Flow KL loss 从 lambda_flow 线性衰减到 0 的 epoch 数")
    p.add_argument("--epochs",       type=int,   default=300)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)

    # Eval / Benchmark
    p.add_argument("--eval_only",  action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--benchmark",  action="store_true")
    p.add_argument("--device",     type=str, default="auto")

    return p.parse_args()


def resolve_device(s):
    if s == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return s


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = resolve_device(args.device)
    seed_everything(args.seed)

    # ── Benchmark ─────────────────────────────────────────────────
    if args.benchmark:
        fd = args.feat_dim or 1024
        print(f"\n--- Inference Benchmark (T={args.segment_len}, D={fd}) ---")
        results = inference_benchmark(
            feat_dim=fd, t_max=args.segment_len, device=device)

        ref_ms = results.get("gated_bimamba2", (None,))[0]
        for name in ["identity", "bilstm", "mstcn", "transformer",
                     "mamba2", "bimamba2", "gated_bimamba2"]:
            if name not in results:
                continue
            entry = results[name]
            if entry[0] is not None:
                ms, params = entry[0], entry[1]
                suffix = ""
                if ref_ms and name != "gated_bimamba2":
                    suffix = f"  ({ms/ref_ms:.1f}× vs GatedBiMamba2)"
                print(f"  {name:18s} {ms:7.1f} ms  {params:>10,} params{suffix}")
            else:
                print(f"  {name:18s}  FAILED: {entry}")
        return

    # ── Data ──────────────────────────────────────────────────────
    train_loader, val_loader, feat_dim = build_multihit_dataloaders(
        features_pt=args.features_pt,
        hit_json=args.hit_json,
        spatial_pt=args.spatial_pt if args.use_cross_attn else None,
        flow_pt=args.flow_pt      if args.use_cross_attn else None,
        clip_data_dir=args.clip_data_dir,
        backbone=args.backbone,
        segment_len=args.segment_len,
        stride=args.stride,
        radius=args.radius,
        batch_size=args.batch_size,
        seed=args.seed,
        eval_only=args.eval_only,
    )

    if args.feat_dim is not None:
        feat_dim = args.feat_dim

    # ── Model ─────────────────────────────────────────────────────
    model = MultiHitModel(
        args.temporal_head,
        feat_dim=feat_dim,
        t_max=args.segment_len,
        use_cross_attn=args.use_cross_attn,
    ).to(device)

    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    cross_attn_str = "CrossAttnPooling" if args.use_cross_attn else "LinearProj"
    print(f"Device: {device} | Head: {args.temporal_head} | "
          f"FeatProj: {cross_attn_str}")
    print(f"T={args.segment_len} D={feat_dim} radius={args.radius} "
          f"lambda_disp={args.lambda_disp} lambda_flow={args.lambda_flow}")
    print(f"Parameters: {trainable:,} trainable / {total_params:,} total")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded: {args.checkpoint}")

    # ── Eval-only ─────────────────────────────────────────────────
    if args.eval_only:
        metrics = evaluate(model, val_loader, device,
                           lambda_disp=args.lambda_disp)
        print(f"\n[Eval] loss={metrics['loss']:.4f} | "
              f"F1@1={metrics['F1@1']*100:.1f}% "
              f"F1@2={metrics['F1@2']*100:.1f}% "
              f"F1@3={metrics['F1@3']*100:.1f}% "
              f"(GT={metrics['total_gt']} Pred={metrics['total_pred']})")
        return

    # ── Training ──────────────────────────────────────────────────
    assert train_loader is not None

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5,
    )

    ckpt_dir = os.path.join(os.path.dirname(__file__) or ".", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # checkpoint 命名: 区分 CrossAttn 和 Linear 实验
    suffix = "_ca" if args.use_cross_attn else ""
    best_tag = f"best_{args.temporal_head}{suffix}_multihit"
    best_f1  = 0.0

    # CSV: 新增 flow_loss 列
    csv_path   = os.path.join(ckpt_dir, f"log_{args.temporal_head}{suffix}.csv")
    csv_fields = [
        "epoch", "lambda_flow_eff",
        "train_loss", "train_cls", "train_disp", "train_flow",
        "val_loss", "val_cls", "val_disp",
        "P@1", "R@1", "F1@1", "P@2", "R@2", "F1@2", "P@3", "R@3", "F1@3",
        "GT", "Pred", "best_F1@1", "train_s", "eval_s", "lr",
    ]
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()
    print(f"CSV: {csv_path}")

    for epoch in range(1, args.epochs + 1):
        # Flow KL loss 线性衰减:
        #   epoch 1:                   lambda_flow_eff = lambda_flow (全量)
        #   epoch flow_anneal_epochs:  lambda_flow_eff = 0
        #   epoch > flow_anneal_epochs: lambda_flow_eff = 0
        if args.use_cross_attn and args.lambda_flow > 0:
            progress = min(epoch / max(args.flow_anneal_epochs, 1), 1.0)
            lambda_flow_eff = args.lambda_flow * (1.0 - progress)
        else:
            lambda_flow_eff = 0.0

        # Train
        t0 = time.time()
        train_loss, train_cls, train_disp, train_flow = train_one_epoch(
            model, train_loader, optimizer, device,
            lambda_disp=args.lambda_disp,
            lambda_flow=lambda_flow_eff,
        )
        train_time = time.time() - t0

        # Eval
        t0 = time.time()
        metrics = evaluate(model, val_loader, device,
                           lambda_disp=args.lambda_disp)
        eval_time = time.time() - t0

        # Best checkpoint (F1@1)
        f1_1    = metrics["F1@1"]
        is_best = f1_1 > best_f1
        if is_best:
            best_f1 = f1_1
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args":                 vars(args),
                "metrics":              metrics,
            }, os.path.join(ckpt_dir, f"{best_tag}.pt"))

        star = " *" if is_best else ""
        flow_str = (f" flow_kl={train_flow:.4f} λ={lambda_flow_eff:.3f}"
                    if args.use_cross_attn else "")
        print(f"[Ep {epoch}/{args.epochs}] "
              f"loss={train_loss:.4f} (cls={train_cls:.4f} "
              f"disp={train_disp:.4f}{flow_str}) | "
              f"F1@1={metrics['F1@1']*100:.1f}% "
              f"F1@2={metrics['F1@2']*100:.1f}% "
              f"F1@3={metrics['F1@3']*100:.1f}% "
              f"(GT={metrics['total_gt']} Pred={metrics['total_pred']}) | "
              f"train={train_time:.1f}s eval={eval_time:.1f}s "
              f"best={best_f1*100:.1f}%{star}")

        csv_writer.writerow({
            "epoch":          epoch,
            "lambda_flow_eff": f"{lambda_flow_eff:.4f}",
            "train_loss":     f"{train_loss:.6f}",
            "train_cls":      f"{train_cls:.6f}",
            "train_disp":     f"{train_disp:.6f}",
            "train_flow":     f"{train_flow:.6f}",
            "val_loss":       f"{metrics['loss']:.6f}",
            "val_cls":        f"{metrics['cls_loss']:.6f}",
            "val_disp":       f"{metrics['disp_loss']:.6f}",
            "P@1":  f"{metrics['P@1']:.4f}",
            "R@1":  f"{metrics['R@1']:.4f}",
            "F1@1": f"{metrics['F1@1']:.4f}",
            "P@2":  f"{metrics['P@2']:.4f}",
            "R@2":  f"{metrics['R@2']:.4f}",
            "F1@2": f"{metrics['F1@2']:.4f}",
            "P@3":  f"{metrics['P@3']:.4f}",
            "R@3":  f"{metrics['R@3']:.4f}",
            "F1@3": f"{metrics['F1@3']:.4f}",
            "GT":   metrics["total_gt"],
            "Pred": metrics["total_pred"],
            "best_F1@1": f"{best_f1:.4f}",
            "train_s":   f"{train_time:.1f}",
            "eval_s":    f"{eval_time:.1f}",
            "lr":        f"{optimizer.param_groups[0]['lr']:.6f}",
        })
        csv_file.flush()

        scheduler.step()

        if epoch == 1:
            ms = single_model_timing(model, feat_dim, args.segment_len, device)
            print(f"  → Inference: {ms:.1f} ms/segment (T={args.segment_len}, B=1)")

    csv_file.close()

    torch.save({
        "epoch":            args.epochs,
        "model_state_dict": model.state_dict(),
        "args":             vars(args),
    }, os.path.join(ckpt_dir, f"last_{args.temporal_head}{suffix}_multihit.pt"))

    print(f"\nDone. Best F1@1={best_f1*100:.1f}% → checkpoints/{best_tag}.pt")


if __name__ == "__main__":
    main()
