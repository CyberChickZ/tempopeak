"""TempoPeak multi-hit training & evaluation.

Trains a temporal head on long sequences with multiple hit events.
Supports two data modes:
  A) Full video:  --features_pt + --hit_json   (FullVideoDataset, segment_len segments)
  B) Clip folder: --clip_data_dir              (ClipFolderDataset, one clip = one sample)

Usage:
  # Train on full video
  python train_multihit.py \\
      --features_pt data/00001_features.pt \\
      --hit_json datasets/v1/clips/0006/00001.json \\
      --temporal_head bimamba2

  # Eval on old clip data
  python train_multihit.py \\
      --clip_data_dir datasets/v1/export \\
      --backbone resnet18 \\
      --eval_only \\
      --checkpoint checkpoints/best_bimamba2_multihit.pt

  # Inference benchmark (all heads)
  python train_multihit.py --benchmark --feat_dim 1024
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from temporal_heads import HEAD_REGISTRY, TransformerHead
from dataloader_multihit import build_multihit_dataloaders
from eval_multihit import soft_cross_entropy, detect_peaks, match_predictions


# ---------------------------------------------------------------------------
# Model (feature-only, no backbone)
# ---------------------------------------------------------------------------

class MultiHitModel(nn.Module):
    """Feature-only model for multi-hit detection.

    All features are projected to 512-d before the temporal head
    (consistent with existing heads that expect 512-d input).

    Input:  [B, T, D]  pre-extracted features
    Output: [B, T]     per-frame hit logits
    """

    def __init__(self, temporal_head_name, feat_dim=1024, t_max=2700):
        super().__init__()

        # Project to 512-d (all heads expect 512-d input)
        if feat_dim != 512:
            self.feat_proj = nn.Linear(feat_dim, 512)
        else:
            self.feat_proj = nn.Identity()

        # Temporal head (TransformerHead needs explicit max_len)
        if temporal_head_name == "transformer":
            self.head = TransformerHead(d_model=512, max_len=t_max)
        else:
            self.head = HEAD_REGISTRY[temporal_head_name]()

        head_out = HEAD_REGISTRY[temporal_head_name].out_dim
        self.fc = nn.Linear(head_out, 1)

    def forward(self, x):
        x = self.feat_proj(x)             # [B, T, 512]
        h = self.head(x)                   # [B, T, D_out]
        return self.fc(h).squeeze(-1)      # [B, T]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    """Train one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        features = batch["features"].to(device)
        target = batch["target"].to(device)
        valid_len = batch["valid_len"].to(device)

        logits = model(features)
        loss = soft_cross_entropy(logits, target, valid_len)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, tolerances=(1, 2, 3),
             sigma_smooth=3.0, height_ratio=0.15, min_distance=10):
    """Evaluate model: peak detection → P/R/F1 at multiple tolerances.

    Returns dict with loss, P@k, R@k, F1@k, total_gt, total_pred.
    """
    model.eval()

    total_loss = 0.0
    n_batches = 0
    matched = {k: 0 for k in tolerances}
    total_gt = 0
    total_pred = 0

    for batch in loader:
        features = batch["features"].to(device)
        target = batch["target"].to(device)
        valid_len = batch["valid_len"].to(device)
        hit_indices = batch["hit_indices"]  # list of list[int]

        logits = model(features)
        loss = soft_cross_entropy(logits, target, valid_len)
        total_loss += loss.item()
        n_batches += 1

        B = logits.shape[0]
        for b in range(B):
            vl = valid_len[b].item()
            logits_b = logits[b, :vl].cpu().numpy()
            gt = hit_indices[b]

            pred = detect_peaks(logits_b, sigma_smooth, height_ratio,
                                min_distance)
            total_pred += len(pred)
            total_gt += len(gt)

            for k in tolerances:
                n_m, _, _ = match_predictions(pred, gt, k)
                matched[k] += n_m

    results = {"loss": total_loss / max(n_batches, 1),
               "total_gt": total_gt, "total_pred": total_pred}
    for k in tolerances:
        p = matched[k] / total_pred if total_pred > 0 else 0.0
        r = matched[k] / total_gt if total_gt > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        results[f"P@{k}"] = p
        results[f"R@{k}"] = r
        results[f"F1@{k}"] = f1

    return results


# ---------------------------------------------------------------------------
# Inference benchmark
# ---------------------------------------------------------------------------

def inference_benchmark(feat_dim=1024, t_max=2700, device="cuda", n_runs=50):
    """Benchmark single-sample inference for all temporal heads.

    Returns dict {head_name: (avg_ms, trainable_params)}.
    """
    heads = ["identity", "bilstm", "mstcn", "transformer", "mamba2", "bimamba2"]
    dummy = torch.randn(1, t_max, feat_dim, device=device)
    results = {}

    for name in heads:
        try:
            model = MultiHitModel(name, feat_dim=feat_dim,
                                  t_max=t_max).to(device)
            model.eval()
            params = sum(p.numel() for p in model.parameters()
                         if p.requires_grad)

            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    model(dummy)
            if device == "cuda" or (hasattr(device, 'type')
                                    and device.type == 'cuda'):
                torch.cuda.synchronize()

            # Timed runs
            t0 = time.time()
            for _ in range(n_runs):
                with torch.no_grad():
                    model(dummy)
            if device == "cuda" or (hasattr(device, 'type')
                                    and device.type == 'cuda'):
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
    """Benchmark a single model, return avg_ms."""
    model.eval()
    dummy = torch.randn(1, t_max, feat_dim, device=device)

    for _ in range(5):
        with torch.no_grad():
            model(dummy)
    if device == "cuda" or (hasattr(device, 'type') and device.type == 'cuda'):
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            model(dummy)
    if device == "cuda" or (hasattr(device, 'type') and device.type == 'cuda'):
        torch.cuda.synchronize()

    return (time.time() - t0) / n_runs * 1000


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("TempoPeak Multi-Hit Training")

    # Data: full-video mode
    p.add_argument("--features_pt", type=str, default=None,
                   help="Pre-extracted features [N, D] .pt file")
    p.add_argument("--hit_json", type=str, default=None,
                   help="JSON with HIT array (global frame indices)")

    # Data: clip folder mode
    p.add_argument("--clip_data_dir", type=str, default=None,
                   help="export/ directory with annotated clips")
    p.add_argument("--backbone", type=str, default="dinov2",
                   choices=["resnet18", "resnet34", "vit_small", "dinov2"])

    # Model
    p.add_argument("--temporal_head", type=str, default="bimamba2",
                   choices=["identity", "bilstm", "mstcn", "transformer",
                            "mamba2", "bimamba2"])
    p.add_argument("--feat_dim", type=int, default=None,
                   help="Feature dimension (auto-detected from data if None)")

    # Training
    p.add_argument("--segment_len", type=int, default=2700)
    p.add_argument("--stride", type=int, default=None,
                   help="Segment stride (default=segment_len, no overlap)")
    p.add_argument("--sigma", type=float, default=2.0,
                   help="Gaussian target sigma")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # Eval / Benchmark
    p.add_argument("--eval_only", action="store_true",
                   help="Evaluate only (no training)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Load model weights from checkpoint")
    p.add_argument("--benchmark", action="store_true",
                   help="Run inference benchmark for all heads")

    # Device
    p.add_argument("--device", type=str, default="auto",
                   help="auto = cuda if available else cpu (never MPS)")

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

    # ── Benchmark mode ────────────────────────────────────────────
    if args.benchmark:
        fd = args.feat_dim or 1024
        print(f"\n--- Inference Benchmark (T={args.segment_len}, B=1, "
              f"D={fd}, device={device}) ---")
        results = inference_benchmark(
            feat_dim=fd, t_max=args.segment_len, device=device)

        bimamba_ms = results.get("bimamba2", (None,))[0]
        for name in ["identity", "bilstm", "mstcn",
                      "transformer", "mamba2", "bimamba2"]:
            if name not in results:
                continue
            entry = results[name]
            if isinstance(entry, tuple) and len(entry) >= 2 and entry[0] is not None:
                ms, params = entry[0], entry[1]
                suffix = ""
                if bimamba_ms and bimamba_ms > 0 and name != "bimamba2":
                    ratio = ms / bimamba_ms
                    suffix = f"  ← {ratio:.0f}× vs BiMamba2" if ratio > 2 else ""
                print(f"  {name:14s} {ms:8.1f} ms  ({params:,} params){suffix}")
            else:
                print(f"  {name:14s}   FAILED: {entry}")
        return

    # ── Data ──────────────────────────────────────────────────────
    train_loader, val_loader, feat_dim = build_multihit_dataloaders(
        features_pt=args.features_pt,
        hit_json=args.hit_json,
        clip_data_dir=args.clip_data_dir,
        backbone=args.backbone,
        segment_len=args.segment_len,
        stride=args.stride,
        sigma=args.sigma,
        batch_size=args.batch_size,
        seed=args.seed,
        eval_only=args.eval_only,
    )

    if args.feat_dim is not None:
        feat_dim = args.feat_dim

    # ── Model ─────────────────────────────────────────────────────
    model = MultiHitModel(
        args.temporal_head, feat_dim=feat_dim, t_max=args.segment_len,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | Head: {args.temporal_head} | "
          f"T={args.segment_len} D={feat_dim}")
    print(f"Parameters: {trainable:,} trainable / {total_params:,} total")

    # Load checkpoint
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Eval-only mode ────────────────────────────────────────────
    if args.eval_only:
        t0 = time.time()
        metrics = evaluate(model, val_loader, device)
        eval_time = time.time() - t0

        print(f"\n[Eval] loss={metrics['loss']:.4f} | "
              f"P@2={metrics['P@2']*100:.1f}% "
              f"R@2={metrics['R@2']*100:.1f}% "
              f"F1@1={metrics['F1@1']*100:.1f}% "
              f"F1@2={metrics['F1@2']*100:.1f}% "
              f"F1@3={metrics['F1@3']*100:.1f}% "
              f"(GT={metrics['total_gt']} Pred={metrics['total_pred']}) "
              f"| eval={eval_time:.1f}s")

        # Single-model inference timing
        ms = single_model_timing(model, feat_dim, args.segment_len, device)
        print(f"Inference: {ms:.1f} ms/segment "
              f"(T={args.segment_len}, B=1)")
        return

    # ── Training ──────────────────────────────────────────────────
    assert train_loader is not None, "No training data"

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5,
    )

    ckpt_dir = os.path.join(os.path.dirname(__file__) or ".", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_tag = f"best_{args.temporal_head}_multihit"
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        train_time = time.time() - t0

        # Eval
        t0 = time.time()
        metrics = evaluate(model, val_loader, device)
        eval_time = time.time() - t0

        # Best check (F1@2 as primary metric)
        f1_2 = metrics["F1@2"]
        is_best = f1_2 > best_f1
        if is_best:
            best_f1 = f1_2
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "metrics": metrics,
            }, os.path.join(ckpt_dir, f"{best_tag}.pt"))

        star = " *" if is_best else ""
        print(f"[Epoch {epoch}/{args.epochs}] loss={train_loss:.4f} | "
              f"P@2={metrics['P@2']*100:.1f}% "
              f"R@2={metrics['R@2']*100:.1f}% "
              f"F1@1={metrics['F1@1']*100:.1f}% "
              f"F1@2={metrics['F1@2']*100:.1f}% "
              f"F1@3={metrics['F1@3']*100:.1f}% "
              f"(GT={metrics['total_gt']} Pred={metrics['total_pred']}) "
              f"| train={train_time:.1f}s eval={eval_time:.1f}s "
              f"| best_F1@2={best_f1*100:.1f}%{star}")

        scheduler.step()

        # After epoch 1: single-sample inference timing
        if epoch == 1:
            ms = single_model_timing(model, feat_dim, args.segment_len,
                                     device)
            print(f"  → Inference: {ms:.1f} ms/segment "
                  f"(T={args.segment_len}, B=1)")

    # Save last
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }, os.path.join(ckpt_dir, f"last_{args.temporal_head}_multihit.pt"))

    print(f"\nDone. Best F1@2={best_f1*100:.1f}% → "
          f"checkpoints/{best_tag}.pt")


if __name__ == "__main__":
    main()
