"""TempoPeak training entry point."""

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from config import parse_args, resolve_device
from dataloader import build_dataloaders
from eval import compute_metrics, gaussian_target, soft_cross_entropy
from model import TempoPeakModel


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model: torch.nn.Module, loader, optimizer,
                    args, device: str) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        frames = batch["frames"].to(device)       # [B, T, 3, IMG, IMG]
        t_gt = batch["t_gt"].to(device)            # [B]
        valid_len = batch["valid_len"].to(device)  # [B]

        logits = model(frames)  # [B, T]

        if args.target_type == "gaussian":
            targets = gaussian_target(t_gt, args.t_max, sigma=args.sigma,
                                      valid_len=valid_len)
            loss = soft_cross_entropy(logits, targets)
        else:
            loss = F.cross_entropy(logits, t_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: str) -> dict[str, float]:
    """Evaluate on val set, return aggregated metrics."""
    model.eval()
    all_logits = []
    all_tgt = []
    all_vlen = []

    for batch in loader:
        frames = batch["frames"].to(device)
        t_gt = batch["t_gt"].to(device)
        valid_len = batch["valid_len"].to(device)
        logits = model(frames)
        all_logits.append(logits)
        all_tgt.append(t_gt)
        all_vlen.append(valid_len)

    if not all_logits:
        return {"mae": 999, "acc1": 0, "acc3": 0, "acc5": 0, "entropy": 0}

    logits = torch.cat(all_logits, dim=0)
    t_gt = torch.cat(all_tgt, dim=0)
    valid_len = torch.cat(all_vlen, dim=0)
    return compute_metrics(logits, t_gt, valid_len=valid_len)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    seed_everything(args.seed)
    print(f"Device: {device} | Head: {args.temporal_head} | T_max: {args.t_max} "
          f"| Target: {args.target_type} | LR: {args.lr}")

    # Data — auto-detect feature mode
    use_features = not args.no_features
    train_loader, val_loader = build_dataloaders(
        args.data_dir, args.t_max, args.batch_size, seed=args.seed,
        use_features=use_features, backbone=args.backbone)

    # Detect if features were actually loaded (check first batch shape)
    sample = train_loader.dataset[train_loader.dataset.samples[0][0]
                                  if hasattr(train_loader.dataset, 'samples')
                                  else 0]
    feature_mode = sample["frames"].dim() == 2  # [T, D] vs [T, 3, H, W]
    feat_dim = sample["frames"].shape[-1] if feature_mode else 512
    mode_str = f"features (D={feat_dim})" if feature_mode else "images"
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Mode: {mode_str}")

    # Model
    model = TempoPeakModel(
        temporal_head=args.temporal_head,
        feat_dim=feat_dim,
        use_backbone=not feature_mode,
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    # Checkpointing
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_tag = f"best_{args.temporal_head}_{args.t_max}"
    best_mae = float("inf")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, args, device)
        metrics = evaluate(model, val_loader, device)

        is_best = metrics["mae"] < best_mae
        if is_best:
            best_mae = metrics["mae"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "metrics": metrics,
            }, os.path.join(ckpt_dir, f"{best_tag}.pt"))

        star = " *" if is_best else ""
        print(f"[Epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} | "
              f"MAE={metrics['mae']:.2f} Acc@1={metrics['acc1']:.1f}% "
              f"Acc@3={metrics['acc3']:.1f}% Acc@5={metrics['acc5']:.1f}% "
              f"Entropy={metrics['entropy']:.3f} | best_MAE={best_mae:.2f}{star}")

    # Save last checkpoint
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }, os.path.join(ckpt_dir, f"last_{args.temporal_head}_{args.t_max}.pt"))

    print(f"\nDone. Best MAE={best_mae:.2f} saved to checkpoints/{best_tag}.pt")


if __name__ == "__main__":
    main()
