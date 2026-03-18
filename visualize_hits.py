"""Visualize hit frame predictions as progress-bar style timelines.

For each val sample, draws N+1 horizontal bars:
  - 1 bar for Ground Truth (green tick)
  - 1 bar per head checkpoint (red/yellow tick for pred)

Vertical lines on each bar mark the hit position.
Yellow = exact match (|error| <= 1), Red = wrong.

Usage:
    python visualize_hits.py --data_dir datasets/v1/export/0001
    python visualize_hits.py --data_dir datasets/v1/export/0001 --n_samples 4
"""

import argparse
import os
import random
import sys
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

from dataloader import build_dataloaders
from model import TempoPeakModel


# Colors (RGB)
COLOR_GT = (0, 200, 0)         # green
COLOR_CORRECT = (255, 200, 0)  # yellow
COLOR_WRONG = (220, 40, 40)    # red
COLOR_BAR_BG = (50, 50, 60)    # dark grey
COLOR_TEXT = (210, 210, 210)    # light grey
COLOR_TICK = (120, 120, 130)   # subtle tick marks

HEAD_COLORS = {
    "identity":    (150, 150, 150),
    "bilstm":      (100, 180, 255),
    "mamba2":      (255, 160, 80),
    "bimamba2":    (220, 100, 255),
    "transformer": (80, 220, 180),
}


def find_checkpoints(ckpt_dir: str, t_max: int = 32) -> OrderedDict:
    """Find all best_*_{t_max}.pt checkpoints, return {head_name: path}."""
    heads = OrderedDict()
    for name in ["identity", "bilstm", "mamba2", "bimamba2", "transformer"]:
        path = os.path.join(ckpt_dir, f"best_{name}_{t_max}.pt")
        if os.path.exists(path):
            heads[name] = path
    return heads


def load_model(ckpt_path: str, feat_dim: int, device: str):
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    state_dict = ckpt["model_state_dict"]
    has_backbone = any(k.startswith("backbone.") for k in state_dict)

    model = TempoPeakModel(
        temporal_head=args["temporal_head"],
        feat_dim=feat_dim,
        use_backbone=has_backbone,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, ckpt.get("metrics", {})


def predict(model, frames, valid_len, device):
    """Run inference, return predicted frame index."""
    with torch.no_grad():
        logits = model(frames.unsqueeze(0).to(device))  # [1, T]

    logits_np = logits.cpu().float().numpy()
    logits_np = gaussian_filter1d(logits_np, sigma=1.5, axis=-1)
    logits_sm = torch.from_numpy(logits_np)

    T = logits_sm.shape[1]
    mask = torch.arange(T) >= valid_len
    logits_sm[0, mask] = float("-inf")

    probs = F.softmax(logits_sm, dim=-1)[0]
    return probs.argmax().item()


def draw_bar(width: int, height: int, valid_len: int, t_max: int,
             hit_pos: int, label: str, color: tuple,
             is_gt: bool = False) -> np.ndarray:
    """Draw one horizontal progress bar with a hit marker.

    Args:
        width: pixel width of the bar area
        height: pixel height of the bar
        valid_len: actual number of frames in this sample
        t_max: maximum sequence length
        hit_pos: frame index to mark
        label: text label (head name)
        color: RGB tuple for the hit marker
        is_gt: if True, use thicker marker
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Layout
    label_w = 120
    bar_x0 = label_w + 10
    bar_x1 = width - 20
    bar_w = bar_x1 - bar_x0
    bar_y0 = 4
    bar_y1 = height - 4
    bar_h = bar_y1 - bar_y0

    # Draw bar background
    cv2.rectangle(canvas, (bar_x0, bar_y0), (bar_x1, bar_y1), COLOR_BAR_BG, -1)
    cv2.rectangle(canvas, (bar_x0, bar_y0), (bar_x1, bar_y1), (80, 80, 90), 1)

    # Draw subtle frame tick marks
    for t in range(0, valid_len, max(1, valid_len // 20)):
        x = bar_x0 + int(t / t_max * bar_w)
        cv2.line(canvas, (x, bar_y1 - 3), (x, bar_y1), COLOR_TICK, 1)

    # Draw valid region boundary
    valid_x = bar_x0 + int(valid_len / t_max * bar_w)
    if valid_len < t_max:
        # Grey out padding region
        cv2.rectangle(canvas, (valid_x, bar_y0), (bar_x1, bar_y1), (35, 35, 40), -1)

    # Draw hit marker
    hit_x = bar_x0 + int(hit_pos / t_max * bar_w)
    marker_w = max(3, bar_w // t_max)
    if is_gt:
        marker_w = max(4, marker_w + 1)

    cv2.rectangle(canvas,
                  (hit_x - marker_w // 2, bar_y0),
                  (hit_x + marker_w // 2, bar_y1),
                  color, -1)

    # Frame number next to marker
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.35
    num_text = str(hit_pos)
    (tw, th), _ = cv2.getTextSize(num_text, font, scale, 1)
    num_x = min(hit_x + marker_w // 2 + 3, bar_x1 - tw - 2)
    cv2.putText(canvas, num_text, (num_x, bar_y0 + th + 2),
                font, scale, color, 1, cv2.LINE_AA)

    # Label
    cv2.putText(canvas, label, (4, height // 2 + 5),
                font, 0.45, color, 1, cv2.LINE_AA)

    return canvas


def draw_sample(t_gt: int, valid_len: int, t_max: int,
                predictions: dict, clip_name: str,
                bar_width: int = 800, bar_height: int = 28) -> np.ndarray:
    """Draw all bars for one sample: GT + each head.

    predictions: {head_name: pred_frame_idx}
    """
    rows = []

    # Header
    header = np.zeros((22, bar_width, 3), dtype=np.uint8)
    cv2.putText(header, f"Clip: {clip_name}  |  valid_len={valid_len}  |  GT hit @ t={t_gt}",
                (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    rows.append(header)

    # GT bar
    gt_bar = draw_bar(bar_width, bar_height, valid_len, t_max,
                      t_gt, "GT", COLOR_GT, is_gt=True)
    rows.append(gt_bar)

    # Head bars
    for head_name, pred in predictions.items():
        error = abs(pred - t_gt)
        correct = error <= 1
        color = COLOR_CORRECT if correct else COLOR_WRONG
        acc_str = "OK" if correct else f"+{error}"
        label = f"{head_name} ({acc_str})"

        bar = draw_bar(bar_width, bar_height, valid_len, t_max,
                       pred, label, color)
        rows.append(bar)

    return np.vstack(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="datasets/v1/export/0001",
                        help="Data dir with pre-extracted features")
    parser.add_argument("--ckpt_dir", default=None,
                        help="Checkpoints dir (auto-detect)")
    parser.add_argument("--output", default="hit_visualization.png")
    parser.add_argument("--n_samples", type=int, default=8,
                        help="Number of val samples to show")
    parser.add_argument("--t_max", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bar_width", type=int, default=900)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Find checkpoints
    ckpt_dir = args.ckpt_dir or os.path.join(os.path.dirname(args.data_dir), "../../checkpoints")
    ckpt_dir = os.path.normpath(ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        ckpt_dir = "checkpoints"
    head_ckpts = find_checkpoints(ckpt_dir, args.t_max)
    if not head_ckpts:
        print(f"No checkpoints found in {ckpt_dir}")
        sys.exit(1)
    print(f"Found {len(head_ckpts)} head checkpoints: {list(head_ckpts.keys())}")

    # Build val dataset (use first checkpoint's args for backbone)
    first_ckpt = torch.load(list(head_ckpts.values())[0], map_location="cpu", weights_only=False)
    backbone = first_ckpt["args"].get("backbone", "resnet18")
    _, val_loader = build_dataloaders(
        args.data_dir, args.t_max, batch_size=1,
        seed=first_ckpt["args"].get("seed", 42),
        use_features=True, backbone=backbone)
    val_ds = val_loader.dataset
    print(f"Val samples: {len(val_ds)} | Backbone: {backbone}")

    # Detect feature dim
    s0 = val_ds[0]
    feat_dim = s0["frames"].shape[-1] if s0["frames"].dim() == 2 else 512

    # Load all models
    models = {}
    for head_name, ckpt_path in head_ckpts.items():
        model, metrics = load_model(ckpt_path, feat_dim, device)
        models[head_name] = model
        acc = metrics.get("acc1", 0)
        print(f"  {head_name}: Acc@1={acc:.1f}%")

    # Select samples: pick a mix of indices
    n_samples = min(args.n_samples, len(val_ds))
    indices = random.sample(range(len(val_ds)), n_samples)

    # Run inference and build visualization
    all_panels = []
    for si, idx in enumerate(indices):
        sample = val_ds[idx]
        frames = sample["frames"]
        t_gt = sample["t_gt"]
        valid_len = sample["valid_len"]
        meta = sample["meta"]
        clip_name = os.path.basename(meta["clip_dir"])

        predictions = {}
        for head_name, model in models.items():
            pred = predict(model, frames, valid_len, device)
            predictions[head_name] = pred

        panel = draw_sample(t_gt, valid_len, args.t_max,
                            predictions, clip_name,
                            bar_width=args.bar_width)
        all_panels.append(panel)

    # Add legend at top
    legend_h = 36
    legend = np.zeros((legend_h, args.bar_width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    x = 10
    cv2.putText(legend, "TempoPeak Hit Detection — Per-Head Comparison",
                (x, 14), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Legend items
    items = [("GT (green)", COLOR_GT),
             ("Correct |e|<=1 (yellow)", COLOR_CORRECT),
             ("Wrong |e|>1 (red)", COLOR_WRONG)]
    x = 10
    for text, color in items:
        cv2.rectangle(legend, (x, 20), (x + 14, 32), color, -1)
        cv2.putText(legend, text, (x + 18, 31), font, 0.33, (170, 170, 170), 1, cv2.LINE_AA)
        x += len(text) * 7 + 30

    # Separator
    sep = np.full((6, args.bar_width, 3), 25, dtype=np.uint8)

    parts = [legend]
    for panel in all_panels:
        parts.append(sep)
        parts.append(panel)

    final = np.vstack(parts)

    # Save
    cv2.imwrite(args.output, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    print(f"\nSaved: {args.output} ({final.shape[1]}x{final.shape[0]})")


if __name__ == "__main__":
    main()
