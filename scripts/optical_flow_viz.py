#!/usr/bin/env python3
"""
Optical Flow Visualization — (t-1, t) frame-pair analysis around hit events.

Uses torchvision RAFT-Large (ECCV 2020 + Sintel-finetuned v2 weights),
which remains a top-tier optical flow method. The architecture is the same
foundation behind SEA-RAFT (ECCV 2024 Oral / Best Paper Candidate) and
FlowDiffuser (CVPR 2024).

For the absolute bleeding edge, see:
  - SEA-RAFT: https://github.com/princeton-vl/SEA-RAFT  (ECCV 2024)
  - FlowDiffuser: CVPR 2024 (diffusion-based flow)
  - UniMatch: TPAMI 2024 (unified matching)

Event cameras (DVS) are a different sensor modality — they capture per-pixel
brightness changes asynchronously and require specialized hardware. Not
applicable to standard frame-based video.

Usage:
    python scripts/optical_flow_viz.py                          # random clip
    python scripts/optical_flow_viz.py --video 0001 --clip 00003  # specific
    python scripts/optical_flow_viz.py --iters 32               # more RAFT iters
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.utils import flow_to_image


def load_clip(clip_dir: Path):
    """Load annot.json and return metadata + frame directory."""
    annot_path = clip_dir / "annot.json"
    with open(annot_path) as f:
        annot = json.load(f)
    frames_dir = clip_dir / "frames"
    assert frames_dir.exists(), f"frames/ not found in {clip_dir}"
    return annot, frames_dir


def load_frame(frames_dir: Path, idx: int) -> np.ndarray:
    """Load a single frame as RGB numpy array."""
    path = frames_dir / f"{idx}.jpg"
    img = cv2.imread(str(path))
    assert img is not None, f"Failed to load {path}"
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_for_raft(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert HWC uint8 numpy → BCHW float tensor, normalized to [-1, 1]."""
    t = torch.from_numpy(img).permute(2, 0, 1).float()  # CHW
    t = t.unsqueeze(0).to(device)  # BCHW
    # RAFT expects [-1, 1] range
    t = 2.0 * (t / 255.0) - 1.0
    return t


def flow_to_hsv(flow_np: np.ndarray) -> np.ndarray:
    """Convert (2, H, W) flow to HSV color-coded image (H, W, 3) uint8.

    Hue = direction, Saturation = 1, Value = magnitude (clamped).
    """
    u, v = flow_np[0], flow_np[1]
    mag = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)

    hsv = np.zeros((*u.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)  # Hue [0,179]
    hsv[..., 1] = 255  # Full saturation
    # Normalize magnitude to [0, 255], clip at 99th percentile for contrast
    mag_clip = np.clip(mag, 0, np.percentile(mag, 99) + 1e-5)
    hsv[..., 2] = (mag_clip / (mag_clip.max() + 1e-5) * 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def draw_flow_arrows(img: np.ndarray, flow_np: np.ndarray, step: int = 24,
                     scale: float = 2.0) -> np.ndarray:
    """Overlay motion vectors as arrows on the image."""
    vis = img.copy()
    h, w = flow_np.shape[1], flow_np.shape[2]
    u, v = flow_np[0], flow_np[1]

    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            dx, dy = u[y, x] * scale, v[y, x] * scale
            mag = np.sqrt(dx**2 + dy**2)
            if mag < 0.5:
                continue
            x2, y2 = int(x + dx), int(y + dy)
            # Color by magnitude: blue→red
            intensity = min(mag / 20.0, 1.0)
            color = (int(255 * intensity), 50, int(255 * (1 - intensity)))
            cv2.arrowedLine(vis, (x, y), (x2, y2), color, 1, tipLength=0.3)
    return vis


@torch.no_grad()
def compute_raft_flow(img1: np.ndarray, img2: np.ndarray, device: torch.device,
                      num_iters: int = 20) -> np.ndarray:
    """Compute optical flow from img1 → img2 using RAFT-Large.

    Returns flow as numpy array (2, H, W).
    """
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True)
    model = model.to(device).eval()

    t1 = preprocess_for_raft(img1, device)
    t2 = preprocess_for_raft(img2, device)

    # RAFT needs dimensions divisible by 8; pad if needed
    _, _, h, w = t1.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        t1 = torch.nn.functional.pad(t1, (0, pad_w, 0, pad_h), mode="replicate")
        t2 = torch.nn.functional.pad(t2, (0, pad_w, 0, pad_h), mode="replicate")

    # Run RAFT — returns list of flow predictions (iterative refinement)
    flow_preds = model(t1, t2, num_flow_updates=num_iters)
    # Take the last (most refined) prediction
    flow = flow_preds[-1]  # (B, 2, H', W')

    # Remove padding
    if pad_h > 0 or pad_w > 0:
        flow = flow[:, :, :h, :w]

    return flow[0].cpu().numpy()  # (2, H, W)


def visualize(img_prev: np.ndarray, img_curr: np.ndarray, flow: np.ndarray,
              hit_frame: int, clip_id: str, out_path: str):
    """Create a 2x3 visualization panel."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle(
        f"RAFT-Large Optical Flow — Clip {clip_id}, "
        f"frames (t-1={hit_frame-1}, t={hit_frame})",
        fontsize=16, fontweight="bold"
    )

    # Row 1: Input frames + overlay
    axes[0, 0].imshow(img_prev)
    axes[0, 0].set_title(f"Frame t-1 = {hit_frame - 1}", fontsize=13)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_curr)
    axes[0, 1].set_title(f"Frame t = {hit_frame} (HIT)", fontsize=13, color="red")
    axes[0, 1].axis("off")

    # Blend: 50% current frame + 50% previous frame
    blend = cv2.addWeighted(img_prev, 0.5, img_curr, 0.5, 0)
    axes[0, 2].imshow(blend)
    axes[0, 2].set_title("Frame Overlay (50/50 blend)", fontsize=13)
    axes[0, 2].axis("off")

    # Row 2: Flow visualizations
    # (a) HSV color wheel
    hsv_img = flow_to_hsv(flow)
    axes[1, 0].imshow(hsv_img)
    axes[1, 0].set_title("Flow Direction (HSV color wheel)", fontsize=13)
    axes[1, 0].axis("off")

    # (b) Flow magnitude heatmap
    mag = np.sqrt(flow[0]**2 + flow[1]**2)
    im = axes[1, 1].imshow(mag, cmap="inferno", vmin=0, vmax=np.percentile(mag, 99))
    axes[1, 1].set_title(f"Flow Magnitude (max={mag.max():.1f} px)", fontsize=13)
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # (c) Motion vectors on current frame
    arrow_img = draw_flow_arrows(img_curr, flow, step=24, scale=2.5)
    axes[1, 2].imshow(arrow_img)
    axes[1, 2].set_title("Motion Vectors on Frame t", fontsize=13)
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Optical flow visualization around hit frames")
    parser.add_argument("--data_root", type=str, default="datasets/v1/export",
                        help="Root directory of annotated clips")
    parser.add_argument("--video", type=str, default=None,
                        help="Video ID (e.g., '0001'). Random if not specified")
    parser.add_argument("--clip", type=str, default=None,
                        help="Clip ID (e.g., '00003'). Random if not specified")
    parser.add_argument("--hit_idx", type=int, default=0,
                        help="Which hit in the clip to visualize (0-indexed, default=0)")
    parser.add_argument("--iters", type=int, default=20,
                        help="Number of RAFT refinement iterations (default=20, max quality=32)")
    parser.add_argument("--output", type=str, default="optical_flow_viz.png",
                        help="Output image path")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda' or 'cpu' (auto-detected if not set)")
    args = parser.parse_args()

    # Resolve paths — use CWD-relative paths (run from project root)
    data_root = Path(args.data_root)

    # Auto-detect device (never MPS)
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Discover available videos and clips
    videos = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    assert videos, f"No video directories found in {data_root}"

    # Pick video
    video_id = args.video if args.video else random.choice(videos)
    video_dir = data_root / video_id
    assert video_dir.exists(), f"Video dir not found: {video_dir}"

    # Pick clip
    clips = sorted([d.name for d in video_dir.iterdir() if d.is_dir()])
    clip_id = args.clip if args.clip else random.choice(clips)
    clip_dir = video_dir / clip_id
    assert clip_dir.exists(), f"Clip dir not found: {clip_dir}"

    print(f"Selected: video={video_id}, clip={clip_id}")

    # Load annotation
    annot, frames_dir = load_clip(clip_dir)
    hits = annot["hits"]
    hitters = annot.get("hitters", [None] * len(hits))
    total_frames = annot["total_frames"]

    assert hits, f"No hits in clip {clip_id}"
    hit_idx = min(args.hit_idx, len(hits) - 1)
    hit_frame = hits[hit_idx]
    hitter = hitters[hit_idx] if hit_idx < len(hitters) else "?"

    print(f"Hit frame: {hit_frame} (hit #{hit_idx+1}/{len(hits)}, hitter=P{hitter})")
    print(f"Total frames: {total_frames}")

    # Ensure t-1 exists
    assert hit_frame > 0, f"Hit frame is 0, no t-1 available"

    # Load frame pair
    img_prev = load_frame(frames_dir, hit_frame - 1)
    img_curr = load_frame(frames_dir, hit_frame)
    print(f"Frame shape: {img_prev.shape}")

    # Compute optical flow
    print(f"Computing RAFT-Large flow ({args.iters} iterations)...")
    flow = compute_raft_flow(img_prev, img_curr, device, num_iters=args.iters)
    print(f"Flow shape: {flow.shape}, range: [{flow.min():.2f}, {flow.max():.2f}]")

    # Flow statistics
    mag = np.sqrt(flow[0]**2 + flow[1]**2)
    print(f"Flow magnitude — mean: {mag.mean():.2f}, median: {np.median(mag):.2f}, "
          f"max: {mag.max():.2f}, 95th: {np.percentile(mag, 95):.2f}")

    # Visualize
    out_path = args.output
    visualize(img_prev, img_curr, flow, hit_frame, f"{video_id}/{clip_id}", out_path)


if __name__ == "__main__":
    main()
