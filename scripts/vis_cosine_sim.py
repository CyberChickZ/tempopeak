"""Vis 5: Temporal cosine similarity curve.

For a clip, compute cosine_similarity(features[t], features[hit_frame]) for all t.
Shows how feature similarity to the hit frame changes over time — peaks at hit frame.

Usage (HPC, no GPU needed):
  python scripts/vis_cosine_sim.py \
      --clip_dir datasets/v1/export/0001/00001 \
      --backbone vit_small \
      --output paper/vis_data/vis5_cosine_sim.npz

Output:
  vis5_cosine_sim.npz with keys:
    cosine_sim: [T] cosine similarity curve
    hit_frames: list of hit frame indices
    total_frames: total frames in clip
    backbone: backbone name
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip_dir", type=str, required=True,
                   help="Clip directory with annot.json and features/")
    p.add_argument("--backbone", type=str, default="vit_small",
                   choices=["resnet18", "resnet34", "vit_small", "dinov2"])
    p.add_argument("--player", type=str, default="p1")
    p.add_argument("--hit_idx", type=int, default=0,
                   help="Which hit to use as reference (0-based)")
    p.add_argument("--output", type=str, default="paper/vis_data/vis5_cosine_sim.npz")
    args = p.parse_args()

    # Load annotation
    annot_path = os.path.join(args.clip_dir, "annot.json")
    with open(annot_path) as f:
        annot = json.load(f)
    hits = annot.get("hits", [])
    total_frames = annot.get("total_frames", 0)

    if not hits:
        print("Error: no hits in annotation")
        return

    hit_frame = hits[args.hit_idx]
    print(f"Clip: {args.clip_dir}")
    print(f"  Total frames: {total_frames}, hits: {hits}")
    print(f"  Reference hit frame: {hit_frame} (index {args.hit_idx})")

    # Load features
    feat_dir = os.path.join(args.clip_dir, "features", args.backbone, args.player)
    if not os.path.exists(feat_dir):
        print(f"Error: feature dir not found: {feat_dir}")
        return

    features = []
    for fi in range(total_frames):
        pt_path = os.path.join(feat_dir, "%d.pt" % fi)
        if os.path.exists(pt_path):
            t = torch.load(pt_path, weights_only=True).float()
            features.append(t)
        elif features:
            features.append(features[-1].clone())  # repeat last
        else:
            print(f"Warning: missing feature {pt_path}")
            return

    features = torch.stack(features)  # [T, D]
    print(f"  Features: {features.shape} ({args.backbone})")

    # Compute cosine similarity to hit frame
    hit_feat = features[hit_frame]  # [D]
    cosine_sim = F.cosine_similarity(features, hit_feat.unsqueeze(0), dim=-1)  # [T]

    cos_np = cosine_sim.numpy()

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(args.output,
             cosine_sim=cos_np,
             hit_frames=np.array(hits),
             hit_ref=hit_frame,
             total_frames=total_frames,
             backbone=args.backbone)

    print(f"\nSaved: {args.output}")
    print(f"  cosine_sim range: [{cos_np.min():.3f}, {cos_np.max():.3f}]")
    print(f"  cosine_sim at hit: {cos_np[hit_frame]:.3f}")

    # Print some stats
    for h in hits:
        if h < len(cos_np):
            print(f"  hit frame {h}: cosine_sim = {cos_np[h]:.3f}")


if __name__ == "__main__":
    main()
