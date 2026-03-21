"""Vis 6 + Vis 7: Softmax probability distributions + signed error distribution.

For each of the 5 temporal heads (T=32, DINOv2 backbone, EDL loss):
  - Vis 6: raw softmax(smoothed_logits) for all val samples → [N_val, T] per head
  - Vis 7: signed error = argmax(smoothed_logits) - gt → [N_val] per head

Also selects 2-3 representative samples:
  - "all_correct": all heads predict within ±1 frame
  - "only_bimamba2": only BiMamba2 is correct (±1), others wrong

Usage (HPC, GPU not required — DINOv2 features pre-extracted):
  python scripts/vis_softmax_errors.py \
      --data_dir datasets/v1/export \
      --output_dir paper/vis_data

Output:
  paper/vis_data/vis6_softmax_{head}.pt     — [N_val, 32] softmax distributions
  paper/vis_data/vis7_errors_{head}.pt      — [N_val] signed errors (int)
  paper/vis_data/vis67_summary.json         — metrics + representative sample indices
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataloader import build_dataloaders
from model import TempoPeakModel


HEADS = ["identity", "bilstm", "mamba2", "bimamba2", "transformer"]


def load_model(head_name, ckpt_path, device):
    """Load checkpoint and reconstruct model."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]

    # Detect feature mode from checkpoint args
    feat_dim = 1024  # DINOv2
    model = TempoPeakModel(
        temporal_head=head_name,
        feat_dim=feat_dim,
        use_backbone=False,  # features pre-extracted
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("metrics", {})


@torch.no_grad()
def run_inference(model, val_loader, device):
    """Run inference, return smoothed softmax probs and predictions.

    Returns:
        all_probs: [N_val, T] softmax distributions (after Gaussian smoothing)
        all_preds: [N_val] predicted frame indices
        all_gt:    [N_val] ground-truth frame indices
        all_vlen:  [N_val] valid lengths
    """
    all_logits = []
    all_gt = []
    all_vlen = []

    for batch in val_loader:
        frames = batch["frames"].to(device)
        t_gt = batch["t_gt"]
        valid_len = batch["valid_len"]

        logits = model(frames)  # [B, T]
        all_logits.append(logits.cpu())
        all_gt.append(t_gt)
        all_vlen.append(valid_len)

    logits = torch.cat(all_logits, dim=0)   # [N_val, T]
    gt = torch.cat(all_gt, dim=0)           # [N_val]
    vlen = torch.cat(all_vlen, dim=0)       # [N_val]

    # Gaussian smoothing (same as eval.py)
    logits_np = logits.float().numpy()
    logits_np = gaussian_filter1d(logits_np, sigma=1.5, axis=-1)
    logits = torch.from_numpy(logits_np)

    # Mask padding to -inf
    B, T = logits.shape
    mask = torch.arange(T).unsqueeze(0) >= vlen.unsqueeze(1)
    logits = logits.masked_fill(mask, float("-inf"))

    probs = F.softmax(logits, dim=-1)       # [N_val, T]
    preds = probs.argmax(dim=-1)            # [N_val]

    return probs, preds, gt, vlen


def find_representative_samples(results, gt):
    """Find representative samples for visualization.

    Returns dict with:
      - "all_correct": index where all heads are within ±1
      - "only_bimamba2": index where only BiMamba2 is correct
    """
    N = len(gt)
    head_correct = {}  # head -> [N] bool

    for head in HEADS:
        preds = results[head]["preds"]
        errors = (preds - gt).abs()
        head_correct[head] = errors <= 1

    reps = {}

    # All correct
    all_mask = torch.ones(N, dtype=torch.bool)
    for head in HEADS:
        all_mask &= head_correct[head]
    indices = all_mask.nonzero(as_tuple=True)[0]
    if len(indices) > 0:
        # Pick one with smallest total error across all heads
        best_idx = indices[0].item()
        reps["all_correct"] = best_idx

    # Only BiMamba2 correct
    bimamba_only = head_correct["bimamba2"].clone()
    for head in HEADS:
        if head != "bimamba2":
            bimamba_only &= ~head_correct[head]
    indices = bimamba_only.nonzero(as_tuple=True)[0]
    if len(indices) > 0:
        reps["only_bimamba2"] = indices[0].item()

    # BiMamba2 correct, at least 2 others wrong
    bimamba_best = head_correct["bimamba2"].clone()
    others_wrong_count = torch.zeros(N, dtype=torch.long)
    for head in HEADS:
        if head != "bimamba2":
            others_wrong_count += (~head_correct[head]).long()
    bimamba_best &= (others_wrong_count >= 2)
    indices = bimamba_best.nonzero(as_tuple=True)[0]
    if len(indices) > 0:
        # Pick one with most others wrong
        best_i = others_wrong_count[indices].argmax()
        reps["bimamba2_best"] = indices[best_i].item()

    return reps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="datasets/v1/export")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--output_dir", type=str, default="paper/vis_data")
    p.add_argument("--t_max", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    # Build val loader (same split as training)
    print("Building val loader...")
    _, val_loader = build_dataloaders(
        args.data_dir, t_max=args.t_max, batch_size=args.batch_size,
        backbone="dinov2")

    # Filter out samples without pre-extracted features (avoid mixed tensor shapes)
    from pathlib import Path
    val_ds = val_loader.dataset
    original_n = len(val_ds)
    filtered = []
    for s in val_ds.samples:
        ci, hi_idx = s[0], s[1]
        clip = val_ds.clips[ci]
        hitter = clip["hitters"][hi_idx] if hi_idx < len(clip["hitters"]) else 0
        pk = "p%d" % hitter if hitter in (1, 2) else "p1"
        feat_dir = Path(clip["clip_dir"]) / "features" / "dinov2" / pk
        if feat_dir.exists():
            filtered.append(s)
    val_ds.samples = filtered
    n_val = len(val_ds)
    print(f"  Val samples: {n_val} (filtered from {original_n}, removed {original_n - n_val} without features)")

    # Run inference for each head
    results = {}
    gt = None

    for head in HEADS:
        ckpt_path = os.path.join(args.ckpt_dir, "best_%s_%d.pt" % (head, args.t_max))
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {head}: {ckpt_path} not found")
            continue

        print(f"\nRunning {head}...")
        try:
            model, saved_metrics = load_model(head, ckpt_path, device)
        except RuntimeError as e:
            print(f"  SKIP {head}: {e}")
            print(f"  (Mamba heads require GPU — re-run with --device cuda)")
            continue
        probs, preds, gt_out, vlen = run_inference(model, val_loader, device)

        if gt is None:
            gt = gt_out
            vlen_saved = vlen

        signed_errors = (preds - gt).long()  # [N_val]
        abs_errors = signed_errors.abs().float()

        mae = abs_errors.mean().item()
        acc1 = (abs_errors <= 1).float().mean().item() * 100
        acc3 = (abs_errors <= 3).float().mean().item() * 100

        results[head] = {
            "probs": probs,      # [N_val, T]
            "preds": preds,      # [N_val]
            "errors": signed_errors,  # [N_val]
            "mae": mae,
            "acc1": acc1,
            "acc3": acc3,
            "saved_metrics": saved_metrics,
        }

        print(f"  MAE={mae:.2f} Acc@1={acc1:.1f}% Acc@3={acc3:.1f}%")
        print(f"  Checkpoint metrics: {saved_metrics}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save outputs
    print(f"\nSaving to {args.output_dir}/...")

    for head in HEADS:
        if head not in results:
            continue
        # Vis 6: softmax distributions
        torch.save(results[head]["probs"],
                   os.path.join(args.output_dir, "vis6_softmax_%s.pt" % head))
        # Vis 7: signed errors
        torch.save(results[head]["errors"],
                   os.path.join(args.output_dir, "vis7_errors_%s.pt" % head))

    # Save ground truth and valid lengths
    torch.save(gt, os.path.join(args.output_dir, "vis67_gt.pt"))
    torch.save(vlen_saved, os.path.join(args.output_dir, "vis67_vlen.pt"))

    # Find representative samples
    reps = find_representative_samples(results, gt)
    print(f"\nRepresentative samples: {reps}")

    # Build summary JSON
    summary = {
        "n_val": n_val,
        "t_max": args.t_max,
        "heads": {},
        "representative_samples": {},
    }
    for head in HEADS:
        if head not in results:
            continue
        summary["heads"][head] = {
            "mae": results[head]["mae"],
            "acc1": results[head]["acc1"],
            "acc3": results[head]["acc3"],
        }

    for label, idx in reps.items():
        sample_info = {
            "index": idx,
            "gt": gt[idx].item(),
            "valid_len": vlen_saved[idx].item(),
        }
        for head in HEADS:
            if head in results:
                sample_info[head] = {
                    "pred": results[head]["preds"][idx].item(),
                    "error": results[head]["errors"][idx].item(),
                }
        summary["representative_samples"][label] = sample_info

    summary_path = os.path.join(args.output_dir, "vis67_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Files saved:")
    for head in HEADS:
        if head in results:
            print(f"  vis6_softmax_{head}.pt  — [{n_val}, {args.t_max}]")
            print(f"  vis7_errors_{head}.pt   — [{n_val}]")
    print(f"  vis67_gt.pt            — [{n_val}]")
    print(f"  vis67_vlen.pt          — [{n_val}]")
    print(f"  vis67_summary.json     — metrics + representative samples")


if __name__ == "__main__":
    main()
