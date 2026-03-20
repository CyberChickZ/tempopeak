"""Extract metrics from all best_*_multihit.pt checkpoints → CSV."""
import csv
import glob
import os
import sys
import torch

ckpt_dir = os.path.join(os.path.dirname(__file__) or ".", "checkpoints")
pattern = os.path.join(ckpt_dir, "best_*_multihit.pt")
files = sorted(glob.glob(pattern))

if not files:
    print(f"No checkpoints found matching: {pattern}")
    sys.exit(1)

rows = []
for f in files:
    ckpt = torch.load(f, map_location="cpu", weights_only=False)
    name = os.path.basename(f)  # best_identity_multihit.pt
    head = name.replace("best_", "").replace("_multihit.pt", "")

    m = ckpt.get("metrics", {})
    a = ckpt.get("args", {})

    rows.append({
        "head": head,
        "best_epoch": ckpt.get("epoch", "?"),
        "val_loss": f"{m.get('loss', 0):.6f}",
        "cls_loss": f"{m.get('cls_loss', 0):.6f}",
        "disp_loss": f"{m.get('disp_loss', 0):.6f}",
        "P@1": f"{m.get('P@1', 0)*100:.1f}",
        "R@1": f"{m.get('R@1', 0)*100:.1f}",
        "F1@1": f"{m.get('F1@1', 0)*100:.1f}",
        "P@2": f"{m.get('P@2', 0)*100:.1f}",
        "R@2": f"{m.get('R@2', 0)*100:.1f}",
        "F1@2": f"{m.get('F1@2', 0)*100:.1f}",
        "P@3": f"{m.get('P@3', 0)*100:.1f}",
        "R@3": f"{m.get('R@3', 0)*100:.1f}",
        "F1@3": f"{m.get('F1@3', 0)*100:.1f}",
        "GT": m.get("total_gt", "?"),
        "Pred": m.get("total_pred", "?"),
        "lr": a.get("lr", "?"),
        "lambda_disp": a.get("lambda_disp", "?"),
        "segment_len": a.get("segment_len", "?"),
        "epochs": a.get("epochs", "?"),
    })

# Sort by F1@1 descending
rows.sort(key=lambda r: float(r["F1@1"]), reverse=True)

# Print table
print(f"\n{'head':14s} {'epoch':>5s} {'F1@1':>6s} {'F1@2':>6s} {'F1@3':>6s} "
      f"{'P@1':>6s} {'R@1':>6s} {'GT':>4s} {'Pred':>5s} {'val_loss':>9s}")
print("-" * 85)
for r in rows:
    print(f"{r['head']:14s} {r['best_epoch']:>5s} {r['F1@1']:>6s} {r['F1@2']:>6s} {r['F1@3']:>6s} "
          f"{r['P@1']:>6s} {r['R@1']:>6s} {str(r['GT']):>4s} {str(r['Pred']):>5s} {r['val_loss']:>9s}")

# Export CSV
csv_path = os.path.join(ckpt_dir, "comparison_6head.csv")
fields = list(rows[0].keys())
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f"\nCSV → {csv_path}")
