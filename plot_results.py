"""
Generate publication-quality plots from 6-head training CSV logs.

Output files (saved to checkpoints/plots/):
  fig1_f1at1_curves.pdf        — F1@1 over epochs, all 6 heads
  fig2_f1at1_curves_smooth.pdf — F1@1 smoothed (EMA α=0.9), all 6 heads
  fig3_best_f1_bar.pdf         — Grouped bar chart: best F1@1/F1@2/F1@3
  fig4_train_val_loss.pdf      — Train vs Val loss, 6 subplots
  fig5_precision_recall_at1.pdf — P@1 and R@1 curves, 6 subplots
  fig6_lr_schedule.pdf         — Cosine annealing LR schedule
  fig7_pred_count.pdf          — Prediction count vs GT over epochs
  fig8_cls_loss_curves.pdf     — Classification loss (train & val), all heads
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
HEADS = ["bilstm", "mstcn", "mamba2", "bimamba2", "transformer", "identity"]
HEAD_LABELS = {
    "bilstm": "BiLSTM",
    "mstcn": "MS-TCN",
    "mamba2": "Mamba-2",
    "bimamba2": "BiMamba-2",
    "transformer": "Transformer",
    "identity": "Identity (no temporal)",
}
HEAD_COLORS = {
    "bilstm": "#2196F3",      # blue
    "mstcn": "#4CAF50",       # green
    "mamba2": "#FF9800",       # orange
    "bimamba2": "#E91E63",     # pink
    "transformer": "#9C27B0",  # purple
    "identity": "#607D8B",     # grey
}
HEAD_MARKERS = {
    "bilstm": "o",
    "mstcn": "s",
    "mamba2": "^",
    "bimamba2": "D",
    "transformer": "v",
    "identity": "x",
}

CKPT_DIR = os.path.join(os.path.dirname(__file__) or ".", "checkpoints")
PLOT_DIR = os.path.join(CKPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── Data Loading ────────────────────────────────────────────────────────────
def load_csv(head):
    path = os.path.join(CKPT_DIR, f"log_{head}.csv")
    if not os.path.exists(path):
        print(f"  ⚠ Missing: {path}")
        return None
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: float(v) for k, v in r.items()})
    return rows


data = {}
for h in HEADS:
    d = load_csv(h)
    if d:
        data[h] = d
        print(f"  Loaded {h}: {len(d)} epochs")

if not data:
    print("No data found!")
    sys.exit(1)


def col(head, key):
    """Extract a column as numpy array."""
    return np.array([r[key] for r in data[head]])


def ema(arr, alpha=0.9):
    """Exponential moving average for smoothing."""
    out = np.zeros_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * arr[i]
    return out


# ── Fig 1: F1@1 Raw Curves ─────────────────────────────────────────────────
print("\n[1/8] F1@1 curves (raw)...")
fig, ax = plt.subplots(figsize=(10, 5))
for h in HEADS:
    if h not in data:
        continue
    epochs = col(h, "epoch")
    f1 = col(h, "F1@1") * 100
    ax.plot(epochs, f1, color=HEAD_COLORS[h], alpha=0.7, linewidth=1.0,
            label=HEAD_LABELS[h])
    # Mark best point
    best_idx = np.argmax(f1)
    ax.scatter(epochs[best_idx], f1[best_idx], color=HEAD_COLORS[h],
               marker=HEAD_MARKERS[h], s=80, zorder=5,
               edgecolors="white" if HEAD_MARKERS[h] != "x" else None,
               linewidths=0.8)
    ax.annotate(f"{f1[best_idx]:.1f}%",
                xy=(epochs[best_idx], f1[best_idx]),
                xytext=(8, 4), textcoords="offset points",
                fontsize=8, color=HEAD_COLORS[h], fontweight="bold")

ax.set_xlabel("Epoch")
ax.set_ylabel("F1@1 (%)")
ax.set_title("Hit Detection F1@1 (±1 frame) — All Temporal Heads")
ax.legend(loc="upper left", framealpha=0.9)
ax.set_xlim(0, 300)
ax.set_ylim(0, None)
path = os.path.join(PLOT_DIR, "fig1_f1at1_curves.pdf")
fig.savefig(path)
fig.savefig(path.replace(".pdf", ".png"))
plt.close(fig)
print(f"  → {path}")

# ── Fig 2: F1@1 Smoothed (EMA) ─────────────────────────────────────────────
print("[2/8] F1@1 curves (EMA smoothed)...")
fig, ax = plt.subplots(figsize=(10, 5))
for h in HEADS:
    if h not in data:
        continue
    epochs = col(h, "epoch")
    f1_raw = col(h, "F1@1") * 100
    f1_sm = ema(f1_raw, alpha=0.92)
    ax.plot(epochs, f1_raw, color=HEAD_COLORS[h], alpha=0.15, linewidth=0.8)
    ax.plot(epochs, f1_sm, color=HEAD_COLORS[h], alpha=0.9, linewidth=2.0,
            label=HEAD_LABELS[h])

ax.set_xlabel("Epoch")
ax.set_ylabel("F1@1 (%)")
ax.set_title("Hit Detection F1@1 (EMA smoothed, α=0.92) — All Temporal Heads")
ax.legend(loc="upper left", framealpha=0.9)
ax.set_xlim(0, 300)
ax.set_ylim(0, None)
path = os.path.join(PLOT_DIR, "fig2_f1at1_curves_smooth.pdf")
fig.savefig(path)
fig.savefig(path.replace(".pdf", ".png"))
plt.close(fig)
print(f"  → {path}")

# ── Fig 3: Best F1 Bar Chart ───────────────────────────────────────────────
print("[3/8] Best F1 bar chart...")
best_f1 = {k: {} for k in ["F1@1", "F1@2", "F1@3"]}
best_epochs = {}
for h in HEADS:
    if h not in data:
        continue
    f1_1 = col(h, "F1@1") * 100
    idx = np.argmax(f1_1)
    best_epochs[h] = int(col(h, "epoch")[idx])
    best_f1["F1@1"][h] = f1_1[idx]
    # F1@2, F1@3 at the same epoch as best F1@1
    best_f1["F1@2"][h] = data[h][idx]["F1@2"] * 100
    best_f1["F1@3"][h] = data[h][idx]["F1@3"] * 100

heads_present = [h for h in HEADS if h in data]
x = np.arange(len(heads_present))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
for i, (metric, offset) in enumerate(
    [("F1@1", -width), ("F1@2", 0), ("F1@3", width)]
):
    vals = [best_f1[metric][h] for h in heads_present]
    bars = ax.bar(x + offset, vals, width, label=metric,
                  color=["#2196F3", "#4CAF50", "#FF9800"][i], alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels([f"{HEAD_LABELS[h]}\n(ep {best_epochs[h]})" for h in heads_present],
                   fontsize=9)
ax.set_ylabel("F1 Score (%)")
ax.set_title("Best F1 Scores at ±1/±2/±3 Frames (at Best F1@1 Epoch)")
ax.legend(loc="upper right")
ax.set_ylim(0, max(max(best_f1["F1@3"].values()), 80) + 5)
path = os.path.join(PLOT_DIR, "fig3_best_f1_bar.pdf")
fig.savefig(path)
fig.savefig(path.replace(".pdf", ".png"))
plt.close(fig)
print(f"  → {path}")

# ── Fig 4: Train vs Val Loss ───────────────────────────────────────────────
print("[4/8] Train vs Val loss subplots...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
axes = axes.flatten()
for i, h in enumerate(HEADS):
    ax = axes[i]
    if h not in data:
        ax.set_title(f"{HEAD_LABELS[h]} (no data)")
        continue
    epochs = col(h, "epoch")
    train_loss = col(h, "train_loss")
    val_loss = col(h, "val_loss")
    ax.plot(epochs, train_loss, color="#2196F3", linewidth=1.0, alpha=0.8,
            label="Train")
    ax.plot(epochs, val_loss, color="#E91E63", linewidth=1.0, alpha=0.8,
            label="Val")
    ax.set_title(HEAD_LABELS[h])
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper right")
    if i >= 3:
        ax.set_xlabel("Epoch")
    if i % 3 == 0:
        ax.set_ylabel("Loss (log)")

fig.suptitle("Train vs Validation Loss — Overfitting Analysis", fontsize=14, y=1.01)
fig.tight_layout()
path = os.path.join(PLOT_DIR, "fig4_train_val_loss.pdf")
fig.savefig(path)
fig.savefig(path.replace(".pdf", ".png"))
plt.close(fig)
print(f"  → {path}")

# ── Fig 5: Precision & Recall @1 ───────────────────────────────────────────
print("[5/8] Precision & Recall @1 subplots...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
axes = axes.flatten()
for i, h in enumerate(HEADS):
    ax = axes[i]
    if h not in data:
        ax.set_title(f"{HEAD_LABELS[h]} (no data)")
        continue
    epochs = col(h, "epoch")
    p1 = ema(col(h, "P@1") * 100, 0.9)
    r1 = ema(col(h, "R@1") * 100, 0.9)
    ax.plot(epochs, p1, color="#2196F3", linewidth=1.5, label="Precision@1")
    ax.plot(epochs, r1, color="#E91E63", linewidth=1.5, label="Recall@1")
    ax.set_title(HEAD_LABELS[h])
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="lower right")
    if i >= 3:
        ax.set_xlabel("Epoch")
    if i % 3 == 0:
        ax.set_ylabel("Score (%)")

fig.suptitle("Precision & Recall @1 (±33ms) — EMA Smoothed", fontsize=14, y=1.01)
fig.tight_layout()
path = os.path.join(PLOT_DIR, "fig5_precision_recall_at1.pdf")
fig.savefig(path)
fig.savefig(path.replace(".pdf", ".png"))
plt.close(fig)
print(f"  → {path}")

# ── Fig 6: Learning Rate Schedule ──────────────────────────────────────────
print("[6/8] LR schedule...")
# All heads have the same LR schedule, just pick one
ref_head = next(iter(data))
fig, ax = plt.subplots(figsize=(8, 3.5))
epochs = col(ref_head, "epoch")
lr = col(ref_head, "lr")
ax.plot(epochs, lr * 1e4, color="#2196F3", linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Learning Rate (×10⁻⁴)")
ax.set_title("CosineAnnealingLR Schedule (3×10⁻⁴ → 1×10⁻⁵)")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
path = os.path.join(PLOT_DIR, "fig6_lr_schedule.pdf")
fig.savefig(path)
fig.savefig(path.replace(".pdf", ".png"))
plt.close(fig)
print(f"  → {path}")

# ── Fig 7: Prediction Count vs GT ──────────────────────────────────────────
print("[7/8] Prediction count vs GT...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
axes = axes.flatten()
for i, h in enumerate(HEADS):
    ax = axes[i]
    if h not in data:
        ax.set_title(f"{HEAD_LABELS[h]} (no data)")
        continue
    epochs = col(h, "epoch")
    gt = col(h, "GT")
    pred = col(h, "Pred")
    ax.plot(epochs, pred, color="#FF9800", linewidth=1.2, alpha=0.8,
            label="Predicted")
    ax.axhline(gt[0], color="#4CAF50", linestyle="--", linewidth=1.5,
               label=f"GT = {int(gt[0])}", alpha=0.8)
    ax.set_title(HEAD_LABELS[h])
    ax.legend(fontsize=8, loc="upper right")
    if i >= 3:
        ax.set_xlabel("Epoch")
    if i % 3 == 0:
        ax.set_ylabel("Count")

fig.suptitle("Predicted Hit Count vs Ground Truth Over Training", fontsize=14, y=1.01)
fig.tight_layout()
path = os.path.join(PLOT_DIR, "fig7_pred_count.pdf")
fig.savefig(path)
fig.savefig(path.replace(".pdf", ".png"))
plt.close(fig)
print(f"  → {path}")

# ── Fig 8: Classification Loss Curves ──────────────────────────────────────
print("[8/8] Classification loss curves...")
fig, ax = plt.subplots(figsize=(10, 5))
for h in HEADS:
    if h not in data:
        continue
    epochs = col(h, "epoch")
    val_cls = ema(col(h, "val_cls"), 0.9)
    ax.plot(epochs, val_cls, color=HEAD_COLORS[h], linewidth=1.5, alpha=0.85,
            label=HEAD_LABELS[h])

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Cls Loss")
ax.set_title("Validation Classification Loss (EMA smoothed) — All Heads")
ax.legend(loc="upper right", framealpha=0.9)
ax.set_xlim(0, 300)
ax.set_yscale("log")
path = os.path.join(PLOT_DIR, "fig8_cls_loss_curves.pdf")
fig.savefig(path)
fig.savefig(path.replace(".pdf", ".png"))
plt.close(fig)
print(f"  → {path}")

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n✅ All 8 figures saved to: {PLOT_DIR}/")
print("   PDF (vector, for paper) + PNG (raster, for preview)")
print("\nFiles:")
for i in range(1, 9):
    fname = [f for f in os.listdir(PLOT_DIR) if f.startswith(f"fig{i}_") and f.endswith(".pdf")]
    if fname:
        print(f"  {fname[0]}")
