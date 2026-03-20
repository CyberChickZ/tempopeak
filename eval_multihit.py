"""Multi-hit evaluation: peak detection, P/R/F1, loss functions.

Peak detection pipeline:
  1. Gaussian-smooth the raw logits (remove frame-level noise)
  2. Normalize to [0,1]
  3. scipy.signal.find_peaks with height & min_distance
  4. Greedy-match predicted peaks to GT peaks within tolerance
  5. Compute Precision / Recall / F1
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def soft_cross_entropy(logits, target, valid_len=None):
    """Soft cross-entropy loss for multi-peak distributions.

    Handles padding: masks both logits and target, renormalizes target
    over valid positions only. Uses -1e9 (not -inf) to avoid NaN from
    0 * -inf in IEEE 754.

    Args:
        logits:    [B, T] raw model output.
        target:    [B, T] normalized multi-Gaussian target.
        valid_len: [B] real frame count per sample (mask padding).

    Returns:
        Scalar loss.
    """
    B, T = logits.shape
    if valid_len is not None:
        mask = torch.arange(T, device=logits.device)[None] >= valid_len[:, None]
        logits = logits.masked_fill(mask, -1e9)
        target = target.masked_fill(mask, 0.0)
        target = target / target.sum(-1, keepdim=True).clamp(min=1e-8)
    log_prob = F.log_softmax(logits, dim=-1)
    return -(target * log_prob).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def detect_peaks(logits_np, sigma_smooth=3.0, height_ratio=0.15,
                 min_distance=10):
    """Detect peaks in a 1-D logits array.

    Args:
        logits_np:    [T] numpy array (raw logits, valid region only).
        sigma_smooth: Gaussian smoothing sigma before peak finding.
        height_ratio: min peak height as fraction of (max - min).
        min_distance: minimum frames between two peaks.

    Returns:
        list[int] — detected peak frame indices (sorted).
    """
    smoothed = gaussian_filter1d(logits_np.astype(np.float64),
                                 sigma=sigma_smooth)
    vmin, vmax = smoothed.min(), smoothed.max()
    if vmax - vmin < 1e-8:
        return []
    normed = (smoothed - vmin) / (vmax - vmin)
    peaks, _ = find_peaks(normed, height=height_ratio, distance=min_distance)
    return sorted(peaks.tolist())


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_predictions(pred, gt, tolerance):
    """Greedy-match predicted peaks to GT peaks within tolerance.

    Greedy: process (pred, gt) pairs in order of ascending distance;
    each pred and each gt can be matched at most once.

    Returns:
        (n_matched, n_pred, n_gt)
    """
    if not pred or not gt:
        return 0, len(pred), len(gt)

    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            d = abs(p - g)
            if d <= tolerance:
                pairs.append((d, i, j))
    pairs.sort()

    used_p, used_g = set(), set()
    for d, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)

    return len(used_p), len(pred), len(gt)


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

def compute_prf1(pred_peaks, gt_peaks, tolerance=2):
    """Compute Precision, Recall, F1 for a single sample."""
    n_matched, n_pred, n_gt = match_predictions(pred_peaks, gt_peaks, tolerance)
    p = n_matched / n_pred if n_pred > 0 else 0.0
    r = n_matched / n_gt if n_gt > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1
