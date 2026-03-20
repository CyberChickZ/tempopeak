"""Multi-hit evaluation: T-DEED style cls+displacement loss, detection, P/R/F1.

Detection pipeline:
  1. Sigmoid on cls_logits → cls probability per frame
  2. Gaussian-smooth cls probabilities (remove frame-level noise)
  3. scipy.signal.find_peaks with height & min_distance
  4. Refine each peak position using displacement prediction
  5. Greedy-match predicted peaks to GT peaks within tolerance
  6. Compute Precision / Recall / F1
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Loss (Focal BCE + Displacement MSE)
# ---------------------------------------------------------------------------

def cls_displacement_loss(cls_logits, disp_logits, cls_target, disp_target,
                          disp_mask, valid_len=None,
                          focal_gamma=2.0, focal_alpha=0.75,
                          lambda_disp=0.01):
    """T-DEED style loss: focal BCE for cls + MSE for displacement.

    Args:
        cls_logits:  [B, T] raw classification logits
        disp_logits: [B, T] raw displacement predictions
        cls_target:  [B, T] binary (0/1)
        disp_target: [B, T] signed offset to nearest hit
        disp_mask:   [B, T] 1.0 for positive frames only
        valid_len:   [B] mask padding
        focal_gamma: focal loss gamma
        focal_alpha: focal loss alpha
        lambda_disp: displacement loss weight

    Returns:
        total_loss, cls_loss, disp_loss (all scalars)
    """
    B, T = cls_logits.shape

    # Valid mask
    if valid_len is not None:
        vmask = (torch.arange(T, device=cls_logits.device)[None]
                 < valid_len[:, None]).float()
    else:
        vmask = torch.ones(B, T, device=cls_logits.device)

    # --- Classification: Focal BCE ---
    p = torch.sigmoid(cls_logits)
    bce = F.binary_cross_entropy_with_logits(
        cls_logits, cls_target, reduction='none')
    p_t = p * cls_target + (1 - p) * (1 - cls_target)
    focal_w = (1 - p_t) ** focal_gamma
    alpha_t = focal_alpha * cls_target + (1 - focal_alpha) * (1 - cls_target)
    cls_loss = (alpha_t * focal_w * bce * vmask).sum() / vmask.sum()

    # --- Displacement: MSE on positive frames only ---
    disp_diff = (disp_logits - disp_target) ** 2
    disp_valid = disp_mask * vmask
    n_pos = disp_valid.sum().clamp(min=1.0)
    disp_loss = (disp_diff * disp_valid).sum() / n_pos

    total = cls_loss + lambda_disp * disp_loss
    return total, cls_loss.item(), disp_loss.item()


# ---------------------------------------------------------------------------
# Detection (cls peak finding + displacement refinement)
# ---------------------------------------------------------------------------

def cls_disp_detect(cls_logits, disp_logits, valid_len,
                    threshold=0.3, min_distance=10):
    """Detect hits from cls + displacement outputs.

    Args:
        cls_logits:  [T] raw classification logits (single sample)
        disp_logits: [T] displacement predictions
        valid_len:   int
        threshold:   sigmoid threshold for cls
        min_distance: min frames between detected hits

    Returns:
        list[int] — detected hit frame indices (sorted)
    """
    T = valid_len
    cls_prob = torch.sigmoid(cls_logits[:T]).cpu().numpy()

    # Smooth cls probabilities
    cls_smooth = gaussian_filter1d(cls_prob.astype(np.float64), sigma=1.5)

    # Find peaks in classification score
    peaks, _ = find_peaks(cls_smooth, height=threshold,
                          distance=min_distance)

    # Refine each peak position using displacement
    disp = disp_logits[:T].cpu().numpy()
    refined = []
    for pk in peaks:
        # displacement = signed offset to nearest hit
        # predicted hit position = pk + disp[pk]
        hit_pos = int(round(pk + disp[pk]))
        hit_pos = max(0, min(hit_pos, T - 1))
        refined.append(hit_pos)

    return sorted(set(refined))


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
