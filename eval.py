"""Evaluation metrics for TempoPeak: MAE, Acc@k, Entropy."""

import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d


def compute_metrics(logits: torch.Tensor, t_gt: torch.Tensor,
                    valid_len: torch.Tensor | None = None) -> dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        logits: [B, T] raw model outputs.
        t_gt: [B] ground-truth hit frame indices.
        valid_len: [B] real frame count per sample (mask padding).

    Returns:
        Dict with mae, acc1, acc3, acc5, entropy.
    """
    # Gaussian smoothing on logits before any computation
    logits_np = logits.cpu().float().numpy()
    logits_np = gaussian_filter1d(logits_np, sigma=1.5, axis=-1)
    logits = torch.from_numpy(logits_np).to(logits.device)

    B, T = logits.shape

    # Mask padding positions to -inf before softmax
    if valid_len is not None:
        mask = torch.arange(T, device=logits.device).unsqueeze(0) >= valid_len.unsqueeze(1)
        logits = logits.masked_fill(mask, float("-inf"))

    probs = F.softmax(logits, dim=-1)  # [B, T]
    preds = probs.argmax(dim=-1)  # [B]

    errors = (preds - t_gt).abs().float()

    mae = errors.mean().item()
    acc1 = (errors <= 1).float().mean().item() * 100
    acc3 = (errors <= 3).float().mean().item() * 100
    acc5 = (errors <= 5).float().mean().item() * 100

    # Entropy of predicted distribution
    log_probs = torch.log(probs + 1e-8)
    entropy = -(probs * log_probs).sum(dim=-1).mean().item()

    return {
        "mae": mae,
        "acc1": acc1,
        "acc3": acc3,
        "acc5": acc5,
        "entropy": entropy,
    }


def gaussian_target(t_gt: torch.Tensor, t_max: int, sigma: float = 2.0,
                    valid_len: torch.Tensor | None = None) -> torch.Tensor:
    """Create Gaussian soft label centred at t_gt.

    Args:
        t_gt: [B] integer ground-truth indices.
        t_max: sequence length T.
        sigma: Gaussian standard deviation.
        valid_len: [B] if given, zero out padding positions before normalising.

    Returns:
        [B, T] normalised probability distribution.
    """
    t = torch.arange(t_max, device=t_gt.device, dtype=torch.float32)  # [T]
    t_gt_f = t_gt.float().unsqueeze(-1)  # [B, 1]
    q = torch.exp(-((t.unsqueeze(0) - t_gt_f) ** 2) / (2 * sigma ** 2))  # [B, T]

    # Zero out padding
    if valid_len is not None:
        mask = torch.arange(t_max, device=q.device).unsqueeze(0) >= valid_len.unsqueeze(1)
        q = q.masked_fill(mask, 0.0)

    q = q / q.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return q


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy loss: -sum q(t) log p(t).

    Args:
        logits: [B, T] raw model outputs.
        targets: [B, T] normalised soft label distribution.

    Returns:
        Scalar loss.
    """
    log_probs = F.log_softmax(logits, dim=-1)  # [B, T]
    loss = -(targets * log_probs).sum(dim=-1).mean()
    return loss


def expected_displacement_loss(logits: torch.Tensor, t_gt: torch.Tensor,
                                valid_len: torch.Tensor | None = None) -> torch.Tensor:
    """Expected Displacement Loss: E[|expected_t - t_gt|].

    Computes softmax over logits to get a probability distribution, then
    calculates the expected timestep and penalises displacement from ground truth.

    Args:
        logits: [B, T] raw model outputs.
        t_gt: [B] ground-truth hit frame indices.
        valid_len: [B] if given, mask padding positions to -inf before softmax.

    Returns:
        Scalar loss.
    """
    B, T = logits.shape
    if valid_len is not None:
        mask = torch.arange(T, device=logits.device).unsqueeze(0) >= valid_len.unsqueeze(1)
        logits = logits.masked_fill(mask, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    t = torch.arange(T, device=logits.device, dtype=torch.float32)
    expected_t = (probs * t).sum(dim=-1)          # [B]
    return (expected_t - t_gt.float()).abs().mean()
