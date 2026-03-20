"""TempoPeak visualization: hook-based intermediate state extraction for all temporal heads.

Zero-invasive approach — monkey-patches / forward hooks applied at inference time only.
No changes to temporal_heads.py, model.py, or train.py.

Usage:
    python visualize.py \
        --checkpoint checkpoints/best_bimamba2_32.pt \
        --data_dir /path/to/data \
        --out_dir viz_outputs/ \
        --num_samples 5 \
        --plots all
"""

import argparse
import os
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from dataloader import build_dataloaders
from model import TempoPeakModel
from temporal_heads import _SSMLayer


# ============================================================================
# Part 1: Monkey-patches and hooks (per head type)
# ============================================================================

def _patched_scan(self, x, dt, A, B_t, C_t):
    """Monkey-patched _SSMLayer._scan that stores per-step hidden states."""
    B_batch, T, D = x.shape
    N = self.d_state

    dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    dB = dt.unsqueeze(-1) * B_t.unsqueeze(2)
    dB_x = dB * x.unsqueeze(-1)

    h = torch.zeros(B_batch, D, N, device=x.device, dtype=x.dtype)
    ys = []
    states = []
    for t in range(T):
        h = dA[:, t] * h + dB_x[:, t]
        ys.append((h * C_t[:, t].unsqueeze(1)).sum(-1))
        states.append(h.detach().clone())

    self._captured_states = torch.stack(states, dim=1)  # [B, T, D, N]
    return torch.stack(ys, dim=1)


def patch_ssm_layers(model):
    """Patch all _SSMLayer instances to capture hidden states."""
    for module in model.modules():
        if isinstance(module, _SSMLayer):
            module._scan = types.MethodType(_patched_scan, module)


def _patched_encoder_layer_forward(self, src, src_mask=None,
                                    src_key_padding_mask=None, **kwargs):
    """Monkey-patched TransformerEncoderLayer.forward with need_weights=True."""
    src2, attn_w = self.self_attn(
        src, src, src,
        need_weights=True, average_attn_weights=False,
        attn_mask=src_mask,
        key_padding_mask=src_key_padding_mask)
    self._last_attn_weights = attn_w  # [B, nhead, T, T]
    src = src + self.dropout1(src2)
    src = self.norm1(src)
    src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    src = src + self.dropout2(src2)
    src = self.norm2(src)
    return src


def patch_transformer(model):
    """Patch TransformerEncoderLayer to capture attention weights."""
    for layer in model.head.encoder.layers:
        layer.forward = types.MethodType(_patched_encoder_layer_forward, layer)


def patch_bilstm(model):
    """Hook BiLSTM to capture per-step fwd/bwd hidden states."""
    lstm = model.head.lstm

    def hook_fn(module, input, output):
        out_tensor = output[0]  # [B, T, 2*hidden]
        D = out_tensor.shape[-1] // 2
        module._fwd_hidden = out_tensor[..., :D].detach()
        module._bwd_hidden = out_tensor[..., D:].detach()

    lstm.register_forward_hook(hook_fn)


def patch_mstcn(model):
    """Hook each MS-TCN stage to capture per-stage activations."""
    for i, stage in enumerate(model.head.stages):
        def make_hook(idx):
            def hook_fn(module, input, output):
                module._captured_activation = output.detach().clone()  # [B, D, T]
            return hook_fn
        stage.register_forward_hook(make_hook(i))


def apply_patches(model, head_name):
    """Apply appropriate patches based on head type."""
    if head_name in ("mamba2", "bimamba2"):
        patch_ssm_layers(model)
    elif head_name == "transformer":
        patch_transformer(model)
    elif head_name == "bilstm":
        patch_bilstm(model)
    elif head_name == "mstcn":
        patch_mstcn(model)
    # identity: no patch needed


# ============================================================================
# Part 2: State extraction helpers
# ============================================================================

def extract_ssm_states(model, head_name):
    """Extract captured SSM states after a forward pass.

    Returns:
        For mamba2: {"ssm_states": [B, T, D, N]} (last layer)
        For bimamba2: {"fwd_states": [B, T, D, N], "bwd_states": [B, T, D, N]}
    """
    if head_name == "mamba2":
        # Last layer's states
        last_layer = list(model.head.layers)[-1]
        if hasattr(last_layer, "_captured_states"):
            return {"ssm_states": last_layer._captured_states}
    elif head_name == "bimamba2":
        fwd_states = bwd_states = None
        # Last fwd layer
        last_fwd = list(model.head.fwd_layers)[-1]
        if hasattr(last_fwd, "_captured_states"):
            fwd_states = last_fwd._captured_states
        # Last bwd layer — states are in reversed time order, flip back
        last_bwd = list(model.head.bwd_layers)[-1]
        if hasattr(last_bwd, "_captured_states"):
            bwd_states = torch.flip(last_bwd._captured_states, [1])
        return {"fwd_states": fwd_states, "bwd_states": bwd_states}
    return {}


def extract_transformer_attn(model):
    """Extract attention weights from last encoder layer."""
    last_layer = model.head.encoder.layers[-1]
    if hasattr(last_layer, "_last_attn_weights"):
        return last_layer._last_attn_weights  # [B, nhead, T, T]
    return None


def extract_bilstm_states(model):
    """Extract BiLSTM per-step hidden states."""
    lstm = model.head.lstm
    result = {}
    if hasattr(lstm, "_fwd_hidden"):
        result["fwd_hidden"] = lstm._fwd_hidden
    if hasattr(lstm, "_bwd_hidden"):
        result["bwd_hidden"] = lstm._bwd_hidden
    return result


def extract_mstcn_stages(model):
    """Extract per-stage activations from MS-TCN."""
    stages = []
    for stage in model.head.stages:
        if hasattr(stage, "_captured_activation"):
            # [B, D, T] -> [B, T, D]
            stages.append(stage._captured_activation.transpose(1, 2))
    return stages


# ============================================================================
# Part 3: Visualization functions
# ============================================================================

def plot_temporal_gradient_saliency(model, batch, head_name, sample_idx,
                                     out_dir, device):
    """Temporal gradient saliency — works for ALL heads, zero hook needed."""
    frames = batch["frames"].to(device)
    t_gt = batch["t_gt"]
    valid_len = batch["valid_len"]

    # Enable gradient on input features
    frames.requires_grad_(True)
    model.zero_grad()

    logits = model(frames)  # [B, T]
    probs = F.softmax(logits, dim=-1)
    pred_t = probs.argmax(dim=-1)  # [B]

    # Backward on predicted frame's logit for each sample in batch
    for b in range(min(frames.shape[0], 3)):  # up to 3 per batch
        if frames.grad is not None:
            frames.grad.zero_()

        logits[b, pred_t[b]].backward(retain_graph=True)
        grad = frames.grad[b]  # [T, ...] or [T, D]

        # Compute per-frame saliency (L2 norm over non-time dims)
        if grad.dim() == 1:
            saliency = grad.abs().cpu().numpy()
        else:
            saliency = grad.flatten(1).norm(dim=-1).cpu().numpy()

        vl = valid_len[b].item()
        saliency = saliency[:vl]
        gt = t_gt[b].item()
        pred = pred_t[b].item()

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.bar(range(vl), saliency, color="steelblue", alpha=0.8)
        ax.axvline(gt, color="red", linestyle="--", linewidth=2, label=f"GT={gt}")
        ax.axvline(pred, color="green", linestyle=":", linewidth=2, label=f"Pred={pred}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Gradient L2 Norm")
        ax.set_title(f"Temporal Gradient Saliency — {head_name}")
        ax.legend()
        fig.tight_layout()

        save_path = os.path.join(out_dir, "saliency",
                                  f"{head_name}_sample_{sample_idx + b}.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    frames.requires_grad_(False)


def plot_probability_distribution(logits, t_gt, valid_len, head_name,
                                   sample_idx, out_dir):
    """Probability distribution snapshots — works for ALL heads."""
    probs = F.softmax(logits, dim=-1).detach().cpu().numpy()

    for b in range(min(logits.shape[0], 3)):
        vl = valid_len[b].item()
        p = probs[b, :vl]
        gt = t_gt[b].item()
        pred = np.argmax(p)

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.fill_between(range(vl), p, alpha=0.4, color="steelblue")
        ax.plot(range(vl), p, color="steelblue", linewidth=1.5)
        ax.axvline(gt, color="red", linestyle="--", linewidth=2, label=f"GT={gt}")
        ax.axvline(pred, color="green", linestyle=":", linewidth=2, label=f"Pred={pred}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("P(hit)")
        ax.set_title(f"Softmax Distribution — {head_name}")
        ax.legend()
        fig.tight_layout()

        save_path = os.path.join(out_dir, "prob_dist",
                                  f"{head_name}_sample_{sample_idx + b}.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


def plot_error_distribution(all_errors, head_name, out_dir):
    """Error distribution histogram across full val set."""
    errors = np.array(all_errors)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(errors.min() - 0.5, errors.max() + 1.5, 1)
    ax.hist(errors, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Signed Error (pred - gt)")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution — {head_name} | "
                 f"MAE={np.abs(errors).mean():.2f} | "
                 f"Acc@1={np.mean(np.abs(errors) <= 1) * 100:.1f}%")
    fig.tight_layout()

    save_path = os.path.join(out_dir, "error_hist", f"{head_name}_errors.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_ssm_hidden_states(model, head_name, t_gt_val, valid_len_val,
                            sample_idx, out_dir):
    """Mamba2/BiMamba2 hidden state trajectory plots."""
    states = extract_ssm_states(model, head_name)
    if not states:
        return

    if head_name == "bimamba2":
        _plot_pincer(states, t_gt_val, valid_len_val, sample_idx, out_dir)
        _plot_ssm_norm(states.get("fwd_states"), "bimamba2_fwd",
                       t_gt_val, valid_len_val, sample_idx, out_dir)
    elif head_name == "mamba2":
        _plot_ssm_norm(states.get("ssm_states"), "mamba2",
                       t_gt_val, valid_len_val, sample_idx, out_dir)

    # PCA projection
    s = states.get("fwd_states") or states.get("ssm_states")
    if s is not None:
        _plot_state_pca(s, head_name, t_gt_val, valid_len_val,
                        sample_idx, out_dir)


def _plot_pincer(states, t_gt, valid_len, sample_idx, out_dir):
    """BiMamba2 forward vs backward 'pincer' plot."""
    fwd = states.get("fwd_states")
    bwd = states.get("bwd_states")
    if fwd is None or bwd is None:
        return

    for b in range(min(fwd.shape[0], 3)):
        vl = valid_len[b].item()
        gt = t_gt[b].item()

        fwd_norm = fwd[b, :vl].flatten(1).norm(dim=-1).cpu().numpy()
        bwd_norm = bwd[b, :vl].flatten(1).norm(dim=-1).cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(vl), fwd_norm, color="dodgerblue", linewidth=2,
                label="Forward scan")
        ax.plot(range(vl), bwd_norm, color="coral", linewidth=2,
                label="Backward scan")
        ax.axvline(gt, color="red", linestyle="--", linewidth=2,
                   label=f"GT hit={gt}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Hidden State L2 Norm")
        ax.set_title("BiMamba2 Forward vs Backward — Pincer Effect")
        ax.legend()
        fig.tight_layout()

        save_path = os.path.join(out_dir, "hidden_states",
                                  f"bimamba2_pincer_sample_{sample_idx + b}.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


def _plot_ssm_norm(states, label, t_gt, valid_len, sample_idx, out_dir):
    """SSM hidden state L2 norm trajectory."""
    if states is None:
        return

    for b in range(min(states.shape[0], 3)):
        vl = valid_len[b].item()
        gt = t_gt[b].item()
        norm_traj = states[b, :vl].flatten(1).norm(dim=-1).cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(range(vl), norm_traj, color="steelblue", linewidth=2)
        ax.axvline(gt, color="red", linestyle="--", linewidth=2,
                   label=f"GT hit={gt}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Hidden State L2 Norm")
        ax.set_title(f"SSM State Trajectory — {label}")
        ax.legend()
        fig.tight_layout()

        save_path = os.path.join(out_dir, "hidden_states",
                                  f"{label}_norm_sample_{sample_idx + b}.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


def _plot_state_pca(states, head_name, t_gt, valid_len, sample_idx, out_dir):
    """PCA projection of SSM/LSTM hidden states."""
    for b in range(min(states.shape[0], 3)):
        vl = valid_len[b].item()
        gt = t_gt[b].item()

        s = states[b, :vl].flatten(1).cpu().numpy()  # [T, D*N]
        if s.shape[0] < 3:
            continue

        pca = PCA(n_components=2)
        proj = pca.fit_transform(s)  # [T, 2]

        fig, ax = plt.subplots(figsize=(6, 6))
        scatter = ax.scatter(proj[:, 0], proj[:, 1],
                              c=np.arange(vl), cmap="viridis",
                              s=40, zorder=2)
        ax.scatter(proj[gt, 0], proj[gt, 1], c="red", s=200,
                   marker="*", zorder=3, label=f"GT hit={gt}")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title(f"Hidden State PCA — {head_name}")
        ax.legend()
        fig.colorbar(scatter, ax=ax, label="Frame index")
        fig.tight_layout()

        save_path = os.path.join(out_dir, "hidden_states",
                                  f"{head_name}_pca_sample_{sample_idx + b}.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


def plot_transformer_attention(model, t_gt, valid_len, sample_idx, out_dir):
    """Transformer attention heatmap."""
    attn = extract_transformer_attn(model)
    if attn is None:
        return

    # Average over heads: [B, nhead, T, T] -> [B, T, T]
    attn_avg = attn.mean(dim=1).detach().cpu().numpy()

    for b in range(min(attn_avg.shape[0], 3)):
        vl = valid_len[b].item()
        gt = t_gt[b].item()
        a = attn_avg[b, :vl, :vl]

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(a, cmap="hot", aspect="auto", origin="lower")
        ax.axvline(gt, color="cyan", linestyle="--", linewidth=2,
                   label=f"GT hit={gt}")
        ax.set_xlabel("Key frame")
        ax.set_ylabel("Query frame")
        ax.set_title("Transformer Attention (avg over heads)")
        ax.legend()
        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        save_path = os.path.join(out_dir, "attention",
                                  f"transformer_attn_sample_{sample_idx + b}.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


def plot_bilstm_states(model, t_gt, valid_len, sample_idx, out_dir):
    """BiLSTM hidden state trajectory."""
    states = extract_bilstm_states(model)
    if not states:
        return

    fwd = states.get("fwd_hidden")
    bwd = states.get("bwd_hidden")

    for b in range(min(fwd.shape[0], 3)):
        vl = valid_len[b].item()
        gt = t_gt[b].item()

        fwd_norm = fwd[b, :vl].norm(dim=-1).cpu().numpy()
        bwd_norm = bwd[b, :vl].norm(dim=-1).cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(vl), fwd_norm, color="dodgerblue", linewidth=2,
                label="Forward LSTM")
        ax.plot(range(vl), bwd_norm, color="coral", linewidth=2,
                label="Backward LSTM")
        ax.axvline(gt, color="red", linestyle="--", linewidth=2,
                   label=f"GT hit={gt}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Hidden State L2 Norm")
        ax.set_title("BiLSTM Forward vs Backward Hidden States")
        ax.legend()
        fig.tight_layout()

        save_path = os.path.join(out_dir, "hidden_states",
                                  f"bilstm_norm_sample_{sample_idx + b}.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    # PCA on concatenated hidden
    if fwd is not None:
        combined = torch.cat([fwd, bwd], dim=-1)  # [B, T, 2*hidden]
        _plot_state_pca(combined, "bilstm", t_gt, valid_len, sample_idx, out_dir)


def plot_mstcn_stages(model, t_gt, valid_len, sample_idx, out_dir):
    """MS-TCN per-stage activation L2 norms."""
    stage_acts = extract_mstcn_stages(model)
    if not stage_acts:
        return

    for b in range(min(stage_acts[0].shape[0], 3)):
        vl = valid_len[b].item()
        gt = t_gt[b].item()

        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ["steelblue", "coral", "seagreen", "mediumpurple"]
        for si, act in enumerate(stage_acts):
            norm = act[b, :vl].norm(dim=-1).cpu().numpy()
            ax.plot(range(vl), norm, color=colors[si % len(colors)],
                    linewidth=2, label=f"Stage {si + 1}")
        ax.axvline(gt, color="red", linestyle="--", linewidth=2,
                   label=f"GT hit={gt}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Activation L2 Norm")
        ax.set_title("MS-TCN Per-Stage Activations")
        ax.legend()
        fig.tight_layout()

        save_path = os.path.join(out_dir, "stages",
                                  f"mstcn_stages_sample_{sample_idx + b}.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


# ============================================================================
# Part 4: Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="TempoPeak visualization")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint .pt file")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to data directory")
    p.add_argument("--out_dir", type=str, default="viz_outputs",
                   help="Output directory for plots")
    p.add_argument("--num_samples", type=int, default=5,
                   help="Max number of val samples to visualize")
    p.add_argument("--plots", type=str, default="all",
                   help="Comma-separated plot types: saliency,prob_dist,"
                        "error_hist,hidden_states,attention,stages,pca "
                        "or 'all'")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]
    head_name = ckpt_args["temporal_head"]
    t_max = ckpt_args["t_max"]
    feat_dim = ckpt_args.get("feat_dim", 512)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Head: {head_name} | T_max: {t_max}")

    # Build model
    model = TempoPeakModel(
        temporal_head=head_name,
        feat_dim=feat_dim,
        use_backbone=False,  # always feature mode for viz
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Apply hooks/patches
    apply_patches(model, head_name)

    # Build val loader
    backbone = ckpt_args.get("backbone", "resnet18")
    _, val_loader = build_dataloaders(
        args.data_dir, t_max, batch_size=4, use_features=True,
        backbone=backbone)
    print(f"Val samples: {len(val_loader.dataset)}")

    # Determine which plots to generate
    if args.plots == "all":
        plot_types = {"saliency", "prob_dist", "error_hist",
                      "hidden_states", "attention", "stages", "pca"}
    else:
        plot_types = set(args.plots.split(","))

    # Create output dirs
    for subdir in ["saliency", "prob_dist", "error_hist",
                   "hidden_states", "attention", "stages"]:
        os.makedirs(os.path.join(args.out_dir, subdir), exist_ok=True)

    # Run inference + visualization
    all_errors = []
    sample_count = 0

    for batch in val_loader:
        if sample_count >= args.num_samples:
            break

        frames = batch["frames"].to(device)
        t_gt = batch["t_gt"]
        valid_len = batch["valid_len"]

        # Forward pass (for hooks to capture states)
        with torch.no_grad():
            logits = model(frames)

        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        errors = (preds - t_gt.to(device)).cpu().numpy()
        all_errors.extend(errors.tolist())

        # --- Universal plots ---
        if "saliency" in plot_types:
            plot_temporal_gradient_saliency(model, batch, head_name,
                                            sample_count, args.out_dir, device)

        if "prob_dist" in plot_types:
            plot_probability_distribution(logits, t_gt, valid_len, head_name,
                                          sample_count, args.out_dir)

        # --- Head-specific plots ---
        if head_name in ("mamba2", "bimamba2") and "hidden_states" in plot_types:
            plot_ssm_hidden_states(model, head_name, t_gt, valid_len,
                                   sample_count, args.out_dir)

        if head_name == "transformer" and "attention" in plot_types:
            plot_transformer_attention(model, t_gt, valid_len,
                                       sample_count, args.out_dir)

        if head_name == "bilstm" and "hidden_states" in plot_types:
            plot_bilstm_states(model, t_gt, valid_len,
                               sample_count, args.out_dir)

        if head_name == "mstcn" and "stages" in plot_types:
            plot_mstcn_stages(model, t_gt, valid_len,
                              sample_count, args.out_dir)

        sample_count += frames.shape[0]

    # --- Error histogram (full val set) ---
    if "error_hist" in plot_types and all_errors:
        # Continue collecting errors for remaining batches
        for batch in val_loader:
            frames = batch["frames"].to(device)
            t_gt = batch["t_gt"]
            with torch.no_grad():
                logits = model(frames)
            preds = F.softmax(logits, dim=-1).argmax(dim=-1)
            errors = (preds - t_gt.to(device)).cpu().numpy()
            all_errors.extend(errors.tolist())

        plot_error_distribution(all_errors, head_name, args.out_dir)

    print(f"\nDone. Plots saved to {args.out_dir}/")
    print(f"Total val samples processed: {len(all_errors)}")


if __name__ == "__main__":
    main()
