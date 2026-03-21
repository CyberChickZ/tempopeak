"""Vis 2: DINOv2 patch attention heatmap.

Extract DINOv2 ViT-L self-attention from the last layer,
visualize CLS→patch attention as a heatmap overlaid on the input image.

Usage (HPC, needs GPU for DINOv2):
  python scripts/vis_dino_attention.py \
      --image1 path/to/hit_frame.jpg \
      --image2 path/to/nonhit_frame.jpg \
      --output_dir paper/vis_data

  # Or from clip directory:
  python scripts/vis_dino_attention.py \
      --clip_dir datasets/v1/export/0001/00001 \
      --player p1 \
      --hit_idx 0 \
      --output_dir paper/vis_data

Output:
  paper/vis_data/vis2_attn_hit.png      — attention heatmap on hit frame
  paper/vis_data/vis2_attn_nonhit.png   — attention heatmap on non-hit frame
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


IMG_SIZE = 518  # DINOv2 native
PATCH_SIZE = 14
GRID_SIZE = IMG_SIZE // PATCH_SIZE  # 37


IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_dinov2(device):
    """Load DINOv2 ViT-L with attention output enabled.

    Uses attn_implementation="eager" to force non-SDPA attention,
    which is required for output_attentions=True to return actual weights.
    (SDPA uses F.scaled_dot_product_attention which doesn't return weights.)
    """
    from transformers import Dinov2Model

    model = Dinov2Model.from_pretrained(
        "facebook/dinov2-large",
        attn_implementation="eager",
    ).to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    return model


def letterbox(img, size):
    """Resize preserving aspect ratio, pad with black."""
    H, W = img.shape[:2]
    scale = size / max(H, W)
    new_w, new_h = int(W * scale), int(H * scale)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def get_attention_map(model, img_tensor, device):
    """Get CLS→patch attention weights from last layer.

    Requires model loaded with attn_implementation="eager" so that
    output_attentions=True returns actual attention weight tensors.

    Returns:
        attn_map: [grid_size, grid_size] attention weights (mean across heads)
    """
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=img_tensor, output_attentions=True)

    if not outputs.attentions:
        raise RuntimeError(
            "output_attentions returned empty. "
            "Ensure model was loaded with attn_implementation='eager'.")

    # Last layer attention: [1, n_heads, 1+n_patches, 1+n_patches]
    last_attn = outputs.attentions[-1]

    # CLS token (index 0) attending to patch tokens (index 1:)
    cls_attn = last_attn[0, :, 0, 1:]  # [n_heads, n_patches]

    # Mean across heads
    attn_mean = cls_attn.mean(dim=0)  # [n_patches]

    # Reshape to grid
    n_patches = attn_mean.shape[0]
    grid = int(n_patches ** 0.5)
    attn_map = attn_mean.reshape(grid, grid).cpu().numpy()

    return attn_map


def overlay_heatmap(img, attn_map, alpha=0.5):
    """Overlay attention heatmap on image."""
    H, W = img.shape[:2]

    # Normalize attention to [0, 1]
    attn_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Upsample to image size
    attn_up = cv2.resize(attn_norm.astype(np.float32), (W, H),
                          interpolation=cv2.INTER_CUBIC)

    # Apply colormap
    heatmap = cv2.applyColorMap((attn_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = (img.astype(np.float32) * (1 - alpha) +
               heatmap.astype(np.float32) * alpha).astype(np.uint8)

    return overlay


def process_image(model, img_path, device):
    """Load image, get attention, return overlay + raw attention."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Letterbox to 518x518
    img_lb = letterbox(img_rgb, IMG_SIZE)

    # Transform for model
    img_tensor = IMAGENET_TRANSFORM(img_lb)

    # Get attention
    attn_map = get_attention_map(model, img_tensor, device)

    # Overlay on original (letterboxed) image
    overlay = overlay_heatmap(img_lb, attn_map)

    return overlay, attn_map, img_lb


def main():
    p = argparse.ArgumentParser()
    # Direct image mode
    p.add_argument("--image1", type=str, default=None,
                   help="Hit frame image path")
    p.add_argument("--image2", type=str, default=None,
                   help="Non-hit frame image path")
    # Clip mode
    p.add_argument("--clip_dir", type=str, default=None,
                   help="Clip directory (auto-select hit/nonhit frames)")
    p.add_argument("--player", type=str, default="p1")
    p.add_argument("--hit_idx", type=int, default=0)
    # Output
    p.add_argument("--output_dir", type=str, default="paper/vis_data")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    # If clip_dir mode, find hit/nonhit frames
    if args.clip_dir:
        import json
        with open(os.path.join(args.clip_dir, "annot.json")) as f:
            annot = json.load(f)
        hits = annot["hits"]
        total = annot["total_frames"]
        hit_frame = hits[args.hit_idx]

        # Non-hit: 10 frames before hit (or after if too close to start)
        nonhit_frame = max(0, hit_frame - 10)
        if abs(nonhit_frame - hit_frame) < 5:
            nonhit_frame = min(total - 1, hit_frame + 10)

        frames_dir = os.path.join(args.clip_dir, "frames")
        args.image1 = os.path.join(frames_dir, "%d.jpg" % hit_frame)
        args.image2 = os.path.join(frames_dir, "%d.jpg" % nonhit_frame)
        print(f"Clip: {args.clip_dir}")
        print(f"  Hit frame: {hit_frame} → {args.image1}")
        print(f"  Non-hit frame: {nonhit_frame} → {args.image2}")

    if not args.image1:
        p.error("Provide --image1 or --clip_dir")

    # Load DINOv2
    print("Loading DINOv2 ViT-L...")
    model = load_dinov2(device)

    # Process hit frame
    print(f"Processing hit frame: {args.image1}")
    overlay1, attn1, raw1 = process_image(model, args.image1, device)
    out1 = os.path.join(args.output_dir, "vis2_attn_hit.png")
    cv2.imwrite(out1, cv2.cvtColor(overlay1, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {out1}")
    print(f"  Attention range: [{attn1.min():.4f}, {attn1.max():.4f}]")

    # Process non-hit frame (optional)
    if args.image2:
        print(f"Processing non-hit frame: {args.image2}")
        overlay2, attn2, raw2 = process_image(model, args.image2, device)
        out2 = os.path.join(args.output_dir, "vis2_attn_nonhit.png")
        cv2.imwrite(out2, cv2.cvtColor(overlay2, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {out2}")
        print(f"  Attention range: [{attn2.min():.4f}, {attn2.max():.4f}]")

    # Save raw attention maps as npz
    save_dict = {"attn_hit": attn1, "img_hit": raw1}
    if args.image2:
        save_dict["attn_nonhit"] = attn2
        save_dict["img_nonhit"] = raw2
    np.savez(os.path.join(args.output_dir, "vis2_attention_maps.npz"), **save_dict)

    print("\nDone.")


if __name__ == "__main__":
    main()
