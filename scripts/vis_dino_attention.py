"""Vis 2: DINOv2 patch feature norm heatmap.

Extract DINOv2 ViT-L patch tokens, compute per-patch L2 norm,
overlay as heatmap on the input image. High norm = DINOv2 encodes
more information for that spatial region.

Usage (HPC, needs GPU for DINOv2):
  python scripts/vis_dino_attention.py \
      --clip_dir datasets/v1/export/0001/00001 \
      --output_dir paper/vis_data

  # Or direct images:
  python scripts/vis_dino_attention.py \
      --image1 path/to/hit.jpg --image2 path/to/nonhit.jpg \
      --output_dir paper/vis_data

Output:
  paper/vis_data/vis2_attn_hit.png
  paper/vis_data/vis2_attn_nonhit.png
  paper/vis_data/vis2_attention_maps.npz
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from torchvision import transforms


IMG_SIZE = 518

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_dinov2(device):
    """Load DINOv2 ViT-L for patch feature extraction."""
    from transformers import Dinov2Model
    model = Dinov2Model.from_pretrained(
        "facebook/dinov2-large",
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


def get_patch_norm_map(model, img_tensor, device):
    """Per-patch feature L2 norm. High = salient content."""
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(pixel_values=img_tensor)
    patch_tokens = outputs.last_hidden_state[:, 1:, :]  # [1, n_patches, 1024]
    norms = patch_tokens[0].norm(dim=-1)  # [n_patches]
    grid = int(norms.shape[0] ** 0.5)
    return norms.reshape(grid, grid).cpu().numpy()


def overlay_heatmap(img, heat_map, alpha=0.5):
    """Overlay heatmap on image with JET colormap."""
    H, W = img.shape[:2]
    norm = (heat_map - heat_map.min()) / (heat_map.max() - heat_map.min() + 1e-8)
    up = cv2.resize(norm.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    cmap = cv2.applyColorMap((up * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
    blend = (img.astype(np.float32) * (1 - alpha) +
             cmap.astype(np.float32) * alpha).astype(np.uint8)
    return blend


def process_image(model, img_path, device):
    """Load image → letterbox → patch norm heatmap overlay."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lb = letterbox(img_rgb, IMG_SIZE)
    img_tensor = IMAGENET_TRANSFORM(img_lb)
    norm_map = get_patch_norm_map(model, img_tensor, device)
    overlay = overlay_heatmap(img_lb, norm_map)
    return overlay, norm_map, img_lb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image1", type=str, default=None)
    p.add_argument("--image2", type=str, default=None)
    p.add_argument("--clip_dir", type=str, default=None)
    p.add_argument("--hit_idx", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="paper/vis_data")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    if args.clip_dir:
        import json
        with open(os.path.join(args.clip_dir, "annot.json")) as f:
            annot = json.load(f)
        hits = annot["hits"]
        total = annot["total_frames"]
        hit_frame = hits[args.hit_idx]
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

    print("Loading DINOv2 ViT-L...")
    model = load_dinov2(device)

    print(f"Processing hit frame: {args.image1}")
    overlay1, norm1, raw1 = process_image(model, args.image1, device)
    out1 = os.path.join(args.output_dir, "vis2_attn_hit.png")
    cv2.imwrite(out1, cv2.cvtColor(overlay1, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {out1}")
    print(f"  Norm range: [{norm1.min():.2f}, {norm1.max():.2f}]")

    if args.image2:
        print(f"Processing non-hit frame: {args.image2}")
        overlay2, norm2, raw2 = process_image(model, args.image2, device)
        out2 = os.path.join(args.output_dir, "vis2_attn_nonhit.png")
        cv2.imwrite(out2, cv2.cvtColor(overlay2, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {out2}")
        print(f"  Norm range: [{norm2.min():.2f}, {norm2.max():.2f}]")

    save_dict = {"norm_hit": norm1, "img_hit": raw1}
    if args.image2:
        save_dict["norm_nonhit"] = norm2
        save_dict["img_nonhit"] = raw2
    np.savez(os.path.join(args.output_dir, "vis2_attention_maps.npz"), **save_dict)

    print("\nDone.")


if __name__ == "__main__":
    main()
