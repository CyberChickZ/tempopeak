"""Extract per-frame DINOv2 features from a full video → single [N, D] .pt file.

No export/ directory needed. Reads .mp4 directly, applies optional court crop,
letterbox resizes, runs DINOv2 ViT-L, saves CLS tokens as fp16.

Usage:
  # Court crop → 518 (main experiment)
  python extract_features_fullvideo.py \
      --video datasets/fullvideo/00001.mp4 \
      --output datasets/fullvideo/00001_court518.pt \
      --crop 100,900,200,1700 --img_size 518 --batch_size 16

  # Court crop → 224 (fast baseline)
  python extract_features_fullvideo.py \
      --video datasets/fullvideo/00001.mp4 \
      --output datasets/fullvideo/00001_court224.pt \
      --crop 100,900,200,1700 --img_size 224 --batch_size 64

  # Full frame → 518 (no crop)
  python extract_features_fullvideo.py \
      --video datasets/fullvideo/00001.mp4 \
      --output datasets/fullvideo/00001_full518.pt \
      --img_size 518 --batch_size 16
"""

import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_dinov2(device):
    """Build frozen DINOv2 ViT-L backbone. Returns (model, feat_dim=1024)."""
    from transformers import AutoModel

    dinov2 = AutoModel.from_pretrained("facebook/dinov2-large")

    class _DINOv2Feats(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            outputs = self.model(pixel_values=x)
            return outputs.last_hidden_state[:, 0, :]  # CLS token [B, 1024]

    backbone = _DINOv2Feats(dinov2).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone, 1024


def letterbox(img, img_size):
    """Resize preserving aspect ratio, pad with black."""
    H, W = img.shape[:2]
    scale = img_size / max(H, W)
    new_w, new_h = int(W * scale), int(H * scale)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    y_off = (img_size - new_h) // 2
    x_off = (img_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def main():
    p = argparse.ArgumentParser(
        description="Extract full-video DINOv2 features (no export/ needed)")
    p.add_argument("--video", required=True, help="Input .mp4 path")
    p.add_argument("--output", required=True, help="Output .pt path ([N, 1024] fp16)")
    p.add_argument("--crop", type=str, default=None,
                   help="Court crop as y1,y2,x1,x2 (e.g. 100,900,200,1700)")
    p.add_argument("--img_size", type=int, default=518,
                   help="Letterbox target size (518=DINOv2 native, 224=fast)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="auto",
                   help="auto = cuda if available else cpu (never MPS)")
    args = p.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"

    # Parse crop
    crop = None
    if args.crop:
        parts = [int(x) for x in args.crop.split(",")]
        assert len(parts) == 4, "Crop must be y1,y2,x1,x2"
        crop = tuple(parts)  # (y1, y2, x1, x2)
        print(f"Court crop: y=[{crop[0]}:{crop[1]}], x=[{crop[2]}:{crop[3]}]")

    # Open video
    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {args.video}")
    print(f"  Frames: {total_frames}, FPS: {fps:.1f}")
    print(f"  img_size: {args.img_size}, batch_size: {args.batch_size}, device: {device}")

    # Build model
    print("Loading DINOv2 ViT-L...")
    backbone, feat_dim = build_dinov2(device)
    print(f"  feat_dim: {feat_dim}")

    # Extract features
    all_features = []
    batch_imgs = []
    t0 = time.time()

    fi = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break

        # BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Court crop
        if crop:
            y1, y2, x1, x2 = crop
            img = img[y1:y2, x1:x2]

        # Letterbox resize
        img = letterbox(img, args.img_size)

        # To tensor + normalize
        tensor = IMAGENET_TRANSFORM(img)
        batch_imgs.append(tensor)
        fi += 1

        # Process batch
        if len(batch_imgs) == args.batch_size:
            batch = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                feats = backbone(batch)  # [B, 1024]
            all_features.append(feats.cpu().half())
            batch_imgs = []

            if fi % (args.batch_size * 20) == 0:
                elapsed = time.time() - t0
                fps_rate = fi / elapsed
                eta = (total_frames - fi) / fps_rate
                print(f"  {fi}/{total_frames} ({fi/total_frames*100:.1f}%) "
                      f"| {fps_rate:.0f} frames/s | ETA {eta:.0f}s")

    cap.release()

    # Last partial batch
    if batch_imgs:
        batch = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            feats = backbone(batch)
        all_features.append(feats.cpu().half())

    # Concatenate and save
    features = torch.cat(all_features, dim=0)  # [N, 1024] fp16
    assert features.shape[0] == fi, f"Expected {fi} frames, got {features.shape[0]}"

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(features, args.output)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nDone. {fi} frames → {features.shape} fp16")
    print(f"  Time: {elapsed:.0f}s ({fi/elapsed:.0f} frames/s)")
    print(f"  File: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
