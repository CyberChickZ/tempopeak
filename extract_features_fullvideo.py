"""Extract per-frame DINOv2 features → single [N, D] .pt file.

Two input modes:
  --video    Read .mp4 directly, optional court crop + letterbox
  --frames_dir  Read pre-exported JPEGs (from annotator CE export), already preprocessed

Usage:
  # Mode 1: From .mp4 with court crop → 518
  python extract_features_fullvideo.py \
      --video datasets/fullvideo/00001.mp4 \
      --output datasets/fullvideo/00001_court518.pt \
      --crop 100,900,200,1700 --img_size 518 --batch_size 16

  # Mode 1: From .mp4 with court crop → 224 (fast baseline)
  python extract_features_fullvideo.py \
      --video datasets/fullvideo/00001.mp4 \
      --output datasets/fullvideo/00001_court224.pt \
      --crop 100,900,200,1700 --img_size 224 --batch_size 64

  # Mode 2: From annotator-exported JPEGs (already cropped+letterboxed)
  python extract_features_fullvideo.py \
      --frames_dir datasets/v1/export/0006/00001/frames \
      --output datasets/fullvideo/00001_ce518.pt \
      --batch_size 16
"""

import argparse
import os
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


# ---------------------------------------------------------------------------
# Frame iterators
# ---------------------------------------------------------------------------

def iter_frames_video(video_path, crop, img_size):
    """Yield (frame_idx, RGB numpy array) from .mp4 with optional crop + letterbox."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {video_path}")
    print(f"  Frames: {total}, FPS: {fps:.1f}")
    if crop:
        print(f"  Court crop: y=[{crop[0]}:{crop[1]}], x=[{crop[2]}:{crop[3]}]")
    print(f"  img_size: {img_size}")

    fi = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if crop:
            y1, y2, x1, x2 = crop
            img = img[y1:y2, x1:x2]
        img = letterbox(img, img_size)
        yield fi, img, total
        fi += 1
    cap.release()


def iter_frames_dir(frames_dir):
    """Yield (frame_idx, RGB numpy array) from pre-exported JPEG directory.

    Frames are already preprocessed (cropped + letterboxed by annotator).
    No additional resize needed.
    """
    # Count frames: 0.jpg, 1.jpg, ...
    files = [f for f in os.listdir(frames_dir) if f.endswith(".jpg")]
    total = len(files)
    print(f"Frames dir: {frames_dir}")
    print(f"  Total frames: {total}")

    if total == 0:
        return

    # Read first frame to show resolution
    sample = cv2.imread(os.path.join(frames_dir, "0.jpg"))
    if sample is not None:
        print(f"  Frame size: {sample.shape[1]}x{sample.shape[0]} (already preprocessed)")

    for fi in range(total):
        jpg_path = os.path.join(frames_dir, f"{fi}.jpg")
        img = cv2.imread(jpg_path)
        if img is None:
            print(f"  WARNING: missing frame {fi}, repeating previous")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        yield fi, img, total


def main():
    p = argparse.ArgumentParser(
        description="Extract full-video DINOv2 features")
    # Input: either --video or --frames_dir
    p.add_argument("--video", type=str, default=None,
                   help="Input .mp4 path (Mode 1: read video directly)")
    p.add_argument("--frames_dir", type=str, default=None,
                   help="Pre-exported JPEG dir (Mode 2: from annotator export)")
    p.add_argument("--output", required=True,
                   help="Output .pt path ([N, 1024] fp16)")
    # Video mode options
    p.add_argument("--crop", type=str, default=None,
                   help="Court crop as y1,y2,x1,x2 (video mode only)")
    p.add_argument("--img_size", type=int, default=518,
                   help="Letterbox target size (video mode only, 518=DINOv2 native)")
    # Common
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="auto",
                   help="auto = cuda if available else cpu (never MPS)")
    args = p.parse_args()

    # Validate input
    if not args.video and not args.frames_dir:
        p.error("Must provide --video or --frames_dir")
    if args.video and args.frames_dir:
        p.error("Provide --video OR --frames_dir, not both")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"

    # Parse crop (video mode only)
    crop = None
    if args.crop:
        parts = [int(x) for x in args.crop.split(",")]
        assert len(parts) == 4, "Crop must be y1,y2,x1,x2"
        crop = tuple(parts)

    # Build model
    print(f"Device: {device}, batch_size: {args.batch_size}")
    print("Loading DINOv2 ViT-L...")
    backbone, feat_dim = build_dinov2(device)
    print(f"  feat_dim: {feat_dim}")

    # Choose frame iterator
    if args.video:
        frame_iter = iter_frames_video(args.video, crop, args.img_size)
    else:
        frame_iter = iter_frames_dir(args.frames_dir)

    # Extract features
    all_features = []
    batch_imgs = []
    t0 = time.time()
    fi_count = 0
    total_frames = 0

    for fi, img, total in frame_iter:
        total_frames = total
        tensor = IMAGENET_TRANSFORM(img)
        batch_imgs.append(tensor)
        fi_count += 1

        # Process batch
        if len(batch_imgs) == args.batch_size:
            batch = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                feats = backbone(batch)  # [B, 1024]
            all_features.append(feats.cpu().half())
            batch_imgs = []

            if fi_count % (args.batch_size * 20) == 0:
                elapsed = time.time() - t0
                fps_rate = fi_count / elapsed
                eta = (total_frames - fi_count) / max(fps_rate, 1)
                print(f"  {fi_count}/{total_frames} ({fi_count/total_frames*100:.1f}%) "
                      f"| {fps_rate:.0f} frames/s | ETA {eta:.0f}s")

    # Last partial batch
    if batch_imgs:
        batch = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            feats = backbone(batch)
        all_features.append(feats.cpu().half())

    if not all_features:
        print("ERROR: No frames processed!")
        return

    # Concatenate and save
    features = torch.cat(all_features, dim=0)  # [N, 1024] fp16

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(features, args.output)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nDone. {fi_count} frames -> {features.shape} fp16")
    print(f"  Time: {elapsed:.0f}s ({fi_count/elapsed:.0f} frames/s)")
    print(f"  File: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
