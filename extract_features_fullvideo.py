"""Extract per-frame DINOv2 features → single [N, D] .pt file.

Two input modes:
  --video      Read .mp4 directly, optional court crop + letterbox
  --frames_dir Read pre-exported JPEGs (from annotator CE export), already preprocessed

Uses PyTorch DataLoader with multiple workers for fast parallel JPEG loading.

Usage:
  # Mode 1: From .mp4 with court crop → 518
  python extract_features_fullvideo.py \
      --video datasets/fullvideo/00001.mp4 \
      --output datasets/fullvideo/00001_court518.pt \
      --crop 100,900,200,1700 --img_size 518 --batch_size 256

  # Mode 2: From annotator-exported JPEGs (already cropped+letterboxed)
  python extract_features_fullvideo.py \
      --frames_dir datasets/v1/export/0006/00001/frames_518 \
      --output datasets/fullvideo/00001_ce518.pt \
      --batch_size 256 --num_workers 8
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
# Dataset classes for DataLoader (parallel loading with num_workers)
# ---------------------------------------------------------------------------

class FramesDirDataset(Dataset):
    """Read pre-exported JPEGs from a directory. Already preprocessed."""

    def __init__(self, frames_dir):
        self.frames_dir = frames_dir
        # Count sequential frames: 0.jpg, 1.jpg, ...
        self.total = 0
        while os.path.exists(os.path.join(frames_dir, f"{self.total}.jpg")):
            self.total += 1
        if self.total == 0:
            # Fallback: count all .jpg files
            self.total = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        jpg_path = os.path.join(self.frames_dir, f"{idx}.jpg")
        img = cv2.imread(jpg_path)
        if img is None:
            # Return black frame if missing
            return torch.zeros(3, 1, 1), idx
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = IMAGENET_TRANSFORM(img)
        return tensor, idx


class VideoDataset(Dataset):
    """Read frames from .mp4 with optional crop + letterbox.

    Pre-reads all frames into memory for random access by workers.
    For very large videos, consider using --frames_dir mode instead.
    """

    def __init__(self, video_path, crop=None, img_size=518):
        self.crop = crop
        self.img_size = img_size
        self.video_path = video_path

        # We can't share cv2.VideoCapture across workers, so each worker
        # will open its own. Store metadata only.
        cap = cv2.VideoCapture(video_path)
        self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # Each worker opens its own VideoCapture (seek is OK for indexed access)
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, img = cap.read()
        cap.release()

        if not ret or img is None:
            return torch.zeros(3, 1, 1), idx

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.crop:
            y1, y2, x1, x2 = self.crop
            img = img[y1:y2, x1:x2]
        img = letterbox(img, self.img_size)
        tensor = IMAGENET_TRANSFORM(img)
        return tensor, idx


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
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8,
                   help="DataLoader workers for parallel JPEG loading")
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

    # Build dataset
    if args.frames_dir:
        dataset = FramesDirDataset(args.frames_dir)
        print(f"Frames dir: {args.frames_dir}")
        # Show frame size from first frame
        sample = cv2.imread(os.path.join(args.frames_dir, "0.jpg"))
        if sample is not None:
            print(f"  Frame size: {sample.shape[1]}x{sample.shape[0]}")
    else:
        dataset = VideoDataset(args.video, crop=crop, img_size=args.img_size)
        print(f"Video: {args.video}")
        print(f"  FPS: {dataset.fps:.1f}")
        if crop:
            print(f"  Court crop: y=[{crop[0]}:{crop[1]}], x=[{crop[2]}:{crop[3]}]")
        print(f"  img_size: {args.img_size}")

    total_frames = len(dataset)
    print(f"  Total frames: {total_frames}")
    print(f"  batch_size: {args.batch_size}, num_workers: {args.num_workers}, device: {device}")

    # DataLoader with parallel workers
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Build model
    print("Loading DINOv2 ViT-L...")
    backbone, feat_dim = build_dinov2(device)
    print(f"  feat_dim: {feat_dim}")

    # Extract features — pre-allocate output tensor
    all_features = torch.zeros(total_frames, feat_dim, dtype=torch.float16)
    t0 = time.time()
    frames_done = 0

    for batch_tensors, batch_indices in loader:
        # batch_tensors: [B, 3, H, W], batch_indices: [B]
        batch_tensors = batch_tensors.to(device)

        with torch.no_grad():
            feats = backbone(batch_tensors)  # [B, 1024]

        # Place features at correct indices (DataLoader may reorder with workers)
        feats_cpu = feats.cpu().half()
        for i, idx in enumerate(batch_indices):
            all_features[idx.item()] = feats_cpu[i]

        frames_done += batch_tensors.shape[0]

        if frames_done % (args.batch_size * 5) == 0 or frames_done == total_frames:
            elapsed = time.time() - t0
            fps_rate = frames_done / max(elapsed, 0.1)
            eta = (total_frames - frames_done) / max(fps_rate, 1)
            print(f"  {frames_done}/{total_frames} ({frames_done/total_frames*100:.1f}%) "
                  f"| {fps_rate:.0f} frames/s | ETA {eta:.0f}s")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(all_features, args.output)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nDone. {total_frames} frames -> {all_features.shape} fp16")
    print(f"  Time: {elapsed:.0f}s ({total_frames/elapsed:.0f} frames/s)")
    print(f"  File: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
