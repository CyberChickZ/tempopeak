"""Extract per-frame DINOv2 features → single .pt file.

Two modes (--mode):
  cls     → [N, 1024] CLS token per frame (default, backward compatible)
  spatial → [N, n_patches, 1024] patch tokens per frame (for CrossAttnPooling)
           n_patches depends on img_size: 196 for 224px, 1369 for 518px

Two input modes:
  --video      Read .mp4 directly (sequential, fast on NFS)
  --frames_dir Read pre-exported JPEGs (thread pool for parallel IO)

Uses threaded prefetch to overlap disk IO and GPU compute.

Usage:
  # CLS tokens (default)
  python extract_features_fullvideo.py \
      --video datasets/v1/export/0006/00001/00001.mp4 \
      --output datasets/v1/export/0006/00001/00001_518.pt \
      --img_size 518 --batch_size 256

  # Spatial patch tokens (for CrossAttnPooling)
  python extract_features_fullvideo.py \
      --video datasets/v1/export/0006/00001/00001.mp4 \
      --output datasets/v1/export/0006/00001/00001_spatial.pt \
      --img_size 224 --batch_size 128 --mode spatial
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from queue import Queue

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_dinov2(device, mode="cls"):
    """Build frozen DINOv2 ViT-L backbone.

    Args:
        mode: "cls"     → returns [B, 1024] CLS token
              "spatial" → returns [B, n_patches, 1024] patch tokens (excl CLS)

    Returns (model, feat_dim=1024).
    """
    from transformers import AutoModel

    dinov2 = AutoModel.from_pretrained("facebook/dinov2-large")

    class _DINOv2Feats(nn.Module):
        def __init__(self, model, mode):
            super().__init__()
            self.model = model
            self.mode = mode

        def forward(self, x):
            outputs = self.model(pixel_values=x)
            if self.mode == "spatial":
                return outputs.last_hidden_state[:, 1:, :]  # [B, n_patches, 1024]
            return outputs.last_hidden_state[:, 0, :]        # [B, 1024]

    backbone = _DINOv2Feats(dinov2, mode).to(device).eval()
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
# Frame reading functions
# ---------------------------------------------------------------------------

def _read_jpeg(args):
    """Read single JPEG → tensor. For ThreadPoolExecutor."""
    idx, path = args
    img = cv2.imread(path)
    if img is None:
        return idx, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return idx, IMAGENET_TRANSFORM(img)


def produce_batches_from_dir(frames_dir, total, batch_size, num_workers, queue):
    """Producer thread: read JPEGs in parallel via ThreadPoolExecutor, push batches to queue.

    Processes in windows of batch_size*4 frames to avoid submitting all 74k jobs
    at once (pool.map creates all futures eagerly → decoded tensors pile up in RAM).
    """
    pool = ThreadPoolExecutor(max_workers=num_workers)

    batch_indices = []
    batch_tensors = []
    WINDOW = batch_size * 4  # submit at most this many frames at a time

    for win_start in range(0, total, WINDOW):
        win_end = min(win_start + WINDOW, total)
        jobs = [(i, os.path.join(frames_dir, f"{i}.jpg"))
                for i in range(win_start, win_end)]

        for idx, tensor in pool.map(_read_jpeg, jobs):
            if tensor is None:
                continue
            batch_indices.append(idx)
            batch_tensors.append(tensor)

            if len(batch_tensors) == batch_size:
                queue.put((torch.stack(batch_tensors), batch_indices))
                batch_indices = []
                batch_tensors = []

    # Last partial batch
    if batch_tensors:
        queue.put((torch.stack(batch_tensors), batch_indices))

    pool.shutdown()
    queue.put(None)  # Sentinel


def produce_batches_from_video(video_path, crop, img_size, batch_size, queue):
    """Producer thread: sequential video read (optimal for NFS), push batches to queue."""
    cap = cv2.VideoCapture(video_path)

    batch_indices = []
    batch_tensors = []
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
        tensor = IMAGENET_TRANSFORM(img)

        batch_indices.append(fi)
        batch_tensors.append(tensor)
        fi += 1

        if len(batch_tensors) == batch_size:
            queue.put((torch.stack(batch_tensors), batch_indices))
            batch_indices = []
            batch_tensors = []

    cap.release()

    if batch_tensors:
        queue.put((torch.stack(batch_tensors), batch_indices))

    queue.put(None)  # Sentinel


def main():
    p = argparse.ArgumentParser(
        description="Extract full-video DINOv2 features")
    p.add_argument("--video", type=str, default=None,
                   help="Input .mp4 path (sequential read, best for NFS)")
    p.add_argument("--frames_dir", type=str, default=None,
                   help="Pre-exported JPEG dir (parallel thread read)")
    p.add_argument("--output", required=True,
                   help="Output .pt path ([N, 1024] fp16)")
    p.add_argument("--crop", type=str, default=None,
                   help="Court crop y1,y2,x1,x2 (video mode only)")
    p.add_argument("--img_size", type=int, default=518,
                   help="Letterbox size (video mode only, 518=DINOv2 native)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=32,
                   help="Thread pool size for JPEG reading (frames_dir mode)")
    p.add_argument("--mode", type=str, default="cls",
                   choices=["cls", "spatial"],
                   help="cls → [N,1024] CLS token; spatial → [N,n_patches,1024] patch tokens")
    p.add_argument("--device", type=str, default="auto",
                   help="auto = cuda if available else cpu (never MPS)")
    args = p.parse_args()

    if not args.video and not args.frames_dir:
        p.error("Must provide --video or --frames_dir")
    if args.video and args.frames_dir:
        p.error("Provide --video OR --frames_dir, not both")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"

    crop = None
    if args.crop:
        parts = [int(x) for x in args.crop.split(",")]
        assert len(parts) == 4, "Crop must be y1,y2,x1,x2"
        crop = tuple(parts)

    # Count total frames
    if args.frames_dir:
        total_frames = 0
        while os.path.exists(os.path.join(args.frames_dir, f"{total_frames}.jpg")):
            total_frames += 1
        if total_frames == 0:
            total_frames = len([f for f in os.listdir(args.frames_dir) if f.endswith(".jpg")])
        sample = cv2.imread(os.path.join(args.frames_dir, "0.jpg"))
        print(f"Frames dir: {args.frames_dir}")
        if sample is not None:
            print(f"  Frame size: {sample.shape[1]}x{sample.shape[0]}")
    else:
        cap = cv2.VideoCapture(args.video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print(f"Video: {args.video}")
        print(f"  FPS: {fps:.1f}, img_size: {args.img_size}")
        if crop:
            print(f"  Court crop: y=[{crop[0]}:{crop[1]}], x=[{crop[2]}:{crop[3]}]")

    print(f"  Total frames: {total_frames}")
    print(f"  batch_size: {args.batch_size}, device: {device}")
    if args.frames_dir:
        print(f"  num_workers: {args.num_workers} (thread pool)")

    # Build model
    print(f"Loading DINOv2 ViT-L (mode={args.mode})...")
    backbone, feat_dim = build_dinov2(device, mode=args.mode)
    print(f"  feat_dim: {feat_dim}, mode: {args.mode}")

    # Determine n_patches dynamically (196 for 224px, 1369 for 518px)
    if args.mode == "spatial":
        if args.frames_dir:
            sample_img = cv2.imread(os.path.join(args.frames_dir, "0.jpg"))
            sh, sw = sample_img.shape[:2]
        else:
            sh = sw = args.img_size
        with torch.no_grad():
            dummy = torch.randn(1, 3, sh, sw, device=device)
            n_patches = backbone(dummy).shape[1]
            del dummy
        grid_side = int(n_patches ** 0.5)
        est_gb = total_frames * n_patches * feat_dim * 2 / 1e9
        print(f"  n_patches: {n_patches} ({grid_side}x{grid_side} grid)")
        print(f"  Estimated output: {est_gb:.1f} GB fp16")

    # Determine output shape
    if args.mode == "spatial":
        feat_shape = (total_frames, n_patches, feat_dim)
    else:
        feat_shape = (total_frames, feat_dim)

    est_bytes = np.prod(feat_shape) * 2  # fp16
    use_chunks = est_bytes > 50e9  # >50 GB → save in chunks to avoid OOM
    CHUNK_FRAMES = 5000  # ~14 GB per chunk for spatial 518

    if use_chunks:
        # Chunked mode: save directly to .npy files, no large pre-allocation
        output_dir = args.output.replace(".pt", "_chunks")
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Chunked mode: {CHUNK_FRAMES} frames/chunk → {output_dir}/")
        print(f"  Total: {est_bytes/1e9:.0f} GB across ~{(total_frames + CHUNK_FRAMES - 1) // CHUNK_FRAMES} chunks")
        # RAM buffer for current chunk only
        if args.mode == "spatial":
            chunk_buf = np.zeros((CHUNK_FRAMES, n_patches, feat_dim), dtype=np.float16)
        else:
            chunk_buf = np.zeros((CHUNK_FRAMES, feat_dim), dtype=np.float16)
        chunk_start = 0
        chunk_idx = 0
        saved_chunks = []
    else:
        all_features = torch.zeros(*feat_shape, dtype=torch.float16)

    # Start producer thread (reads frames → pushes batches to queue)
    # Queue size 2 = double buffering: producer fills next batch while GPU processes current
    queue = Queue(maxsize=2)

    if args.frames_dir:
        producer = Thread(target=produce_batches_from_dir,
                          args=(args.frames_dir, total_frames, args.batch_size,
                                args.num_workers, queue))
    else:
        producer = Thread(target=produce_batches_from_video,
                          args=(args.video, crop, args.img_size, args.batch_size, queue))
    producer.start()

    # Consumer: GPU inference
    t0 = time.time()
    frames_done = 0

    def _save_chunk(chunk_buf, chunk_start, actual_len, output_dir, chunk_idx):
        """Save one chunk as .npy file."""
        chunk_path = os.path.join(output_dir, "chunk_%05d.npy" % chunk_idx)
        np.save(chunk_path, chunk_buf[:actual_len])
        return {"path": chunk_path, "start": chunk_start,
                "end": chunk_start + actual_len, "frames": actual_len}

    while True:
        item = queue.get()
        if item is None:
            break

        batch_tensors, batch_indices = item
        batch_tensors = batch_tensors.to(device)

        with torch.no_grad():
            feats = backbone(batch_tensors)

        feats_cpu = feats.cpu().half()

        if use_chunks:
            for i, idx in enumerate(batch_indices):
                local = idx - chunk_start
                if local >= CHUNK_FRAMES:
                    # Save current chunk, start new one
                    info = _save_chunk(chunk_buf, chunk_start,
                                       CHUNK_FRAMES, output_dir, chunk_idx)
                    saved_chunks.append(info)
                    print(f"    → saved {info['path']} ({info['frames']} frames)")
                    chunk_idx += 1
                    chunk_start += CHUNK_FRAMES
                    chunk_buf[:] = 0
                    local = idx - chunk_start
                chunk_buf[local] = feats_cpu[i].numpy()
        else:
            for i, idx in enumerate(batch_indices):
                all_features[idx] = feats_cpu[i]

        frames_done += len(batch_indices)
        elapsed = time.time() - t0
        fps_rate = frames_done / max(elapsed, 0.1)
        eta = (total_frames - frames_done) / max(fps_rate, 1)
        print(f"  {frames_done}/{total_frames} ({frames_done/total_frames*100:.1f}%) "
              f"| {fps_rate:.0f} frames/s | ETA {eta:.0f}s")

    producer.join()

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if use_chunks:
        # Save final partial chunk
        remaining = total_frames - chunk_start
        if remaining > 0:
            info = _save_chunk(chunk_buf, chunk_start,
                               remaining, output_dir, chunk_idx)
            saved_chunks.append(info)
            print(f"    → saved {info['path']} ({info['frames']} frames)")
        del chunk_buf

        # Write manifest
        import json
        manifest = {
            "total_frames": total_frames,
            "feat_shape": list(feat_shape),
            "dtype": "float16",
            "chunks": saved_chunks,
        }
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        elapsed = time.time() - t0
        total_size = sum(os.path.getsize(c["path"]) for c in saved_chunks)
        print(f"\nDone. {total_frames} frames → {len(saved_chunks)} chunks")
        print(f"  Time: {elapsed:.0f}s ({total_frames/elapsed:.0f} frames/s)")
        print(f"  Dir: {output_dir}/ ({total_size/1e9:.1f} GB total)")
        print(f"  Manifest: {manifest_path}")
    else:
        torch.save(all_features, args.output)
        elapsed = time.time() - t0
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"\nDone. {total_frames} frames → {feat_shape} fp16")
        print(f"  Time: {elapsed:.0f}s ({total_frames/elapsed:.0f} frames/s)")
        print(f"  File: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
