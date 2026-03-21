"""Extract per-frame Farneback optical flow magnitude → [N, 196] fp16 .pt file.

Each frame's flow magnitude is spatially averaged into a 14×14 grid matching
DINOv2's patch layout (img_size/14 pixels per patch). This produces a [196]
vector per frame where each element = mean flow magnitude within that patch.

Frame 0 gets zeros (no previous frame to compute flow from).

Output: [N, 196] fp16 tensor (~28 MB for 74463 frames).

Usage:
  python extract_flow.py \
      --video datasets/v1/export/0006/00001/00001.mp4 \
      --output datasets/v1/export/0006/00001/00001_flow.pt
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description="Extract Farneback flow → [N,196] fp16")
    p.add_argument("--video", required=True, help="Input .mp4 path")
    p.add_argument("--output", required=True, help="Output .pt path")
    p.add_argument("--grid", type=int, default=14,
                   help="Spatial grid size (14 = DINOv2 patch layout)")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    G = args.grid
    N_patches = G * G  # 196

    print(f"Video: {args.video}")
    print(f"  {W}x{H} @ {fps:.1f} fps, {total} frames")
    print(f"  Grid: {G}x{G} = {N_patches} patches")
    print(f"  Output: [N, {N_patches}] fp16")

    all_flow = torch.zeros(total, N_patches, dtype=torch.float16)

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: cannot read first frame")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    fi = 1  # frame 0 stays zeros

    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        # flow: [H, W, 2] — dx, dy per pixel
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)  # [H, W]

        # Average magnitude into G×G grid
        # Reshape into grid cells and take mean
        pH = H // G
        pW = W // G
        # Crop to exact grid size (drop remainder pixels)
        mag_crop = mag[:pH * G, :pW * G]
        # Reshape: [G, pH, G, pW] → mean over (pH, pW) → [G, G]
        grid = mag_crop.reshape(G, pH, G, pW).mean(axis=(1, 3))  # [G, G]
        # Per-frame max normalization to [0, 1]
        gmax = grid.max()
        if gmax > 0:
            grid = grid / gmax

        all_flow[fi] = torch.from_numpy(grid.reshape(N_patches).astype(np.float32)).half()

        prev_gray = gray
        fi += 1

        if fi % 5000 == 0:
            elapsed = time.time() - t0
            rate = fi / elapsed
            eta = (total - fi) / rate
            print(f"  {fi}/{total} ({fi/total*100:.1f}%) "
                  f"| {rate:.0f} frames/s | ETA {eta:.0f}s")

    cap.release()

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(all_flow, args.output)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nDone. {fi} frames → {all_flow.shape} fp16")
    print(f"  Time: {elapsed:.0f}s ({fi/elapsed:.0f} frames/s)")
    print(f"  File: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
