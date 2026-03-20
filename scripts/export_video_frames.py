"""Export full video frames + annot.json into export/ format (no YOLO needed).

Creates:
  datasets/v1/export/{video_id}/{clip_id}/
    ├── frames/{0..N}.jpg
    └── annot.json

Usage:
  python scripts/export_video_frames.py \
      --video /path/to/00001.mp4 \
      --hit_json /path/to/00001.json \
      --out_dir datasets/v1/export/0006/00001
"""

import argparse
import json
import os

import cv2


def main():
    p = argparse.ArgumentParser(
        description="Export video frames + annot.json (no YOLO)")
    p.add_argument("--video", required=True, help="Input .mp4 path")
    p.add_argument("--hit_json", default=None,
                   help="JSON with HIT/hits array (global frame indices)")
    p.add_argument("--out_dir", required=True,
                   help="Output clip dir, e.g. datasets/v1/export/0006/00001")
    p.add_argument("--quality", type=int, default=85,
                   help="JPEG quality (default 85)")
    args = p.parse_args()

    # Load hits
    hits = []
    if args.hit_json:
        with open(args.hit_json) as f:
            data = json.load(f)
        hits = sorted(data.get("HIT", data.get("hits", [])))
        print(f"Loaded {len(hits)} hits from {args.hit_json}")

    # Open video
    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {args.video}")
    print(f"  Frames: {total_frames}, FPS: {fps:.1f}")

    # Create output dirs
    frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Export frames
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.quality]
    fi = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        out_path = os.path.join(frames_dir, f"{fi}.jpg")
        cv2.imwrite(out_path, img, encode_params)
        fi += 1
        if fi % 1000 == 0:
            print(f"  {fi}/{total_frames} frames exported...")
    cap.release()
    print(f"  {fi} frames exported to {frames_dir}/")

    # Create annot.json — all hits assigned to hitter=1 (→ p1)
    annot = {
        "clip": os.path.basename(args.out_dir),
        "video": os.path.basename(os.path.dirname(args.out_dir)),
        "total_frames": fi,
        "hits": hits,
        "hitters": [1] * len(hits),
        "frames": {},
    }
    annot_path = os.path.join(args.out_dir, "annot.json")
    with open(annot_path, "w") as f:
        json.dump(annot, f, indent=2)
    print(f"  annot.json: {len(hits)} hits, hitters=all p1")

    print(f"\nDone. Next steps:")
    print(f"  1. extract_features.py --data_dir={os.path.dirname(os.path.dirname(args.out_dir))} --backbone=dinov2")
    print(f"  2. train_multihit.py --clip_data_dir=...")


if __name__ == "__main__":
    main()
