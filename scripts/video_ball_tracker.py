# video_ball_tracker.py — Full-Video Ball Tracking (HF Sam3VideoModel, text-only prompt)
# Based on HuggingFace "Pre-loaded Video Inference" documentation
import os
import json
import torch
import numpy as np
from PIL import Image
from accelerate import Accelerator
from transformers.video_utils import load_video
from transformers import Sam3VideoModel, Sam3VideoProcessor

HF_LOCAL_MODEL = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/models/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7"
VIDEO_PATH     = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/datasets/serve/00001.mp4"
OUT_DIR        = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/outputs/ball_tracker"
OUT_JSON       = os.path.join(OUT_DIR, "tracks.json")

os.makedirs(OUT_DIR, exist_ok=True)

device = Accelerator().device
dtype  = torch.bfloat16

print("Loading Sam3VideoModel...")
model     = Sam3VideoModel.from_pretrained(HF_LOCAL_MODEL, local_files_only=True).to(device, dtype=dtype)
processor = Sam3VideoProcessor.from_pretrained(HF_LOCAL_MODEL, local_files_only=True)

print("Loading video frames...")
video_frames, _ = load_video(VIDEO_PATH)
print(f"  Total frames: {len(video_frames)}")

# ── Initialize video inference session ────────────────────────────────────────
print("Initializing video session...")
inference_session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    processing_device="cpu",
    video_storage_device="cpu",
    dtype=dtype,
)

# ── Add text prompt: "ball" ────────────────────────────────────────────────────
print("Adding text prompt: 'ball'...")
inference_session = processor.add_text_prompt(
    inference_session=inference_session,
    text="ball",
)

# ── Propagate through entire video ────────────────────────────────────────────
print("Propagating in video...")

def mask_centroid(mask: torch.Tensor):
    """Return [cx, cy] pixel coords from a 2D boolean/float mask, or None."""
    ys, xs = torch.where(mask > 0.5)
    if len(xs) == 0:
        return None
    return [float(xs.float().mean()), float(ys.float().mean())]

tracks: dict = {}
for model_outputs in model.propagate_in_video_iterator(
    inference_session=inference_session, max_frame_num_to_track=len(video_frames)
):
    frame_idx = model_outputs.frame_idx
    processed = processor.postprocess_outputs(inference_session, model_outputs)

    obj_ids = processed["object_ids"]   # Tensor of int
    scores  = processed["scores"]       # Tensor of float
    masks   = processed["masks"]        # [N, 1, H, W]

    frame_data = {}
    for i, obj_id in enumerate(obj_ids.tolist()):
        mask = masks[i]
        if mask.dim() == 4:
            mask = mask[0, 0]  # [H, W]
        elif mask.dim() == 3:
            mask = mask[0]     # [H, W]

        centroid = mask_centroid(mask)
        score    = float(scores[i]) if i < len(scores) else 0.0
        frame_data[str(obj_id)] = {"centroid": centroid, "score": round(score, 4)}

    tracks[frame_idx] = frame_data
    if frame_idx % 30 == 0:
        print(f"  frame {frame_idx:04d}: {frame_data}")

print(f"\nTotal tracked frames: {len(tracks)}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
with open(OUT_JSON, "w") as f:
    json.dump(tracks, f, indent=2)
print(f"Saved tracks to: {OUT_JSON}")

# ── OpenCV Visualization ──────────────────────────────────────────────────────
import cv2

OUT_MP4 = os.path.join(OUT_DIR, "ball_tracker_vis.mp4")
print(f"Generating visualization to {OUT_MP4}...")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
out_vid = cv2.VideoWriter(OUT_MP4, fourcc, fps, (w, h))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    pts = tracks.get(frame_idx, {})
    for obj_id, data in pts.items():
        centroid = data.get("centroid")
        if centroid is not None:
            cx, cy = int(centroid[0]), int(centroid[1])
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
            cv2.putText(frame, f"ball s:{data.get('score', 0):.2f}",
                        (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, f"Frame: {frame_idx}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out_vid.write(frame)
    frame_idx += 1

cap.release()
out_vid.release()
print("Visualization saved successfully!")
