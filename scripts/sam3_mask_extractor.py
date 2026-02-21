# sam3_video_text_prompt_ball_track.py — Ball + Racket Tracking (HF Sam3VideoModel)
# Ref: HF Doc "Pre-loaded Video Inference" section
#
# Key facts (confirmed from HPC inspection):
#   processed["masks"]            shape: [N, H, W]  dtype: bool  device: cuda
#   processed["boxes"]            shape: [N, 4]     dtype: float  device: cuda  (XYXY abs)
#   processed["scores"]           shape: [N]        dtype: float  device: cpu
#   processed["object_ids"]       shape: [N]        dtype: int64  device: cpu
#   processed["prompt_to_obj_ids"]: {'ball': [0,1,2], 'racket': [3,4]}  etc.
#
# Design:
#   - ONE session, TWO text prompts (ball + racket) → shared backbone + memory
#   - Per prompt: keep only top-1 score instance
#   - JSON stores per-frame centroids + scores keyed by prompt name
#   - OpenCV MP4: ball = red, racket = blue
#
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
OUT_DIR        = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/outputs/sam3_video_ball_track"
OUT_JSON       = os.path.join(OUT_DIR, "tracks.json")
OUT_MP4        = os.path.join(OUT_DIR, "ball_track_vis.mp4")

os.makedirs(OUT_DIR, exist_ok=True)

device = Accelerator().device
dtype  = torch.bfloat16

print("Loading Sam3VideoModel...")
model     = Sam3VideoModel.from_pretrained(HF_LOCAL_MODEL, local_files_only=True).to(device, dtype=dtype)
processor = Sam3VideoProcessor.from_pretrained(HF_LOCAL_MODEL, local_files_only=True)

print("Loading video frames...")
video_frames, _ = load_video(VIDEO_PATH)
print(f"  Total frames: {len(video_frames)}")

# ── Initialize a SINGLE video session ─────────────────────────────────────────
print("Initializing video session...")
inference_session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    processing_device="cpu",
    video_storage_device="cpu",
    dtype=dtype,
)

# ── Add BOTH prompts to the same session (shared backbone + memory bank) ───────
print("Adding text prompt: 'ball'...")
inference_session = processor.add_text_prompt(
    inference_session=inference_session,
    text="ball",
)

print("Adding text prompt: 'racket'...")
inference_session = processor.add_text_prompt(
    inference_session=inference_session,
    text="racket",
)

# After this, processed["prompt_to_obj_ids"] will look like:
#   {'ball': [0, 1, 2], 'racket': [3, 4]}

# ── Helper ────────────────────────────────────────────────────────────────────
def mask_centroid(mask: torch.Tensor):
    """mask: 2D bool [H, W] → [cx, cy] float or None."""
    ys, xs = torch.where(mask)
    if len(xs) == 0:
        return None
    return [float(xs.float().mean()), float(ys.float().mean())]


# ── Propagate through whole video ─────────────────────────────────────────────
print("Propagating through video...")
# tracks: {frame_idx: {str(obj_id): {prompt, centroid, score}}}
tracks: dict = {}

for model_outputs in model.propagate_in_video_iterator(
    inference_session=inference_session,
    max_frame_num_to_track=len(video_frames),
):
    frame_idx = model_outputs.frame_idx
    processed = processor.postprocess_outputs(inference_session, model_outputs)

    obj_ids_list      = processed["object_ids"].tolist()   # list[int]
    scores_list       = processed["scores"].tolist()        # list[float]
    masks_tensor      = processed["masks"]                  # [N, H, W] bool, cuda
    prompt_to_obj_ids = processed["prompt_to_obj_ids"]     # {'ball':[...], 'racket':[...]}

    # Invert: obj_id → prompt label
    id_to_prompt = {}
    for label, ids in prompt_to_obj_ids.items():
        for oid in ids:
            id_to_prompt[oid] = label

    frame_data = {}
    for i, obj_id in enumerate(obj_ids_list):
        centroid = mask_centroid(masks_tensor[i])
        frame_data[str(obj_id)] = {
            "prompt":   id_to_prompt.get(obj_id, "unknown"),
            "centroid": centroid,
            "score":    round(scores_list[i], 4),
        }

    tracks[frame_idx] = frame_data

    if frame_idx % 30 == 0:
        print(f"  frame {frame_idx:04d}  objects={list(frame_data.keys())}")

print(f"\nTotal tracked frames: {len(tracks)}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
with open(OUT_JSON, "w") as f:
    json.dump(tracks, f, indent=2)
print(f"Saved tracks → {OUT_JSON}")

# ── OpenCV MP4 Visualization ───────────────────────────────────────────────────
import cv2

print(f"Generating MP4 visualization → {OUT_MP4}...")
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
out_vid = cv2.VideoWriter(OUT_MP4, fourcc, fps, (w, h))

LABEL_COLORS = {
    "ball":    (0, 0, 255),    # red
    "racket":  (255, 64, 0),   # orange-blue
    "unknown": (0, 255, 255),  # yellow
}

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    for obj_id, info in tracks.get(frame_idx, {}).items():
        c = info.get("centroid")
        if c is not None:
            cx, cy = int(c[0]), int(c[1])
            label = info.get("prompt", "unknown")
            col   = LABEL_COLORS.get(label, (200, 200, 200))
            cv2.circle(frame, (cx, cy), 8, col, -1)
            cv2.putText(frame, f"{label}#{obj_id} {info.get('score', 0):.2f}",
                        (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    cv2.putText(frame, f"Frame: {frame_idx}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out_vid.write(frame)
    frame_idx += 1

cap.release()
out_vid.release()
print("MP4 visualization saved successfully!")
