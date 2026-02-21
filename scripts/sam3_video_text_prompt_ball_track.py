# sam3_video_text_prompt_ball_track.py — Full-Video Ball Tracking (HF Sam3VideoModel)
# Ref: HF Doc "Pre-loaded Video Inference" section
#
# Key facts confirmed from HPC inspection:
#   processed["masks"]    shape: [N, H, W]  dtype: bool  device: cuda
#   processed["boxes"]    shape: [N, 4]     dtype: float  device: cuda
#   processed["scores"]   shape: [N]        dtype: float  device: cpu
#   processed["object_ids"] shape: [N]      dtype: int64  device: cpu
#   processed["prompt_to_obj_ids"] = {'ball': [0, 1, 2]}
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

# ── Initialize video session ───────────────────────────────────────────────────
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

# ── Helper: mask centroid ──────────────────────────────────────────────────────
def mask_centroid(mask: torch.Tensor):
    """mask: 2D bool [H, W] → [cx, cy] or None."""
    ys, xs = torch.where(mask)
    if len(xs) == 0:
        return None
    return [float(xs.float().mean()), float(ys.float().mean())]

def overlay_masks_on_pil(pil_img, masks_tensor, color):
    """masks_tensor: [N, H, W] bool on any device."""
    image = pil_img.convert("RGBA")
    for mask in masks_tensor.cpu():
        mask_np = mask.numpy().astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_np)
        overlay = Image.new("RGBA", image.size, color + (0,))
        overlay.putalpha(mask_img.point(lambda v: int(v * 0.5)))
        image = Image.alpha_composite(image, overlay)
    return image.convert("RGB")

# ── Propagate ─────────────────────────────────────────────────────────────────
print("Propagating through video...")
tracks: dict = {}
meta_path = os.path.join(OUT_DIR, "per_frame_meta.txt")

with open(meta_path, "w") as meta_f:
    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session,
        max_frame_num_to_track=len(video_frames),
    ):
        frame_idx = model_outputs.frame_idx
        processed = processor.postprocess_outputs(inference_session, model_outputs)

        # All fields to cpu for safe indexing
        obj_ids = processed["object_ids"].tolist()        # list[int]
        scores  = processed["scores"].tolist()            # list[float]
        masks   = processed["masks"]                      # [N, H, W] bool, cuda
        boxes   = processed["boxes"].cpu().tolist()       # [N, 4] xyxy abs coords

        frame_data = {}
        for i, obj_id in enumerate(obj_ids):
            m = masks[i]          # [H, W] bool
            centroid = mask_centroid(m)
            score    = round(scores[i], 4) if i < len(scores) else 0.0
            frame_data[str(obj_id)] = {
                "centroid": centroid,
                "score": score,
                "box": boxes[i] if i < len(boxes) else None,
            }

        tracks[frame_idx] = frame_data

        meta_f.write(
            f"frame {frame_idx:04d}  n={len(obj_ids)}  "
            f"ids={obj_ids}  scores={[round(s,3) for s in scores]}\n"
        )

        # Save mask overlay every 10th frame for visual check
        if frame_idx % 10 == 0:
            print(f"  frame {frame_idx:04d}: {frame_data}")
            pil_frame = video_frames[frame_idx]
            if isinstance(pil_frame, np.ndarray):
                pil_frame = Image.fromarray(pil_frame)
            vis = overlay_masks_on_pil(pil_frame, masks, color=(255, 0, 0))
            vis.save(os.path.join(OUT_DIR, f"frame_{frame_idx:05d}.jpg"))

print(f"\nTotal tracked frames: {len(tracks)}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
with open(OUT_JSON, "w") as f:
    json.dump(tracks, f, indent=2)
print(f"Saved tracks → {OUT_JSON}")

# ── OpenCV Visualization (MP4) ─────────────────────────────────────────────────
import cv2

print(f"Generating MP4 visualization → {OUT_MP4}...")
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
out_vid = cv2.VideoWriter(OUT_MP4, fourcc, fps, (w, h))

colors = [(0, 0, 255), (0, 255, 0), (255, 128, 0)]   # up to 3 instances

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    for i, (obj_id, data) in enumerate(tracks.get(frame_idx, {}).items()):
        c = data.get("centroid")
        if c is not None:
            cx, cy = int(c[0]), int(c[1])
            col = colors[i % len(colors)]
            cv2.circle(frame, (cx, cy), 8, col, -1)
            cv2.putText(frame, f"ball#{obj_id} {data.get('score',0):.2f}",
                        (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    cv2.putText(frame, f"Frame: {frame_idx}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out_vid.write(frame)
    frame_idx += 1

cap.release()
out_vid.release()
print("MP4 visualization saved successfully!")
