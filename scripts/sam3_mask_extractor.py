# sam3_mask_extractor.py
# ONE video → [videoname].json + [videoname].npz
# Optional: --vis to render MP4

import os
import json
import argparse
import torch
import numpy as np
from transformers.video_utils import load_video
from transformers import Sam3VideoModel, Sam3VideoProcessor

# ───────────────────────────────────────────────
# Args
# ───────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--vis", action="store_true")
args = parser.parse_args()

# ───────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────
HF_LOCAL_MODEL = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/models/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7"
VIDEO_NAME     = "00001"
VIDEO_PATH     = f"/nfs/hpc/share/zhanhaoc/hpe/tempopeak/datasets/serve/{VIDEO_NAME}.mp4"
OUT_DIR        = f"/nfs/hpc/share/zhanhaoc/hpe/tempopeak/outputs/sam3_mask_extractor/"

OUT_JSON = os.path.join(OUT_DIR, f"{VIDEO_NAME}.json")
OUT_MASK = os.path.join(OUT_DIR, f"{VIDEO_NAME}.npz")
OUT_MP4  = os.path.join(OUT_DIR, f"{VIDEO_NAME}_vis.mp4")

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype  = torch.bfloat16

# ───────────────────────────────────────────────
# Load model
# ───────────────────────────────────────────────
print("Loading model...")
model = Sam3VideoModel.from_pretrained(
    HF_LOCAL_MODEL,
    local_files_only=True
).to(device, dtype=dtype)

processor = Sam3VideoProcessor.from_pretrained(
    HF_LOCAL_MODEL,
    local_files_only=True
)

print("Loading video...")
video_frames, _ = load_video(VIDEO_PATH)
print("Total frames:", len(video_frames))

# ───────────────────────────────────────────────
# Init session
# ───────────────────────────────────────────────
session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    processing_device="cpu",
    video_storage_device="cpu",
    dtype=dtype,
)

for prompt in ("ball", "racket"):
    session = processor.add_text_prompt(
        inference_session=session,
        text=prompt
    )

# ───────────────────────────────────────────────
# Helper
# ───────────────────────────────────────────────
def mask_centroid(mask):
    ys, xs = torch.where(mask)
    if len(xs) == 0:
        return None
    return [float(xs.float().mean()), float(ys.float().mean())]

# ───────────────────────────────────────────────
# Propagate
# ───────────────────────────────────────────────
print("Propagating...")

tracks = {}
all_masks = []
mask_frame_indices = []
mask_object_ids = []

mask_counter = 0

for model_outputs in model.propagate_in_video_iterator(
    inference_session=session,
    max_frame_num_to_track=len(video_frames),
):
    frame_idx = model_outputs.frame_idx
    processed = processor.postprocess_outputs(session, model_outputs)

    obj_ids      = processed["object_ids"].tolist()
    scores       = processed["scores"].tolist()
    masks_tensor = processed["masks"]  # [N, H, W]
    boxes        = processed["boxes"].cpu().tolist()
    p2o          = processed["prompt_to_obj_ids"]

    # Read the dynamic tracking score from raw model outputs
    tracker_scores_dict = getattr(model_outputs, "obj_id_to_tracker_score", {})
    if not tracker_scores_dict and isinstance(model_outputs, dict):
        tracker_scores_dict = model_outputs.get("obj_id_to_tracker_score", {})

    id_to_prompt = {}
    for label, ids in p2o.items():
        for oid in ids:
            id_to_prompt[oid] = label

    frame_data = {}

    for i, obj_id in enumerate(obj_ids):
        centroid = mask_centroid(masks_tensor[i])

        # Use tracker score if available, otherwise fallback to static prompt score
        dyn_score = tracker_scores_dict.get(obj_id, scores[i])
        if hasattr(dyn_score, "item"):
            dyn_score = dyn_score.item()

        if float(dyn_score) <= 0.1:
            continue

        frame_data[str(obj_id)] = {
            "prompt":   id_to_prompt.get(obj_id, "unknown"),
            "score":    round(float(dyn_score), 4),
            "centroid": centroid,
            "box":      boxes[i],
            "mask_idx": mask_counter
        }

        # 保存 mask 到统一数组
        all_masks.append(masks_tensor[i].cpu().numpy())
        mask_frame_indices.append(frame_idx)
        mask_object_ids.append(obj_id)

        mask_counter += 1

    tracks[frame_idx] = frame_data

    if frame_idx % 30 == 0:
        print("frame", frame_idx)

print("Total masks:", len(all_masks))

# ───────────────────────────────────────────────
# Save JSON
# ───────────────────────────────────────────────
with open(OUT_JSON, "w") as f:
    json.dump(tracks, f, indent=2)

print("Saved:", OUT_JSON)

# ───────────────────────────────────────────────
# Save compressed masks
# ───────────────────────────────────────────────
np.savez_compressed(
    OUT_MASK,
    masks=np.array(all_masks, dtype=np.bool_),
    frame_indices=np.array(mask_frame_indices, dtype=np.int32),
    object_ids=np.array(mask_object_ids, dtype=np.int32),
)

print("Saved:", OUT_MASK)

# ───────────────────────────────────────────────
# Visualization (optional)
# ───────────────────────────────────────────────
if args.vis:
    import cv2

    print("Rendering video...")

    data = np.load(OUT_MASK)
    masks_np = data["masks"]

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        OUT_MP4,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    LABEL_COLORS = {
        "ball":   (0, 0, 255),
        "racket": (255, 128, 0),
    }

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for obj_id, info in tracks.get(frame_idx, {}).items():
            midx = info["mask_idx"]
            mask = masks_np[midx]

            color = LABEL_COLORS.get(info["prompt"], (200,200,200))
            overlay = np.zeros_like(frame)
            overlay[mask] = color

            frame = cv2.addWeighted(frame, 1.0, overlay, 0.4, 0)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("Saved:", OUT_MP4)

else:
    print("Done (no visualization)")