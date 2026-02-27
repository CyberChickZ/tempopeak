# scripts/sam3_mask_extractor.py
# ONE video -> [videoname].json + [videoname].npz (+ optional vis mp4)

import os
import json
import argparse
import time
import platform
import numpy as np
import torch
from accelerate import Accelerator
from transformers.video_utils import load_video
from transformers import Sam3VideoModel, Sam3VideoProcessor


# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--hf_local_model", type=str, required=True)
parser.add_argument("--video_name", type=str, required=True)
parser.add_argument("--video_path", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--prompts", nargs="+", default=["ball", "racket"])

parser.add_argument("--dtype", type=str, default="bf16")
parser.add_argument("--max_frames", type=int, default=-1)

parser.add_argument("--tracker_score_min", type=float, default=0.10)
parser.add_argument("--static_score_min", type=float, default=-1.0)
parser.add_argument("--mask_area_min", type=int, default=1)

parser.add_argument("--max_jump_px", type=float, default=-1.0)
parser.add_argument("--max_lost", type=int, default=0)

parser.add_argument("--force_memory_update", action="store_true")

args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
out_json = os.path.join(args.out_dir, f"{args.video_name}.json")
out_npz = os.path.join(args.out_dir, f"{args.video_name}.npz")

# -------------------------
# Dtype
# -------------------------
torch_dtype = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}[args.dtype]

# -------------------------
# Helpers
# -------------------------
def mask_centroid(mask):
    ys, xs = torch.where(mask)
    if xs.numel() == 0:
        return None
    return (float(xs.float().mean()), float(ys.float().mean()))

def l2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return float((dx * dx + dy * dy) ** 0.5)

def _ensure_low_res_mask_1x1(mask_any):
    if mask_any.ndim == 2:
        mask_any = mask_any.unsqueeze(0).unsqueeze(0)
    elif mask_any.ndim == 3:
        mask_any = mask_any.unsqueeze(0)
    return mask_any.float()

# -------------------------
# Load model
# -------------------------
accelerator = Accelerator()
device = accelerator.device

model = Sam3VideoModel.from_pretrained(
    args.hf_local_model,
    local_files_only=True,
).to(device, dtype=torch_dtype)

processor = Sam3VideoProcessor.from_pretrained(
    args.hf_local_model,
    local_files_only=True,
)

# -------------------------
# Load video
# -------------------------
video_frames, _ = load_video(args.video_path)
num_frames = len(video_frames)
if args.max_frames > 0:
    num_frames = min(num_frames, args.max_frames)

# -------------------------
# Init session
# -------------------------
session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    dtype=torch_dtype,
)

# 🔴 启用 delay 模式（核心）
session.set_delay_memory_update_until_verified(True)

for p in args.prompts:
    session = processor.add_text_prompt(inference_session=session, text=p)

# -------------------------
# External tracking state
# -------------------------
state = {}
tracks = {}
all_masks = []
mask_frame_indices = []
mask_object_ids = []

# -------------------------
# Main loop
# -------------------------
for frame_idx in range(num_frames):

    # Forward
    model_outputs = model(
        inference_session=session,
        frame_idx=int(frame_idx),
        reverse=False,
    )

    pp = processor.postprocess_outputs(session, model_outputs)
    obj_ids = pp["object_ids"].tolist()
    masks = pp["masks"]

    static_scores = dict(model_outputs.obj_id_to_score or {})
    tracker_scores = dict(model_outputs.obj_id_to_tracker_score or {})
    low_res_masks = model_outputs.obj_id_to_mask or {}

    allow_ids = set()
    reject_ids = set()

    # ---------------------
    # Decide allow / reject
    # ---------------------
    for i, obj_id in enumerate(obj_ids):

        mask = masks[i]
        area = int(mask.sum().item())
        if area < args.mask_area_min:
            reject_ids.add(obj_id)
            continue

        centroid = mask_centroid(mask)
        if centroid is None:
            reject_ids.add(obj_id)
            continue

        if tracker_scores.get(obj_id, 0.0) < args.tracker_score_min:
            reject_ids.add(obj_id)
            continue

        if args.static_score_min > 0 and static_scores.get(obj_id, 0.0) < args.static_score_min:
            reject_ids.add(obj_id)
            continue

        prev = state.get(obj_id)
        if args.max_jump_px > 0 and prev is not None:
            if l2(centroid, prev["centroid"]) > args.max_jump_px:
                reject_ids.add(obj_id)
                continue

        allow_ids.add(obj_id)

    # ---------------------
    # Memory gating
    # ---------------------
    for oid in allow_ids:
        session.set_memory_update_allowed(frame_idx, oid, True)

    for oid in reject_ids:
        session.set_memory_update_allowed(frame_idx, oid, False)
        if args.max_lost == 0:
            session.remove_object(oid)

    # 🔴 唯一写 memory 的地方
    session.commit_frame_memory(frame_idx)

    # ---------------------
    # Update state + save
    # ---------------------
    frame_data = {}

    for i, obj_id in enumerate(obj_ids):

        if obj_id not in allow_ids:
            continue

        mask = masks[i]
        centroid = mask_centroid(mask)

        state[obj_id] = {"centroid": centroid}

        if obj_id in low_res_masks:
            state[obj_id]["last_low_res_mask"] = _ensure_low_res_mask_1x1(
                low_res_masks[obj_id]
            ).to(device)

        frame_data[str(obj_id)] = {
            "centroid": centroid,
            "tracker_score": float(tracker_scores.get(obj_id, 0.0)),
            "static_score": float(static_scores.get(obj_id, 0.0)),
        }

        all_masks.append(mask.cpu().numpy().astype(np.bool_))
        mask_frame_indices.append(frame_idx)
        mask_object_ids.append(obj_id)

    tracks[str(frame_idx)] = frame_data

# -------------------------
# Save
# -------------------------
json_payload = {
    "_meta": {
        "video_name": args.video_name,
        "num_frames": num_frames,
    }
}
json_payload.update(tracks)

with open(out_json, "w") as f:
    json.dump(json_payload, f, indent=2)

np.savez_compressed(
    out_npz,
    masks=np.array(all_masks, dtype=np.bool_),
    frame_indices=np.array(mask_frame_indices, dtype=np.int32),
    object_ids=np.array(mask_object_ids, dtype=np.int32),
)

print("Saved:", out_json)
print("Saved:", out_npz)