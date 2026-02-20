import os
import json
import torch
from pathlib import Path

# NOTE: This script assumes facebookresearch/sam3 is installed.
import sam3
from sam3.model_builder import build_sam3_video_predictor

# -------------------------
# Config 
# -------------------------
VIDEO_PATH = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/datasets/serve/00001.mp4"
OUT_JSON = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/outputs/smoke_sam3_official_tracks.json"

gpus_to_use = range(torch.cuda.device_count()) if torch.cuda.is_available() else []

print(f"Building SAM3 predictor using GPUs: {gpus_to_use}")
predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

print(f"Starting session for video: {VIDEO_PATH}")
response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=VIDEO_PATH,
    )
)
session_id = response["session_id"]

# Helper to mask out center
def mask_centroid(mask_2d: torch.Tensor):
    ysxs = torch.nonzero(mask_2d > 0.0)
    if ysxs.numel() == 0:
        return None
    y = ysxs[:, 0].float().mean().item()
    x = ysxs[:, 1].float().mean().item()
    return [x, y]

# We want to track "ball" and "racket"
# We add them as text prompts to frame 0
print("Adding text prompt: 'ball'")
response_ball = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text="ball",
    )
)

print("Adding text prompt: 'racket'")
response_racket = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text="racket",
    )
)

print("Propagating in video...")
outputs_per_frame = {}
tracks = {}

# Process responses to extract masks
for response in predictor.handle_stream_request(
    request=dict(
        type="propagate_in_video",
        session_id=session_id,
    )
):
    frame_idx = response["frame_index"]
    out = response["outputs"]
    outputs_per_frame[frame_idx] = out
    
    frame_coords = {}
    for obj_id, mask_dict in out.items():
        # Usually mask_dict has 'mask' key as a boolean tensor [H, W] or similar.
        if isinstance(mask_dict, dict) and 'mask' in mask_dict:
            centroid = mask_centroid(mask_dict['mask'])
            score = mask_dict.get('score', 0.0)
            frame_coords[str(obj_id)] = {"centroid": centroid, "score": float(score) if isinstance(score, (float, torch.Tensor)) else score}
        elif isinstance(mask_dict, torch.Tensor):
            frame_coords[str(obj_id)] = {"centroid": mask_centroid(mask_dict), "score": 1.0}
    
    tracks[frame_idx] = frame_coords
    
# We need to map the object IDs to our concepts
# In official SAM3, the API automatically assigns object IDs for each detected instance across the "person"/"ball" text.
# The `outputs_per_frame` will be a dict of frame_idx -> outputs.
# Let's inspect frame 0 to see which object ID is which.

# Close session
_ = predictor.handle_request(
    request=dict(
        type="close_session",
        session_id=session_id,
    )
)
predictor.shutdown()

Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(tracks, f, indent=2)

print(f"Tracked frames: {len(tracks)}")
print(f"Wrote to: {OUT_JSON}")

# -------------------------
# Visualization 
# -------------------------
import cv2

OUT_MP4 = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/outputs/smoke_sam3_official_vis.mp4"
print(f"Generating visualization to {OUT_MP4}...")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_vid = cv2.VideoWriter(OUT_MP4, fourcc, fps, (w, h))

frame_idx = 0
colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0)] # Up to 4 distinct instances mapped

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    pts = tracks.get(frame_idx, {})
    color_idx = 0
    
    for obj_id, data in pts.items():
        centroid = data.get("centroid")
        if centroid is not None:
            cx, cy = int(centroid[0]), int(centroid[1])
            col = colors[color_idx % len(colors)]
            cv2.circle(frame, (cx, cy), 8, col, -1) 
            # In official tracking, we aren't sure which ID is explicit, so we draw all:
            cv2.putText(frame, f"ID:{obj_id}", (cx+10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            color_idx += 1
            
    cv2.putText(frame, f"Frame: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out_vid.write(frame)
    frame_idx += 1

cap.release()
out_vid.release()
print(f"Visualization saved successfully!")
