import os
import json
import torch
from pathlib import Path
from accelerate import Accelerator
from transformers.video_utils import load_video
from transformers import Sam3VideoModel, Sam3VideoProcessor

# -------------------------
# Config 
# -------------------------
HF_LOCAL_MODEL = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/models/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7"
VIDEO_PATH = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/datasets/serve/00001.mp4"
OUT_JSON = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/outputs/smoke_sam3_hf_video_pcs_tracks.json"

device = Accelerator().device
print("Device:", device)

print("Loading PCS Video model...")
model = Sam3VideoModel.from_pretrained(HF_LOCAL_MODEL, local_files_only=True).to(device, dtype=torch.bfloat16)
processor = Sam3VideoProcessor.from_pretrained(HF_LOCAL_MODEL, local_files_only=True)

print("Loading video frames...")
video_frames, _ = load_video(VIDEO_PATH)

print("Initializing video inference session...")
inference_session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    processing_device="cpu",
    video_storage_device="cpu",
    dtype=torch.bfloat16,
)

# Add text prompts for tracking
# We can add both "ball" and "racket" on the first frame.
print("Adding 'ball' text prompt...")
inference_session = processor.add_text_prompt(
    inference_session=inference_session,
    text="ball",
    frame_idx=0
)

print("Adding 'racket' text prompt...")
inference_session = processor.add_text_prompt(
    inference_session=inference_session,
    text="racket",
    frame_idx=0
)

def mask_centroid(mask_2d: torch.Tensor):
    # mask_2d: [H,W], float/bool
    ysxs = torch.nonzero(mask_2d > 0.0)
    if ysxs.numel() == 0:
        return None
    y = ysxs[:, 0].float().mean().item()
    x = ysxs[:, 1].float().mean().item()
    return [x, y]

print("Propagating and processing frames...")
outputs_per_frame = {}
tracks = {}

for model_outputs in model.propagate_in_video_iterator(
    inference_session=inference_session, max_frame_num_to_track=150
):
    frame_idx = model_outputs.frame_idx
    processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
    outputs_per_frame[frame_idx] = processed_outputs
    
    # Process for JSON dump
    frame_coords = {}
    obj_ids = processed_outputs["object_ids"]
    scores = processed_outputs["scores"]
    masks = processed_outputs["masks"]
    
    for i in range(len(obj_ids)):
        obj_id = obj_ids[i].item()
        score = scores[i].item()
        mask = masks[i]
        
        # Squeeze down to 2D
        if mask.dim() == 3:
            mask = mask[0]
            
        centroid = mask_centroid(mask)
        frame_coords[str(obj_id)] = {"centroid": centroid, "score": score}
        
    tracks[frame_idx] = frame_coords
    
    if frame_idx % 10 == 0:
        print(f"Processed frame {frame_idx}")

Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(tracks, f, indent=2)

print(f"Tracked frames: {len(tracks)}")
print(f"Wrote to: {OUT_JSON}")

# -------------------------
# Visualization 
# -------------------------
import cv2

OUT_MP4 = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/outputs/smoke_sam3_hf_video_pcs_vis.mp4"
print(f"Generating visualization to {OUT_MP4}...")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_vid = cv2.VideoWriter(OUT_MP4, fourcc, fps, (w, h))

frame_idx = 0
colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0)]

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
            # Write Label and Score
            scorestr = f"{data.get('score', 0):.2f}"
            cv2.putText(frame, f"ID:{obj_id} s:{scorestr}", (cx+10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            color_idx += 1
            
    cv2.putText(frame, f"Frame: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out_vid.write(frame)
    frame_idx += 1

cap.release()
out_vid.release()
print("Visualization saved successfully!")
