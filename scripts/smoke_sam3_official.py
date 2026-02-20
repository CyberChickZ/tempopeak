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

