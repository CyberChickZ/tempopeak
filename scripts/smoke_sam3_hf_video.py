# import os

# # ---------------------------------
# # HuggingFace cache path (fixed)
# # ---------------------------------
# HF_CACHE_DIR = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/models"
# os.makedirs(HF_CACHE_DIR, exist_ok=True)

import torch
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
from accelerate import Accelerator
from transformers.video_utils import load_video

# -------------------------
# Device
# -------------------------
device = Accelerator().device
print("Device:", device)

# -------------------------
# Load Model (uncomment for first time download)
# -------------------------
# model = Sam3TrackerVideoModel.from_pretrained(
#     "facebook/sam3",
#     cache_dir=HF_CACHE_DIR
# ).to(device)

# processor = Sam3TrackerVideoProcessor.from_pretrained(
#     "facebook/sam3",
#     cache_dir=HF_CACHE_DIR
# )
HF_LOCAL_MODEL = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/models/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7"

model = Sam3TrackerVideoModel.from_pretrained(
    HF_LOCAL_MODEL,
    local_files_only=True
).to(device)

processor = Sam3TrackerVideoProcessor.from_pretrained(
    HF_LOCAL_MODEL,
    local_files_only=True
)

# -------------------------
# Load Video
# -------------------------
video_path = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/datasets/serve/00001.mp4"
video_frames, _ = load_video(video_path)
print("Loaded frames:", len(video_frames))

# -------------------------
# Initialize Video Session
# -------------------------
inference_session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
)

# -------------------------
# Add Point Prompt
# -------------------------
# 在第0帧给一个点(先随便点, smoke只验证能跑通)
ann_frame_idx = 0
ann_obj_id = 1
points = [[[[210, 350]]]]
labels = [[[1]]]

processor.add_inputs_to_inference_session(
    inference_session=inference_session,
    frame_idx=ann_frame_idx,
    obj_ids=ann_obj_id,
    input_points=points,
    input_labels=labels,
)

# -------------------------
# Run Inference (HF style)
# -------------------------
# print([m for m in dir(model) if "propagate" in m])

video_segments = {}

for sam3_tracker_video_output in model.propagate_in_video_iterator(inference_session):

    video_res_masks = processor.post_process_masks(
        [sam3_tracker_video_output.pred_masks],
        original_sizes=[[inference_session.video_height, inference_session.video_width]],
        binarize=False,
    )[0]

    video_segments[sam3_tracker_video_output.frame_idx] = video_res_masks

print("Processed frames:", len(video_segments))

# example
first_key = sorted(video_segments.keys())[0]
print("Example frame:", first_key)
print("Mask shape:", video_segments[first_key].shape)
