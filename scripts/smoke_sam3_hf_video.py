import json
from pathlib import Path

import torch
from accelerate import Accelerator
from transformers.video_utils import load_video
from transformers import (
    Sam3VideoModel, Sam3VideoProcessor,
    Sam3TrackerVideoModel, Sam3TrackerVideoProcessor,
)

# -------------------------
# Fixed paths
# -------------------------
HF_LOCAL_MODEL = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/models/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7"
VIDEO_PATH = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/datasets/serve/00001.mp4"
OUT_JSON = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/outputs/smoke_sam3_tracks.json"

# -------------------------
# Config (smoke first)
# -------------------------
SCAN_INIT_MAX_FRAMES = 30   # initial scan window for "first time found"
MAX_TRACK_FRAMES = 130      # cap for smoke
PCS_SCORE_TH = 0.0          # keep it loose for dataset stage
USE_FP32 = True             # keep fp32 to avoid bf16 mismatch issues

def mask_centroid(mask_2d: torch.Tensor):
    # mask_2d: [H,W], float/bool
    ysxs = torch.nonzero(mask_2d > 0.0)
    if ysxs.numel() == 0:
        return None
    y = ysxs[:, 0].float().mean().item()
    x = ysxs[:, 1].float().mean().item()
    return [x, y]

def pick_instance_by_box_area(boxes_xyxy: torch.Tensor, scores: torch.Tensor, mode: str):
    # mode: "small" for ball, "large" for racket
    n = int(scores.shape[0])
    if n == 0:
        return None
    # top-k by score then choose by area
    topk = min(5, n)
    _, idxs = torch.topk(scores, k=topk, largest=True)
    cand = []
    for i in idxs.tolist():
        x1, y1, x2, y2 = boxes_xyxy[i].tolist()
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        cand.append((i, area))
    if mode == "small":
        return sorted(cand, key=lambda t: t[1])[0][0]
    return sorted(cand, key=lambda t: t[1], reverse=True)[0][0]

device = Accelerator().device
print("Device:", device)

# -------------------------
# Load video frames (needs av installed)
# -------------------------
video_frames, _ = load_video(VIDEO_PATH)
print("Loaded frames:", len(video_frames))

dtype = torch.float32 if USE_FP32 else torch.bfloat16

# -------------------------
# Load PCS model/processor (text -> instance masks per frame)
# -------------------------
pcs_model = Sam3VideoModel.from_pretrained(HF_LOCAL_MODEL, local_files_only=True).to(device, dtype=dtype)
pcs_proc = Sam3VideoProcessor.from_pretrained(HF_LOCAL_MODEL, local_files_only=True)

# -------------------------
# Load Tracker model/processor (points -> track identities)
# -------------------------
trk_model = Sam3TrackerVideoModel.from_pretrained(HF_LOCAL_MODEL, local_files_only=True).to(device, dtype=dtype)
trk_proc = Sam3TrackerVideoProcessor.from_pretrained(HF_LOCAL_MODEL, local_files_only=True)

# -------------------------
# Init tracker session
# -------------------------
trk_session = trk_proc.init_video_session(
    video=video_frames,
    inference_device=device,
)

# object id convention (fixed)
BALL_ID = 1
RACKET_ID = 2

have_ball = False
have_racket = False

tracks = {}  # frame_idx -> {"ball": [x,y] or None, "racket": [x,y] or None}

# -------------------------
# Helper: run PCS on a single frame_idx with a single text
# returns centroid or None
# -------------------------
def pcs_detect_one(frame_idx: int, text: str, pick_mode: str):
    # Build a true single-frame "video" to do independent image-level prediction
    # This avoids any temporal context overhead from the rest of the video.
    single_frame = [video_frames[frame_idx]]
    sess = pcs_proc.init_video_session(
        video=single_frame,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )
    # The prompt is applied to the only frame, which is index 0
    sess = pcs_proc.add_text_prompt(inference_session=sess, text=text, frame_idx=0)

    out_frame = None
    for out in pcs_model.propagate_in_video_iterator(inference_session=sess):
        # We only have one frame
        out_frame = pcs_proc.postprocess_outputs(sess, out)
        break
    
    if out_frame is None:
        return None

    scores = out_frame["scores"]
    boxes = out_frame["boxes"]
    masks = out_frame["masks"]

    if scores.numel() == 0:
        return None

    # filter by score threshold
    keep = scores >= PCS_SCORE_TH
    if keep.sum().item() == 0:
        return None

    scores_k = scores[keep]
    boxes_k = boxes[keep]
    masks_k = masks[keep]

    inst_i = pick_instance_by_box_area(boxes_k, scores_k, mode=pick_mode)
    if inst_i is None:
        return None

    m = masks_k[inst_i]
    if m.dim() == 3:
        m = m[0]
    return mask_centroid(m)

# -------------------------
# Step A: initial scan to acquire at least one prompt (ball/racket)
# -------------------------
init_start_frame = None
for fi in range(min(SCAN_INIT_MAX_FRAMES, len(video_frames))):
    if not have_ball:
        c = pcs_detect_one(fi, "ball", pick_mode="small")
        if c is not None:
            points = [[[[c[0], c[1]]]]]
            labels = [[[1]]]
            trk_proc.add_inputs_to_inference_session(
                inference_session=trk_session,
                frame_idx=fi,
                obj_ids=BALL_ID,
                input_points=points,
                input_labels=labels,
            )
            have_ball = True
            init_start_frame = fi if init_start_frame is None else min(init_start_frame, fi)
            print("init acquire ball at frame", fi, "point", c)

    if not have_racket:
        c = pcs_detect_one(fi, "racket", pick_mode="large")
        if c is not None:
            points = [[[[c[0], c[1]]]]]
            labels = [[[1]]]
            trk_proc.add_inputs_to_inference_session(
                inference_session=trk_session,
                frame_idx=fi,
                obj_ids=RACKET_ID,
                input_points=points,
                input_labels=labels,
            )
            have_racket = True
            init_start_frame = fi if init_start_frame is None else min(init_start_frame, fi)
            print("init acquire racket at frame", fi, "point", c)

    if have_ball and have_racket:
        break

if init_start_frame is None:
    raise RuntimeError("failed to acquire any prompt in initial scan window")

# Tracker requires running forward once on a frame with inputs to set start frame
_ = trk_model(inference_session=trk_session, frame_idx=init_start_frame)

# -------------------------
# Step B: tracking loop + per-frame PCS re-acquire
# -------------------------
last_ball = None
last_racket = None

for out in trk_model.propagate_in_video_iterator(trk_session):
    fi = int(out.frame_idx)
    if fi >= MAX_TRACK_FRAMES:
        break

    # tracker masks -> centroid
    masks = trk_proc.post_process_masks(
        [out.pred_masks],
        original_sizes=[[trk_session.video_height, trk_session.video_width]],
        binarize=False,
    )[0]

    ball_c = None
    racket_c = None

    # masks is usually [num_obj, 1, H, W] but be tolerant
    if have_ball:
        try:
            m = masks[0]
            if m.dim() == 3:
                m = m[0]
            ball_c = mask_centroid(m)
        except Exception:
            ball_c = None

    if have_racket:
        try:
            j = 1 if have_ball else 0
            m = masks[j]
            if m.dim() == 3:
                m = m[0]
            racket_c = mask_centroid(m)
        except Exception:
            racket_c = None

    # per-frame PCS detect (re-acquire when missing)
    if have_ball and ball_c is None:
        c = pcs_detect_one(fi, "ball", pick_mode="small")
        if c is not None:
            points = [[[[c[0], c[1]]]]]
            labels = [[[1]]]
            trk_proc.add_inputs_to_inference_session(
                inference_session=trk_session,
                frame_idx=fi,
                obj_ids=BALL_ID,
                input_points=points,
                input_labels=labels,
            )
            # run once to apply update on this frame
            _ = trk_model(inference_session=trk_session, frame_idx=fi)
            ball_c = c

    if have_racket and racket_c is None:
        c = pcs_detect_one(fi, "racket", pick_mode="large")
        if c is not None:
            points = [[[[c[0], c[1]]]]]
            labels = [[[1]]]
            trk_proc.add_inputs_to_inference_session(
                inference_session=trk_session,
                frame_idx=fi,
                obj_ids=RACKET_ID,
                input_points=points,
                input_labels=labels,
            )
            _ = trk_model(inference_session=trk_session, frame_idx=fi)
            racket_c = c

    tracks[fi] = {"ball": ball_c, "racket": racket_c}
    last_ball = ball_c if ball_c is not None else last_ball
    last_racket = racket_c if racket_c is not None else last_racket

Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_JSON).write_text(json.dumps(tracks, indent=2), encoding="utf-8")
print("Tracked frames:", len(tracks))
print("Wrote:", OUT_JSON)
