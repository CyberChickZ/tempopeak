# sam3_mask_extractor.py
# ONE video -> [videoname].json + [videoname].npz (+ optional vis mp4)
# Deterministic protocol: uses Sam3VideoSegmentationOutput fields only (no hasattr/getattr fallbacks).
#
# Output JSON schema:
#   {
#     "_meta": {all resolved args + runtime info},
#     "0": { "<obj_id>": {...}, ... },   # frame 0
#     "1": { ... },                      # frame 1
#     ...
#   }
#
# NPZ schema:
#   masks:         [M, H, W] bool
#   frame_indices: [M] int32
#   object_ids:    [M] int32
#
# Notes:
# - Boxes are computed from masks (XYXY absolute pixel coords).
# - Labels are provided explicitly via --obj_id_to_label (no black-box prompt mapping).
# - Optional motion control layer: max_jump_px, ema_alpha, max_lost, predict_on_reject.

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
from contextlib import closing

# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser()

# IO
parser.add_argument("--hf_local_model", type=str, default="/nfs/hpc/share/zhanhaoc/hpe/tempopeak/models/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7")
parser.add_argument("--video_name", type=str, default="00001")
parser.add_argument("--video_path", type=str, default="/nfs/hpc/share/zhanhaoc/hpe/tempopeak/datasets/serve/00001.mp4")
parser.add_argument("--out_dir", type=str, default="/nfs/hpc/share/zhanhaoc/hpe/tempopeak/outputs/sam3_mask_extractor/")
parser.add_argument("--vis", action="store_true")

# Prompts (session initialization only; labels are controlled by obj_id_to_label)
parser.add_argument("--prompts", nargs="+", default=["ball", "racket"])

# Runtime
parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
parser.add_argument("--processing_device", type=str, default="cpu")
parser.add_argument("--video_storage_device", type=str, default="cpu")
parser.add_argument("--max_frames", type=int, default=-1, help="<=0 means full video")

# Filtering thresholds
parser.add_argument("--tracker_score_min", type=float, default=0.10, help="min obj_id_to_tracker_score to keep")
parser.add_argument("--mask_area_min", type=int, default=1, help="min number of true pixels in mask to keep")

# Motion control (external tracking layer)
parser.add_argument("--max_jump_px", type=float, default=-1.0, help="<=0 disables jump rejection")
parser.add_argument("--ema_alpha", type=float, default=1.0, help="1.0 disables smoothing; typical 0.5~0.8")
parser.add_argument("--max_lost", type=int, default=2, help="how many consecutive rejects before we stop predicting")
parser.add_argument("--predict_on_reject", action="store_true", help="if reject, advance centroid by last velocity (state only)")

# Labeling (explicit protocol, no inference)
parser.add_argument(
    "--obj_id_to_label",
    type=str,
    default="",
    help='Comma-separated mapping: "0:ball,1:ball,2:racket". If empty, label="unknown".',
)

args = parser.parse_args()

# -------------------------
# Fail-fast checks
# -------------------------
if not os.path.isdir(args.hf_local_model):
    raise FileNotFoundError(f"hf_local_model not found: {args.hf_local_model}")
if not os.path.isfile(args.video_path):
    raise FileNotFoundError(f"video_path not found: {args.video_path}")

os.makedirs(args.out_dir, exist_ok=True)

out_json = os.path.join(args.out_dir, f"{args.video_name}.json")
out_npz  = os.path.join(args.out_dir, f"{args.video_name}.npz")
out_mp4  = os.path.join(args.out_dir, f"{args.video_name}_vis.mp4")

# -------------------------
# Dtype
# -------------------------
if args.dtype == "bf16":
    torch_dtype = torch.bfloat16
elif args.dtype == "fp16":
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

# -------------------------
# Helpers
# -------------------------
def parse_obj_id_to_label(s: str) -> dict:
    m = {}
    s = s.strip()
    if not s:
        return m
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        k, v = p.split(":")
        m[int(k.strip())] = v.strip()
    return m

OBJ_ID_TO_LABEL = parse_obj_id_to_label(args.obj_id_to_label)

def mask_centroid(mask_bool: torch.Tensor):
    # mask_bool: [H, W] bool
    ys, xs = torch.where(mask_bool)
    if xs.numel() == 0:
        return None
    x = float(xs.float().mean().item())
    y = float(ys.float().mean().item())
    return (x, y)

def mask_box_xyxy(mask_bool: torch.Tensor):
    ys, xs = torch.where(mask_bool)
    if xs.numel() == 0:
        return None
    x0 = int(xs.min().item())
    y0 = int(ys.min().item())
    x1 = int(xs.max().item())
    y1 = int(ys.max().item())
    return (x0, y0, x1, y1)

def l2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return float((dx * dx + dy * dy) ** 0.5)

# -------------------------
# Load model + processor
# -------------------------
accelerator = Accelerator()
device = accelerator.device

print("Loading model...")
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
print("Loading video...")
video_frames, _ = load_video(args.video_path)
num_frames = len(video_frames)

if args.max_frames > 0:
    num_track_frames = min(num_frames, args.max_frames)
else:
    num_track_frames = num_frames

print("Total frames:", num_frames, "Tracking frames:", num_track_frames)

# -------------------------
# Init session
# -------------------------
session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    processing_device=args.processing_device,
    video_storage_device=args.video_storage_device,
    dtype=torch_dtype,
)

for p in args.prompts:
    session = processor.add_text_prompt(inference_session=session, text=p)

# -------------------------
# Tracking state (external control layer)
# -------------------------
# state[obj_id] = {
#   "last_centroid": (x,y),
#   "last_velocity": (vx,vy),
#   "lost_count": int,
# }
state = {}

# Running stats for area rules (populated during propagation)
# racket_area_stats = list of accepted racket pixel-areas so far
racket_area_stats = []

# -------------------------
# Main loop
# -------------------------
tracks = {}

all_masks = []
mask_frame_indices = []
mask_object_ids = []
mask_counter = 0

t0 = time.time()
print("Propagating...")

iterator = model.propagate_in_video_iterator(
    inference_session=session,
    max_frame_num_to_track=num_track_frames,
)

with closing(iterator) as it:
    for model_outputs in it:
        # Deterministic protocol: Sam3VideoSegmentationOutput fields
        frame_idx = int(model_outputs.frame_idx)

        obj_ids = list(model_outputs.object_ids)  # List[int]
        removed = set(model_outputs.removed_obj_ids)  # Set[int]
        suppressed = set(model_outputs.suppressed_obj_ids)  # Set[int]

        obj_id_to_mask = dict(model_outputs.obj_id_to_mask)  # Dict[int -> Tensor[H,W]]
        obj_id_to_score = dict(model_outputs.obj_id_to_score)  # Dict[int -> float]
        obj_id_to_tracker_score = dict(model_outputs.obj_id_to_tracker_score)  # Dict[int -> float]

        frame_data = {}

        for obj_id in obj_ids:
            if obj_id in removed:
                print(f"Frame {frame_idx} [Obj {obj_id}] DROP: removed")
                continue
            if obj_id in suppressed:
                print(f"Frame {frame_idx} [Obj {obj_id}] DROP: suppressed")
                continue

            mask = obj_id_to_mask[obj_id].squeeze()  # Tensor[1,H,W] <BOOL> on device

            area = int(mask.sum().item())
            if area < args.mask_area_min:
                print(f"Frame {frame_idx} [Obj {obj_id}] DROP: area {area} < mask_area_min {args.mask_area_min}")
                continue

            centroid = mask_centroid(mask)
            if centroid is None:
                print(f"Frame {frame_idx} [Obj {obj_id}] DROP: centroid is None")
                continue

            box = mask_box_xyxy(mask)
            if box is None:
                print(f"Frame {frame_idx} [Obj {obj_id}] DROP: box is None")
                continue

            tracker_score = float(obj_id_to_tracker_score[obj_id])
            static_score = float(obj_id_to_score[obj_id])

            if tracker_score < args.tracker_score_min:
                print(f"Frame {frame_idx} [Obj {obj_id}] DROP: tracker_score {tracker_score:.3f} < {args.tracker_score_min}")
                continue

            label = OBJ_ID_TO_LABEL.get(obj_id, "unknown")

            # ======================================================
            # Motion gating / smoothing (rule-based, sequential)
            # Add new rules here. Each rule can `continue` to reject.
            # ======================================================
            prev_state = state.get(obj_id)

            # -------------------------------------------------------
            # Rule 1: Distance gating
            #   Sequential sub-rules applied in order.
            #   A rejection anywhere skips to the next obj_id.
            # -------------------------------------------------------

            # Rule 1 • ALL: max centroid displacement between frames
            DIST_MAX_ALL_PX = 100.0
            if prev_state is not None:
                dist = l2(centroid, prev_state["last_centroid"])
                if dist > DIST_MAX_ALL_PX:
                    # advance lost counter + optional prediction
                    prev_state["lost_count"] += 1
                    if args.predict_on_reject and prev_state["lost_count"] <= args.max_lost:
                        vx, vy = prev_state["last_velocity"]
                        pc = prev_state["last_centroid"]
                        prev_state["last_centroid"] = (pc[0] + vx, pc[1] + vy)
                    continue  # DROP

            # Rule 1 • BALL:  placeholder for ball-specific distance rules
            # if label == "ball":
            #     pass  # e.g. max speed from physics

            # Rule 1 • RACKET: placeholder for racket-specific distance rules
            # if label == "racket":
            #     pass

            # -------------------------------------------------------
            # Rule 2: Area gating
            #   Sequential sub-rules applied in order.
            # -------------------------------------------------------

            # Rule 2 • ALL: placeholder for global area constraints
            # e.g. if area < AREA_GLOBAL_MIN: continue

            # Rule 2 • BALL: reject if area exceeds avg accepted racket area
            if label == "ball" and len(racket_area_stats) > 0:
                avg_racket_area = sum(racket_area_stats) / len(racket_area_stats)
                if area > avg_racket_area:
                    print(f"Frame {frame_idx} [Obj {obj_id}] DROP: area {area} > avg_racket_area {avg_racket_area:.1f}")
                    continue  # DROP: ball bigger than average racket -> likely wrong object

            # Rule 2 • RACKET: placeholder for racket-specific area rules
            # if label == "racket":
            #     pass

            # -------------------------------------------------------
            # All rules passed  →  accept detection, update state
            # -------------------------------------------------------

            # Update racket area stats for the ball area gate
            if label == "racket":
                racket_area_stats.append(area)

            if prev_state is None:
                state[obj_id] = {
                    "last_centroid": centroid,
                    "last_velocity": (0.0, 0.0),
                    "lost_count": 0,
                }
            else:
                prev_state["lost_count"] = 0
                pc = prev_state["last_centroid"]

                # EMA smoothing on centroid
                if args.ema_alpha < 1.0:
                    a = float(args.ema_alpha)
                    centroid = (a * centroid[0] + (1.0 - a) * pc[0],
                                a * centroid[1] + (1.0 - a) * pc[1])

                prev_state["last_velocity"] = (centroid[0] - pc[0], centroid[1] - pc[1])
                prev_state["last_centroid"] = centroid

            frame_data[str(obj_id)] = {
                "label": label,
                "tracker_score": round(tracker_score, 6),
                "static_score": round(static_score, 6),
                "centroid": [round(float(centroid[0]), 3), round(float(centroid[1]), 3)],
                "box_xyxy": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                "mask_idx": int(mask_counter),
            }

            all_masks.append(mask.detach().to("cpu").numpy().astype(np.bool_))
            mask_frame_indices.append(frame_idx)
            mask_object_ids.append(int(obj_id))
            mask_counter += 1

        tracks[str(frame_idx)] = frame_data

        if frame_idx % 30 == 0:
            print("frame", frame_idx, "masks", mask_counter)

t1 = time.time()
print("Total masks:", len(all_masks))
print("Time (s):", round(t1 - t0, 2))

# -------------------------
# Save JSON (with exported params)
# -------------------------
meta = {
    "time_unix": time.time(),
    "hostname": platform.node(),
    "platform": platform.platform(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "device": str(device),
    "dtype": args.dtype,
    "video_name": args.video_name,
    "video_path": args.video_path,
    "num_frames_total": int(num_frames),
    "num_frames_tracked": int(num_track_frames),
    "hf_local_model": args.hf_local_model,
    "out_dir": args.out_dir,
    "out_json": out_json,
    "out_npz": out_npz,
    "out_mp4": out_mp4 if args.vis else "",
    "prompts": list(args.prompts),
    "obj_id_to_label": OBJ_ID_TO_LABEL,
    "tracker_score_min": float(args.tracker_score_min),
    "mask_area_min": int(args.mask_area_min),
    "max_jump_px": float(args.max_jump_px),
    "ema_alpha": float(args.ema_alpha),
    "max_lost": int(args.max_lost),
    "predict_on_reject": bool(args.predict_on_reject),
    "processing_device": args.processing_device,
    "video_storage_device": args.video_storage_device,
}

json_payload = {"_meta": meta}
json_payload.update(tracks)

with open(out_json, "w") as f:
    json.dump(json_payload, f, indent=2)

print("Saved:", out_json)

# -------------------------
# Save NPZ
# -------------------------
np.savez_compressed(
    out_npz,
    masks=np.array(all_masks, dtype=np.bool_),
    frame_indices=np.array(mask_frame_indices, dtype=np.int32),
    object_ids=np.array(mask_object_ids, dtype=np.int32),
)

print("Saved:", out_npz)

# -------------------------
# Visualization (optional)
# -------------------------
if args.vis:
    import cv2

    print("Rendering video...")

    data = np.load(out_npz)
    masks_np = data["masks"]

    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        out_mp4,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    LABEL_COLORS = {
        "ball": (0, 0, 255),
        "racket": (255, 128, 0),
        "unknown": (200, 200, 200),
    }

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_key = str(frame_idx)
        per_frame = tracks.get(frame_key, {})

        for obj_id_str, info in per_frame.items():
            midx = int(info["mask_idx"])
            mask = masks_np[midx]

            label = info.get("label", "unknown")
            color = LABEL_COLORS.get(label, (200, 200, 200))

            overlay = np.zeros_like(frame)
            overlay[mask] = color
            frame = cv2.addWeighted(frame, 1.0, overlay, 0.4, 0)

            # box
            x0, y0, x1, y1 = info["box_xyxy"]
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)

            # text
            txt = f"id={obj_id_str} {label} ts={info['tracker_score']:.3f}"
            cv2.putText(frame, txt, (x0, max(0, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("Saved:", out_mp4)
else:
    print("Done (no visualization)")