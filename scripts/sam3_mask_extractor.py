# scripts/sam3_mask_extractor.py
# ONE video -> [videoname].json + [videoname].npz (+ optional vis mp4)
#
# Deterministic protocol (PP-first):
# - Geometry MUST come from processor.postprocess_outputs() (video-resolution boolean masks).
# - Scores MUST come from raw model_outputs metadata (obj_id_to_score + obj_id_to_tracker_score),
#   but only for PP-kept obj_ids.
# - Labels come from PP "prompt_to_obj_ids" (no explicit obj_id_to_label protocol).
# - If an obj_id is not mapped to any prompt, label="unknown" (for manual review).
#
# Output JSON schema:
#   {
#     "_meta": {...},
#     "0": { "<obj_id>": {...}, ... },
#     "1": { ... },
#     ...
#   }
#
# NPZ schema:
#   masks:         [M, H, W] bool
#   frame_indices: [M] int32
#   object_ids:    [M] int32

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
parser.add_argument(
    "--hf_local_model",
    type=str,
    default="/nfs/hpc/share/zhanhaoc/hpe/tempopeak/models/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7",
)
parser.add_argument("--video_name", type=str, default="00001")
parser.add_argument("--video_path", type=str, default="/nfs/hpc/share/zhanhaoc/hpe/tempopeak/datasets/serve/00001.mp4")
parser.add_argument("--out_dir", type=str, default="/nfs/hpc/share/zhanhaoc/hpe/tempopeak/outputs/sam3_mask_extractor/")
parser.add_argument("--vis", action="store_true")

# Prompts (session initialization; labels come from PP prompt_to_obj_ids)
parser.add_argument("--prompts", nargs="+", default=["ball", "racket"])

# Runtime
parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
parser.add_argument("--processing_device", type=str, default="cpu")
parser.add_argument("--video_storage_device", type=str, default="cpu")
parser.add_argument("--max_frames", type=int, default=-1, help="<=0 means full video")

# Filtering thresholds (PP masks + raw scores metadata)
parser.add_argument("--tracker_score_min", type=float, default=0.10, help="min obj_id_to_tracker_score to keep")
parser.add_argument("--static_score_min", type=float, default=-1.0, help="<=0 disables; else min obj_id_to_score to keep")
parser.add_argument("--mask_area_min", type=int, default=1, help="min number of true pixels in mask to keep")

# Motion control (external gating only)
parser.add_argument("--max_jump_px", type=float, default=-1.0, help="<=0 disables jump rejection; else max centroid jump")
parser.add_argument("--ema_alpha", type=float, default=1.0, help="1.0 disables smoothing; typical 0.5~0.8")
parser.add_argument("--max_lost", type=int, default=0, help="reject-only mode if 0 (no prediction state carry)")
parser.add_argument("--predict_on_reject", action="store_true", help="if reject and max_lost>0, advance centroid by velocity")

# Score fusion for dataset confidence (optional)
parser.add_argument(
    "--quality_score_mode",
    type=str,
    choices=["none", "mul", "min"],
    default="mul",
    help='How to compute "quality_score" from (static_score, tracker_score): none|mul|min',
)

# Debug
parser.add_argument("--print_every", type=int, default=30, help="print progress every N frames (<=0 disables)")
parser.add_argument("--debug_first_frames", type=int, default=1, help="print mapping for first K frames")

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
out_npz = os.path.join(args.out_dir, f"{args.video_name}.npz")
out_mp4 = os.path.join(args.out_dir, f"{args.video_name}_vis.mp4")


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
def mask_centroid(mask_bool: torch.Tensor):
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


def compute_quality_score(static_score: float, tracker_score: float, mode: str) -> float:
    if mode == "none":
        return -1.0
    if mode == "mul":
        return float(static_score * tracker_score)
    if mode == "min":
        return float(min(static_score, tracker_score))
    raise ValueError(f"Unknown quality_score_mode: {mode}")


def build_obj_id_to_label_from_pp(prompt_to_obj_ids: dict, prompt_order: list):
    """
    Deterministic reverse map:
    - If an obj_id appears in multiple prompts, take the first prompt in prompt_order.
    - If not present in any prompt, it will be labeled as "unknown" later.
    """
    obj_id_to_label = {}
    for p in prompt_order:
        ids = prompt_to_obj_ids.get(p, [])
        for oid in ids:
            if oid not in obj_id_to_label:
                obj_id_to_label[oid] = p
    return obj_id_to_label


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
# Tracking state (external gating only)
# -------------------------
state = {}  # state[obj_id] = {"last_centroid": (x,y), "last_velocity": (vx,vy), "lost_count": int}


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
        frame_idx = int(model_outputs.frame_idx)

        # PP FIRST: geometry + prompt_to_obj_ids come from PP only
        pp = processor.postprocess_outputs(session, model_outputs)

        obj_ids = pp["object_ids"].tolist()  # List[int]
        masks = pp["masks"]                  # Tensor[N, H, W] bool (device)
        prompt_to_obj_ids = pp.get("prompt_to_obj_ids", {})  # dict[str, list[int]]  (PP-maintained)

        obj_id_to_label = build_obj_id_to_label_from_pp(prompt_to_obj_ids, args.prompts)

        if frame_idx < args.debug_first_frames:
            print(f"[debug] frame={frame_idx} prompt_to_obj_ids={prompt_to_obj_ids}")
            print(f"[debug] frame={frame_idx} obj_ids={obj_ids}")

        # Scores from raw metadata (dict keyed by obj_id)
        obj_id_to_static_score = dict(model_outputs.obj_id_to_score)
        obj_id_to_tracker_score = dict(model_outputs.obj_id_to_tracker_score)

        # Removed / suppressed (raw semantics)
        removed = set(model_outputs.removed_obj_ids) if model_outputs.removed_obj_ids is not None else set()
        suppressed = set(model_outputs.suppressed_obj_ids) if model_outputs.suppressed_obj_ids is not None else set()

        frame_data = {}

        for i, obj_id in enumerate(obj_ids):
            if obj_id in removed:
                continue
            if obj_id in suppressed:
                continue

            mask = masks[i]

            # PP geometry filters
            area = int(mask.sum().item())
            if area < args.mask_area_min:
                continue

            centroid = mask_centroid(mask)
            if centroid is None:
                continue

            box = mask_box_xyxy(mask)
            if box is None:
                continue

            # Raw scores
            static_score = float(obj_id_to_static_score.get(obj_id, 0.0))
            tracker_score = float(obj_id_to_tracker_score.get(obj_id, 0.0))

            if tracker_score < args.tracker_score_min:
                continue
            if args.static_score_min > 0.0 and static_score < args.static_score_min:
                continue

            quality_score = compute_quality_score(static_score, tracker_score, args.quality_score_mode)

            # Label from PP mapping
            label = obj_id_to_label.get(obj_id, "unknown")

            # Optional motion gating (no EMA or lost prediction unless you enable them)
            prev_state = state.get(obj_id)

            if args.max_jump_px > 0.0 and prev_state is not None:
                dist = l2(centroid, prev_state["last_centroid"])
                if dist > float(args.max_jump_px):
                    prev_state["lost_count"] += 1
                    if args.predict_on_reject and args.max_lost > 0 and prev_state["lost_count"] <= args.max_lost:
                        vx, vy = prev_state["last_velocity"]
                        pc = prev_state["last_centroid"]
                        prev_state["last_centroid"] = (pc[0] + vx, pc[1] + vy)
                    continue

            # Accept -> update state
            if prev_state is None:
                state[obj_id] = {
                    "last_centroid": centroid,
                    "last_velocity": (0.0, 0.0),
                    "lost_count": 0,
                }
            else:
                prev_state["lost_count"] = 0
                pc = prev_state["last_centroid"]

                if args.ema_alpha < 1.0:
                    a = float(args.ema_alpha)
                    centroid = (a * centroid[0] + (1.0 - a) * pc[0], a * centroid[1] + (1.0 - a) * pc[1])

                prev_state["last_velocity"] = (centroid[0] - pc[0], centroid[1] - pc[1])
                prev_state["last_centroid"] = centroid

            frame_data[str(obj_id)] = {
                "label": label,
                "tracker_score": round(tracker_score, 6),
                "static_score": round(static_score, 6),
                "quality_score": round(float(quality_score), 6) if quality_score >= 0 else -1.0,
                "centroid": [round(float(centroid[0]), 3), round(float(centroid[1]), 3)],
                "box_xyxy": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                "mask_idx": int(mask_counter),
                "mask_area": int(area),
            }

            # Save mask
            all_masks.append(mask.detach().to("cpu").numpy().astype(np.bool_))
            mask_frame_indices.append(frame_idx)
            mask_object_ids.append(int(obj_id))
            mask_counter += 1

        tracks[str(frame_idx)] = frame_data

        if args.print_every > 0 and (frame_idx % args.print_every == 0):
            print("frame", frame_idx, "kept_masks", mask_counter, "kept_objs_this_frame", len(frame_data))

t1 = time.time()
print("Total kept masks:", len(all_masks))
print("Time (s):", round(t1 - t0, 2))


# -------------------------
# Save JSON
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
    "tracker_score_min": float(args.tracker_score_min),
    "static_score_min": float(args.static_score_min),
    "mask_area_min": int(args.mask_area_min),
    "max_jump_px": float(args.max_jump_px),
    "ema_alpha": float(args.ema_alpha),
    "max_lost": int(args.max_lost),
    "predict_on_reject": bool(args.predict_on_reject),
    "quality_score_mode": str(args.quality_score_mode),
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

        per_frame = tracks.get(str(frame_idx), {})

        for obj_id_str, info in per_frame.items():
            midx = int(info["mask_idx"])
            if midx < 0 or midx >= masks_np.shape[0]:
                continue

            mask = masks_np[midx]

            label = info.get("label", "unknown")
            color = LABEL_COLORS.get(label, (200, 200, 200))

            overlay = np.zeros_like(frame)
            overlay[mask] = color
            frame = cv2.addWeighted(frame, 1.0, overlay, 0.4, 0)

            x0, y0, x1, y1 = info["box_xyxy"]
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)

            qs = float(info.get("quality_score", -1.0))
            ts = float(info.get("tracker_score", 0.0))
            ss = float(info.get("static_score", 0.0))
            txt = f"id={obj_id_str} {label} q={qs:.3f} ts={ts:.3f} ss={ss:.3f}"
            cv2.putText(
                frame,
                txt,
                (x0, max(0, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("Saved:", out_mp4)
else:
    print("Done (no visualization)")