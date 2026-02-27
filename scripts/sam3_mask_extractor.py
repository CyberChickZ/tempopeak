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
# Plan A (hard object removal on reject):
# - If a mask is rejected by motion gating (large jump, lost handling),
#   we call session.remove_object(obj_id) in the SAME frame after PP.
# - This ensures the object (including memory) is removed from the inference session,
#   so it will NOT participate in future attention/memory.
#
# Post-processing (JSON+NPZ consistent):
# - delete: remove track entries from JSON, and remove all corresponding masks from NPZ.
# - fusion: merge tracks with the same label if separated by < max_gap frames.
#          "Other" tracks are fused into the first (base) track.
#          If fusion causes collisions in the same frame, keep the better one and drop the other
#          (and also drop its NPZ mask).
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
from collections import defaultdict

import numpy as np
import torch
from accelerate import Accelerator
from transformers.video_utils import load_video
from transformers import Sam3VideoModel, Sam3VideoProcessor


# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser()

# IO
parser.add_argument("--hf_local_model", type=str, required=True)
parser.add_argument("--video_name", type=str, required=True)
parser.add_argument("--video_path", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
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
parser.add_argument(
    "--static_score_min",
    type=float,
    default=-1.0,
    help="<=0 disables; else min obj_id_to_score to keep",
)
parser.add_argument("--mask_area_min", type=int, default=1, help="min number of true pixels in mask to keep")

# Motion control (Plan A: reject => hard remove object from session)
parser.add_argument("--max_jump_px", type=float, default=-1.0, help="<=0 disables; else max centroid jump allowed")
parser.add_argument("--max_lost", type=int, default=0, help="allowed consecutive rejects before removal")
parser.add_argument("--ema_alpha", type=float, default=1.0, help="1.0 disables smoothing; typical 0.5~0.8")

# Score fusion for dataset confidence (optional)
parser.add_argument(
    "--quality_score_mode",
    type=str,
    choices=["none", "mul", "min"],
    default="mul",
    help='How to compute "quality_score" from (static_score, tracker_score): none|mul|min',
)

# Post-processing switches
parser.add_argument(
    "--post_process_rm",
    action="store_true",
    help="Delete tracks that are too short (< rm_min_len) or static (<= rm_static_px avg move).",
)
parser.add_argument(
    "--post_process_fusion",
    action="store_true",
    help="Fuse tracks with the same label if separated by < fusion_max_gap frames.",
)

# Post-processing params (explicit, stable defaults)
parser.add_argument("--rm_min_len", type=int, default=15)
parser.add_argument("--rm_static_px", type=float, default=5.0)
parser.add_argument("--fusion_max_gap", type=int, default=5)
parser.add_argument("--fusion_skip_unknown", action="store_true", help="Do not fuse label=unknown tracks.")

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


# -------------------------
# Label binding (ROBUST prompt key matching)
# -------------------------
def _canon_prompt(s: str) -> str:
    # normalize common prompt formatting differences: case, underscores, extra spaces
    return str(s).strip().lower().replace("_", " ")


def build_obj_id_to_label_from_pp(prompt_to_obj_ids: dict, prompt_order: list):
    """
    Robust reverse map using canonicalized prompt keys.

    Why:
      - HF processor may output prompt_to_obj_ids keys in a normalized form that differs from user args.prompts
        (e.g., "Tennis_Ball" vs "tennis ball" vs "tennis_ball").
      - Old exact string matching -> empty mapping -> all labels become "unknown".

    Policy:
      - Canonical key: lower + strip + '_'->' '
      - If an obj_id appears in multiple prompts, keep the first prompt in prompt_order.
      - Stored label is the ORIGINAL prompt string from prompt_order (so your dataset labels are stable).
    """
    prompt_to_obj_ids = prompt_to_obj_ids or {}

    # canonical user prompt -> original user prompt (first wins)
    canon_to_label = {}
    for p in prompt_order:
        cp = _canon_prompt(p)
        if cp not in canon_to_label:
            canon_to_label[cp] = p

    # reverse mapping from pp
    obj_id_to_label = {}
    for raw_key, ids in prompt_to_obj_ids.items():
        ck = _canon_prompt(raw_key)
        label = canon_to_label.get(ck, None)
        if label is None:
            continue
        for oid in ids:
            if oid not in obj_id_to_label:
                obj_id_to_label[oid] = label

    return obj_id_to_label


def _score_tuple(info: dict):
    """
    Higher is better. Used to resolve fusion collisions within the same frame.
    Priority:
      1) quality_score
      2) tracker_score
      3) static_score
      4) mask_area
    """
    q = float(info.get("quality_score", -1.0))
    ts = float(info.get("tracker_score", 0.0))
    ss = float(info.get("static_score", 0.0))
    area = int(info.get("mask_area", 0))
    return (q, ts, ss, area)


def build_track_history(tracks_dict: dict):
    """
    Returns:
      history[obj_id_str] = list of (frame_idx, centroid_xy, label)
      plus quick stats: start, end, len
    """
    history = defaultdict(list)
    for frame_idx_str, frame_data in tracks_dict.items():
        f = int(frame_idx_str)
        for oid_str, info in frame_data.items():
            history[oid_str].append((f, info["centroid"], info["label"]))
    for oid_str in history:
        history[oid_str].sort(key=lambda x: x[0])
    return history


def compute_track_stats(history_list):
    frames = [x[0] for x in history_list]
    start_f = frames[0]
    end_f = frames[-1]
    n = len(frames)
    total_move = 0.0
    for i in range(1, n):
        c1 = history_list[i - 1][1]
        c2 = history_list[i][1]
        total_move += float(((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5)
    avg_move = total_move / (n - 1) if n > 1 else 0.0
    label = history_list[0][2] if n > 0 else "unknown"
    return {
        "start": int(start_f),
        "end": int(end_f),
        "len": int(n),
        "avg_move": float(avg_move),
        "label": str(label),
    }


def plan_deletions(track_history: dict, rm_min_len: int, rm_static_px: float):
    delete_set = set()
    for oid_str, hist in track_history.items():
        st = compute_track_stats(hist)
        if st["len"] < int(rm_min_len):
            delete_set.add(oid_str)
            continue
        if st["avg_move"] <= float(rm_static_px):
            delete_set.add(oid_str)
            continue
    return delete_set


def plan_fusions(track_history: dict, delete_set: set, max_gap: int, skip_unknown: bool):
    """
    Greedy fusions per label.
    If same label and 0 < next.start - prev.end < max_gap, fuse next into prev base.
    Returns fusion_map: old_id_str -> base_id_str
    """
    label_to_tracks = defaultdict(list)
    for oid_str, hist in track_history.items():
        if oid_str in delete_set:
            continue
        st = compute_track_stats(hist)
        if skip_unknown and st["label"] == "unknown":
            continue
        label_to_tracks[st["label"]].append(
            {"id": oid_str, "start": st["start"], "end": st["end"], "label": st["label"]}
        )

    fusion_map = {}
    for label, items in label_to_tracks.items():
        items.sort(key=lambda x: x["start"])
        if len(items) <= 1:
            continue

        base = items[0]
        for j in range(1, len(items)):
            cur = items[j]
            gap = int(cur["start"] - base["end"])
            if 0 < gap < int(max_gap):
                fusion_map[cur["id"]] = base["id"]
                base = {"id": base["id"], "start": base["start"], "end": max(base["end"], cur["end"]), "label": label}
            else:
                base = cur

    return fusion_map


def apply_delete_and_fusion(tracks_dict: dict, delete_set: set, fusion_map: dict):
    """
    Applies delete and fusion to tracks_dict (JSON layer), preserving labels.
    Also resolves same-frame collisions by keeping the better entry and dropping the other.
    Returns:
      new_tracks
      dropped_old_mask_indices: set[int]
    """
    new_tracks = {}
    dropped_old_mask_indices = set()

    for frame_idx_str, frame_data in tracks_dict.items():
        out_frame = {}
        for oid_str, info in frame_data.items():
            if oid_str in delete_set:
                dropped_old_mask_indices.add(int(info["mask_idx"]))
                continue

            final_id = fusion_map.get(oid_str, oid_str)

            if final_id not in out_frame:
                out_frame[final_id] = info
                continue

            # Collision: same final_id in same frame.
            # Keep the better one, drop the other (and drop its NPZ mask).
            keep = out_frame[final_id]
            cand = info

            if _score_tuple(cand) > _score_tuple(keep):
                dropped_old_mask_indices.add(int(keep["mask_idx"]))
                out_frame[final_id] = cand
            else:
                dropped_old_mask_indices.add(int(cand["mask_idx"]))

        new_tracks[frame_idx_str] = out_frame

    return new_tracks, dropped_old_mask_indices


def rebuild_npz_and_reindex(
    tracks_dict: dict,
    all_masks_list,
    mask_frame_indices_list,
    mask_object_ids_list,
    dropped_old_mask_indices: set,
):
    """
    Rebuild NPZ arrays from the final tracks_dict.
    Ensures:
      - deleted masks are removed
      - fused ids are reflected in NPZ object_ids
      - mask_idx values in JSON are reindexed to [0..M-1] consistent with new NPZ
    """
    keep_old_indices = []
    # Deterministic order: by frame, then by object id numeric if possible, else string
    frame_keys = sorted(tracks_dict.keys(), key=lambda x: int(x))
    for fkey in frame_keys:
        frame_data = tracks_dict[fkey]
        obj_keys = sorted(frame_data.keys(), key=lambda s: int(s) if s.isdigit() else s)
        for oid_str in obj_keys:
            info = frame_data[oid_str]
            old_idx = int(info["mask_idx"])
            if old_idx in dropped_old_mask_indices:
                raise RuntimeError(
                    f"mask_idx {old_idx} is marked dropped but still referenced in tracks (frame {fkey}, id {oid_str})."
                )
            keep_old_indices.append(old_idx)

    if len(set(keep_old_indices)) != len(keep_old_indices):
        raise RuntimeError(
            "Duplicate mask_idx detected in final tracks. This indicates fusion collision was not resolved correctly."
        )

    old_to_new = {}
    for new_i, old_i in enumerate(keep_old_indices):
        old_to_new[int(old_i)] = int(new_i)

    new_masks = []
    new_frame_indices = []
    new_object_ids = []

    for old_i in keep_old_indices:
        new_masks.append(all_masks_list[old_i])
        new_frame_indices.append(mask_frame_indices_list[old_i])
        new_object_ids.append(mask_object_ids_list[old_i])

    # overwrite object_ids from tracks (single source of truth)
    old_idx_to_final_obj = {}
    for fkey in frame_keys:
        frame_data = tracks_dict[fkey]
        for oid_str, info in frame_data.items():
            old_idx = int(info["mask_idx"])
            old_idx_to_final_obj[old_idx] = int(oid_str)

    for old_i in keep_old_indices:
        final_oid = old_idx_to_final_obj.get(int(old_i), None)
        if final_oid is None:
            raise RuntimeError(f"Kept mask_idx {old_i} not found in tracks during NPZ rebuild.")
        new_object_ids[old_to_new[int(old_i)]] = int(final_oid)

    # Update mask_idx in tracks to new indices
    for fkey in frame_keys:
        frame_data = tracks_dict[fkey]
        for oid_str, info in frame_data.items():
            old_idx = int(info["mask_idx"])
            info["mask_idx"] = int(old_to_new[old_idx])

    return tracks_dict, new_masks, new_frame_indices, new_object_ids


# -------------------------
# Load model + processor
# -------------------------
accelerator = Accelerator()
device = accelerator.device

print("Loading model...")
model = Sam3VideoModel.from_pretrained(args.hf_local_model, local_files_only=True).to(device, dtype=torch_dtype)
processor = Sam3VideoProcessor.from_pretrained(args.hf_local_model, local_files_only=True)

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
# Tracking state (external gating)
# -------------------------
state = {}

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

for frame_idx in range(num_track_frames):
    model_outputs = model(inference_session=session, frame_idx=int(frame_idx), reverse=False)

    # PP FIRST
    pp = processor.postprocess_outputs(session, model_outputs)

    obj_ids = pp["object_ids"].tolist()
    masks = pp["masks"]  # Tensor[N, H, W] bool
    prompt_to_obj_ids = pp.get("prompt_to_obj_ids", {})

    obj_id_to_label = build_obj_id_to_label_from_pp(prompt_to_obj_ids, args.prompts)

    if frame_idx < args.debug_first_frames:
        print(f"[debug] frame={frame_idx} prompt_to_obj_ids={prompt_to_obj_ids}")
        print(f"[debug] frame={frame_idx} obj_ids={obj_ids}")
        # extra debug: show canonical keys side-by-side (first few frames only)
        if prompt_to_obj_ids:
            pp_keys = list(prompt_to_obj_ids.keys())
            print(f"[debug] frame={frame_idx} pp_prompt_keys={pp_keys}")
            print(f"[debug] frame={frame_idx} pp_prompt_keys_canon={[_canon_prompt(k) for k in pp_keys]}")
        print(f"[debug] frame={frame_idx} args_prompts={list(args.prompts)}")
        print(f"[debug] frame={frame_idx} args_prompts_canon={[_canon_prompt(p) for p in args.prompts]}")

    obj_id_to_static_score = dict(model_outputs.obj_id_to_score) if model_outputs.obj_id_to_score is not None else {}
    obj_id_to_tracker_score = (
        dict(model_outputs.obj_id_to_tracker_score) if model_outputs.obj_id_to_tracker_score is not None else {}
    )

    removed = set(model_outputs.removed_obj_ids) if model_outputs.removed_obj_ids is not None else set()
    suppressed = set(model_outputs.suppressed_obj_ids) if model_outputs.suppressed_obj_ids is not None else set()

    frame_data = {}
    hard_remove_obj_ids = []

    for i, obj_id in enumerate(obj_ids):
        if obj_id in removed:
            continue
        if obj_id in suppressed:
            continue

        mask = masks[i]

        area = int(mask.sum().item())
        if area < args.mask_area_min:
            continue

        centroid = mask_centroid(mask)
        if centroid is None:
            continue

        box = mask_box_xyxy(mask)
        if box is None:
            continue

        static_score = float(obj_id_to_static_score.get(obj_id, 0.0))
        tracker_score = float(obj_id_to_tracker_score.get(obj_id, 0.0))

        if tracker_score < args.tracker_score_min:
            continue
        if args.static_score_min > 0.0 and static_score < args.static_score_min:
            continue

        prev = state.get(obj_id)
        if args.max_jump_px > 0.0 and prev is not None:
            dist = l2(centroid, prev["last_centroid"])
            if dist > float(args.max_jump_px):
                prev["lost_count"] += 1
                if args.max_lost <= 0 or prev["lost_count"] > int(args.max_lost):
                    hard_remove_obj_ids.append(int(obj_id))
                continue

        if prev is None:
            state[obj_id] = {"last_centroid": centroid, "lost_count": 0, "last_velocity": (0.0, 0.0)}
            prev = state[obj_id]
        else:
            prev["lost_count"] = 0
            pc = prev["last_centroid"]

            if args.ema_alpha < 1.0:
                a = float(args.ema_alpha)
                centroid = (a * centroid[0] + (1.0 - a) * pc[0], a * centroid[1] + (1.0 - a) * pc[1])

            prev["last_velocity"] = (centroid[0] - pc[0], centroid[1] - pc[1])
            prev["last_centroid"] = centroid

        quality_score = compute_quality_score(static_score, tracker_score, args.quality_score_mode)
        label = obj_id_to_label.get(obj_id, "unknown")

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

        all_masks.append(mask.detach().to("cpu").numpy().astype(np.bool_))
        mask_frame_indices.append(int(frame_idx))
        mask_object_ids.append(int(obj_id))
        mask_counter += 1

    if hard_remove_obj_ids:
        for oid in hard_remove_obj_ids:
            try:
                session.remove_object(int(oid))
            except Exception as e:
                raise RuntimeError(f"session.remove_object({oid}) failed at frame {frame_idx}: {e}") from e
            state.pop(int(oid), None)

    tracks[str(frame_idx)] = frame_data

    if args.print_every > 0 and (frame_idx % args.print_every == 0):
        print("frame", frame_idx, "kept_masks", mask_counter, "kept_objs_this_frame", len(frame_data))

t1 = time.time()
print("Total kept masks before post-processing:", len(all_masks))
print("Time (s):", round(t1 - t0, 2))


# -------------------------
# Post-processing: delete and fuse (JSON + NPZ consistent)
# -------------------------
if args.post_process_rm or args.post_process_fusion:
    print("Post-processing tracks...")
    track_history = build_track_history(tracks)

    delete_set = set()
    if args.post_process_rm:
        delete_set = plan_deletions(track_history, args.rm_min_len, args.rm_static_px)
        if delete_set:
            print("Delete tracks:", len(delete_set))

    fusion_map = {}
    if args.post_process_fusion:
        fusion_map = plan_fusions(track_history, delete_set, args.fusion_max_gap, args.fusion_skip_unknown)
        if fusion_map:
            print("Fuse tracks:", len(fusion_map))

    if delete_set or fusion_map:
        tracks, dropped_old_mask_indices = apply_delete_and_fusion(tracks, delete_set, fusion_map)

        tracks, all_masks, mask_frame_indices, mask_object_ids = rebuild_npz_and_reindex(
            tracks,
            all_masks,
            mask_frame_indices,
            mask_object_ids,
            dropped_old_mask_indices,
        )

    print("Total kept masks after post-processing:", len(all_masks))


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
    "max_lost": int(args.max_lost),
    "ema_alpha": float(args.ema_alpha),
    "quality_score_mode": str(args.quality_score_mode),
    "processing_device": args.processing_device,
    "video_storage_device": args.video_storage_device,
    "plan": "A_hard_remove_on_reject",
    "post_process_rm": bool(args.post_process_rm),
    "post_process_fusion": bool(args.post_process_fusion),
    "rm_min_len": int(args.rm_min_len),
    "rm_static_px": float(args.rm_static_px),
    "fusion_max_gap": int(args.fusion_max_gap),
    "fusion_skip_unknown": bool(args.fusion_skip_unknown),
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

    out = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

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
            cv2.putText(frame, txt, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("Saved:", out_mp4)
else:
    print("Done (no visualization)")