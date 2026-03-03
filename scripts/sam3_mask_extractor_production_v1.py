# =============================================================================
# sam3_mask_extractor_production_v1.py
#
# Stable production dataset pipeline (v1.0)
# - Canonical label system (no underscore/space risk)
# - Deterministic mask protocol (PP-first)
# - 3 postprocess stages: rm / fusion / predict
# - Hit score computation (stable)
# - Hard reset per video (gc + empty_cache + model.reset_state)
# =============================================================================

import os
import json
import argparse
import time
import gc
import platform
from collections import defaultdict

import numpy as np
import torch
from accelerate import Accelerator
from transformers.video_utils import load_video
from transformers import Sam3VideoModel, Sam3VideoProcessor


# =============================================================================
# Label canonicalization (NO underscore risk)
# =============================================================================
def canonical_label(s: str) -> str:
    """
    统一 label:
    - 全部小写
    - 去掉空格
    - 去掉下划线
    例如 "Tennis_Ball" / "tennis ball" / "tennis_ball" -> "tennisball"
    """
    return str(s).strip().lower().replace(" ", "").replace("_", "")


# =============================================================================
# Args (每个参数详细中文说明)
# =============================================================================
parser = argparse.ArgumentParser(
    description="SAM3 dataset extraction pipeline (production v1.0)"
)

# -----------------------------
# IO
# -----------------------------
parser.add_argument(
    "--hf_local_model",
    type=str,
    required=False,
    default="/nfs/hpc/share/zhanhaoc/hpe/tempopeak/models/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7",
    help=(
        "本地 HuggingFace SAM3 模型 snapshot 路径. "
        "必须是目录, 且已下载到本机/HPC (例如 .../models--facebook--sam3/snapshots/<hash>)."
    ),
)
parser.add_argument(
    "--video_dir",
    type=str,
    default="/nfs/hpc/share/zhanhaoc/hpe/tempopeak/datasets/serve",
    required=False,
    help=(
        "待处理视频目录. 目录内所有 .mp4 会被逐个处理. "
        "建议只放同一数据规格的视频, 便于产出一致的 json/npz."
    ),
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="/nfs/hpc/share/zhanhaoc/hpe/tempopeak/outputs/sam3_prod_v1/serve",
    required=False,
    help=(
        "输出目录. 每个视频会生成: <name>.json, <name>.npz, "
        "如果开启 --vis 则额外生成 <name>_vis.mp4."
    ),
)
parser.add_argument(
    "--vis",
    action="store_true",
    help="是否输出可视化 mp4. 仅用于调试, 会显著增加运行时间与 IO.",
)

# -----------------------------
# Prompt
# -----------------------------
parser.add_argument(
    "--prompts",
    nargs="+",
    default=["ball", "racket"],
    help=(
        "初始化 session 的文本 prompts. "
        "示例: --prompts ball racket. "
        "注意: 后续 label 会做 canonical 化, 不会保留下划线/空格."
    ),
)

# -----------------------------
# Precision / device
# -----------------------------
parser.add_argument(
    "--dtype",
    type=str,
    choices=["bf16", "fp16", "fp32"],
    default="bf16",
    help=(
        "推理精度. H100 推荐 bf16. "
        "fp16 可能更快但在个别算子上更不稳. fp32 最稳但慢."
    ),
)
parser.add_argument(
    "--processing_device",
    type=str,
    default="cpu",
    help=(
        "processor.postprocess_outputs 的计算设备. "
        "推荐 cpu (更稳, 且避免显存波动)."
    ),
)
parser.add_argument(
    "--video_storage_device",
    type=str,
    default="cpu",
    help=(
        "视频帧缓存设备. 推荐 cpu (避免视频长导致显存被占)."
    ),
)
parser.add_argument(
    "--max_frames",
    type=int,
    default=-1,
    help=(
        "最多处理多少帧. <=0 表示处理完整视频. "
        "用于快速 smoke test 或调参."
    ),
)

# -----------------------------
# Filtering thresholds
# -----------------------------
parser.add_argument(
    "--tracker_score_min",
    type=float,
    default=0.10,
    help=(
        "最小 tracker_score 阈值. "
        "当 obj 的 tracker_score < 该值时, 该 obj 会在该帧被丢弃(不写入 json/npz)."
    ),
)
parser.add_argument(
    "--static_score_min",
    type=float,
    default=-1.0,
    help=(
        "最小 static_score 阈值. <=0 表示禁用. "
        "否则当 obj 的 static_score < 该值时, 该 obj 会被丢弃."
    ),
)
parser.add_argument(
    "--mask_area_min",
    type=int,
    default=1,
    help=(
        "最小 mask 面积阈值(像素数). "
        "面积过小通常是噪声/碎片, 建议 >=1 或更大."
    ),
)

# -----------------------------
# Motion gating (可选, reject 可触发 hard remove)
# -----------------------------
parser.add_argument(
    "--max_jump_px",
    type=float,
    default=-1.0,
    help=(
        "最大允许的 centroid 跳变像素距离. <=0 表示禁用. "
        "启用后: 若当前帧 centroid 与上一帧 centroid 距离超过该阈值, 该帧视为 reject."
    ),
)
parser.add_argument(
    "--max_lost",
    type=int,
    default=0,
    help=(
        "允许连续 reject 的最大次数. "
        "当 reject 次数 > max_lost 时, 触发 session.remove_object(obj_id) 做硬删除(清 memory). "
        "注意: max_jump_px 必须启用才会触发 lost 逻辑."
    ),
)
parser.add_argument(
    "--ema_alpha",
    type=float,
    default=1.0,
    help=(
        "centroid EMA 平滑系数. 1.0 表示不平滑. "
        "典型: 0.5~0.8. 值越小越平滑但滞后更大."
    ),
)

# -----------------------------
# Postprocess switches (3 stages)
# -----------------------------
parser.add_argument(
    "--post_process_rm",
    action="store_true",
    help=(
        "后处理阶段1: 删除轨迹. "
        "删除过短轨迹(len < rm_min_len) 或近似静止轨迹(avg_move <= rm_static_px)."
    ),
)
parser.add_argument(
    "--rm_min_len",
    type=int,
    default=15,
    help="rm 阶段参数: 最小轨迹长度阈值. 小于该长度的轨迹会被删除.",
)
parser.add_argument(
    "--rm_static_px",
    type=float,
    default=5.0,
    help="rm 阶段参数: 平均位移阈值(px). <= 该值认为静止轨迹并删除.",
)

parser.add_argument(
    "--post_process_fusion",
    action="store_true",
    help=(
        "后处理阶段2: 融合同 label 的轨迹. "
        "如果两个轨迹间断帧数 gap < fusion_max_gap, 则把后者 fusion 到前者(同一 id)."
    ),
)
parser.add_argument(
    "--fusion_max_gap",
    type=int,
    default=5,
    help="fusion 阶段参数: 最大允许 gap(帧). gap < 该值才 fusion.",
)
parser.add_argument(
    "--fusion_skip_unknown",
    action="store_true",
    help="fusion 阶段参数: 如果启用, label=unknown 的轨迹不参与 fusion.",
)

parser.add_argument(
    "--post_process_predict",
    action="store_true",
    help=(
        "后处理阶段3: gap prediction. "
        "对同一 obj_id 在两次出现之间的短 gap 进行插值补帧(含 mask 形状插值)."
    ),
)
parser.add_argument(
    "--predict_max_gap",
    type=int,
    default=15,
    help="predict 阶段参数: 仅补 <= 该长度的 gap(帧).",
)

# -----------------------------
# Debug
# -----------------------------
parser.add_argument(
    "--print_every",
    type=int,
    default=30,
    help="每隔多少帧打印一次进度. <=0 表示不打印.",
)
parser.add_argument(
    "--debug_first_frames",
    type=int,
    default=1,
    help="前多少帧打印 prompt_to_obj_ids / label 分布等 debug 信息.",
)

args = parser.parse_args()

# =============================================================================
# Fail-fast checks
# =============================================================================
if not os.path.isdir(args.hf_local_model):
    raise FileNotFoundError(f"hf_local_model not found: {args.hf_local_model}")
if not os.path.isdir(args.video_dir):
    raise FileNotFoundError(f"video_dir not found: {args.video_dir}")
os.makedirs(args.out_dir, exist_ok=True)

# =============================================================================
# Dtype
# =============================================================================
if args.dtype == "bf16":
    torch_dtype = torch.bfloat16
elif args.dtype == "fp16":
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

# =============================================================================
# Helpers
# =============================================================================
def _to_int_list(x):
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        return [int(v) for v in x.detach().cpu().tolist()]
    return [int(v) for v in list(x)]

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

def build_obj_id_to_label_from_pp(prompt_to_obj_ids: dict, prompts_order: list):
    """
    使用 PP 的 prompt_to_obj_ids 建立 obj_id -> label 映射.
    label 会 canonical 化, 从源头规避下划线/空格风险.
    """
    prompt_to_obj_ids = prompt_to_obj_ids or {}

    canon_user_prompts = [canonical_label(p) for p in prompts_order]
    canon_set = set(canon_user_prompts)

    obj_id_to_label = {}

    for raw_prompt, ids in prompt_to_obj_ids.items():
        cp = canonical_label(raw_prompt)
        if cp not in canon_set:
            continue
        for oid in _to_int_list(ids):
            if int(oid) not in obj_id_to_label:
                obj_id_to_label[int(oid)] = cp

    return obj_id_to_label

def build_track_history(tracks_dict: dict):
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
    return {"start": int(start_f), "end": int(end_f), "len": int(n), "avg_move": float(avg_move), "label": str(label)}

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
    label_to_tracks = defaultdict(list)
    for oid_str, hist in track_history.items():
        if oid_str in delete_set:
            continue
        st = compute_track_stats(hist)
        if skip_unknown and st["label"] == "unknown":
            continue
        label_to_tracks[st["label"]].append({"id": oid_str, "start": st["start"], "end": st["end"], "label": st["label"]})

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

def _score_tuple(info: dict):
    ts = float(info.get("tracker_score", 0.0))
    ss = float(info.get("static_score", 0.0))
    area = int(info.get("mask_area", 0))
    return (ts, ss, area)

def apply_delete_and_fusion(tracks_dict: dict, delete_set: set, fusion_map: dict):
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

            keep = out_frame[final_id]
            cand = info
            if _score_tuple(cand) > _score_tuple(keep):
                dropped_old_mask_indices.add(int(keep["mask_idx"]))
                out_frame[final_id] = cand
            else:
                dropped_old_mask_indices.add(int(cand["mask_idx"]))

        new_tracks[frame_idx_str] = out_frame

    return new_tracks, dropped_old_mask_indices

def rebuild_npz_and_reindex(tracks_dict: dict, all_masks_list, mask_frame_indices_list, mask_object_ids_list, dropped_old_mask_indices: set):
    keep_old_indices = []
    frame_keys = sorted(tracks_dict.keys(), key=lambda x: int(x))
    for fkey in frame_keys:
        frame_data = tracks_dict[fkey]
        for oid_str, info in frame_data.items():
            old_idx = int(info["mask_idx"])
            if old_idx in dropped_old_mask_indices:
                raise RuntimeError(f"mask_idx {old_idx} dropped but still referenced (frame={fkey}, id={oid_str}).")
            keep_old_indices.append(old_idx)

    if len(set(keep_old_indices)) != len(keep_old_indices):
        raise RuntimeError("Duplicate mask_idx detected in final tracks.")

    old_to_new = {int(old_i): int(new_i) for new_i, old_i in enumerate(keep_old_indices)}

    new_masks = [all_masks_list[old_i] for old_i in keep_old_indices]
    new_frame_indices = [mask_frame_indices_list[old_i] for old_i in keep_old_indices]
    new_object_ids = [mask_object_ids_list[old_i] for old_i in keep_old_indices]

    old_idx_to_final_obj = {}
    for fkey in frame_keys:
        frame_data = tracks_dict[fkey]
        for oid_str, info in frame_data.items():
            old_idx_to_final_obj[int(info["mask_idx"])] = int(oid_str)

    for old_i in keep_old_indices:
        final_oid = old_idx_to_final_obj.get(int(old_i), None)
        if final_oid is None:
            raise RuntimeError(f"Kept mask_idx {old_i} not found in tracks during NPZ rebuild.")
        new_object_ids[old_to_new[int(old_i)]] = int(final_oid)

    for fkey in frame_keys:
        frame_data = tracks_dict[fkey]
        for oid_str, info in frame_data.items():
            info["mask_idx"] = int(old_to_new[int(info["mask_idx"])])

    return tracks_dict, new_masks, new_frame_indices, new_object_ids

def compute_hit_scores(tracks: dict):
    """
    稳定 hit_score:
    - 只用 centroid 距离 (ball centroid 到 racket centroid)
    - 找到 raw 分数最高的 peak_frame
    - 用时间高斯在 peak 周围扩散
    """
    ball_label = canonical_label("ball")
    racket_label = canonical_label("racket")

    peak_frame = None
    peak_raw = -1.0

    frame_keys = sorted(tracks.keys(), key=lambda s: int(s))
    per_frame_raw = {}

    for fk in frame_keys:
        frame = tracks.get(fk, {})
        if not frame:
            continue

        best_ball = None
        best_ball_ts = -1.0
        best_racket = None
        best_racket_ts = -1.0

        for oid_str, info in frame.items():
            lab = str(info.get("label", "unknown"))
            ts = float(info.get("tracker_score", 0.0))
            if lab == ball_label and ts > best_ball_ts:
                best_ball_ts = ts
                best_ball = info
            elif lab == racket_label and ts > best_racket_ts:
                best_racket_ts = ts
                best_racket = info

        if best_ball is None or best_racket is None:
            continue

        bx, by = float(best_ball["centroid"][0]), float(best_ball["centroid"][1])
        rx, ry = float(best_racket["centroid"][0]), float(best_racket["centroid"][1])

        dist = float(((bx - rx) ** 2 + (by - ry) ** 2) ** 0.5)
        sigma = 20.0
        raw = float(np.exp(-(dist * dist) / (2.0 * sigma * sigma)))

        per_frame_raw[int(fk)] = raw
        if raw > peak_raw:
            peak_raw = raw
            peak_frame = int(fk)

    # init
    for fk in frame_keys:
        frame = tracks.get(fk, {})
        for oid_str, info in frame.items():
            if str(info.get("label", "unknown")) == ball_label:
                info["hit_score"] = 0.0

    hit_dbg = {"status": "no_peak"}
    if peak_frame is None:
        return tracks, hit_dbg

    # temporal gaussian
    sigma_t = 3.0
    peak_cap = 0.92
    hit_dbg = {"status": "ok", "peak_frame": int(peak_frame), "peak_raw": float(peak_raw)}

    for t, raw in per_frame_raw.items():
        dt = int(t - peak_frame)
        g = float(np.exp(-(dt * dt) / (2.0 * sigma_t * sigma_t)))
        s = float(min(peak_cap, peak_raw * g))

        fk = str(t)
        frame = tracks.get(fk, {})
        for oid_str, info in frame.items():
            if str(info.get("label", "unknown")) == ball_label:
                info["hit_score"] = float(s)

    return tracks, hit_dbg


# =============================================================================
# Load model + processor
# =============================================================================
accelerator = Accelerator()
device = accelerator.device

print("Loading model...")
model = Sam3VideoModel.from_pretrained(args.hf_local_model, local_files_only=True).to(device, dtype=torch_dtype)
processor = Sam3VideoProcessor.from_pretrained(args.hf_local_model, local_files_only=True)

# =============================================================================
# Main loop
# =============================================================================
import glob
mp4_files = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
print(f"Found {len(mp4_files)} videos in {args.video_dir}")

for video_path in mp4_files:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_json = os.path.join(args.out_dir, f"{video_name}.json")
    out_npz = os.path.join(args.out_dir, f"{video_name}.npz")
    out_mp4 = os.path.join(args.out_dir, f"{video_name}_vis.mp4")

    print("\n=========================================")
    print(f"Processing {video_name}...")
    print("=========================================")

    # hard reset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # load video
    video_frames, _ = load_video(video_path)
    num_frames = len(video_frames)
    if args.max_frames > 0:
        num_track_frames = min(num_frames, int(args.max_frames))
    else:
        num_track_frames = num_frames
    print("Total frames:", num_frames, "Tracking frames:", num_track_frames)

    # reset model state per video (important)
    if hasattr(model, "reset_state"):
        model.reset_state()

    # init session
    session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device=args.processing_device,
        video_storage_device=args.video_storage_device,
        dtype=torch_dtype,
    )
    for p in args.prompts:
        session = processor.add_text_prompt(inference_session=session, text=p)

    # storage
    state = {}  # per obj motion state
    tracks = {}
    all_masks = []
    mask_frame_indices = []
    mask_object_ids = []
    mask_counter = 0

    t0 = time.time()
    print("Tracking...")

    for frame_idx in range(num_track_frames):
        with torch.no_grad():
            model_outputs = model(
                inference_session=session,
                frame_idx=int(frame_idx),
                reverse=False,
            )

        pp = processor.postprocess_outputs(session, model_outputs)

        obj_ids = _to_int_list(pp.get("object_ids", []))
        masks = pp.get("masks", None)
        prompt_to_obj_ids = pp.get("prompt_to_obj_ids", {}) or {}

        if masks is None:
            tracks[str(frame_idx)] = {}
            continue

        obj_id_to_label = build_obj_id_to_label_from_pp(prompt_to_obj_ids, args.prompts)

        if frame_idx < int(args.debug_first_frames):
            dist = defaultdict(int)
            for oid in obj_ids:
                dist[obj_id_to_label.get(int(oid), "unknown")] += 1
            print(f"[debug] frame={frame_idx} prompt_to_obj_ids={prompt_to_obj_ids}")
            print(f"[debug] frame={frame_idx} label_dist={dict(dist)}")

        # raw scores (optional)
        obj_id_to_static_score = dict(model_outputs.obj_id_to_score) if getattr(model_outputs, "obj_id_to_score", None) is not None else {}
        obj_id_to_tracker_score = dict(model_outputs.obj_id_to_tracker_score) if getattr(model_outputs, "obj_id_to_tracker_score", None) is not None else {}

        removed = set(_to_int_list(getattr(model_outputs, "removed_obj_ids", None)))
        suppressed = set(_to_int_list(getattr(model_outputs, "suppressed_obj_ids", None)))

        frame_data = {}
        hard_remove_obj_ids = []

        for i, obj_id in enumerate(obj_ids):
            obj_id = int(obj_id)
            if obj_id in removed or obj_id in suppressed:
                continue

            mask = masks[i]
            area = int(mask.sum().item())
            if area < int(args.mask_area_min):
                continue

            centroid = mask_centroid(mask)
            if centroid is None:
                continue

            box = mask_box_xyxy(mask)
            if box is None:
                continue

            static_score = float(obj_id_to_static_score.get(obj_id, 0.0))
            tracker_score = float(obj_id_to_tracker_score.get(obj_id, 0.0))

            if tracker_score < float(args.tracker_score_min):
                continue
            if float(args.static_score_min) > 0.0 and static_score < float(args.static_score_min):
                continue

            # motion gating (optional)
            prev = state.get(obj_id)
            if float(args.max_jump_px) > 0.0 and prev is not None:
                dist = l2(centroid, prev["last_centroid"])
                if dist > float(args.max_jump_px):
                    prev["lost_count"] += 1
                    if int(args.max_lost) <= 0 or prev["lost_count"] > int(args.max_lost):
                        hard_remove_obj_ids.append(int(obj_id))
                    continue

            if prev is None:
                state[obj_id] = {"last_centroid": centroid, "lost_count": 0}
                prev = state[obj_id]
            else:
                prev["lost_count"] = 0
                pc = prev["last_centroid"]

                if float(args.ema_alpha) < 1.0:
                    a = float(args.ema_alpha)
                    centroid = (a * centroid[0] + (1.0 - a) * pc[0], a * centroid[1] + (1.0 - a) * pc[1])

                prev["last_centroid"] = centroid

            label = obj_id_to_label.get(int(obj_id), "unknown")
            # 强制 canonical 化 (规避 underscore 风险)
            label = canonical_label(label) if label != "unknown" else "unknown"

            frame_data[str(obj_id)] = {
                "label": label,
                "tracker_score": round(float(tracker_score), 6),
                "static_score": round(float(static_score), 6),
                "centroid": [round(float(centroid[0]), 3), round(float(centroid[1]), 3)],
                "box_xyxy": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                "mask_idx": int(mask_counter),
                "mask_area": int(area),
            }

            all_masks.append(mask.detach().to("cpu").numpy().astype(np.bool_))
            mask_frame_indices.append(int(frame_idx))
            mask_object_ids.append(int(obj_id))
            mask_counter += 1

        # hard remove objects (clear session memory)
        if hard_remove_obj_ids:
            for oid in hard_remove_obj_ids:
                session.remove_object(int(oid))
                state.pop(int(oid), None)

        tracks[str(frame_idx)] = frame_data

        if int(args.print_every) > 0 and (frame_idx % int(args.print_every) == 0):
            print("frame", frame_idx, "kept_masks", mask_counter, "kept_objs_this_frame", len(frame_data))

        # free per-frame refs
        model_outputs = None
        pp = None

    print("Tracking done. Total kept masks:", len(all_masks))
    print("Time (s):", round(time.time() - t0, 2))

# =============================================================================
# Postprocess (FORCED APPLY VERSION)
# =============================================================================

    print("\n===== POST PROCESS START =====")
    orig_track_count = sum(len(v) for v in tracks.values())
    print("original total track entries:", orig_track_count)

    dropped_old_mask_indices = set()

    if args.post_process_rm or args.post_process_fusion:

        print("Building track history...")
        track_history = build_track_history(tracks)
        print("total unique track ids:", len(track_history))

        delete_set = set()
        if args.post_process_rm:
            print("Running rm stage ...")
            delete_set = plan_deletions(
                track_history,
                int(args.rm_min_len),
                float(args.rm_static_px),
            )
            print("rm delete tracks:", len(delete_set))

        fusion_map = {}
        if args.post_process_fusion:
            print("Running fusion stage ...")
            fusion_map = plan_fusions(
                track_history,
                delete_set,
                int(args.fusion_max_gap),
                bool(args.fusion_skip_unknown),
            )
            print("fusion pairs:", len(fusion_map))

        # 🔥 强制执行 apply
        print("Applying delete/fusion ...")
        tracks, dropped_old_mask_indices = apply_delete_and_fusion(
            tracks,
            delete_set,
            fusion_map,
        )

        print("dropped mask count:", len(dropped_old_mask_indices))

        # 🔥 强制 rebuild
        print("Rebuilding NPZ ...")
        tracks, all_masks, mask_frame_indices, mask_object_ids = rebuild_npz_and_reindex(
            tracks,
            all_masks,
            mask_frame_indices,
            mask_object_ids,
            dropped_old_mask_indices,
        )

    else:
        print("Post-process disabled.")

    new_track_count = sum(len(v) for v in tracks.values())
    print("new total track entries:", new_track_count)
    print("POST PROCESS DELTA:", orig_track_count - new_track_count)
    print("===== POST PROCESS END =====\n")

    # =============================================================================
    # Hit score (always)
    # =============================================================================
    print("Computing hit_score ...")
    tracks, hit_dbg = compute_hit_scores(tracks)

    # =============================================================================
    # Save JSON + NPZ
    # =============================================================================
    meta = {
        "time_unix": float(time.time()),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "dtype": str(args.dtype),
        "video_name": video_name,
        "video_path": video_path,
        "num_frames_total": int(num_frames),
        "num_frames_tracked": int(num_track_frames),
        "hf_local_model": args.hf_local_model,
        "out_dir": args.out_dir,
        "prompts": [str(p) for p in args.prompts],
        "canonical_prompts": [canonical_label(p) for p in args.prompts],
        "tracker_score_min": float(args.tracker_score_min),
        "static_score_min": float(args.static_score_min),
        "mask_area_min": int(args.mask_area_min),
        "max_jump_px": float(args.max_jump_px),
        "max_lost": int(args.max_lost),
        "ema_alpha": float(args.ema_alpha),
        "post_process_rm": bool(args.post_process_rm),
        "post_process_fusion": bool(args.post_process_fusion),
        "post_process_predict": bool(args.post_process_predict),
        "rm_min_len": int(args.rm_min_len),
        "rm_static_px": float(args.rm_static_px),
        "fusion_max_gap": int(args.fusion_max_gap),
        "fusion_skip_unknown": bool(args.fusion_skip_unknown),
        "predict_max_gap": int(args.predict_max_gap),
        "hit_debug": hit_dbg,
    }

    json_payload = {"_meta": meta}
    json_payload.update(tracks)

    with open(out_json, "w") as f:
        json.dump(json_payload, f, indent=2)
    print("Saved:", out_json)

    np.savez_compressed(
        out_npz,
        masks=np.array(all_masks, dtype=np.bool_),
        frame_indices=np.array(mask_frame_indices, dtype=np.int32),
        object_ids=np.array(mask_object_ids, dtype=np.int32),
    )
    print("Saved:", out_npz)

    # =============================================================================
    # Vis (optional): v1.0 不内置, 避免引入 cv2 依赖与额外 IO
    # =============================================================================
    if args.vis:
        print("Warning: --vis is enabled, but v1.0 does not render mp4 (deliberately).")
        print("If you need visualization, I will add a controlled cv2 renderer in v1.1.")

    # cleanup per video
    del session
    del state
    del tracks
    del all_masks
    del mask_frame_indices
    del mask_object_ids
    del video_frames
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("All done.")