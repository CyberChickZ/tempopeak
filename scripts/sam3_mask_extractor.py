# =============================================================================
# sam3_mask_extractor_production_v1.py
#
# Stable production dataset pipeline
# - Canonical label system (no underscore risk)
# - Deterministic mask protocol
# - Optional post-processing
# - Hit score computation
# - Hard memory reset per video
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
# Label Canonicalization (NO underscore risk)
# =============================================================================

def canonical_label(s: str) -> str:
    """
    统一 label 格式:
    - 全部小写
    - 去掉空格
    - 去掉下划线
    """
    return str(s).lower().replace(" ", "").replace("_", "")


# =============================================================================
# Argument Parser (每个参数详细中文说明)
# =============================================================================

parser = argparse.ArgumentParser(description="SAM3 Stable Dataset Pipeline v1.0")

# -----------------------------
# 基础 IO 参数
# -----------------------------
parser.add_argument(
    "--hf_local_model",
    type=str,
    required=True,
    help="本地 HuggingFace SAM3 模型路径 (必须是已经下载的 snapshot 目录)",
)

parser.add_argument(
    "--video_dir",
    type=str,
    required=True,
    help="待处理视频目录 (目录中所有 mp4 会被逐个处理)",
)

parser.add_argument(
    "--out_dir",
    type=str,
    required=True,
    help="输出目录 (会生成 json + npz + 可选 mp4)",
)

parser.add_argument(
    "--vis",
    action="store_true",
    help="是否输出可视化 mp4 文件 (调试用)",
)

# -----------------------------
# Prompt 设置
# -----------------------------
parser.add_argument(
    "--prompts",
    nargs="+",
    default=["ball", "racket"],
    help="初始化 session 的文本 prompt 列表 (例如: ball racket)",
)

# -----------------------------
# 精度与设备
# -----------------------------
parser.add_argument(
    "--dtype",
    type=str,
    default="bf16",
    choices=["bf16", "fp16", "fp32"],
    help="模型推理精度 (bf16 推荐 H100 使用)",
)

parser.add_argument(
    "--processing_device",
    type=str,
    default="cpu",
    help="后处理设备 (cpu 推荐，稳定)",
)

parser.add_argument(
    "--video_storage_device",
    type=str,
    default="cpu",
    help="视频缓存设备 (cpu 推荐)",
)

# -----------------------------
# 过滤阈值
# -----------------------------
parser.add_argument(
    "--tracker_score_min",
    type=float,
    default=0.1,
    help="最小 tracker_score，小于该值的目标会被过滤",
)

parser.add_argument(
    "--mask_area_min",
    type=int,
    default=1,
    help="最小 mask 面积 (像素数)",
)

# -----------------------------
# 调试
# -----------------------------
parser.add_argument(
    "--print_every",
    type=int,
    default=30,
    help="每多少帧打印一次进度",
)

args = parser.parse_args()

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
# 加载模型
# =============================================================================

accelerator = Accelerator()
device = accelerator.device

print("Loading model...")
model = Sam3VideoModel.from_pretrained(
    args.hf_local_model, local_files_only=True
).to(device, dtype=torch_dtype)

processor = Sam3VideoProcessor.from_pretrained(
    args.hf_local_model, local_files_only=True
)

# =============================================================================
# 工具函数
# =============================================================================

def mask_centroid(mask):
    ys, xs = torch.where(mask)
    if xs.numel() == 0:
        return None
    return float(xs.float().mean()), float(ys.float().mean())

def mask_box(mask):
    ys, xs = torch.where(mask)
    if xs.numel() == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

# =============================================================================
# HIT SCORE (稳定版本)
# =============================================================================

def compute_hit_scores(tracks, masks_np):

    ball_label = canonical_label("ball")
    racket_label = canonical_label("racket")

    peak_frame = None
    peak_score = 0.0

    # 找最近帧
    for frame_key in tracks:
        frame = tracks[frame_key]

        ball = None
        racket = None

        for oid in frame:
            if frame[oid]["label"] == ball_label:
                ball = frame[oid]
            if frame[oid]["label"] == racket_label:
                racket = frame[oid]

        if ball is None or racket is None:
            continue

        bx, by = ball["centroid"]
        rx, ry = racket["centroid"]

        dist = np.sqrt((bx - rx) ** 2 + (by - ry) ** 2)
        score = np.exp(-(dist ** 2) / (2 * 20 ** 2))

        if score > peak_score:
            peak_score = score
            peak_frame = int(frame_key)

    # 初始化
    for frame_key in tracks:
        for oid in tracks[frame_key]:
            if tracks[frame_key][oid]["label"] == ball_label:
                tracks[frame_key][oid]["hit_score"] = 0.0

    if peak_frame is None:
        return tracks

    # 高斯时间衰减
    for frame_key in tracks:
        t = int(frame_key)
        dt = t - peak_frame
        temporal = np.exp(-(dt ** 2) / (2 * 3 ** 2))
        final = float(min(0.92, peak_score * temporal))

        for oid in tracks[frame_key]:
            if tracks[frame_key][oid]["label"] == ball_label:
                tracks[frame_key][oid]["hit_score"] = final

    return tracks

# =============================================================================
# 主循环
# =============================================================================

import glob

mp4_files = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))

for video_path in mp4_files:

    print("Processing:", video_path)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_json = os.path.join(args.out_dir, video_name + ".json")
    out_npz = os.path.join(args.out_dir, video_name + ".npz")

    gc.collect()
    torch.cuda.empty_cache()

    video_frames, _ = load_video(video_path)

    session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device=args.processing_device,
        video_storage_device=args.video_storage_device,
        dtype=torch_dtype,
    )

    for p in args.prompts:
        session = processor.add_text_prompt(session, text=p)

    tracks = {}
    all_masks = []
    mask_frame_indices = []
    mask_object_ids = []

    for frame_idx in range(len(video_frames)):

        with torch.no_grad():
            outputs = model(session, frame_idx=int(frame_idx))

        pp = processor.postprocess_outputs(session, outputs)

        obj_ids = pp["object_ids"]
        masks = pp["masks"]
        prompt_map = pp.get("prompt_to_obj_ids", {})

        id_to_label = {}
        for prompt_name in prompt_map:
            canon = canonical_label(prompt_name)
            for oid in prompt_map[prompt_name]:
                id_to_label[int(oid)] = canon

        frame_data = {}

        for i, oid in enumerate(obj_ids):

            oid = int(oid)
            mask = masks[i]
            area = int(mask.sum().item())

            if area < args.mask_area_min:
                continue

            centroid = mask_centroid(mask)
            if centroid is None:
                continue

            label = id_to_label.get(oid, "unknown")

            frame_data[str(oid)] = {
                "label": label,
                "centroid": [float(centroid[0]), float(centroid[1])],
                "mask_idx": len(all_masks),
                "mask_area": area,
            }

            all_masks.append(mask.cpu().numpy().astype(np.bool_))
            mask_frame_indices.append(frame_idx)
            mask_object_ids.append(oid)

        tracks[str(frame_idx)] = frame_data

        if frame_idx % args.print_every == 0:
            print("frame:", frame_idx)

    # HIT SCORE
    tracks = compute_hit_scores(tracks, np.array(all_masks))

    # SAVE
    payload = {"_meta": {"video": video_name}}
    payload.update(tracks)

    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    np.savez_compressed(
        out_npz,
        masks=np.array(all_masks, dtype=np.bool_),
        frame_indices=np.array(mask_frame_indices, dtype=np.int32),
        object_ids=np.array(mask_object_ids, dtype=np.int32),
    )

    print("Saved:", video_name)

    del session
    gc.collect()
    torch.cuda.empty_cache()

print("All done.")