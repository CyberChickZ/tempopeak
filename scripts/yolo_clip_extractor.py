"""
yolo_clip_extractor.py — YOLO26 detection → hit event detection → clip cutting

Processes a folder of mp4 videos:
1. YOLO26 detects ball (class 32) and racket (class 38) per frame
2. Computes ball-racket distance, identifies hit events
3. Cuts clip mp4s around each event
4. Calls callback(type="clip", ...) for each clip (SSE integration)

Standalone usage:
    python scripts/yolo_clip_extractor.py --folder /path/to/videos

fps is validated to {15, 30, 60} only. Others are skipped.
"""

import os
import sys
import math
import glob
import argparse

import cv2

VALID_FPS = {15, 30, 60}
HIT_THRESH = 150
MIN_EVENT_GAP = 30
PAD_BEFORE = 15
PAD_AFTER = 15


def get_validated_fps(cap) -> int:
    raw = cap.get(cv2.CAP_PROP_FPS)
    rounded = round(raw)
    if rounded not in VALID_FPS:
        raise ValueError(f"Unsupported fps: {raw}, rounded={rounded}. Only {VALID_FPS} allowed.")
    return rounded


def box_centroid(box):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def process_single_video(video_path, video_name, tmp_dir, model, callback, conf=0.25):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        callback(type="skip", video_name=video_name, reason="cannot open video")
        return

    try:
        fps = get_validated_fps(cap)
    except ValueError as e:
        callback(type="skip", video_name=video_name, reason=str(e))
        cap.release()
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"[yolo] {video_name}: {total_frames} frames @ {fps} fps, {W}x{H}")

    results = model.predict(source=video_path, classes=[32, 38], conf=conf, stream=True, verbose=False)
    frame_distances = {}

    for frame_idx, r in enumerate(results):
        balls = [box for box in r.boxes if int(box.cls.item()) == 32]
        rackets = [box for box in r.boxes if int(box.cls.item()) == 38]
        if balls and rackets:
            min_dist = min(euclidean(box_centroid(b), box_centroid(rk)) for b in balls for rk in rackets)
            frame_distances[frame_idx] = min_dist
        if frame_idx % 60 == 0:
            callback(type="progress", video_name=video_name, frame=frame_idx, total=total_frames)

    print(f"[yolo] {video_name}: detection done, {len(frame_distances)} frames with ball+racket")

    hit_frames = sorted(f for f, d in frame_distances.items() if d < HIT_THRESH)
    if not hit_frames:
        print(f"[yolo] {video_name}: no hit events found")
        return

    events = []
    for f in hit_frames:
        if events and f - events[-1]["end"] < MIN_EVENT_GAP:
            events[-1]["end"] = f
            if frame_distances.get(f, 999) < frame_distances.get(events[-1]["peak"], 999):
                events[-1]["peak"] = f
        else:
            events.append({"start": f, "end": f, "peak": f})

    print(f"[yolo] {video_name}: {len(events)} hit events detected")

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for clip_idx, event in enumerate(events):
        clip_start = max(0, event["start"] - PAD_BEFORE)
        clip_end = min(total_frames - 1, event["end"] + PAD_AFTER)
        num_frames = clip_end - clip_start + 1
        clip_id = f"{video_name}_clip_{clip_idx:04d}"
        out_path = os.path.join(tmp_dir, f"{clip_id}.mp4")

        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        writer.release()

        print(f"[yolo] clip {clip_id}: frames {clip_start}-{clip_end} ({num_frames} frames)")
        callback(type="clip", clip_id=clip_id, path=out_path, num_frames=num_frames)

    cap.release()


def process_folder(folder, tmp_dir, callback, model_name="yolo26x.pt", conf=0.25):
    from ultralytics import YOLO

    mp4s = sorted(glob.glob(os.path.join(folder, "*.mp4")))
    if not mp4s:
        callback(type="skip", video_name="(none)", reason=f"No .mp4 files in {folder}")
        return

    print(f"[yolo] Loading model: {model_name}")
    model = YOLO(model_name)

    for mp4_path in mp4s:
        video_name = os.path.splitext(os.path.basename(mp4_path))[0]
        try:
            process_single_video(mp4_path, video_name, tmp_dir, model, callback, conf=conf)
        except Exception as e:
            print(f"[yolo] ERROR processing {video_name}: {e}")
            callback(type="skip", video_name=video_name, reason=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO26 clip extractor")
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--tmp_dir", type=str, default="/tmp/clips")
    parser.add_argument("--model", type=str, default="yolo26x.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)

    def cli_callback(**kwargs):
        t = kwargs.get("type", "")
        if t == "clip":
            print(f"  -> CLIP: {kwargs['clip_id']} ({kwargs['num_frames']} frames) -> {kwargs['path']}")
        elif t == "skip":
            print(f"  -> SKIP: {kwargs['video_name']}: {kwargs['reason']}")
        elif t == "progress":
            print(f"  -> {kwargs['video_name']}: frame {kwargs['frame']}/{kwargs['total']}")

    process_folder(args.folder, args.tmp_dir, cli_callback, model_name=args.model, conf=args.conf)
    print("Done.")
