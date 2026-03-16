"""
yolo_clip_extractor.py — YOLO26 detection → hit event detection → JPEG frame extraction

Processes a folder of mp4 videos:
1. YOLO26 detects ball (class 32) and racket (class 38) per frame
2. Computes ball-racket distance, identifies hit events
3. Extracts JPEG frames around each event to data/clips/{clip_id}/
4. Optionally overlays bounding boxes on extracted frames
5. Writes meta.json per clip (includes peak_frame, export_pad)
6. Calls callback(type="clip", ...) for each clip

Standalone usage:
    python scripts/yolo_clip_extractor.py --folder /path/to/videos

fps is validated to {15, 30, 60} only. Others are skipped.
"""

import os
import sys
import math
import glob
import json
import argparse
from typing import Callable

import cv2

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLIPS_DIR = os.path.join(_PROJECT_ROOT, "data", "clips")

VALID_FPS = {15, 30, 60}
MODEL_NAME = "yolo26x.pt"


def get_validated_fps(cap) -> int:
    raw = cap.get(cv2.CAP_PROP_FPS)
    rounded = round(raw)
    if rounded not in VALID_FPS:
        raise ValueError(f"Unsupported fps: {raw}, rounded={rounded}. Only {VALID_FPS} allowed.")
    return rounded


def box_edge_distance(box1_xyxy, box2_xyxy) -> float:
    """Minimum distance between edges of two AABBs. Returns 0 if overlapping."""
    x1a, y1a, x2a, y2a = box1_xyxy
    x1b, y1b, x2b, y2b = box2_xyxy
    dx = max(0, max(x1a, x1b) - min(x2a, x2b))
    dy = max(0, max(y1a, y1b) - min(y2a, y2b))
    return math.sqrt(dx * dx + dy * dy)


def nearest_edge_points(box1, box2):
    """
    Returns the closest point pair on the edges of two AABBs: (pt1, pt2).
    Each point is (int, int). Works by clamping each box's center onto the other box.
    """
    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2
    cx_b = (x1b + x2b) / 2
    cy_b = (y1b + y2b) / 2
    cx_a = (x1a + x2a) / 2
    cy_a = (y1a + y2a) / 2
    pt1 = (int(max(x1a, min(x2a, cx_b))), int(max(y1a, min(y2a, cy_b))))
    pt2 = (int(max(x1b, min(x2b, cx_a))), int(max(y1b, min(y2b, cy_a))))
    return pt1, pt2


def _draw_boxes(frame, boxes_data: dict) -> "np.ndarray":
    """Draw colored bounding boxes (no line). Returns a copy of frame."""
    frame = frame.copy()
    for x1, y1, x2, y2, conf in boxes_data.get("balls", []):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ball {conf:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for x1, y1, x2, y2, conf in boxes_data.get("rackets", []):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 255), 2)
        cv2.putText(frame, f"racket {conf:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)
    return frame


def process_single_video(video_path: str, video_name: str, model, callback: Callable,
                         hit_thresh: float = 150, pad_before: int = 15, pad_after: int = 15,
                         export_pad: int = 32, conf: float = 0.25, iou: float = 0.7,
                         imgsz: int = 640, show_boxes: bool = True):
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

    results = model.predict(
        source=video_path,
        classes=[32, 38],
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        stream=True,
        verbose=False,
    )

    frame_distances: dict[int, float] = {}
    # {frame_idx: {balls, rackets}} — all detections, for drawing colored boxes
    frame_boxes: dict[int, dict] = {}
    # {frame_idx: {dist, ball_box, racket_box, ball_conf, racket_conf}} — best pair, for peak line
    frame_detections: dict[int, dict] = {}

    for frame_idx, r in enumerate(results):
        balls = [box for box in r.boxes if int(box.cls.item()) == 32]
        rackets = [box for box in r.boxes if int(box.cls.item()) == 38]

        if balls or rackets:
            ball_data = [
                (int(b.xyxy[0][0]), int(b.xyxy[0][1]),
                 int(b.xyxy[0][2]), int(b.xyxy[0][3]), float(b.conf.item()))
                for b in balls
            ]
            racket_data = [
                (int(b.xyxy[0][0]), int(b.xyxy[0][1]),
                 int(b.xyxy[0][2]), int(b.xyxy[0][3]), float(b.conf.item()))
                for b in rackets
            ]
            frame_boxes[frame_idx] = {"balls": ball_data, "rackets": racket_data}

            if balls and rackets:
                best_dist = float("inf")
                best_b_box = best_rk_box = None
                best_b_conf = best_rk_conf = 0.0
                for b in balls:
                    bx = (int(b.xyxy[0][0]), int(b.xyxy[0][1]),
                          int(b.xyxy[0][2]), int(b.xyxy[0][3]))
                    for rk in rackets:
                        rkx = (int(rk.xyxy[0][0]), int(rk.xyxy[0][1]),
                               int(rk.xyxy[0][2]), int(rk.xyxy[0][3]))
                        d = box_edge_distance(bx, rkx)
                        if d < best_dist:
                            best_dist = d
                            best_b_box, best_b_conf = bx, float(b.conf.item())
                            best_rk_box, best_rk_conf = rkx, float(rk.conf.item())
                frame_distances[frame_idx] = best_dist
                frame_detections[frame_idx] = {
                    "dist": best_dist,
                    "ball_box": best_b_box,
                    "racket_box": best_rk_box,
                    "ball_conf": best_b_conf,
                    "racket_conf": best_rk_conf,
                }

        if frame_idx % 60 == 0:
            callback(type="progress", video_name=video_name, frame=frame_idx, total=total_frames)

    print(f"[yolo] {video_name}: detection done, {len(frame_distances)} frames with ball+racket")

    hit_frames = sorted(f for f, d in frame_distances.items() if d < hit_thresh)
    if not hit_frames:
        print(f"[yolo] {video_name}: no hit events found")
        return

    events = []
    for f in hit_frames:
        if events and f - events[-1]["end"] < 30:
            events[-1]["end"] = f
            if frame_distances.get(f, 999) < frame_distances.get(events[-1]["peak"], 999):
                events[-1]["peak"] = f
        else:
            events.append({"start": f, "end": f, "peak": f})

    print(f"[yolo] {video_name}: {len(events)} hit events detected")

    cap = cv2.VideoCapture(video_path)

    for clip_idx, event in enumerate(events):
        clip_start = max(0, event["start"] - pad_before)
        clip_end = min(total_frames - 1, event["end"] + pad_after)
        num_frames = clip_end - clip_start + 1
        peak_frame_rel = event["peak"] - clip_start  # relative to clip start
        clip_id = f"{video_name}_clip_{clip_idx:04d}"
        clip_dir = os.path.join(CLIPS_DIR, clip_id)
        os.makedirs(clip_dir, exist_ok=True)

        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            abs_frame = clip_start + i

            if show_boxes:
                # Draw colored boxes on all frames with detections
                if abs_frame in frame_boxes:
                    frame = _draw_boxes(frame, frame_boxes[abs_frame])
                # Peak frame: additionally draw nearest-edge line + distance
                if i == peak_frame_rel and abs_frame in frame_detections:
                    det = frame_detections[abs_frame]
                    pt1, pt2 = nearest_edge_points(det["ball_box"], det["racket_box"])
                    if abs_frame not in frame_boxes:
                        frame = frame.copy()
                    cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
                    mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                    cv2.putText(frame, f"{det['dist']:.0f}px",
                                (mid[0] + 4, mid[1] - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            cv2.imwrite(os.path.join(clip_dir, f"frame_{i:03d}.jpg"), frame)

        meta = {
            "clip_id": clip_id,
            "video_name": video_name,
            "source_video": os.path.abspath(video_path),
            "start_frame": clip_start,
            "num_frames": num_frames,
            "fps": fps,
            "peak_frame": peak_frame_rel,
            "export_pad": export_pad,
        }
        with open(os.path.join(clip_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[yolo] clip {clip_id}: frames {clip_start}-{clip_end} ({num_frames} frames), peak={event['peak']}")
        callback(type="clip", clip_id=clip_id, num_frames=num_frames, fps=fps,
                 peak_frame=peak_frame_rel)

    cap.release()


def process_folder(folder: str, callback: Callable, hit_thresh: float = 150,
                   pad_before: int = 15, pad_after: int = 15, export_pad: int = 32,
                   conf: float = 0.25, iou: float = 0.7, imgsz: int = 640,
                   show_boxes: bool = True) -> None:
    from ultralytics import YOLO

    mp4s = sorted(glob.glob(os.path.join(folder, "*.mp4")))
    if not mp4s:
        callback(type="skip", video_name="(none)", reason=f"No .mp4 files in {folder}")
        callback(type="done")
        return

    print(f"[yolo] Loading model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    for mp4_path in mp4s:
        video_name = os.path.splitext(os.path.basename(mp4_path))[0]
        try:
            process_single_video(
                mp4_path, video_name, model, callback,
                hit_thresh=hit_thresh, pad_before=pad_before, pad_after=pad_after,
                export_pad=export_pad, conf=conf, iou=iou, imgsz=imgsz, show_boxes=show_boxes,
            )
        except Exception as e:
            print(f"[yolo] ERROR processing {video_name}: {e}")
            callback(type="skip", video_name=video_name, reason=str(e))

    callback(type="done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO26 clip extractor")
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--hit_thresh", type=float, default=150)
    parser.add_argument("--pad_before", type=int, default=15)
    parser.add_argument("--pad_after", type=int, default=15)
    parser.add_argument("--export_pad", type=int, default=32)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no_boxes", action="store_true")
    args = parser.parse_args()

    os.makedirs(CLIPS_DIR, exist_ok=True)

    def cli_callback(**kwargs):
        t = kwargs.get("type", "")
        if t == "clip":
            print(f"  -> CLIP: {kwargs['clip_id']} ({kwargs['num_frames']} frames, peak={kwargs.get('peak_frame')})")
        elif t == "skip":
            print(f"  -> SKIP: {kwargs['video_name']}: {kwargs['reason']}")
        elif t == "progress":
            print(f"  -> {kwargs['video_name']}: frame {kwargs['frame']}/{kwargs['total']}")
        elif t == "done":
            print("  -> DONE")

    process_folder(
        args.folder, cli_callback,
        hit_thresh=args.hit_thresh, pad_before=args.pad_before, pad_after=args.pad_after,
        export_pad=args.export_pad, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
        show_boxes=not args.no_boxes,
    )
    print("Done.")
