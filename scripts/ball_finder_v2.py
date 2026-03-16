"""
ball_finder_v2.py — 检测 + 过滤 + 连接 id
1. YOLO 检测球
2. 同帧 NMS — 保留最高 conf，删除 IoU > 5% 的重叠框
3. 静止过滤 — 同位置连续 5 帧 IoU > 5% → 删除（背景误检）
4. 连接 id — 下一帧 120px 内最近的球 → 同一 id
"""
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict

VIDEO_PATH  = "/Users/harryzhang/git/tempopeak/datasets/serve/00011.mp4"
OUTPUT_PATH = "/Users/harryzhang/git/tempopeak/outputs/ball_finder_v2.mp4"
MODEL_PATH  = "/Users/harryzhang/git/tempopeak/sam3_annotator/server/yolo26s.pt"

CONF        = 0.3
IMGSZ       = 960
MAX_LINK_PX = 120   # max px between frames to link same id
STATIC_IOU  = 0.95  # iou threshold for "same position"
STATIC_N    = 5     # consecutive frames to be considered static → remove

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────
def iou(a, b):
    """a, b: [x1,y1,x2,y2]"""
    ix1, iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2, iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (ua + 1e-9)

def nms_same_frame(detections):
    """detections: list of [x1,y1,x2,y2,conf]. Return filtered list."""
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: -d[4])
    kept = []
    for d in detections:
        if all(iou(d[:4], k[:4]) < STATIC_IOU for k in kept):
            kept.append(d)
    return kept

def center(box):
    return np.array([(box[0]+box[2])/2, (box[1]+box[3])/2], dtype=float)

# ── pass 1: detect + per-frame NMS ───────────────────────────────────────────
print("Pass 1: detection + NMS...")
model  = YOLO(MODEL_PATH)
cap    = cv2.VideoCapture(VIDEO_PATH)
fps    = cap.get(cv2.CAP_PROP_FPS)
W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

raw_detections = []   # per frame: list of [x1,y1,x2,y2,conf]

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, classes=[32], conf=CONF,
                            imgsz=IMGSZ, verbose=False, device="mps")
    dets = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
            c = float(box.conf[0])
            dets.append([float(x1),float(y1),float(x2),float(y2),c])
    raw_detections.append(nms_same_frame(dets))

    frame_idx += 1
    if frame_idx % 30 == 0:
        pct = frame_idx/total*100
        print(f"\r  {pct:.0f}% ({frame_idx}/{total})", end="", flush=True)

cap.release()
print(f"\n  done. {frame_idx} frames")

# ── static filter: remove boxes that stay in same spot ≥ STATIC_N frames ─────
print("Static filter...")
# for each frame, track how many consecutive frames each box has been "static"
filtered = [list(dets) for dets in raw_detections]
# rolling: for each detection in frame t, count consecutive overlap with prev frames
static_count = []   # parallel to filtered, per box
prev_boxes   = []

for fi, dets in enumerate(filtered):
    new_prev = []
    keep     = []
    for d in dets:
        # how many consecutive frames was something in this position?
        consec = 1
        for pbox, pcnt in prev_boxes:
            if iou(d[:4], pbox) >= STATIC_IOU:
                consec = pcnt + 1
                break
        if consec < STATIC_N:
            keep.append((d, consec))
            new_prev.append((d[:4], consec))
        # else: drop — static background detection
    filtered[fi] = [k[0] for k in keep]
    prev_boxes   = [(b, c) for b, c in new_prev]

n_removed = sum(len(raw_detections[i]) - len(filtered[i]) for i in range(len(filtered)))
print(f"  removed {n_removed} static detections")

# ── id linking: greedy nearest-neighbour across frames ───────────────────────
print("Linking ids...")
next_id    = 0
# last known position per active id: {id: [x1,y1,x2,y2]}
active_ids = {}   # id -> last_box
frame_ids  = []   # per frame: list of (box, id)

for fi, dets in enumerate(filtered):
    assigned = []
    used_ids = set()

    # sort dets by conf descending
    for d in sorted(dets, key=lambda x: -x[4]):
        best_id   = None
        best_dist = MAX_LINK_PX

        for tid, last_box in active_ids.items():
            if tid in used_ids:
                continue
            dist = float(np.linalg.norm(center(d[:4]) - center(last_box)))
            if dist < best_dist:
                best_dist = dist
                best_id   = tid

        if best_id is None:
            best_id = next_id
            next_id += 1

        used_ids.add(best_id)
        active_ids[best_id] = d[:4]
        assigned.append((d, best_id))

    # expire ids not seen this frame
    active_ids = {tid: box for tid, box in active_ids.items() if tid in used_ids}
    frame_ids.append(assigned)

# ── pass 2: render ────────────────────────────────────────────────────────────
print("Pass 2: rendering...")
cap    = cv2.VideoCapture(VIDEO_PATH)
writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

# collect trail per id
trails     = defaultdict(list)
TRAIL_LEN  = 45
COLORS     = [(0,255,255),(0,200,255),(255,200,0),(0,255,128),(255,100,100)]

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    assigned = frame_ids[frame_idx] if frame_idx < len(frame_ids) else []

    for d, tid in assigned:
        x1,y1,x2,y2,conf = int(d[0]),int(d[1]),int(d[2]),int(d[3]),d[4]
        cx,cy = (x1+x2)//2, (y1+y2)//2
        color = COLORS[tid % len(COLORS)]

        trails[tid].append((cx,cy))
        if len(trails[tid]) > TRAIL_LEN:
            trails[tid].pop(0)

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"#{tid} {conf:.2f}", (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, (cx,cy), 4, color, -1)

    # draw trails
    for tid, pts in trails.items():
        color = COLORS[tid % len(COLORS)]
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            c = tuple(int(v*alpha) for v in color)
            cv2.line(frame, pts[i-1], pts[i], c, 2)

    n = len(assigned)
    cv2.putText(frame, f"balls: {n}", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if n else (0,80,200), 2)
    cv2.putText(frame, f"#{frame_idx}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    writer.write(frame)
    frame_idx += 1
    if frame_idx % 50 == 0:
        pct = frame_idx/total*100
        print(f"\r  {pct:.0f}%", end="", flush=True)

cap.release()
writer.release()
print(f"\nDone → {OUTPUT_PATH}")