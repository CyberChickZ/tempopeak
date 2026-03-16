"""
ball_finder_v3.py — 检测 + 过滤 + 连接 id + 速度预测插值
"""
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict

VIDEO_PATH  = "/Users/harryzhang/git/tempopeak/datasets/serve/00018.mp4"
OUTPUT_PATH = "/Users/harryzhang/git/tempopeak/outputs/ball_finder_v3.mp4"
MODEL_PATH  = "/Users/harryzhang/git/tempopeak/sam3_annotator/server/yolo26s.pt"

CONF        = 0.3
IMGSZ       = 960
MAX_LINK_PX = 120
STATIC_IOU  = 0.95
STATIC_N    = 5
MAX_COAST   = 3     # 最多用速度预测插值几帧，超过就放弃这个 id

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────
def iou(a, b):
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter   = max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter == 0: return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/(ua+1e-9)

def nms_same_frame(dets):
    if not dets: return []
    dets = sorted(dets, key=lambda d: -d[4])
    kept = []
    for d in dets:
        if all(iou(d[:4], k[:4]) < STATIC_IOU for k in kept):
            kept.append(d)
    return kept

def center(box):
    return np.array([(box[0]+box[2])/2, (box[1]+box[3])/2], dtype=float)

def box_size(box):
    return box[2]-box[0], box[3]-box[1]

def shift_box(box, vel):
    """Translate box by velocity vector."""
    x1,y1,x2,y2 = box
    return [x1+vel[0], y1+vel[1], x2+vel[0], y2+vel[1]]

# ── pass 1: detect ────────────────────────────────────────────────────────────
print("Pass 1: detection...")
model  = YOLO(MODEL_PATH)
cap    = cv2.VideoCapture(VIDEO_PATH)
fps    = cap.get(cv2.CAP_PROP_FPS)
W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

raw = []
fi  = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    results = model.predict(frame, classes=[32], conf=CONF,
                            imgsz=IMGSZ, verbose=False, device="mps")
    dets = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
            c = float(box.conf[0])
            dets.append([float(x1),float(y1),float(x2),float(y2),c])
    raw.append(nms_same_frame(dets))
    fi += 1
    if fi % 30 == 0:
        print(f"\r  {fi/total*100:.0f}% ({fi}/{total})", end="", flush=True)
cap.release()
print(f"\n  {fi} frames")

# ── static filter ─────────────────────────────────────────────────────────────
print("Static filter...")
filtered  = [list(d) for d in raw]
prev_info = []   # list of (box, count)

for fi, dets in enumerate(filtered):
    keep     = []
    new_prev = []
    for d in dets:
        consec = 1
        for pbox, pcnt in prev_info:
            if iou(d[:4], pbox) >= STATIC_IOU:
                consec = pcnt + 1
                break
        if consec < STATIC_N:
            keep.append(d)
            new_prev.append((d[:4], consec))
    filtered[fi] = keep
    prev_info    = new_prev

# ── id linking + velocity coasting ───────────────────────────────────────────
print("Linking ids + velocity prediction...")

next_id = 0

class Track:
    def __init__(self, tid, box):
        self.tid       = tid
        self.box       = box           # last confirmed box
        self.vel       = np.zeros(2)   # pixels/frame velocity
        self.coast     = 0             # frames since last real detection
        self.predicted = False         # is current box predicted?
        self.history   = [box]         # for velocity estimation (last 2 real)

    def update(self, box):
        new_c = center(box)
        old_c = center(self.box)
        self.vel       = new_c - old_c
        self.box       = box
        self.coast     = 0
        self.predicted = False
        self.history.append(box)

    def predict_next(self):
        self.box       = shift_box(self.box, self.vel)
        self.coast    += 1
        self.predicted = True

active   = {}   # tid -> Track
frame_ids = []  # per frame: list of (box, tid, is_predicted)

for fi, dets in enumerate(filtered):
    assigned   = []
    used_tids  = set()
    used_dets  = set()

    # try to match detections to active tracks (closest center within MAX_LINK_PX)
    det_list = sorted(dets, key=lambda x: -x[4])
    for d in det_list:
        dc = center(d[:4])
        best_tid  = None
        best_dist = MAX_LINK_PX
        for tid, trk in active.items():
            if tid in used_tids: continue
            dist = float(np.linalg.norm(dc - center(trk.box)))
            if dist < best_dist:
                best_dist = dist
                best_tid  = tid
        if best_tid is not None:
            active[best_tid].update(d[:4])
            used_tids.add(best_tid)
            assigned.append((d[:4], best_tid, False))
        else:
            # new track
            trk = Track(next_id, d[:4])
            active[next_id] = trk
            used_tids.add(next_id)
            assigned.append((d[:4], next_id, False))
            next_id += 1

    # coast unmatched tracks: predict next position using velocity
    for tid, trk in list(active.items()):
        if tid in used_tids:
            continue
        if trk.coast < MAX_COAST and np.linalg.norm(trk.vel) > 0.5:
            trk.predict_next()
            assigned.append((trk.box, tid, True))   # predicted box
        else:
            del active[tid]   # too long without detection → expire

    frame_ids.append(assigned)

# ── pass 2: render ────────────────────────────────────────────────────────────
print("Pass 2: rendering...")
cap    = cv2.VideoCapture(VIDEO_PATH)
writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

trails    = defaultdict(list)
TRAIL_LEN = 60
COLORS    = [(0,255,255),(0,200,255),(255,200,0),(0,255,128),(255,100,100)]

fi = 0
while True:
    ret, frame = cap.read()
    if not ret: break

    assigned = frame_ids[fi] if fi < len(frame_ids) else []

    for box, tid, predicted in assigned:
        x1,y1,x2,y2 = map(int, box)
        cx,cy = (x1+x2)//2, (y1+y2)//2
        color = COLORS[tid % len(COLORS)]

        trails[tid].append((cx, cy, predicted))
        if len(trails[tid]) > TRAIL_LEN:
            trails[tid].pop(0)

        if predicted:
            # dashed box for predicted frames
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 1)
            cv2.putText(frame, f"#{tid} pred", (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        else:
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"#{tid}", (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, (cx,cy), 4, color, -1)

    # trails: solid for real, dotted for predicted
    for tid, pts in trails.items():
        color = COLORS[tid % len(COLORS)]
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            c = tuple(int(v*alpha) for v in color)
            thickness = 1 if pts[i][2] else 2   # thin if predicted
            cv2.line(frame, pts[i-1][:2], pts[i][:2], c, thickness)

    n_real = sum(1 for _,_,p in assigned if not p)
    n_pred = sum(1 for _,_,p in assigned if p)
    cv2.putText(frame, f"real:{n_real} pred:{n_pred}", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"#{fi}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    writer.write(frame)
    fi += 1
    if fi % 50 == 0:
        print(f"\r  {fi/total*100:.0f}%", end="", flush=True)

cap.release()
writer.release()
print(f"\nDone → {OUTPUT_PATH}")