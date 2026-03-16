"""
ball_finder_v4.py
"""
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
from scipy.signal import find_peaks
from scipy.ndimage import convolve1d

VIDEO_PATH  = "/Users/harryzhang/git/tempopeak/datasets/serve/00006.mp4"
OUTPUT_PATH = "/Users/harryzhang/git/tempopeak/outputs/ball_finder_v4.mp4"
MODEL_PATH  = "/Users/harryzhang/git/tempopeak/sam3_annotator/server/yolo26s.pt"

CONF             = 0.3
IMGSZ            = 960
MAX_LINK_PX      = 120
SAME_FRAME_IOU   = 0.5   # 同帧去重：两个框重叠>50%就是同一个球
STATIC_IOU       = 0.95  # 跨帧静止检测：上抛顶点合理，连续5帧静止才删
STATIC_N         = 5
MAX_COAST        = 3
VEL_COAST_FACTOR = 1.2

SMOOTH_SIGMA    = 3
PEAK_MIN_DIST   = 10
PEAK_PROMINENCE = 0.25
TRAIL_LEN       = 60

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def iou(a, b):
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter   = max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter == 0: return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/(ua+1e-9)

def nms_same_frame(dets, thresh=0.5):
    """同帧去重：IoU > thresh 保留高conf的那个"""
    if not dets: return []
    dets = sorted(dets, key=lambda d: -d[4])
    kept = []
    for d in dets:
        if all(iou(d[:4], k[:4]) < thresh for k in kept):
            kept.append(d)
    return kept

def center(box):
    return np.array([(box[0]+box[2])/2, (box[1]+box[3])/2], dtype=float)

def shift_box(box, vel):
    x1,y1,x2,y2 = box
    return [x1+vel[0], y1+vel[1], x2+vel[0], y2+vel[1]]

def in_expanded_box(pred_box, real_center, factor=1.2):
    cx = (pred_box[0]+pred_box[2])/2
    cy = (pred_box[1]+pred_box[3])/2
    hw = (pred_box[2]-pred_box[0])/2 * factor
    hh = (pred_box[3]-pred_box[1])/2 * factor
    return abs(real_center[0]-cx) <= hw and abs(real_center[1]-cy) <= hh

def smooth_normalize(series, sigma):
    arr = np.array(series, dtype=np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9: return arr
    arr = (arr - mn) / (mx - mn)
    half = int(sigma * 3)
    k = np.exp(-0.5*(np.arange(-half, half+1)/sigma)**2)
    k /= k.sum()
    return convolve1d(arr, k, mode="reflect")

# ── pass 1: detect ────────────────────────────────────────────────────────────
print("Pass 1: detection...")
model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(VIDEO_PATH)
fps   = cap.get(cv2.CAP_PROP_FPS)
W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
            dets.append([float(x1),float(y1),float(x2),float(y2),float(box.conf[0])])
    # 同帧去重用 SAME_FRAME_IOU
    raw.append(nms_same_frame(dets, thresh=SAME_FRAME_IOU))
    fi += 1
    if fi % 30 == 0:
        print(f"\r  {fi/total*100:.0f}% ({fi}/{total})", end="", flush=True)
cap.release()
print(f"\n  {fi} frames")

# ── 跨帧静止过滤 (STATIC_IOU=0.95, 允许上抛顶点) ─────────────────────────────
filtered  = [list(d) for d in raw]
prev_info = []
for fi, dets in enumerate(filtered):
    keep, new_prev = [], []
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
print("Linking ids...")
next_id = 0

class Track:
    def __init__(self, tid, box):
        self.tid       = tid
        self.box       = box
        self.vel       = np.zeros(2)
        self.prev_vel  = np.zeros(2)
        self.coast     = 0
        self.predicted = False

    def update(self, box):
        new_c         = center(box)
        old_c         = center(self.box)
        self.prev_vel = self.vel.copy()
        self.vel      = new_c - old_c
        self.box      = box
        self.coast    = 0
        self.predicted = False

    def predict_next(self):
        self.box      = shift_box(self.box, self.vel)
        self.coast   += 1
        self.predicted = True

    def link_threshold(self):
        spd = float(np.linalg.norm(self.vel))
        return spd * VEL_COAST_FACTOR if spd > 2.0 else MAX_LINK_PX

    def delta_v(self):
        return float(np.linalg.norm(self.vel - self.prev_vel))

active    = {}
frame_ids = []
frame_dv  = []

for fi, dets in enumerate(filtered):
    assigned  = []
    used_tids = set()

    # 匹配真实检测
    for d in sorted(dets, key=lambda x: -x[4]):
        dc        = center(d[:4])
        best_tid  = None
        best_dist = None
        for tid, trk in active.items():
            if tid in used_tids: continue
            thresh = trk.link_threshold()
            dist   = float(np.linalg.norm(dc - center(trk.box)))
            if dist < thresh:
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_tid  = tid
        if best_tid is not None:
            active[best_tid].update(d[:4])
            used_tids.add(best_tid)
            assigned.append((d[:4], best_tid, False))
        else:
            trk = Track(next_id, d[:4])
            active[next_id] = trk
            used_tids.add(next_id)
            assigned.append((d[:4], next_id, False))
            next_id += 1

    # coast：预测位置，如果附近有真实检测就 snap
    real_centers = [center(d[:4]) for d in dets]
    for tid, trk in list(active.items()):
        if tid in used_tids: continue
        if trk.coast >= MAX_COAST or np.linalg.norm(trk.vel) <= 0.5:
            del active[tid]
            continue

        pred_box = shift_box(trk.box, trk.vel)

        # snap 到附近真实检测
        matched_real = None
        for rc in real_centers:
            if in_expanded_box(pred_box, rc, factor=1.2):
                pc = center(pred_box)
                matched_real = min(dets, key=lambda d: np.linalg.norm(center(d[:4])-pc))
                break

        if matched_real is not None:
            active[tid].update(matched_real[:4])
            used_tids.add(tid)
            assigned.append((matched_real[:4], tid, False))
        else:
            trk.predict_next()
            assigned.append((trk.box, tid, True))

    frame_ids.append(assigned)

    dv = max(
        (active[tid].delta_v() for _, tid, pred in assigned
         if not pred and tid in active),
        default=0.0
    )
    # 在 frame_ids 循环末尾加这段，替换掉原来的 frame_dv.append

    frame_dv.append(dv)

    # track log
    real_count = sum(1 for _,_,p in assigned if not p)
    pred_count = sum(1 for _,_,p in assigned if p)
    if real_count + pred_count > 0:
        detail = " ".join(
            f"id={tid}({'P' if pred else 'R'} vel={np.linalg.norm(active[tid].vel):.1f}px)"
            if tid in active else f"id={tid}(expired)"
            for _, tid, pred in assigned
        )
        print(f"  f{fi:04d} real={real_count} pred={pred_count} dv={dv:.1f} | {detail}")

# ── hit detection ─────────────────────────────────────────────────────────────
print("Hit detection...")
smoothed = smooth_normalize(frame_dv, SMOOTH_SIGMA)
peaks, _ = find_peaks(
    smoothed,
    distance=PEAK_MIN_DIST,
    prominence=PEAK_PROMINENCE * (smoothed.max() - smoothed.min()),
)
hit_frames = set(peaks.tolist())
print(f"  {len(hit_frames)} hit candidates")

# ── pass 2: render ────────────────────────────────────────────────────────────
print("Pass 2: rendering...")
cap    = cv2.VideoCapture(VIDEO_PATH)
writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
trails = defaultdict(list)
COLORS = [(0,255,255),(0,200,255),(255,200,0),(0,255,128),(255,100,100)]
BAR_W  = 260

fi = 0
while True:
    ret, frame = cap.read()
    if not ret: break

    assigned = frame_ids[fi] if fi < len(frame_ids) else []

    for box, tid, predicted in assigned:
        x1,y1,x2,y2 = map(int, box)
        cx,cy = (x1+x2)//2, (y1+y2)//2
        color = COLORS[tid % len(COLORS)]
        trails[tid].append((cx,cy,predicted))
        if len(trails[tid]) > TRAIL_LEN: trails[tid].pop(0)

        if predicted:
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 1)
            cv2.putText(frame, f"#{tid} pred", (x1,y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        else:
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"#{tid}", (x1,y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, (cx,cy), 4, color, -1)

    for tid, pts in trails.items():
        color = COLORS[tid % len(COLORS)]
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            c = tuple(int(v*alpha) for v in color)
            cv2.line(frame, pts[i-1][:2], pts[i][:2], c,
                     1 if pts[i][2] else 2)

    dv  = float(smoothed[fi]) if fi < len(smoothed) else 0.0
    x0  = W - BAR_W - 15
    cv2.rectangle(frame, (x0,15), (x0+BAR_W,38), (30,30,30), -1)
    cv2.rectangle(frame, (x0,15), (x0+int(dv*BAR_W),38), (0,200,255), -1)
    cv2.putText(frame, f"dv: {dv:.3f}", (x0,12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,255), 1)

    if fi in hit_frames:
        cv2.rectangle(frame, (0,0), (W,H), (0,0,255), 10)
        cv2.putText(frame, "HIT", (W//2-60, H//2),
                    cv2.FONT_HERSHEY_DUPLEX, 3.0, (0,0,255), 6)

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