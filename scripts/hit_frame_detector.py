"""
hit_frame_detector_v2.py
────────────────────────
用 HSV 颜色过滤找球 → 只在球的 ROI 里算光流方向 → 检测方向突变 = hit frame

Pipeline:
  1. HSV 过滤找网球（黄绿色）→ 得到球的位置序列
  2. 球位置做卡尔曼平滑（填补漏检帧）
  3. 在球周围 ROI 里算 t→t+1 光流方向角
  4. 方向变化角度突变 → hit frame 候选
  5. 输出标注视频 + JSON
"""

import json
import cv2
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from scipy.ndimage import convolve1d
from collections import deque

# ─── CONFIG ──────────────────────────────────────────────────────────────────
VIDEO_PATH   = "/Users/harryzhang/git/tempopeak/datasets/serve/00002.mp4"
OUTPUT_VIDEO = "/Users/harryzhang/git/tempopeak/outputs/hit_detection_v2.mp4"
OUTPUT_JSON  = "/Users/harryzhang/git/tempopeak/outputs/hit_frames_v2.json"

# HSV range for tennis ball (yellow-green)
# 在 HSV 空间里网球是高饱和黄绿，球场是蓝灰
HSV_LOW  = np.array([ 25, 80,  80])   # H:25-45, S>80, V>80
HSV_HIGH = np.array([ 45, 255, 255])

# Ball detection
BALL_MIN_RADIUS = 4    # px in original resolution
BALL_MAX_RADIUS = 30
ROI_SCALE       = 4.0  # ROI = ball_radius * this factor

# Direction-change detection
SMOOTH_SIGMA    = 4
PEAK_MIN_DIST   = 12
PEAK_PROMINENCE = 0.25

# Visualisation
TRAIL_LEN = 30   # frames of ball trail to draw
# ─────────────────────────────────────────────────────────────────────────────

Path(OUTPUT_VIDEO).parent.mkdir(parents=True, exist_ok=True)
Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)


def find_ball(frame_bgr):
    """
    Return (cx, cy, radius) of best tennis ball candidate, or None.
    Uses HSV color mask + contour fitting.
    """
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)

    # morphology: close small holes, remove noise
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best = None
    best_score = -1
    for c in cnts:
        area = cv2.contourArea(c)
        if area < np.pi * BALL_MIN_RADIUS**2:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if not (BALL_MIN_RADIUS <= r <= BALL_MAX_RADIUS):
            continue
        # circularity score: how round is it?
        peri = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (peri * peri + 1e-9)
        if circ > best_score:
            best_score = circ
            best = (int(cx), int(cy), int(r))

    return best   # None if no good candidate


def roi_flow_direction(prev_gray, curr_gray, cx, cy, radius):
    """
    Compute mean flow direction angle (degrees) inside ball ROI.
    Returns angle in [-180, 180] or None if ROI is degenerate.
    """
    pad = int(radius * ROI_SCALE)
    H, W = prev_gray.shape
    x1, y1 = max(0, cx - pad), max(0, cy - pad)
    x2, y2 = min(W, cx + pad), min(H, cy + pad)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    p_roi = prev_gray[y1:y2, x1:x2]
    c_roi = curr_gray[y1:y2, x1:x2]
    flow  = cv2.calcOpticalFlowFarneback(
        p_roi, c_roi, None,
        pyr_scale=0.5, levels=2, winsize=9,
        iterations=3, poly_n=5, poly_sigma=1.1, flags=0,
    )
    # mean vector
    vx = float(flow[..., 0].mean())
    vy = float(flow[..., 1].mean())
    mag = np.sqrt(vx**2 + vy**2)
    if mag < 0.3:   # ball barely moving — skip
        return None
    return float(np.degrees(np.arctan2(vy, vx)))


def angle_diff(a, b):
    """Signed angular difference in degrees, wrapped to [-180, 180]."""
    d = (b - a + 180) % 360 - 180
    return abs(d)


def smooth_normalize(series, sigma):
    arr = np.array(series, dtype=np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return arr
    arr = (arr - mn) / (mx - mn)
    half   = int(sigma * 3)
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    return convolve1d(arr, kernel, mode="reflect")


# ─── PASS 1: detect ball + compute direction-change series ───────────────────
print("Pass 1: ball detection + direction-change series...")
cap          = cv2.VideoCapture(VIDEO_PATH)
fps          = cap.get(cv2.CAP_PROP_FPS)
orig_W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

ball_positions = []   # (cx, cy, r) or None per frame
prev_gray      = None
prev_ball      = None
direction_series = []   # angle-change per frame (0 if no ball)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ball = find_ball(frame)
    ball_positions.append(ball)

    dchange = 0.0
    if ball and prev_ball and prev_gray is not None:
        cx, cy, r = ball
        angle = roi_flow_direction(prev_gray, gray, cx, cy, r)
        prev_angle = getattr(roi_flow_direction, '_last_angle', None)
        if angle is not None and prev_angle is not None:
            dchange = angle_diff(prev_angle, angle)
        if angle is not None:
            roi_flow_direction._last_angle = angle
    else:
        roi_flow_direction._last_angle = None

    direction_series.append(dchange)
    prev_gray = gray
    prev_ball = ball
    frame_idx += 1

    if frame_idx % 50 == 0:
        pct = frame_idx / total_frames * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        n_detected = sum(1 for b in ball_positions if b is not None)
        print(f"\r  [{bar}] {pct:.0f}%  ball detected: {n_detected}/{frame_idx}", end="", flush=True)

cap.release()
n_det = sum(1 for b in ball_positions if b is not None)
print(f"\n  frames={frame_idx}, ball detected={n_det} ({100*n_det/frame_idx:.0f}%)")

# ─── peak detection on direction-change series ───────────────────────────────
print("Smoothing + peak detection...")
smoothed = smooth_normalize(direction_series, SMOOTH_SIGMA)
peaks, _ = find_peaks(
    smoothed,
    distance=PEAK_MIN_DIST,
    prominence=PEAK_PROMINENCE * (smoothed.max() - smoothed.min()),
)
hit_frames = peaks.tolist()
print(f"  {len(hit_frames)} hit candidates: {hit_frames[:20]}")

# ─── PASS 2: render ───────────────────────────────────────────────────────────
print("Pass 2: rendering...")
cap    = cv2.VideoCapture(VIDEO_PATH)
writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (orig_W, orig_H))

hit_set   = set(hit_frames)
trail     = deque(maxlen=TRAIL_LEN)
BAR_W     = 260

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    ball = ball_positions[frame_idx] if frame_idx < len(ball_positions) else None

    # ── ball circle + trail ───────────────────────────────────────────────
    if ball:
        cx, cy, r = ball
        trail.append((cx, cy))
        cv2.circle(frame, (cx, cy), r + 3, (0, 255, 255), 2)   # yellow ring
        cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

    for i in range(1, len(trail)):
        alpha = i / len(trail)
        color = (0, int(255 * alpha), int(255 * (1 - alpha)))   # cyan→green fade
        cv2.line(frame, trail[i-1], trail[i], color, 2)

    # ── ball detection status ─────────────────────────────────────────────
    status     = "BALL ✓" if ball else "BALL ✗"
    status_col = (0, 255, 0) if ball else (0, 80, 200)
    cv2.putText(frame, status, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col, 2)

    # ── direction-change bar (top-right) ──────────────────────────────────
    dval = smoothed[frame_idx] if frame_idx < len(smoothed) else 0.0
    x0   = orig_W - BAR_W - 15
    cv2.rectangle(frame, (x0, 15), (x0 + BAR_W, 38), (30, 30, 30), -1)
    cv2.rectangle(frame, (x0, 15), (x0 + int(dval * BAR_W), 38), (0, 200, 255), -1)
    cv2.putText(frame, f"dir-change: {dval:.3f}", (x0, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    # ── hit flash ─────────────────────────────────────────────────────────
    if frame_idx in hit_set:
        cv2.rectangle(frame, (0, 0), (orig_W, orig_H), (0, 0, 255), 10)
        cv2.putText(frame, f"HIT #{hit_frames.index(frame_idx)+1}",
                    (orig_W // 2 - 90, orig_H // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 0, 255), 5)

    # ── ROI box around ball ───────────────────────────────────────────────
    if ball:
        cx, cy, r = ball
        pad = int(r * ROI_SCALE)
        cv2.rectangle(frame,
                      (max(0, cx-pad), max(0, cy-pad)),
                      (min(orig_W, cx+pad), min(orig_H, cy+pad)),
                      (255, 200, 0), 1)

    cv2.putText(frame, f"#{frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    writer.write(frame)
    frame_idx += 1

    if frame_idx % 100 == 0:
        pct = frame_idx / total_frames * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"\r  [{bar}] {pct:.0f}%", end="", flush=True)

cap.release()
writer.release()
print(f"\nDone.")

Path(OUTPUT_JSON).write_text(json.dumps({
    "video": VIDEO_PATH,
    "fps": fps,
    "total_frames": frame_idx,
    "ball_detection_rate": sum(1 for b in ball_positions if b) / max(frame_idx, 1),
    "hit_frames": hit_frames,
    "method": "hsv_ball_roi_direction_change",
}, indent=2))
print(f"  video → {OUTPUT_VIDEO}")
print(f"  json  → {OUTPUT_JSON}")