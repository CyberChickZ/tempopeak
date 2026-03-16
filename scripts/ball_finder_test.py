"""
ball_finder_test.py — 只找球，打印每帧所有 box 信息
"""
import cv2
from ultralytics import YOLO
from pathlib import Path

VIDEO_PATH  = "/Users/harryzhang/git/tempopeak/datasets/serve/00011.mp4"
OUTPUT_PATH = "/Users/harryzhang/git/tempopeak/outputs/ball_finder_test_yolo26s.mp4"
MODEL_PATH  = "/Users/harryzhang/git/tempopeak/sam3_annotator/server/yolo26s.pt"

CONF  = 0.1
IMGSZ = 960

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
model = YOLO(MODEL_PATH)

cap    = cv2.VideoCapture(VIDEO_PATH)
fps    = cap.get(cv2.CAP_PROP_FPS)
W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

frame_idx = 0
detected  = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, classes=[32], conf=CONF,
                            imgsz=IMGSZ, verbose=False, device="mps")
    boxes = results[0].boxes

    if boxes and len(boxes):
        detected += 1
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf_val = float(box.conf[0])
            cx, cy   = (x1+x2)//2, (y1+y2)//2
            w, h     = x2-x1, y2-y1
            # ── log ──────────────────────────────────────────────────────
            print(f"  f{frame_idx:04d} box=({cx:4d},{cy:4d}) "
                  f"size=({w}x{h}) conf={conf_val:.2f}")
            # ─────────────────────────────────────────────────────────────
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
            cv2.circle(frame, (cx,cy), 4, (0,255,255), -1)
            cv2.putText(frame, f"ball {conf_val:.2f}", (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
        status = f"BALL ✓ ({len(boxes)})"
        col    = (0, 255, 0)
    else:
        status = "BALL ✗"
        col    = (0, 80, 200)

    cv2.putText(frame, status,          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    cv2.putText(frame, f"#{frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, f"det rate: {detected}/{frame_idx+1}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()
print(f"\nDone. detection rate: {detected}/{frame_idx} ({100*detected/frame_idx:.0f}%)")
print(f"→ {OUTPUT_PATH}")