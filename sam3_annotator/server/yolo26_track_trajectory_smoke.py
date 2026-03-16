###############################################################################
# Tennis Tracking — raw detection + tracking, no post-processing
# Model: yolo26l_refined.pt (COCO original ids)
###############################################################################

import cv2
from collections import defaultdict
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

VIDEO_PATH  = "/Users/harryzhang/git/tempopeak/datasets/raw/Serve-Compilation-Slow-Motion-Alcaraz-Dj.mp4"
MODEL_PATH  = "/Users/harryzhang/git/tempopeak/sam3_annotator/server/yolo26l_refined.pt"
OUTPUT_PATH = "/Users/harryzhang/git/tempopeak/outputs/tennis_tracking_result.mp4"
Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

TRACKER_CFG = str(Path(__file__).parent / "bytetrack_tennis.yaml")

CONF  = 0.02
IOU   = 0.5
IMGSZ = 1280

BALL_CLASS   = 32   # sports ball (COCO original)
RACKET_CLASS = 38   # tennis racket (COCO original)
PERSON_CLASS = 0    # person (COCO original)

CLASS_COLOR = {
    BALL_CLASS:   (0, 255, 0),
    RACKET_CLASS: (255, 100, 0),
    PERSON_CLASS: (180, 180, 180)
}

CLASS_NAME = {
    BALL_CLASS:   "ball",
    RACKET_CLASS: "racket",
    PERSON_CLASS: "person",
}

MAX_TRAIL = 64

# -----------------------------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------------------------

model = YOLO(MODEL_PATH)

# -----------------------------------------------------------------------------
# TRACKING
# -----------------------------------------------------------------------------

tracks = defaultdict(list)   # track_id -> [(cx, cy), ...]

vid_stride = 2
results = model.track(
    source=VIDEO_PATH,
    conf=CONF,
    iou=IOU,
    imgsz=IMGSZ,
    tracker=TRACKER_CFG,
    stream=True,
    device="mps",
    half=True,       # fp16 — ~20-30% faster on MPS
    vid_stride=vid_stride,    # skip every other frame — 2x speed
    verbose=False,
    classes=[BALL_CLASS, RACKET_CLASS, PERSON_CLASS],  # keep only 3 classes
)

writer = None

# Get total frames for progress bar
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
expected_frames = (total_frames + vid_stride - 1) // vid_stride if total_frames > 0 else None
if expected_frames is not None and expected_frames > 3000:
    expected_frames = 3000

for frame_idx, r in enumerate(tqdm(results, total=expected_frames, desc="Tracking")):
    if frame_idx >= 3000:
        break
    # Use original image and draw boxes manually so we can enforce our own colors and labels
    frame = r.orig_img.copy()

    if writer is None:
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(
            OUTPUT_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (w, h),
        )

    if r.boxes is not None and r.boxes.id is not None:
        boxes   = r.boxes.xyxy.cpu().numpy()
        ids     = r.boxes.id.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confs   = r.boxes.conf.cpu().numpy()

        for box, track_id, cls, conf in zip(boxes, ids, classes, confs):
            cls = int(cls)
            tid = int(track_id)

            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            tracks[tid].append((cx, cy))
            if len(tracks[tid]) > MAX_TRAIL:
                tracks[tid].pop(0)

            color = CLASS_COLOR.get(cls, (200, 200, 200))
            
            # Use our dictionary to explicitly fetch the class name 
            label = f"{CLASS_NAME.get(cls, str(cls))} #{tid} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            pts = tracks[tid]
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                faded = tuple(int(c * alpha) for c in color)
                thickness = 2 if cls == BALL_CLASS else 1
                cv2.line(frame, pts[i - 1], pts[i], faded, thickness)

    cv2.putText(frame, f"#{frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    writer.write(frame)

writer.release()
print("Saved:", OUTPUT_PATH)