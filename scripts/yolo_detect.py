"""
yolo_detect.py — 最简 YOLOv8 网球检测脚本
输出格式与 sam3_mask_extractor 完全兼容（JSON + 可视化 MP4）

用法:
  python scripts/yolo_detect.py \
    --video_path datasets/serve/00001.mp4 \
    --video_name 00001 \
    --out_dir outputs/sam3_mask_extractor

COCO class IDs:
  32 = sports ball
  38 = tennis racket
"""

import os
import json
import argparse

parser = argparse.ArgumentParser(description="YOLOv8 tennis ball/racket detector")
parser.add_argument("--video_path", type=str, required=True)
parser.add_argument("--video_name", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--model", type=str, default="yolov8x.pt", help="YOLO model weights")
parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
parser.add_argument("--vis", action="store_true", help="Save annotated video")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

from ultralytics import YOLO

model = YOLO(args.model)

# COCO classes: 32=sports ball, 38=tennis racket
CLASSES = [32, 38]
LABEL_MAP = {32: "ball", 38: "racket"}

results = model.predict(
    source=args.video_path,
    classes=CLASSES,
    conf=args.conf,
    stream=True,
    verbose=False,
)

tracks = {}

for frame_idx, r in enumerate(results):
    frame_data = {}

    for i, box in enumerate(r.boxes):
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        label = LABEL_MAP.get(cls_id, "unknown")
        obj_id = f"{cls_id}_{i}"

        frame_data[obj_id] = {
            "label": label,
            "centroid": [round(cx, 3), round(cy, 3)],
            "box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": round(conf, 4),
            "mask_idx": -1,
            "mask_area": int((x2 - x1) * (y2 - y1)),
        }

    tracks[str(frame_idx)] = frame_data

    if frame_idx % 30 == 0:
        print(f"frame {frame_idx}: {len(frame_data)} detections")

# Save JSON
out_json = os.path.join(args.out_dir, f"{args.video_name}_yolo.json")
meta = {
    "video_name": args.video_name,
    "model": args.model,
    "conf": args.conf,
    "detector": "yolov8",
    "frames": frame_idx + 1,
}
payload = {"_meta": meta}
payload.update(tracks)

with open(out_json, "w") as f:
    json.dump(payload, f, indent=2)
print("Saved:", out_json)

# Optional: save annotated video
if args.vis:
    print("Rendering annotated video...")
    results_vis = model.predict(
        source=args.video_path,
        classes=CLASSES,
        conf=args.conf,
        save=True,
        project=args.out_dir,
        name=f"{args.video_name}_yolo_vis",
        exist_ok=True,
    )
    print("Done. Check:", os.path.join(args.out_dir, f"{args.video_name}_yolo_vis"))

print("All done.")
