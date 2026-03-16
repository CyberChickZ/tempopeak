"""
Fine-tune yolo26l.pt on tempopeak-refine-1 COCO dataset.

Input:  datasets/yolo_refine/tempopeak-refine-1.zip
Output: sam3_annotator/server/yolo26l_refined.pt

Classes (COCO original ids — do NOT remap):
  0  = person
  32 = sports ball
  38 = tennis racket

=> In tracking script use:
   BALL_CLASS   = 32
   RACKET_CLASS = 38
   PERSON_CLASS = 0
   MODEL_PATH   = 'sam3_annotator/server/yolo26l_refined.pt'
"""

import json
import os
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

# ─── Paths ───────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
ZIP_PATH   = REPO_ROOT / "datasets/yolo_refine/tempopeak-refine-1.zip"
WORK_DIR   = REPO_ROOT / "datasets/yolo_refine/work"
MODEL_PATH = REPO_ROOT / "sam3_annotator/server/yolo26l.pt"   # ← changed x→l
OUTPUT_DIR = REPO_ROOT / "sam3_annotator/server"

TRAIN_RATIO = 0.85
SEED        = 42

# ─── Fine-tune hyperparams ───────────────────────────────────────────────────
EPOCHS   = 100
IMGSZ    = 1280
BATCH    = 8        # l model is lighter, batch=8 should fit
LR0      = 0.001
LRF      = 0.01
PATIENCE = 30

# ─── COCO 80 class names (original order, index = COCO id mapping) ───────────
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]


# ─── Step 1: Extract zip ─────────────────────────────────────────────────────
print("Extracting dataset...")
extract_dir = WORK_DIR / "raw"
shutil.rmtree(extract_dir, ignore_errors=True)
extract_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    zf.extractall(extract_dir)

coco_json = extract_dir / "dataset/_annotations.coco.json"
img_src   = extract_dir / "dataset/images"

with open(coco_json) as f:
    coco = json.load(f)

# Keep original COCO id — no remapping
your_classes = {c["id"]: c["name"] for c in coco["categories"]}
print(f"Your annotated classes: {your_classes}")
# Expected: {0: 'person', 32: 'sports ball', 38: 'tennis racket'}

img_info   = {img["id"]: img for img in coco["images"]}
ann_by_img = defaultdict(list)
for ann in coco["annotations"]:
    ann_by_img[ann["image_id"]].append(ann)


# ─── Step 2: Convert to YOLO label format (original COCO ids) ────────────────
print("Converting COCO → YOLO labels...")
labels_dir = WORK_DIR / "labels_all"
labels_dir.mkdir(parents=True, exist_ok=True)

for img in coco["images"]:
    iid  = img["id"]
    w, h = img["width"], img["height"]
    stem = Path(img["file_name"]).stem

    lines = []
    for ann in ann_by_img[iid]:
        cls_id = ann["category_id"]          # original COCO id: 0, 32, or 38
        bx, by, bw, bh = ann["bbox"]
        cx = (bx + bw / 2) / w
        cy = (by + bh / 2) / h
        nw = bw / w
        nh = bh / h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    (labels_dir / f"{stem}.txt").write_text("\n".join(lines))


# ─── Step 3: Train / val split ───────────────────────────────────────────────
print("Splitting train/val...")
all_ids = [img["id"] for img in coco["images"]]
random.seed(SEED)
random.shuffle(all_ids)

n_train   = max(1, int(len(all_ids) * TRAIN_RATIO))
train_ids = set(all_ids[:n_train])
val_ids   = set(all_ids[n_train:])
print(f"  train={len(train_ids)}, val={len(val_ids)}")

for split, ids in [("train", train_ids), ("val", val_ids)]:
    img_dst = WORK_DIR / f"images/{split}"
    lbl_dst = WORK_DIR / f"labels/{split}"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    for iid in ids:
        fname = img_info[iid]["file_name"]
        stem  = Path(fname).stem
        shutil.copy(img_src / fname,             img_dst / fname)
        shutil.copy(labels_dir / f"{stem}.txt",  lbl_dst / f"{stem}.txt")


# ─── Step 4: dataset.yaml — nc=80, original COCO class names ─────────────────
yaml_path = WORK_DIR / "dataset.yaml"
yaml_path.write_text(f"""\
path: {WORK_DIR}
train: images/train
val:   images/val

nc: 80
names: {COCO_NAMES}
""")
print(f"dataset.yaml → {yaml_path}")


# ─── Step 5: Fine-tune ───────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nLoading model: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))

print(f"Starting fine-tuning (epochs={EPOCHS}, lr0={LR0}, imgsz={IMGSZ})...")
results = model.train(
    data         = str(yaml_path),
    epochs       = EPOCHS,
    imgsz        = IMGSZ,
    batch        = BATCH,
    lr0          = LR0,
    lrf          = LRF,
    patience     = PATIENCE,
    freeze       = 6,       # freeze backbone early layers, protect pretrained features
    degrees      = 5,
    translate    = 0.05,
    scale        = 0.3,
    fliplr       = 0.5,
    mosaic       = 0.5,
    mixup        = 0.0,
    cache        = True,
    workers      = 0,
    amp          = False,
    project      = str(OUTPUT_DIR),
    name         = "yolo26l_refined",
    exist_ok     = True,
    device       = "mps",
)

best_weights = Path(results.save_dir) / "weights/best.pt"
out_path     = OUTPUT_DIR / "yolo26l_refined.pt"
shutil.copy(best_weights, out_path)
print(f"\nDone. Best weights → {out_path}")
print("\nUpdate tracking script:")
print("  MODEL_PATH   = 'sam3_annotator/server/yolo26l_refined.pt'")
print("  BALL_CLASS   = 32")
print("  RACKET_CLASS = 38")
print("  PERSON_CLASS = 0")