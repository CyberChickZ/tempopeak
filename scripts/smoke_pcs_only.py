import os
import torch
import numpy as np
from PIL import Image
from transformers.video_utils import load_video
from transformers import Sam3Model, Sam3Processor

HF_LOCAL_MODEL = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/models/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7"
VIDEO_PATH = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/datasets/serve/00001.mp4"
OUT_DIR = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/outputs/pcs_samples"

os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

print("Loading model for Text-to-Mask (PCS)...")
model = Sam3Model.from_pretrained(HF_LOCAL_MODEL, local_files_only=True).to(device, dtype=dtype)
processor = Sam3Processor.from_pretrained(HF_LOCAL_MODEL, local_files_only=True)

print("Loading video...")
video_frames, _ = load_video(VIDEO_PATH)

# 只取一帧（比如视频开头，或者中间某帧，这里选第 10 帧作为测试）
sample_indices = [10]

def overlay_masks(image, masks, color):
    # image: PIL Image
    # masks: [N, H, W] boolean/float
    image = image.convert("RGBA")
    masks = masks.cpu().numpy()
    
    for mask in masks:
        # Avoid issues where masks might be soft probabilities, we threshold it:
        mask_bool = mask > 0.0
        mask_img = Image.fromarray((mask_bool * 255).astype(np.uint8))
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask_img.point(lambda v: int(v * 0.5))  # 50% transparency
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image.convert("RGB")

print(f"Sampling {len(sample_indices)} frames: {sample_indices}")

for idx in sample_indices:
    frame = video_frames[idx]
    # Handle both np.ndarray and PIL inputs returned by load_video
    if isinstance(frame, np.ndarray):
        pil_img = Image.fromarray(frame)
    else:
        pil_img = frame
        
    print(f"Processing frame {idx}...")
    vis_img = pil_img.copy()

    # --- 移除了 Ball，只专属检测 Racket ---
    print(f"Detecting 'racket' on frame {idx}...")
    inputs = processor(images=pil_img, text="racket", return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    
    with torch.no_grad():
        outputs = model(**inputs)

    # 这里的 target_sizes 需要是 (height, width)
    target_sizes = [pil_img.size[::-1]] 

    # 将 threshold 设置为 0.05，看看除了高分还能召回哪些潜在的球拍
    res_racket = processor.post_process_instance_segmentation(
        outputs, threshold=0.05, mask_threshold=0.5, target_sizes=target_sizes
    )[0]

    if len(res_racket["masks"]) > 0:
        # blue for racket
        vis_img = overlay_masks(vis_img, res_racket["masks"], color=(0, 0, 255))
        
    out_path = os.path.join(OUT_DIR, f"frame_{idx:04d}_racket_only.jpg")
    vis_img.save(out_path)
    
    r_scores = [round(s, 3) for s in res_racket["scores"].tolist()] if len(res_racket["scores"]) > 0 else []
    print(f"  -> Saved {out_path}")
    print(f"     Racket scores: {r_scores}")
    
print("All frames processed successfully!")
