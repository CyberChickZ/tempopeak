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

# Randomly sample 10 evenly spaced frames
step = max(1, len(video_frames) // 10)
sample_indices = list(range(0, len(video_frames), step))[:10]

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

    # 1. Detect Ball
    inputs = processor(images=pil_img, text="ball", return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # target_sizes needs to be (height, width)
    # PIL image size is (width, height)
    target_sizes = [pil_img.size[::-1]] 
    res_ball = processor.post_process_instance_segmentation(
        outputs, threshold=0.0, mask_threshold=0.5, target_sizes=target_sizes
    )[0]
    
    if len(res_ball["masks"]) > 0:
        # red for ball
        vis_img = overlay_masks(vis_img, res_ball["masks"], color=(255, 0, 0)) 
        
    # 2. Detect Racket
    inputs = processor(images=pil_img, text="racket", return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    with torch.no_grad():
        outputs = model(**inputs)

    res_racket = processor.post_process_instance_segmentation(
        outputs, threshold=0.0, mask_threshold=0.5, target_sizes=target_sizes
    )[0]

    if len(res_racket["masks"]) > 0:
        # blue for racket
        vis_img = overlay_masks(vis_img, res_racket["masks"], color=(0, 0, 255))
        
    out_path = os.path.join(OUT_DIR, f"frame_{idx:04d}.jpg")
    vis_img.save(out_path)
    
    b_scores = [round(s, 3) for s in res_ball["scores"].tolist()] if len(res_ball["scores"]) > 0 else []
    r_scores = [round(s, 3) for s in res_racket["scores"].tolist()] if len(res_racket["scores"]) > 0 else []
    print(f"  -> Saved {out_path}")
    print(f"     Ball scores: {b_scores}")
    print(f"     Racket scores: {r_scores}")
    
print("All frames processed successfully!")
