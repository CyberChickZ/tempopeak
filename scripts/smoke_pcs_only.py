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

# Use a specific clear frame where racket and ball might be visible
frame_idx = 45 # Assuming frame 45 is a mid-swing frame
frame = video_frames[frame_idx]

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

# Handle both np.ndarray and PIL inputs returned by load_video
if isinstance(frame, np.ndarray):
    pil_img = Image.fromarray(frame)
else:
    pil_img = frame

# Extensively test alternative prompts
PROMPTS = [
    "ball", "tennis ball", "yellow tennis ball", "sphere", "round object", "sports ball",
    "racket", "tennis racket", "strings", "racket handle", "sports equipment", "tennis player",
    "person", "man", "shoes", "net", "tennis net"
]

print(f"Testing {len(PROMPTS)} different prompts on frame {frame_idx} (Threshold 0.4)...")

for prompt_str in PROMPTS:
    vis_img = pil_img.copy()

    inputs = processor(images=pil_img, text=prompt_str, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = [pil_img.size[::-1]] 
    res = processor.post_process_instance_segmentation(
        outputs, threshold=0.4, mask_threshold=0.5, target_sizes=target_sizes
    )[0]
    
    scores = []
    if len(res["masks"]) > 0:
        vis_img = overlay_masks(vis_img, res["masks"], color=(0, 255, 0)) # Green for all test prompts
        scores = [round(s, 3) for s in res["scores"].tolist()]
        
    out_filename = prompt_str.replace(" ", "_").lower() + ".jpg"
    out_path = os.path.join(OUT_DIR, out_filename)
    vis_img.save(out_path)
    
    print(f"[{prompt_str}] -> {len(scores)} masks, Max Score: {max(scores) if scores else 0.0}")

print("All diagnostic prompt frames saved successfully!")
