import os
import torch
import numpy as np
from PIL import Image
from transformers.video_utils import load_video
from transformers import Sam3VideoModel, Sam3VideoProcessor
from accelerate import Accelerator

HF_LOCAL_MODEL = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/models/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7"
VIDEO_PATH = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/datasets/serve/00001.mp4"
OUT_DIR = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/outputs/sam3_video_ball_track"

os.makedirs(OUT_DIR, exist_ok=True)

acc = Accelerator()
device = acc.device
dtype = torch.bfloat16

print("Loading Sam3VideoModel...")
model = Sam3VideoModel.from_pretrained(HF_LOCAL_MODEL, local_files_only=True).to(device, dtype=dtype)
processor = Sam3VideoProcessor.from_pretrained(HF_LOCAL_MODEL, local_files_only=True)

print("Loading video frames...")
video_frames, _ = load_video(VIDEO_PATH)
print("num_frames:", len(video_frames))

print("Init video session...")
inference_session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    processing_device="cpu",
    video_storage_device="cpu",
    dtype=dtype,
)

print("Add text prompt: ball")
inference_session = processor.add_text_prompt(
    inference_session=inference_session,
    text="ball",
)

def overlay_masks(image_pil, masks, color):
    image = image_pil.convert("RGBA")
    m = masks.cpu().numpy()
    for mask in m:
        mask_bool = mask > 0.0
        mask_img = Image.fromarray((mask_bool * 255).astype(np.uint8))
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask_img.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image.convert("RGB")

print("Propagating through video...")
meta_path = os.path.join(OUT_DIR, "per_frame_meta.txt")
with open(meta_path, "w") as f:
    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session,
        max_frame_num_to_track=len(video_frames),
    ):
        frame_idx = model_outputs.frame_idx
        processed = processor.postprocess_outputs(inference_session, model_outputs)

        # processed fields (per doc): object_ids, scores, boxes, masks
        obj_ids = processed["object_ids"].tolist() if "object_ids" in processed else []
        scores = processed["scores"].tolist() if "scores" in processed else []
        boxes = processed["boxes"].cpu().tolist() if "boxes" in processed else []
        masks = processed["masks"]  # torch tensor, [N, H, W]

        f.write(f"frame {frame_idx}  n={len(obj_ids)}  obj_ids={obj_ids}  scores={[round(x,3) for x in scores]}\n")

        # save a visualization every frame (adjust later if too slow)
        frame = video_frames[frame_idx]
        pil_img = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
        vis = pil_img.copy()
        if masks is not None and masks.numel() > 0:
            vis = overlay_masks(vis, masks, color=(255, 0, 0))
        vis.save(os.path.join(OUT_DIR, f"frame_{frame_idx:05d}.jpg"))

print("Done. meta:", meta_path)
