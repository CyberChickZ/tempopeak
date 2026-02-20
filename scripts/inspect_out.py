import torch
from accelerate import Accelerator
from transformers.video_utils import load_video
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

device = Accelerator().device
HF_LOCAL_MODEL = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/models/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7"
VIDEO_PATH = "/nfs/hpc/share/zhanhaoc/hpe/tempopeak/datasets/serve/00001.mp4"

video_frames, _ = load_video(VIDEO_PATH)
trk_model = Sam3TrackerVideoModel.from_pretrained(HF_LOCAL_MODEL, local_files_only=True).to(device, dtype=torch.bfloat16)
trk_proc = Sam3TrackerVideoProcessor.from_pretrained(HF_LOCAL_MODEL, local_files_only=True)

trk_session = trk_proc.init_video_session(video=video_frames, inference_device=device)

points = [[[[656.0, 577.0]]]]
labels = [[[1]]]
trk_proc.add_inputs_to_inference_session(
    inference_session=trk_session,
    frame_idx=0,
    obj_ids=1,
    input_points=points,
    input_labels=labels,
)

out = trk_model(inference_session=trk_session, frame_idx=0)
print("Keys in output:", dir(out))
if hasattr(out, 'iou_predictions'):
    print("iou_predictions:", out.iou_predictions)
if hasattr(out, 'scores'):
    print("scores:", out.scores)
if hasattr(out, 'object_scores'):
    print("object_scores:", out.object_scores)
