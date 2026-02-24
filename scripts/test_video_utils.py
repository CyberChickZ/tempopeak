from transformers.video_utils import load_video
import cv2

path = "/Users/harryzhang/git/tempopeak/datasets/serve/00001.mp4"
video_frames, _ = load_video(path)
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Native: {n_frames} frames at {fps} fps")
print(f"load_video: {len(video_frames)} frames")
