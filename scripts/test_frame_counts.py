import cv2
import json

path = "/Users/harryzhang/git/tempopeak/datasets/serve/00001.mp4"
cap = cv2.VideoCapture(path)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

d = json.load(open("/Users/harryzhang/git/tempopeak/outputs/sam3_mask_extractor/00001.json"))
print(f"Video {path}: {n_frames} frames at {fps} fps")
print(f"JSON has {len(d)} frames. Max key: {max([int(k) for k in d.keys()])}")
