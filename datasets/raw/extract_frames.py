import cv2
import os

video_path = "Serve-Compilation-Slow-Motion-Alcaraz-Dj.mp4"
output_dir = "Serve-Compilation-Slow-Motion-Alcaraz-Dj_frames"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if total_frames < 200:
    print(f"Video has fewer than 200 frames ({total_frames}).")
    step = 1
    num_frames = total_frames
else:
    step = total_frames / 200
    num_frames = 200

print(f"Total frames: {total_frames}, Extracting {num_frames} frames.")

count = 0
for i in range(num_frames):
    frame_idx = int(i * step)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        out_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(out_path, frame)
        count += 1
    else:
        print(f"Failed to read frame at {frame_idx}")

cap.release()
print(f"Successfully extracted {count} frames to {output_dir}")
