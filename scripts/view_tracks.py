import cv2
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="/nfs/hpc/share/zhanhaoc/hpe/tempopeak/datasets/serve/00001.mp4")
    parser.add_argument("--tracks", type=str, default="/nfs/hpc/share/zhanhaoc/hpe/tempopeak/outputs/smoke_sam3_tracks.json")
    parser.add_argument("--out", type=str, default="/nfs/hpc/share/zhanhaoc/hpe/tempopeak/outputs/smoke_sam3_tracks_vis.mp4")
    args = parser.parse_args()

    with open(args.tracks, "r") as f:
        tracks = json.load(f)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Using mp4v codec for mp4 output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        idx_str = str(frame_idx)
        if idx_str in tracks:
            pts = tracks[idx_str]
            ball = pts.get("ball")
            racket = pts.get("racket")

            if ball is not None:
                bx, by = int(ball[0]), int(ball[1])
                cv2.circle(frame, (bx, by), 8, (0, 0, 255), -1) # Red for ball
                cv2.putText(frame, "Ball", (bx+10, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            if racket is not None:
                rx, ry = int(racket[0]), int(racket[1])
                cv2.circle(frame, (rx, ry), 8, (255, 0, 0), -1) # Blue for racket
                cv2.putText(frame, "Racket", (rx+10, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        cv2.putText(frame, f"Frame: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved JSON coordinates visualization to {args.out}")

if __name__ == "__main__":
    main()
