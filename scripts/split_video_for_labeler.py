"""Split a long video into segments for the HIT Annotator.

No YOLO needed. Just ffmpeg to cut .mp4 segments + JSON stubs.

Usage:
  # Split 40-min match into 90s segments, with existing detections for review
  python scripts/split_video_for_labeler.py \
      --video data/00001.mp4 \
      --hit_json data/00001_hits.json \
      --out_dir data/labeler_workdir \
      --segment_sec 90

  # Just split, no pre-existing hits
  python scripts/split_video_for_labeler.py \
      --video data/00001.mp4 \
      --out_dir data/labeler_workdir

Then open HIT Annotator → set workdir to data/labeler_workdir.
"""

import argparse
import json
import math
import os
import subprocess
import sys


def get_video_info(video_path):
    """Get fps and total frames via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Fallback to opencv
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return fps, total

    info = json.loads(result.stdout)
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            fps_str = s.get("r_frame_rate", "30/1")
            num, den = map(int, fps_str.split("/"))
            fps = num / den
            total = int(s.get("nb_frames", 0))
            if total == 0:
                # Try duration * fps
                dur = float(s.get("duration", 0))
                total = int(dur * fps)
            return fps, total
    raise RuntimeError(f"No video stream found in {video_path}")


def split_video(video_path, out_dir, segment_sec, hit_frames=None):
    """Split video into segments and create JSON annotation stubs.

    Args:
        video_path: path to input .mp4
        out_dir: output directory (workdir for HIT Annotator)
        segment_sec: segment duration in seconds
        hit_frames: list of global frame indices (existing detections)
    """
    os.makedirs(out_dir, exist_ok=True)

    fps, total_frames = get_video_info(video_path)
    segment_frames = int(segment_sec * fps)
    n_segments = math.ceil(total_frames / segment_frames)

    if hit_frames is None:
        hit_frames = []
    hit_frames = sorted(hit_frames)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Video: {video_path}")
    print(f"  FPS: {fps}, Total frames: {total_frames}")
    print(f"  Segment: {segment_sec}s = {segment_frames} frames")
    print(f"  Segments: {n_segments}")
    print(f"  Hit frames: {len(hit_frames)}")
    print()

    for i in range(n_segments):
        start_frame = i * segment_frames
        start_sec = start_frame / fps
        end_frame = min((i + 1) * segment_frames, total_frames)
        duration_sec = (end_frame - start_frame) / fps

        seg_name = f"{video_name}_seg{i:03d}"
        mp4_out = os.path.join(out_dir, f"{seg_name}.mp4")
        json_out = os.path.join(out_dir, f"{seg_name}.json")

        # Map global hit frames to segment-local indices
        local_hits = [h - start_frame for h in hit_frames
                      if start_frame <= h < end_frame]

        # Cut segment with ffmpeg (copy codec = fast, no re-encode)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "warning",
            "-ss", f"{start_sec:.3f}",
            "-i", video_path,
            "-t", f"{duration_sec:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            mp4_out,
        ]
        subprocess.run(cmd, check=True)

        # Create JSON stub
        data = {
            "video": os.path.basename(video_path),
            "segment": i,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "fps": fps,
            "HIT": [],              # confirmed hits (empty, user fills in)
            "notest_HIT": local_hits,  # detections to review
        }
        with open(json_out, "w") as f:
            json.dump(data, f, indent=2)

        n_hits = len(local_hits)
        hit_str = f" ({n_hits} detections to review)" if n_hits else ""
        print(f"  [{i+1}/{n_segments}] {seg_name}.mp4  "
              f"frames {start_frame}-{end_frame}{hit_str}")

    print(f"\nDone. Workdir: {out_dir}")
    print(f"Open HIT Annotator → set workdir to: {os.path.abspath(out_dir)}")


def main():
    p = argparse.ArgumentParser(
        description="Split video into segments for HIT Annotator (no YOLO)")
    p.add_argument("--video", required=True,
                   help="Input video path (.mp4)")
    p.add_argument("--out_dir", required=True,
                   help="Output workdir for HIT Annotator")
    p.add_argument("--segment_sec", type=float, default=90,
                   help="Segment duration in seconds (default: 90)")
    p.add_argument("--hit_json", type=str, default=None,
                   help="JSON with HIT/hits array of global frame indices "
                        "(pre-existing detections to review)")
    args = p.parse_args()

    # Load existing hit frames if provided
    hit_frames = None
    if args.hit_json:
        with open(args.hit_json) as f:
            data = json.load(f)
        hit_frames = data.get("HIT", data.get("hits", []))
        print(f"Loaded {len(hit_frames)} hit frames from {args.hit_json}")

    split_video(args.video, args.out_dir, args.segment_sec, hit_frames)


if __name__ == "__main__":
    main()
