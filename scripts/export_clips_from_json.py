#!/usr/bin/env python3
"""
Re-export clips from an existing Hit Annotator JSON annotation.

Usage:
    python scripts/export_clips_from_json.py \
        --json datasets/serve/00023.json \
        --video "datasets/v1/1(hit-clip-no person).mp4" \
        --outdir datasets/serve \
        --fps 30
"""
import argparse
import glob
import json
import os
import subprocess
import sys


def scan_max_num(outdir):
    max_num = 0
    for ext in ("*.json", "*.mp4"):
        for p in glob.glob(os.path.join(outdir, ext)):
            base = os.path.splitext(os.path.basename(p))[0]
            try:
                num = int(base)
                if num > max_num:
                    max_num = num
            except ValueError:
                pass
    return max_num


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to annotation JSON")
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    with open(args.json) as f:
        ann = json.load(f)

    cut_points = ann.get("cut_points", [])
    ignored_set = set(ann.get("ignored_segments", []))
    hit_frames = ann.get("HIT", [])

    boundaries = [0] + cut_points + [None]  # None = end of video (use total frames)
    # If no cut_points, just one segment = whole video
    # We need total frames to set the last boundary — derive from video duration
    video_path = args.video.strip()  # remove accidental trailing spaces

    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-show_entries", "format=duration",
         "-of", "json", video_path],
        capture_output=True, text=True
    )
    probe = json.loads(result.stdout)
    total_frames = None
    streams = probe.get("streams", [])
    if streams:
        nb = streams[0].get("nb_frames")
        if nb and nb != "N/A":
            total_frames = int(nb)
    if total_frames is None:
        fmt = probe.get("format", {})
        duration = float(fmt.get("duration", 0))
        if duration == 0:
            sys.exit(f"ERROR: could not read video duration. Check path:\n  {video_path}")
        total_frames = round(duration * args.fps)
    boundaries[-1] = total_frames

    os.makedirs(args.outdir, exist_ok=True)
    next_num = scan_max_num(args.outdir) + 1

    segments = []
    for i in range(len(boundaries) - 1):
        if i in ignored_set:
            continue
        start = boundaries[i]
        end = boundaries[i + 1]
        hits = [h - start for h in hit_frames if start <= h < end]
        segments.append({"start": start, "end": end, "hits": hits})

    print(f"Source: {args.video}")
    print(f"Total frames: {total_frames}, cut points: {len(cut_points)}, "
          f"segments: {len(boundaries)-1}, ignored: {len(ignored_set)}, "
          f"to export: {len(segments)}")
    print(f"Output dir: {args.outdir}, starting from: {next_num:05d}\n")

    for i, seg in enumerate(segments):
        num = next_num + i
        clip_name = f"{num:05d}.mp4"
        json_name = f"{num:05d}.json"
        clip_path = os.path.join(args.outdir, clip_name)
        json_path = os.path.join(args.outdir, json_name)

        start_sec = seg["start"] / args.fps
        duration_sec = (seg["end"] - seg["start"]) / args.fps

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i", video_path,
            "-t", str(duration_sec),
            "-c", "copy",
            clip_path
        ]
        print(f"[{i+1}/{len(segments)}] {clip_name} "
              f"(frames {seg['start']}–{seg['end']}, {len(seg['hits'])} HITs)", end=" ")
        sys.stdout.flush()
        subprocess.run(cmd, check=True, capture_output=True)

        with open(json_path, "w") as f:
            json.dump({"video": clip_name, "HIT": seg["hits"]}, f, indent=2)
        print("✓")

    print(f"\nDone. Exported {len(segments)} clips: "
          f"{next_num:05d} – {next_num+len(segments)-1:05d}")


if __name__ == "__main__":
    main()
