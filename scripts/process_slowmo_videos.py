import os
import subprocess
from pathlib import Path

def process_videos(input_dir, output_dir, speed_up_factor=4.0, target_fps=60):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() == '.mp4']

    for video_file in sorted(video_files):
        out_file = output_path / video_file.name
        
        # pts multiplier: e.g. for 4x speed, setpts=0.25*PTS
        pts_multiplier = 1.0 / speed_up_factor

        print(f"Processing: {video_file.name} -> {out_file.name}")
        
        cmd = [
            "ffmpeg",
            "-y", # Overwrite if exists
            "-i", str(video_file),
            "-filter:v", f"setpts={pts_multiplier}*PTS",
            "-r", str(target_fps),
            "-an", # No audio
            str(out_file)
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  Done: {out_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"  Error processing {video_file.name}: {e}")

if __name__ == "__main__":
    input_directory = "/Users/harryzhang/git/tempopeak/data/[clip]Serve-Compilation-Slow-Motion-Alcaraz-Dj"
    output_directory = "/Users/harryzhang/git/tempopeak/data/[clip]Serve-Compilation-Slow-Motion-Alcaraz-Dj_normal"
    
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    print("Starting batch video processing...\n")
    
    process_videos(input_directory, output_directory, speed_up_factor=4.0, target_fps=60)
    print("\nAll processing complete.")
