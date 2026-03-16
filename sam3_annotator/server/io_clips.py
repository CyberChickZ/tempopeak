"""
io_clips.py — ClipStore: manage clip state during annotation and final export.

Clips are stored as JPEG frames in data/clips/{clip_id}/.
Annotation state is persisted to data/state.json as {clip_id: hit_frame}.
"""

import os
import json

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CLIPS_DIR = os.path.join(_PROJECT_ROOT, "data", "clips")
STATE_PATH = os.path.join(_PROJECT_ROOT, "data", "state.json")


class ClipStore:
    def __init__(self):
        self._clips: dict[str, dict] = {}
        self._rejected: set[str] = set()
        self._insert_order: list[str] = []

    def reset(self):
        self._clips.clear()
        self._rejected.clear()
        self._insert_order.clear()

    def scan_existing(self) -> None:
        """Scan CLIPS_DIR for existing clip subdirs and restore annotation state."""
        if not os.path.isdir(CLIPS_DIR):
            return
        clip_ids = sorted(
            d for d in os.listdir(CLIPS_DIR)
            if os.path.isdir(os.path.join(CLIPS_DIR, d))
        )
        for clip_id in clip_ids:
            meta_path = os.path.join(CLIPS_DIR, clip_id, "meta.json")
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                continue
            self._clips[clip_id] = {
                "path": os.path.join(CLIPS_DIR, clip_id),
                "num_frames": meta["num_frames"],
                "fps": meta.get("fps", 30),
                "peak_frame": meta.get("peak_frame", 0),
                "hit_frame": None,
                "annotated": False,
            }
            self._insert_order.append(clip_id)
        self._load_state()

    def add_clip(self, clip_id: str, num_frames: int, fps: int, peak_frame: int = 0) -> None:
        self._clips[clip_id] = {
            "path": os.path.join(CLIPS_DIR, clip_id),
            "num_frames": num_frames,
            "fps": fps,
            "peak_frame": peak_frame,
            "hit_frame": None,
            "annotated": False,
        }
        self._insert_order.append(clip_id)

    def annotate(self, clip_id: str, hit_frame: int) -> bool:
        if clip_id not in self._clips:
            return False
        info = self._clips[clip_id]
        if not (0 <= hit_frame < info["num_frames"]):
            return False
        info["hit_frame"] = hit_frame
        info["annotated"] = True
        self._rejected.discard(clip_id)
        self._save_state()
        return True

    def reject(self, clip_id: str) -> None:
        if clip_id in self._clips:
            self._clips[clip_id]["annotated"] = False
            self._clips[clip_id]["hit_frame"] = None
        self._rejected.add(clip_id)
        self._save_state()

    def undo(self, clip_id: str) -> None:
        if clip_id in self._clips:
            self._clips[clip_id]["annotated"] = False
            self._clips[clip_id]["hit_frame"] = None
        self._rejected.discard(clip_id)
        self._save_state()

    def get_list(self) -> list[dict]:
        return [
            {
                "clip_id": cid,
                "num_frames": self._clips[cid]["num_frames"],
                "annotated": self._clips[cid]["annotated"],
                "hit_frame": self._clips[cid]["hit_frame"],
                "peak_frame": self._clips[cid].get("peak_frame", 0),
            }
            for cid in self._insert_order
            if cid not in self._rejected
        ]

    def get_clip_path(self, clip_id: str) -> str | None:
        if clip_id in self._clips:
            return self._clips[clip_id]["path"]
        return None

    def export(self, out_dir: str) -> dict:
        import cv2
        os.makedirs(out_dir, exist_ok=True)

        annotated = [
            (cid, self._clips[cid])
            for cid in self._insert_order
            if cid not in self._rejected and self._clips[cid]["annotated"]
        ]

        annotations = {}
        for new_idx, (old_id, info) in enumerate(annotated, start=1):
            new_name = f"clip_{new_idx:04d}"
            dst = os.path.join(out_dir, f"{new_name}.mp4")

            meta_path = os.path.join(CLIPS_DIR, old_id, "meta.json")
            with open(meta_path) as f:
                meta = json.load(f)

            source_video = meta["source_video"]
            fps = meta["fps"]
            export_pad = meta.get("export_pad", 32)
            peak_frame_rel = meta.get("peak_frame", 0)

            cap = cv2.VideoCapture(source_video)
            total_source = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Export window centered on peak frame
            abs_peak = meta["start_frame"] + peak_frame_rel
            export_start = max(0, abs_peak - export_pad)
            export_end = min(total_source - 1, abs_peak + export_pad)
            export_num = export_end - export_start + 1

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(dst, fourcc, fps, (W, H))
            cap.set(cv2.CAP_PROP_POS_FRAMES, export_start)
            for _ in range(export_num):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
            writer.release()
            cap.release()

            # hit_frame relative to export window
            abs_hit = meta["start_frame"] + info["hit_frame"]
            annotations[new_name] = max(0, abs_hit - export_start)

        out_json = os.path.join(out_dir, "annotations.json")
        with open(out_json, "w") as f:
            json.dump(annotations, f, indent=2)

        return {"exported": len(annotations), "out_dir": out_dir}

    def _load_state(self) -> None:
        if not os.path.exists(STATE_PATH):
            return
        try:
            with open(STATE_PATH) as f:
                state = json.load(f)
            for clip_id, hit_frame in state.items():
                if clip_id in self._clips:
                    self._clips[clip_id]["hit_frame"] = hit_frame
                    self._clips[clip_id]["annotated"] = True
        except Exception:
            pass

    def _save_state(self) -> None:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        state = {
            cid: self._clips[cid]["hit_frame"]
            for cid in self._insert_order
            if cid not in self._rejected and self._clips[cid]["annotated"]
        }
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
