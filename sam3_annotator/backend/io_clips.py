"""
io_clips.py — ClipStore: manage clip state during annotation and final export.

Intermediate state lives in memory. Export writes only annotated clips
with sequential numbering (no gaps).
"""

import os
import json
import shutil

VALID_FPS = {15, 30, 60}


def get_validated_fps(cap) -> int:
    """Round raw fps to nearest int and reject if not in {15, 30, 60}."""
    raw = cap.get(5)  # cv2.CAP_PROP_FPS = 5
    rounded = round(raw)
    if rounded not in VALID_FPS:
        raise ValueError(f"Unsupported fps: {raw}, rounded={rounded}. Only {VALID_FPS} allowed.")
    return rounded


class ClipStore:
    def __init__(self):
        self._clips: dict[str, dict] = {}
        self._rejected: set[str] = set()
        self._insert_order: list[str] = []

    def reset(self):
        self._clips.clear()
        self._rejected.clear()
        self._insert_order.clear()

    def add_clip(self, clip_id: str, path: str, num_frames: int) -> None:
        self._clips[clip_id] = {
            "path": path,
            "num_frames": num_frames,
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
        return True

    def reject(self, clip_id: str) -> None:
        if clip_id in self._clips:
            self._clips[clip_id]["annotated"] = False
            self._clips[clip_id]["hit_frame"] = None
        self._rejected.add(clip_id)

    def undo(self, clip_id: str) -> None:
        if clip_id in self._clips:
            self._clips[clip_id]["annotated"] = False
            self._clips[clip_id]["hit_frame"] = None
        self._rejected.discard(clip_id)

    def get_clip_path(self, clip_id: str) -> str | None:
        if clip_id in self._clips:
            return self._clips[clip_id]["path"]
        return None

    def get_all(self) -> list[dict]:
        """Return non-rejected clips in insertion order for the frontend."""
        result = []
        for cid in self._insert_order:
            if cid in self._rejected:
                continue
            info = self._clips[cid]
            result.append({
                "clip_id": cid,
                "num_frames": info["num_frames"],
                "annotated": info["annotated"],
                "hit_frame": info["hit_frame"],
            })
        return result

    def export(self, out_dir: str) -> dict:
        """Export only annotated clips with sequential numbering."""
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
            shutil.copy2(info["path"], dst)
            annotations[new_name] = info["hit_frame"]

        out_json = os.path.join(out_dir, "annotations.json")
        with open(out_json, "w") as f:
            json.dump(annotations, f, indent=2)

        return {"exported": len(annotations), "out_dir": out_dir}
