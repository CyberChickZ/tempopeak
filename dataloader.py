"""ClipDataset — reads Person Labeler exports or raw clip data for training."""

import json
import os
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ImageNet normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ClipDataset(Dataset):
    """Dataset of tennis clips with hit-frame labels.

    Supports two data formats:
      1. Person Labeler export: {dir}/annot.json + frames/*.jpg
      2. Raw: {dir}/{name}.json + {name}.mp4
    """

    def __init__(self, data_dir: str, t_max: int = 32, img_size: int = 224):
        self.t_max = t_max
        self.img_size = img_size

        # Build index: list of (clip_dir_or_json_path, hit_frame, total_frames, format)
        self.samples: list[dict] = []
        self._scan(data_dir)
        if not self.samples:
            raise RuntimeError(f"No clips found in {data_dir}")

    def _scan(self, root: str) -> None:
        """Recursively find all clips and build (clip, hit) index."""
        root = Path(root)

        # Format 1: annot.json exports
        for annot_path in sorted(root.rglob("annot.json")):
            self._index_annot(annot_path)

        # Format 2: raw {name}.json + {name}.mp4 pairs
        for json_path in sorted(root.rglob("*.json")):
            if json_path.name == "annot.json" or json_path.name.endswith("_det.json"):
                continue
            mp4_path = json_path.with_suffix(".mp4")
            if mp4_path.exists():
                self._index_raw(json_path, mp4_path)

    def _index_annot(self, annot_path: Path) -> None:
        """Index a Person Labeler export (annot.json)."""
        with open(annot_path) as f:
            meta = json.load(f)
        clip_dir = annot_path.parent
        hits = meta.get("hits", [])
        total = meta.get("total_frames")
        if not total:
            # Try to infer from frames/ dir or companion mp4
            mp4 = self._find_mp4(clip_dir, meta.get("video", ""))
            total = self._get_frame_count(mp4) if mp4 else None
        if not total or not hits:
            return
        for hit in hits:
            self.samples.append({
                "clip_dir": str(clip_dir),
                "hit": int(hit),
                "total_frames": int(total),
                "format": "annot",
                "annot": meta,
                "mp4": str(self._find_mp4(clip_dir, meta.get("video", "")) or ""),
            })

    def _index_raw(self, json_path: Path, mp4_path: Path) -> None:
        """Index a raw clip ({name}.json + {name}.mp4)."""
        with open(json_path) as f:
            meta = json.load(f)
        hits = meta.get("hits", [])
        total = meta.get("total_frames")
        if not total:
            total = self._get_frame_count(str(mp4_path))
        if not total or not hits:
            return
        for hit in hits:
            self.samples.append({
                "clip_dir": str(json_path.parent),
                "hit": int(hit),
                "total_frames": int(total),
                "format": "raw",
                "meta": meta,
                "mp4": str(mp4_path),
                "json_path": str(json_path),
            })

    @staticmethod
    def _find_mp4(clip_dir: Path, video_name: str) -> Optional[str]:
        """Find companion mp4 file."""
        if video_name:
            p = clip_dir / video_name
            if p.exists():
                return str(p)
        for ext in ("*.mp4", "*.MP4"):
            found = list(clip_dir.glob(ext))
            if found:
                return str(found[0])
        return None

    @staticmethod
    def _get_frame_count(mp4_path: str) -> Optional[int]:
        """Get total frame count from an mp4."""
        if not mp4_path or not os.path.exists(mp4_path):
            return None
        cap = cv2.VideoCapture(mp4_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n if n > 0 else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        hit = sample["hit"]
        total = sample["total_frames"]
        all_hits = self._get_all_hits(sample)

        # Pick random window of t_max containing the target hit
        start = self._pick_window(hit, total, all_hits)
        end = start + self.t_max  # exclusive

        # Load frames
        frames = self._load_frames(sample, start, end)

        # hit position within the window
        t_gt = hit - start

        meta = {
            "clip_dir": sample["clip_dir"],
            "hit": hit,
            "start": start,
            "total_frames": total,
        }
        return {"frames": frames, "t_gt": t_gt, "meta": meta}

    def _get_all_hits(self, sample: dict) -> list[int]:
        """Get all hits for this clip."""
        if sample["format"] == "annot":
            return sample["annot"].get("hits", [])
        return sample["meta"].get("hits", [])

    def _pick_window(self, hit: int, total: int, all_hits: list[int]) -> int:
        """Pick a random window start so hit is in [start, start+t_max-1].

        Hard constraint: window boundaries must be >= 3 frames from any OTHER hit.
        """
        other_hits = [h for h in all_hits if h != hit]

        # hit must appear in window → start ∈ [hit - t_max + 1, hit]
        lo = max(0, hit - self.t_max + 1)
        hi = min(hit, total - self.t_max)
        hi = max(lo, hi)  # ensure lo <= hi

        # Filter: window [s, s+t_max-1] must be >=3 frames from other hits
        valid = []
        for s in range(lo, hi + 1):
            win_start, win_end = s, s + self.t_max - 1
            ok = True
            for oh in other_hits:
                if win_start <= oh <= win_end:
                    # Other hit is inside window — check distance from edges
                    if oh - win_start < 3 or win_end - oh < 3:
                        ok = False
                        break
            if ok:
                valid.append(s)

        if valid:
            return random.choice(valid)
        # Fallback: just center the hit
        return max(0, min(hit - self.t_max // 2, total - self.t_max))

    def _load_frames(self, sample: dict, start: int, end: int) -> torch.Tensor:
        """Load and preprocess frames [start, end) → [T, 3, 224, 224]."""
        total = sample["total_frames"]
        frames = []

        if sample["format"] == "annot":
            frames = self._load_annot_frames(sample, start, end, total)
        else:
            frames = self._load_mp4_frames(sample["mp4"], start, end, total)

        # Stack → [T, H, W, 3] uint8
        arr = np.stack(frames, axis=0)

        # Resize, normalise, to tensor
        return self._preprocess(arr)

    def _load_annot_frames(self, sample: dict, start: int, end: int,
                           total: int) -> list[np.ndarray]:
        """Load from frames/ JPEG dir, fill gaps from mp4."""
        frames_dir = Path(sample["clip_dir"]) / "frames"
        mp4 = sample["mp4"]
        result = []

        for i in range(start, end):
            fi = max(0, min(i, total - 1))  # clamp for edge padding
            jpg = frames_dir / f"{fi}.jpg"
            if jpg.exists():
                img = cv2.imread(str(jpg))
                if img is not None:
                    result.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    continue
            # Fallback to mp4
            if mp4:
                result.append(self._read_single_frame(mp4, fi))
            else:
                # Black frame as last resort
                result.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
        return result

    def _load_mp4_frames(self, mp4_path: str, start: int, end: int,
                         total: int) -> list[np.ndarray]:
        """Load consecutive frames from mp4."""
        cap = cv2.VideoCapture(mp4_path)
        result = []

        # Seek to start (clamped)
        seek_start = max(0, start)
        cap.set(cv2.CAP_PROP_POS_FRAMES, seek_start)

        for i in range(start, end):
            fi = max(0, min(i, total - 1))
            if fi < seek_start or fi >= total:
                # Edge padding: duplicate edge frame
                if result:
                    result.append(result[-1].copy())
                else:
                    result.append(np.zeros((240, 320, 3), dtype=np.uint8))
                continue

            ret, frame = cap.read()
            if ret:
                result.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            elif result:
                result.append(result[-1].copy())
            else:
                result.append(np.zeros((240, 320, 3), dtype=np.uint8))

        cap.release()
        return result

    @staticmethod
    def _read_single_frame(mp4_path: str, frame_idx: int) -> np.ndarray:
        """Read a single frame from mp4."""
        cap = cv2.VideoCapture(mp4_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.zeros((240, 320, 3), dtype=np.uint8)

    def _preprocess(self, frames: np.ndarray) -> torch.Tensor:
        """Resize to 224x224, normalise with ImageNet stats. [T,H,W,3] → [T,3,224,224]."""
        T = frames.shape[0]
        out = np.zeros((T, self.img_size, self.img_size, 3), dtype=np.float32)
        for t in range(T):
            out[t] = cv2.resize(frames[t], (self.img_size, self.img_size)).astype(np.float32)

        # [T, H, W, 3] → [T, 3, H, W], scale to [0,1], normalise
        out = out / 255.0
        out = (out - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        tensor = torch.from_numpy(out).permute(0, 3, 1, 2).float()  # [T, 3, 224, 224]
        return tensor


def build_dataloaders(data_dir: str, t_max: int, batch_size: int,
                      seed: int = 42) -> tuple:
    """Build train/val DataLoaders with 80/20 split by clip name."""
    full_ds = ClipDataset(data_dir, t_max=t_max)

    # Get unique clip dirs, sorted for deterministic split
    clip_dirs = sorted(set(s["clip_dir"] for s in full_ds.samples))
    n_train = int(len(clip_dirs) * 0.8)
    train_dirs = set(clip_dirs[:n_train])
    val_dirs = set(clip_dirs[n_train:])

    train_indices = [i for i, s in enumerate(full_ds.samples) if s["clip_dir"] in train_dirs]
    val_indices = [i for i, s in enumerate(full_ds.samples) if s["clip_dir"] in val_dirs]

    train_ds = torch.utils.data.Subset(full_ds, train_indices)
    val_ds = torch.utils.data.Subset(full_ds, val_indices)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, generator=g,
        collate_fn=_collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )
    return train_loader, val_loader


def _collate(batch: list[dict]) -> dict:
    """Custom collate: stack frames and t_gt, keep meta as list."""
    frames = torch.stack([b["frames"] for b in batch])  # [B, T, 3, H, W]
    t_gt = torch.tensor([b["t_gt"] for b in batch], dtype=torch.long)  # [B]
    meta = [b["meta"] for b in batch]
    return {"frames": frames, "t_gt": t_gt, "meta": meta}
