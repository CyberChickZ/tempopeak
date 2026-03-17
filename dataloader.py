"""ClipDataset v3 — enumerate all valid (start, n) windows at build time.

Supports two loading modes:
  1. Image mode: load JPEGs, bbox crop, resize 448, return [T, 3, 448, 448]
  2. Feature mode: load pre-extracted .pt tensors, return [T, D] (e.g. D=512)
     Feature mode is auto-detected when features_p*/ dirs exist alongside frames/.
"""

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
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --- Core constants ---
HIT_MARGIN = 4              # exclude ±4 frames around neighbouring hits
MIN_LEN = 8                 # minimum window length
MAX_WINDOWS_PER_HIT = 50    # train: max windows sampled per hit
IMG_SIZE = 448               # crop resize target
BBOX_PAD = 1.5               # longest side multiplier for square crop


def _enumerate_windows(hit: int, lo: int, hi: int,
                       t_max: int, min_len: int) -> list[tuple[int, int]]:
    """Enumerate all valid (start, n) pairs for a hit within [lo, hi]."""
    windows = []
    available = hi - lo + 1
    for n in range(min_len, min(t_max, available) + 1):
        start_lo = max(lo, hit - n + 1)
        start_hi = min(hit, hi - n + 1)
        for start in range(start_lo, start_hi + 1):
            windows.append((start, n))
    return windows


class ClipDataset(Dataset):
    """Dataset of tennis clips with hit-frame labels and person-crop augmentation."""

    def __init__(self, data_dir: str, t_max: int = 32,
                 is_val: bool = False,
                 use_features: bool = True, backbone: str = "resnet18",
                 preload: bool = True):
        self.t_max = t_max
        self.is_val = is_val
        self.use_features = use_features  # auto-downgrade if features/ not found
        self.backbone = backbone

        # Build index: list of clip metadata dicts
        self.clips: list[dict] = []
        self._scan(data_dir)
        if not self.clips:
            raise RuntimeError(f"No clips found in {data_dir}")

        # Preload all features into RAM
        self._feature_cache: dict[str, torch.Tensor] = {}
        if preload and self.use_features:
            self._preload_features()

        # Expand to (clip_idx, hit_idx, start, n) samples
        self.samples: list[tuple[int, int, int, int]] = []
        self._build_samples()

    def _scan(self, root: str) -> None:
        """Recursively find all clips."""
        root = Path(root)

        # Format 1: annot.json exports
        for annot_path in sorted(root.rglob("annot.json")):
            clip = self._parse_annot(annot_path)
            if clip:
                self.clips.append(clip)

        # Format 2: raw {name}.json + {name}.mp4 pairs
        for json_path in sorted(root.rglob("*.json")):
            if json_path.name == "annot.json" or json_path.name.endswith("_det.json"):
                continue
            mp4_path = json_path.with_suffix(".mp4")
            if mp4_path.exists():
                clip = self._parse_raw(json_path, mp4_path)
                if clip:
                    self.clips.append(clip)

    def _parse_annot(self, annot_path: Path) -> Optional[dict]:
        """Parse a Person Labeler export (annot.json)."""
        with open(annot_path) as f:
            meta = json.load(f)
        clip_dir = annot_path.parent
        hits = meta.get("hits", [])
        hitters = meta.get("hitters", [0] * len(hits))
        total = meta.get("total_frames")
        if not total:
            mp4 = self._find_mp4(clip_dir, meta.get("video", ""))
            total = self._get_frame_count(mp4) if mp4 else None
        if not total or not hits:
            return None

        # Compute valid intervals for each hit
        intervals = self._compute_intervals(hits, total)

        return {
            "clip_dir": str(clip_dir),
            "hits": hits,
            "hitters": hitters,
            "total_frames": total,
            "intervals": intervals,
            "format": "annot",
            "frames_dict": meta.get("frames", {}),
            "mp4": str(self._find_mp4(clip_dir, meta.get("video", "")) or ""),
            "clip_id": str(annot_path),
        }

    def _parse_raw(self, json_path: Path, mp4_path: Path) -> Optional[dict]:
        """Parse a raw clip ({name}.json + {name}.mp4)."""
        with open(json_path) as f:
            meta = json.load(f)
        hits = meta.get("hits", [])
        hitters = meta.get("hitters", [0] * len(hits))
        total = meta.get("total_frames")
        if not total:
            total = self._get_frame_count(str(mp4_path))
        if not total or not hits:
            return None

        intervals = self._compute_intervals(hits, total)

        return {
            "clip_dir": str(json_path.parent),
            "hits": hits,
            "hitters": hitters,
            "total_frames": total,
            "intervals": intervals,
            "format": "raw",
            "frames_dict": meta.get("frames", {}),
            "mp4": str(mp4_path),
            "clip_id": str(json_path),
        }

    @staticmethod
    def _compute_intervals(hits: list[int], total: int) -> list[tuple[int, int]]:
        """Compute valid [lo, hi] interval for each hit (Section 3.2)."""
        intervals = []
        for i, h in enumerate(hits):
            lo = 0 if i == 0 else hits[i - 1] + HIT_MARGIN + 1
            hi = (total - 1) if i == len(hits) - 1 else hits[i + 1] - HIT_MARGIN - 1
            lo = max(0, lo)
            hi = min(total - 1, hi)
            intervals.append((lo, hi))
        return intervals

    def _build_samples(self) -> None:
        """Enumerate all valid (start, n) windows for each hit."""
        for ci, clip in enumerate(self.clips):
            for hi_idx in range(len(clip["hits"])):
                hit = clip["hits"][hi_idx]
                lo, hi_bound = clip["intervals"][hi_idx]
                available = hi_bound - lo + 1
                if available < MIN_LEN:
                    continue

                windows = _enumerate_windows(hit, lo, hi_bound,
                                             self.t_max, MIN_LEN)
                if not windows:
                    continue

                if self.is_val:
                    # Pick the longest window with hit closest to center
                    best_n = max(w[1] for w in windows)
                    candidates = [w for w in windows if w[1] == best_n]
                    best = min(candidates,
                               key=lambda w: abs(w[0] + w[1] // 2 - hit))
                    self.samples.append((ci, hi_idx, best[0], best[1]))
                else:
                    selected = random.sample(
                        windows, min(len(windows), MAX_WINDOWS_PER_HIT))
                    for start, n in selected:
                        self.samples.append((ci, hi_idx, start, n))

    def _preload_features(self) -> None:
        """Load all .pt feature files into RAM at startup."""
        print("Preloading features into RAM...", flush=True)
        for clip in self.clips:
            for hitter in set(clip["hitters"]):
                player_key = f"p{hitter}" if hitter in (1, 2) else "p1"
                feat_dir = Path(clip["clip_dir"]) / "features" / self.backbone / player_key
                if not feat_dir.exists():
                    continue
                for pt_path in sorted(feat_dir.glob("*.pt")):
                    key = str(pt_path)
                    if key not in self._feature_cache:
                        self._feature_cache[key] = torch.load(pt_path, weights_only=True)
        print(f"Cached {len(self._feature_cache)} feature tensors.", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        ci, hi_idx, start, n = self.samples[idx]
        clip = self.clips[ci]
        hit = clip["hits"][hi_idx]
        hitter = clip["hitters"][hi_idx] if hi_idx < len(clip["hitters"]) else 0
        end = start + n

        # Load data (features or images)
        player_key = f"p{hitter}" if hitter in (1, 2) else "p1"
        feat_dir = Path(clip["clip_dir"]) / "features" / self.backbone / player_key

        if self.use_features and feat_dir.exists():
            data = self._load_features(clip, feat_dir, start, end)
            mode = "features"
        else:
            data = self._load_cropped_frames(clip, start, end, hit, hitter)
            mode = "images"

        # Pad to t_max
        valid_len = n
        if n < self.t_max:
            pad = data[-1:].expand(self.t_max - n, *data.shape[1:])
            data = torch.cat([data, pad], dim=0)

        t_gt = hit - start

        meta = {
            "clip_dir": clip["clip_dir"],
            "hit": hit,
            "hitter": hitter,
            "start": start,
            "valid_len": valid_len,
            "total_frames": clip["total_frames"],
            "mode": mode,
        }
        return {"frames": data, "t_gt": t_gt, "valid_len": valid_len, "meta": meta}

    def _load_features(self, clip: dict, feat_dir: Path,
                       start: int, end: int) -> torch.Tensor:
        """Load pre-extracted feature vectors [T, D] from cache or disk."""
        total = clip["total_frames"]
        feats = []
        for i in range(start, end):
            fi = max(0, min(i, total - 1))
            key = str(feat_dir / f"{fi}.pt")
            if key in self._feature_cache:
                feats.append(self._feature_cache[key])
            elif feats:
                feats.append(feats[-1].clone())
            else:
                raise FileNotFoundError(f"Missing feature: {key}")
        return torch.stack(feats)  # [T, D]

    def _load_cropped_frames(self, clip: dict, start: int, end: int,
                             hit: int, hitter: int) -> torch.Tensor:
        """Load frames, crop around hitter bbox, resize to IMG_SIZE."""
        total = clip["total_frames"]
        frames_dir = Path(clip["clip_dir"]) / "frames"
        frames_dict = clip["frames_dict"]
        mp4 = clip["mp4"]

        # Determine which player to crop (hitter: 1=p1, 2=p2, 0=fallback to p1)
        player_key = f"p{hitter}" if hitter in (1, 2) else "p1"

        result = []
        cap = None

        for i in range(start, end):
            fi = max(0, min(i, total - 1))

            # Load raw frame
            img = None
            jpg = frames_dir / f"{fi}.jpg"
            if clip["format"] == "annot" and jpg.exists():
                img = cv2.imread(str(jpg))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if img is None and mp4:
                if cap is None:
                    cap = cv2.VideoCapture(mp4)
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if ret:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if img is None:
                img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                result.append(img)
                continue

            # Get bbox and crop
            bbox = self._get_bbox(frames_dict, fi, player_key)
            crop = self._crop_square(img, bbox)
            result.append(crop)

        if cap is not None:
            cap.release()

        # Stack and preprocess
        arr = np.stack(result, axis=0)  # [T, IMG_SIZE, IMG_SIZE, 3]
        return self._preprocess(arr)

    @staticmethod
    def _get_bbox(frames_dict: dict, frame_idx: int,
                  player_key: str) -> Optional[dict]:
        """Get bbox for a frame, with nearest-frame fallback."""
        fi_str = str(frame_idx)
        frame_data = frames_dict.get(fi_str)

        if frame_data:
            bbox = frame_data.get(player_key)
            if bbox:
                return bbox
            # Fallback: try the other player
            other = "p2" if player_key == "p1" else "p1"
            bbox = frame_data.get(other)
            if bbox:
                return bbox

        # Nearest annotated frame fallback
        if not frames_dict:
            return None
        annotated = sorted(int(k) for k in frames_dict.keys() if k.isdigit())
        if not annotated:
            return None

        # Binary search for nearest
        import bisect
        pos = bisect.bisect_left(annotated, frame_idx)
        candidates = []
        if pos < len(annotated):
            candidates.append(annotated[pos])
        if pos > 0:
            candidates.append(annotated[pos - 1])
        nearest = min(candidates, key=lambda x: abs(x - frame_idx))

        frame_data = frames_dict.get(str(nearest), {})
        return frame_data.get(player_key) or frame_data.get(
            "p2" if player_key == "p1" else "p1")

    @staticmethod
    def _crop_square(img: np.ndarray, bbox: Optional[dict]) -> np.ndarray:
        """Crop square around bbox center, side = max(w,h)*BBOX_PAD, resize to IMG_SIZE."""
        H, W = img.shape[:2]

        if bbox is None:
            # No bbox: center crop the full image
            side = min(H, W)
            cy, cx = H // 2, W // 2
        else:
            x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            bw, bh = x2 - x1, y2 - y1
            side = int(max(bw, bh) * BBOX_PAD)

        half = side // 2
        # Clamp to image boundaries
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(W, x0 + side)
        y1 = min(H, y0 + side)
        # Re-adjust if clamping shrunk the box
        x0 = max(0, x1 - side)
        y0 = max(0, y1 - side)

        crop = img[y0:y1, x0:x1]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            crop = img  # fallback to full image

        resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        return resized

    def _preprocess(self, frames: np.ndarray) -> torch.Tensor:
        """[T, H, W, 3] uint8 → [T, 3, IMG_SIZE, IMG_SIZE] float normalised."""
        out = frames.astype(np.float32) / 255.0
        out = (out - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(out).permute(0, 3, 1, 2).float()
        return tensor

    # --- Helpers ---
    @staticmethod
    def _find_mp4(clip_dir: Path, video_name: str) -> Optional[str]:
        if video_name:
            p = clip_dir / video_name
            if p.exists():
                return str(p)
            p = clip_dir / (video_name + ".mp4")
            if p.exists():
                return str(p)
        for ext in ("*.mp4", "*.MP4"):
            found = list(clip_dir.glob(ext))
            if found:
                return str(found[0])
        return None

    @staticmethod
    def _get_frame_count(mp4_path: str) -> Optional[int]:
        if not mp4_path or not os.path.exists(mp4_path):
            return None
        cap = cv2.VideoCapture(mp4_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n if n > 0 else None


def build_dataloaders(data_dir: str, t_max: int, batch_size: int,
                      seed: int = 42, use_features: bool = True,
                      backbone: str = "resnet18") -> tuple:
    """Build train/val DataLoaders with 80/20 split by clip."""
    kw = dict(use_features=use_features, backbone=backbone)

    # Build full index just to get clip list (no preload needed)
    probe = ClipDataset(data_dir, t_max=t_max, is_val=True, preload=False, **kw)
    clip_ids = sorted(set(c["clip_id"] for c in probe.clips))
    n_train = max(1, int(len(clip_ids) * 0.8))
    train_ids = set(clip_ids[:n_train])

    # Build separate datasets
    train_ds = ClipDataset(data_dir, t_max=t_max, **kw)
    val_ds = ClipDataset(data_dir, t_max=t_max, is_val=True, **kw)

    # Filter samples by clip split
    train_ds.samples = [
        s for s in train_ds.samples
        if train_ds.clips[s[0]]["clip_id"] in train_ids
    ]
    val_ds.samples = [
        s for s in val_ds.samples
        if val_ds.clips[s[0]]["clip_id"] not in train_ids
    ]

    if not val_ds.samples:
        val_ds.samples = [
            s for s in val_ds.samples
        ] or train_ds.samples[:len(train_ds.samples) // 5]

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, generator=g, collate_fn=_collate,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=_collate,
        pin_memory=True,
    )
    return train_loader, val_loader


def _collate(batch: list[dict]) -> dict:
    """Custom collate: stack frames and t_gt, keep meta as list."""
    frames = torch.stack([b["frames"] for b in batch])      # [B, T, 3, H, W]
    t_gt = torch.tensor([b["t_gt"] for b in batch], dtype=torch.long)
    valid_len = torch.tensor([b["valid_len"] for b in batch], dtype=torch.long)
    meta = [b["meta"] for b in batch]
    return {"frames": frames, "t_gt": t_gt, "valid_len": valid_len, "meta": meta}
