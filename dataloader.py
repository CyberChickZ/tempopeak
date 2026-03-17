"""ClipDataset v2 — bbox-crop + variable-length windows + stratified hit position."""

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

# --- Core constants (Section 3.4) ---
HIT_MARGIN = 4     # exclude ±4 frames around neighbouring hits
MIN_LEN = 8        # minimum window length
AUG_K_TRAIN = 5    # each hit registered K times for train
AUG_K_VAL = 1      # val: 1 deterministic sample per hit
IMG_SIZE = 448      # crop resize target
BBOX_PAD = 1.5      # longest side multiplier for square crop


class ClipDataset(Dataset):
    """Dataset of tennis clips with hit-frame labels and person-crop augmentation."""

    def __init__(self, data_dir: str, t_max: int = 32,
                 aug_k: int = AUG_K_TRAIN, is_val: bool = False):
        self.t_max = t_max
        self.aug_k = aug_k
        self.is_val = is_val

        # Build index: list of clip metadata dicts
        self.clips: list[dict] = []
        self._scan(data_dir)
        if not self.clips:
            raise RuntimeError(f"No clips found in {data_dir}")

        # Expand to (clip_idx, hit_idx, aug_id) samples
        self.samples: list[tuple[int, int, int]] = []
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
        """Register (clip_idx, hit_idx, aug_id) for each hit × K augmentations."""
        for ci, clip in enumerate(self.clips):
            for hi in range(len(clip["hits"])):
                lo, hi_bound = clip["intervals"][hi]
                available = hi_bound - lo + 1
                if available < MIN_LEN:
                    continue  # skip hits with too little context
                for aug_id in range(self.aug_k):
                    self.samples.append((ci, hi, aug_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        ci, hi, aug_id = self.samples[idx]
        clip = self.clips[ci]
        hit = clip["hits"][hi]
        hitter = clip["hitters"][hi] if hi < len(clip["hitters"]) else 0
        lo, hi_bound = clip["intervals"][hi]
        available = hi_bound - lo + 1

        # Step 2: sample window length
        max_n = min(self.t_max, available)
        if self.is_val:
            n = max_n  # val: use max length, deterministic
        else:
            n = random.randint(MIN_LEN, max_n)

        # Step 3: compute start with stratified hit position
        start = self._pick_start(hit, lo, hi_bound, n, aug_id)
        end = start + n  # exclusive

        # Step 4: load frames with bbox crop
        frames = self._load_cropped_frames(clip, start, end, hit, hitter)

        # Step 5: pad to t_max
        valid_len = n
        if n < self.t_max:
            pad_frame = frames[-1:].expand(self.t_max - n, -1, -1, -1)
            frames = torch.cat([frames, pad_frame], dim=0)

        # t_gt relative to window
        t_gt = hit - start

        meta = {
            "clip_dir": clip["clip_dir"],
            "hit": hit,
            "hitter": hitter,
            "start": start,
            "valid_len": valid_len,
            "total_frames": clip["total_frames"],
        }
        return {"frames": frames, "t_gt": t_gt, "valid_len": valid_len, "meta": meta}

    def _pick_start(self, hit: int, lo: int, hi: int,
                    n: int, aug_id: int) -> int:
        """Pick window start with stratified tier positioning (Section 3.2)."""
        # start must satisfy: lo <= start, start + n - 1 <= hi, start <= hit < start + n
        start_lo = max(lo, hit - n + 1)
        start_hi = min(hit, hi - n + 1)
        start_hi = max(start_lo, start_hi)

        if self.is_val:
            # Deterministic: center hit in window
            return max(start_lo, min(hit - n // 2, start_hi))

        tier = aug_id % 3
        span = start_hi - start_lo
        if span <= 0:
            return start_lo

        if tier == 0:
            # Hit in front 1/3 of window → start close to hit (high start)
            t_lo = start_lo + (span * 2) // 3
            t_hi = start_hi
        elif tier == 1:
            # Hit in middle
            t_lo = start_lo + span // 3
            t_hi = start_lo + (span * 2) // 3
        else:
            # Hit in back 2/3 → start far before hit (low start)
            t_lo = start_lo
            t_hi = start_lo + span // 3

        t_lo = max(start_lo, t_lo)
        t_hi = min(start_hi, t_hi)
        t_hi = max(t_lo, t_hi)
        return random.randint(t_lo, t_hi)

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
                      seed: int = 42) -> tuple:
    """Build train/val DataLoaders with 80/20 split by clip."""
    # Build full index just to get clip list
    probe = ClipDataset(data_dir, t_max=t_max, aug_k=1)
    clip_ids = sorted(set(c["clip_id"] for c in probe.clips))
    n_train = max(1, int(len(clip_ids) * 0.8))
    train_ids = set(clip_ids[:n_train])

    # Build separate datasets with correct aug_k
    train_ds = ClipDataset(data_dir, t_max=t_max, aug_k=AUG_K_TRAIN)
    val_ds = ClipDataset(data_dir, t_max=t_max, aug_k=AUG_K_VAL, is_val=True)

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
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=_collate,
    )
    return train_loader, val_loader


def _collate(batch: list[dict]) -> dict:
    """Custom collate: stack frames and t_gt, keep meta as list."""
    frames = torch.stack([b["frames"] for b in batch])      # [B, T, 3, H, W]
    t_gt = torch.tensor([b["t_gt"] for b in batch], dtype=torch.long)
    valid_len = torch.tensor([b["valid_len"] for b in batch], dtype=torch.long)
    meta = [b["meta"] for b in batch]
    return {"frames": frames, "t_gt": t_gt, "valid_len": valid_len, "meta": meta}
