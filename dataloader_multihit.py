"""Multi-hit dataloader for TempoPeak (T-DEED style cls+displacement targets).

Two dataset classes:
  FullVideoDataset  — pre-extracted [N,D] features + HIT JSON → fixed-length segments
  ClipFolderDataset — scan export/ clip folders, one sample per player per clip

Usage:
  # Full video mode (training on 00001.mp4)
  train_loader, val_loader, feat_dim = build_multihit_dataloaders(
      features_pt="data/00001_features.pt",
      hit_json="datasets/v1/clips/0006/00001.json",
  )

  # Clip folder mode (eval on old annotated clips)
  _, val_loader, feat_dim = build_multihit_dataloaders(
      clip_data_dir="datasets/v1/export",
      backbone="resnet18",
      eval_only=True,
  )
"""

import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Target generation (T-DEED style: classification + displacement)
# ---------------------------------------------------------------------------

def cls_displacement_target(hit_indices, T, radius=5):
    """Classification + displacement target for T-DEED style detection.

    Args:
        hit_indices: list[int] hit frame positions (segment-local).
        T: sequence length.
        radius: frames within ±radius of a hit are positive.

    Returns:
        cls_target:  [T] float, 1.0 if within radius of any hit, else 0.0
        disp_target: [T] float, signed offset to nearest hit (0 if no hits)
        disp_mask:   [T] float, 1.0 for positive frames (only regress near hits)
    """
    cls_target = torch.zeros(T)
    disp_target = torch.zeros(T)
    disp_mask = torch.zeros(T)

    if not hit_indices:
        return cls_target, disp_target, disp_mask

    hits = torch.tensor(hit_indices, dtype=torch.float32)
    t = torch.arange(T, dtype=torch.float32)

    # For each frame, find distance to nearest hit
    # [T, 1] - [1, N_hits] → [T, N_hits]
    dists = t.unsqueeze(1) - hits.unsqueeze(0)
    abs_dists = dists.abs()
    nearest_idx = abs_dists.argmin(dim=1)       # [T]
    nearest_disp = dists[torch.arange(T), nearest_idx]  # [T] signed

    # Classification: positive if within radius
    cls_target = (abs_dists.min(dim=1).values <= radius).float()

    # Displacement: signed offset to nearest hit (only valid for positive frames)
    disp_target = nearest_disp
    disp_mask = cls_target  # only regress on positive frames

    return cls_target, disp_target, disp_mask


# ---------------------------------------------------------------------------
# Dataset 1: Full Video → segments
# ---------------------------------------------------------------------------

class FullVideoDataset(Dataset):
    """Single long video cut into fixed-length segments.

    Input:
        features_pt — pre-extracted features [N, D] (one row per frame)
        hit_json    — JSON with "HIT" (or "hits") array of global frame indices

    Each segment is a training sample with 0+ hits mapped to local indices.
    """

    def __init__(self, features_pt, hit_json,
                 segment_len=2700, stride=None, radius=5):
        self.segment_len = segment_len
        self.radius = radius

        # Load features
        self.features = torch.load(features_pt, weights_only=True).float()
        if self.features.dim() == 3:
            self.features = self.features.squeeze(1)  # [N,1,D] → [N,D]
        assert self.features.dim() == 2, \
            f"Expected [N, D] features, got {self.features.shape}"
        self.N, self.feat_dim = self.features.shape

        # Load hits
        with open(hit_json) as f:
            data = json.load(f)
        self.all_hits = sorted(data.get("HIT", data.get("hits", [])))

        # Create segments
        if stride is None:
            stride = segment_len

        self.segments = []  # [(start, end, [local_hits])]
        for start in range(0, self.N, stride):
            end = min(start + segment_len, self.N)
            local_hits = [h - start for h in self.all_hits
                          if start <= h < end]
            self.segments.append((start, end, local_hits))

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        start, end, hits = self.segments[idx]
        valid_len = end - start

        feats = self.features[start:end]  # [valid_len, D]

        # Pad to segment_len
        if valid_len < self.segment_len:
            pad = feats[-1:].expand(self.segment_len - valid_len, -1)
            feats = torch.cat([feats, pad])

        cls_target, disp_target, disp_mask = cls_displacement_target(
            hits, self.segment_len, radius=self.radius)

        return {
            "features": feats,           # [segment_len, D]
            "cls_target": cls_target,    # [segment_len]
            "disp_target": disp_target,  # [segment_len]
            "disp_mask": disp_mask,      # [segment_len]
            "hit_indices": hits,         # list[int] (segment-local)
            "valid_len": valid_len,      # int
            "segment_start": start,      # int (global offset)
        }


# ---------------------------------------------------------------------------
# Dataset 2: Clip Folder (export/ directory)
# ---------------------------------------------------------------------------

class ClipFolderDataset(Dataset):
    """Scan export/ directory for annotated clip folders.

    Directory structure:
        data_dir/
        ├── {video_id}/
        │   ├── {clip_id}/
        │   │   ├── annot.json
        │   │   └── features/{backbone}/{player}/*.pt

    Each (clip, player) pair = one sample. Supports multi-hit per player.
    """

    def __init__(self, data_dir, backbone="dinov2", radius=5, max_len=2700):
        self.backbone = backbone
        self.radius = radius
        self.max_len = max_len
        self.data_dir = Path(data_dir)
        self.feat_dim = None  # auto-detected

        self.samples = []  # [(clip_dir, player_key, [hit_frames], total_frames)]

        for video_dir in sorted(self.data_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            for clip_dir in sorted(video_dir.iterdir()):
                if not clip_dir.is_dir():
                    continue
                annot_path = clip_dir / "annot.json"
                if not annot_path.exists():
                    continue

                with open(annot_path) as f:
                    annot = json.load(f)

                hits = annot.get("hits", [])
                hitters = annot.get("hitters", [])
                total_frames = annot.get("total_frames", 0)

                if total_frames == 0 or len(hits) == 0:
                    continue

                # Group hits by player
                player_hits = {}
                for h, p in zip(hits, hitters):
                    pkey = f"p{p}"
                    player_hits.setdefault(pkey, []).append(h)

                for pkey, phits in player_hits.items():
                    feat_dir = clip_dir / "features" / backbone / pkey
                    if not feat_dir.exists():
                        continue

                    # Auto-detect feat_dim from first file
                    if self.feat_dim is None:
                        pts = sorted(feat_dir.glob("*.pt"))
                        if pts:
                            t = torch.load(pts[0], weights_only=True)
                            self.feat_dim = t.shape[-1] if t.dim() > 0 else t.numel()

                    self.samples.append(
                        (str(clip_dir), pkey, sorted(phits), total_frames))

        if self.feat_dim is None:
            self.feat_dim = 512  # fallback

        print(f"ClipFolderDataset: {len(self.samples)} samples, "
              f"feat_dim={self.feat_dim}, backbone={backbone}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip_dir, player, hits, total = self.samples[idx]
        clip_dir = Path(clip_dir)
        feat_dir = clip_dir / "features" / self.backbone / player

        # Load features frame by frame
        feats = []
        for fi in range(total):
            pt = feat_dir / f"{fi}.pt"
            if pt.exists():
                t = torch.load(pt, weights_only=True)
                if t.dim() > 1:
                    t = t.squeeze()  # [1,D] → [D]
                feats.append(t)
            elif feats:
                feats.append(feats[-1].clone())  # missing → repeat prev

        if not feats:
            feats = [torch.zeros(self.feat_dim)]

        feats = torch.stack(feats)  # [T_clip, D]
        T = feats.shape[0]
        valid_len = min(T, self.max_len)

        # Pad or truncate
        if T < self.max_len:
            pad = feats[-1:].expand(self.max_len - T, -1)
            feats = torch.cat([feats, pad])
        elif T > self.max_len:
            feats = feats[:self.max_len]
            hits = [h for h in hits if h < self.max_len]

        cls_target, disp_target, disp_mask = cls_displacement_target(
            hits, self.max_len, radius=self.radius)

        return {
            "features": feats,           # [max_len, D]
            "cls_target": cls_target,    # [max_len]
            "disp_target": disp_target,  # [max_len]
            "disp_mask": disp_mask,      # [max_len]
            "hit_indices": hits,         # list[int]
            "valid_len": valid_len,      # int
            "segment_start": 0,          # always 0 for clips
        }


# ---------------------------------------------------------------------------
# Collate + DataLoader factory
# ---------------------------------------------------------------------------

def _collate(batch):
    """Custom collate: hit_indices stays as list of lists."""
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "cls_target": torch.stack([b["cls_target"] for b in batch]),
        "disp_target": torch.stack([b["disp_target"] for b in batch]),
        "disp_mask": torch.stack([b["disp_mask"] for b in batch]),
        "hit_indices": [b["hit_indices"] for b in batch],
        "valid_len": torch.tensor([b["valid_len"] for b in batch]),
        "segment_start": torch.tensor([b["segment_start"] for b in batch]),
    }


def build_multihit_dataloaders(
    # Full video mode
    features_pt=None,
    hit_json=None,
    # Clip folder mode
    clip_data_dir=None,
    backbone="dinov2",
    # Common
    segment_len=2700,
    stride=None,
    radius=5,
    batch_size=4,
    val_split=0.2,
    seed=42,
    eval_only=False,
):
    """Build train/val DataLoaders for multi-hit training.

    Returns:
        (train_loader, val_loader, feat_dim)
        train_loader is None when eval_only=True.
    """
    if features_pt and hit_json:
        dataset = FullVideoDataset(
            features_pt, hit_json,
            segment_len=segment_len, stride=stride, radius=radius,
        )
        feat_dim = dataset.feat_dim
    elif clip_data_dir:
        dataset = ClipFolderDataset(
            clip_data_dir, backbone=backbone,
            radius=radius, max_len=segment_len,
        )
        feat_dim = dataset.feat_dim
    else:
        raise ValueError("Provide (features_pt + hit_json) or clip_data_dir")

    print(f"Dataset: {len(dataset)} samples, feat_dim={feat_dim}")

    if eval_only:
        val_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            collate_fn=_collate, pin_memory=True,
        )
        return None, val_loader, feat_dim

    # Train/val split
    n = len(dataset)
    n_val = max(1, int(n * val_split))
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=gen,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_collate, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate, pin_memory=True,
    )
    return train_loader, val_loader, feat_dim
