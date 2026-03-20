"""Multi-hit dataloader for TempoPeak.

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
# Target generation
# ---------------------------------------------------------------------------

def multi_gaussian_target(hit_indices, T, sigma=2.0):
    """Create normalized multi-peak Gaussian target distribution.

    Args:
        hit_indices: list[int] hit frame positions (segment-local).
        T: sequence length.
        sigma: Gaussian std.

    Returns:
        [T] tensor, sums to 1.0 (uniform if no hits).
    """
    t = torch.arange(T, dtype=torch.float32)
    target = torch.zeros(T)
    for h in hit_indices:
        target += torch.exp(-0.5 * ((t - h) / sigma) ** 2)
    if target.sum() > 0:
        target = target / target.sum()
    else:
        target = torch.ones(T) / T  # uniform (no-hit segment)
    return target


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
                 segment_len=2700, stride=None, sigma=2.0):
        self.segment_len = segment_len
        self.sigma = sigma

        # Load features
        self.features = torch.load(features_pt, weights_only=True)
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

        target = multi_gaussian_target(hits, self.segment_len, self.sigma)

        return {
            "features": feats,           # [segment_len, D]
            "target": target,            # [segment_len]
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

    def __init__(self, data_dir, backbone="dinov2", sigma=2.0, max_len=2700):
        self.backbone = backbone
        self.sigma = sigma
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

        target = multi_gaussian_target(hits, self.max_len, self.sigma)

        return {
            "features": feats,           # [max_len, D]
            "target": target,            # [max_len]
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
        "target": torch.stack([b["target"] for b in batch]),
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
    sigma=2.0,
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
            segment_len=segment_len, stride=stride, sigma=sigma,
        )
        feat_dim = dataset.feat_dim
    elif clip_data_dir:
        dataset = ClipFolderDataset(
            clip_data_dir, backbone=backbone,
            sigma=sigma, max_len=segment_len,
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
