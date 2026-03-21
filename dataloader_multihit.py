"""Multi-hit dataloader for TempoPeak (T-DEED style cls+displacement targets).

两个 Dataset:
  FullVideoDataset  — 单视频 [N,D] 或 [N,196,D] features + HIT JSON → 固定长度 segment
  ClipFolderDataset — 扫描 export/ clip 目录, 每个 (clip, player) 一个样本

spatial/flow 模式:
  传入 spatial_pt + flow_pt 时, __getitem__ 额外返回:
    patches:  [segment_len, n_patches, 1024]  fp32
    flow_mag: [segment_len, n_patches]        fp32
  CrossAttnPooling 在 train_multihit.py 里消费这两个字段.
  推理时不传 spatial_pt/flow_pt, 退化为普通 CLS token 模式.

  支持 .pt 和 .npy 格式:
    .pt  → 加载到 RAM (适合 <50GB 文件)
    .npy → numpy mmap 按需读取 (适合 200+ GB 的 518-native spatial)

使用:
  train_loader, val_loader, feat_dim = build_multihit_dataloaders(
      features_pt="data/00001_ce224.pt",
      spatial_pt="data/00001_spatial518.npy",  # 可选, .pt 或 .npy
      flow_pt="data/00001_flow518.pt",          # 可选
      hit_json="data/annot.json",
  )
"""

import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Target generation (T-DEED style)
# ---------------------------------------------------------------------------

def cls_displacement_target(hit_indices, T, radius=5):
    """分类 + displacement 目标.

    Args:
        hit_indices: list[int] segment-local hit 帧位置
        T:           序列长度
        radius:      ±radius 帧内为正样本

    Returns:
        cls_target:  [T] float, 1.0 if within radius of any hit
        disp_target: [T] float, signed offset to nearest hit
        disp_mask:   [T] float, 1.0 for positive frames only

    Tensor 示意 (T=10, hit_indices=[4], radius=2):
      t:          0  1  2  3  4  5  6  7  8  9
      cls_target: 0  0  1  1  1  1  1  0  0  0
      disp_target:0  0  2  1  0 -1 -2  0  0  0
      disp_mask:  0  0  1  1  1  1  1  0  0  0
    """
    cls_target  = torch.zeros(T)
    disp_target = torch.zeros(T)
    disp_mask   = torch.zeros(T)

    if not hit_indices:
        return cls_target, disp_target, disp_mask

    hits = torch.tensor(hit_indices, dtype=torch.float32)
    t    = torch.arange(T, dtype=torch.float32)

    # [T, N_hits] distance matrix
    dists     = t.unsqueeze(1) - hits.unsqueeze(0)
    abs_dists = dists.abs()

    nearest_idx  = abs_dists.argmin(dim=1)             # [T]
    nearest_disp = dists[torch.arange(T), nearest_idx] # [T] signed

    cls_target  = (abs_dists.min(dim=1).values <= radius).float()
    disp_target = nearest_disp
    disp_mask   = cls_target

    return cls_target, disp_target, disp_mask


# ---------------------------------------------------------------------------
# Dataset 1: Full Video → fixed-length segments
# ---------------------------------------------------------------------------

class FullVideoDataset(Dataset):
    """单长视频切成固定长度 segment.

    CLS token 模式 (默认):
      features_pt shape: [N, 1024]
      __getitem__ 返回 features: [segment_len, 1024]

    Spatial 模式 (传入 spatial_pt + flow_pt):
      spatial_pt shape: [N, 196, 1024]
      flow_pt shape:    [N, 196]
      __getitem__ 额外返回:
        patches:  [segment_len, 196, 1024]
        flow_mag: [segment_len, 196]
    """

    @staticmethod
    def _load_spatial_or_flow(path):
        """Load .pt or .npy file. Returns (data, is_mmap).

        .npy files are loaded with mmap_mode='r' for zero-RAM usage (ideal for 200+ GB).
        .pt files are loaded normally into RAM.
        """
        import numpy as np
        if path.endswith(".npy"):
            data = np.load(path, mmap_mode="r")
            return data, True
        else:
            data = torch.load(path, weights_only=True)
            return data, False

    def __init__(self, features_pt, hit_json,
                 spatial_pt=None,
                 flow_pt=None,
                 segment_len=2700,
                 stride=None,
                 radius=5,
                 frame_range=None):
        self.segment_len = segment_len
        self.radius = radius

        # --- 加载 CLS features ---
        all_features = torch.load(features_pt, weights_only=True).float()
        if all_features.dim() == 3:
            all_features = all_features.squeeze(1)  # [N,1,D] → [N,D]
        assert all_features.dim() == 2, \
            f"features_pt 期望 [N,D], 实际 {all_features.shape}"

        # --- 加载 hits ---
        with open(hit_json) as f:
            data = json.load(f)
        all_hits = sorted(data.get("HIT", data.get("hits", [])))

        # --- frame_range 裁剪 (temporal train/val split) ---
        if frame_range is not None:
            start_f, end_f = frame_range
            all_features = all_features[start_f:end_f]
            all_hits = [h - start_f for h in all_hits
                        if start_f <= h < end_f]
        else:
            start_f = 0

        self.features = all_features
        self.all_hits = all_hits
        self.N, self.feat_dim = self.features.shape

        # --- 可选: spatial patch tokens [N, n_patches, 1024] ---
        self.spatial = None
        self._spatial_npy = None  # numpy mmap reference (if .npy)
        self._spatial_range = None  # frame_range for mmap slicing
        if spatial_pt is not None:
            print(f"加载 spatial tokens: {spatial_pt} ...")
            sp, is_mmap = self._load_spatial_or_flow(spatial_pt)
            assert sp.ndim == 3, \
                f"spatial_pt 期望 [N, n_patches, D], 实际 {sp.shape}"
            n_total = sp.shape[0]
            if frame_range is not None and not is_mmap:
                sp = sp[start_f:end_f]
            if is_mmap:
                # mmap mode: store numpy reference, slice in __getitem__
                self._spatial_npy = sp
                self._spatial_range = (start_f, end_f) if frame_range else (0, n_total)
                effective_n = (self._spatial_range[1] - self._spatial_range[0])
                assert effective_n == self.N, \
                    f"spatial_pt 帧数 {effective_n} 与 features_pt {self.N} 不匹配"
                print(f"  spatial: {sp.shape} mmap (0 GB RAM, reads on demand)")
            else:
                assert sp.shape[0] == self.N, \
                    f"spatial_pt 帧数 {sp.shape[0]} 与 features_pt {self.N} 不匹配"
                self.spatial = sp
                print(f"  spatial: {self.spatial.shape}  "
                      f"({self.spatial.numel()*2/1e9:.1f} GB fp16-equiv in RAM)")

        # --- 可选: optical flow [N, n_patches] ---
        self.flow = None
        if flow_pt is not None:
            print(f"加载 flow: {flow_pt} ...")
            fl, _ = self._load_spatial_or_flow(flow_pt)
            if isinstance(fl, torch.Tensor):
                fl = fl.float()
            else:
                fl = torch.from_numpy(fl.copy() if hasattr(fl, 'copy') else fl).float()
            assert fl.dim() == 2, \
                f"flow_pt 期望 [N, n_patches], 实际 {fl.shape}"
            if frame_range is not None:
                fl = fl[start_f:end_f]
            assert fl.shape[0] == self.N, \
                f"flow_pt 帧数 {fl.shape[0]} 与 features_pt {self.N} 不匹配"
            self.flow = fl

        # --- spatial-flow 一致性校验 ---
        sp_patches = None
        if self.spatial is not None:
            sp_patches = self.spatial.shape[1]
        elif self._spatial_npy is not None:
            sp_patches = self._spatial_npy.shape[1]
        if sp_patches is not None and self.flow is not None:
            assert sp_patches == self.flow.shape[1], \
                f"spatial n_patches={sp_patches} != flow n_patches={self.flow.shape[1]}"

        # --- 切 segment ---
        if stride is None:
            stride = 900  # 67% overlap for segment_len=2700

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

        # --- CLS features ---
        feats = self.features[start:end]              # [valid_len, D]
        if valid_len < self.segment_len:
            pad = feats[-1:].expand(self.segment_len - valid_len, -1)
            feats = torch.cat([feats, pad])            # [segment_len, D]

        cls_target, disp_target, disp_mask = cls_displacement_target(
            hits, self.segment_len, radius=self.radius)

        result = {
            "features":     feats,           # [segment_len, D]
            "cls_target":   cls_target,      # [segment_len]
            "disp_target":  disp_target,     # [segment_len]
            "disp_mask":    disp_mask,       # [segment_len]
            "hit_indices":  hits,            # list[int]
            "valid_len":    valid_len,       # int
            "segment_start": start,          # int (global offset)
        }

        # --- Spatial patch tokens (可选, 支持 mmap) ---
        if self.spatial is not None or self._spatial_npy is not None:
            if self._spatial_npy is not None:
                # mmap mode: read only the needed segment from disk
                s0, _ = self._spatial_range
                patches = torch.from_numpy(
                    self._spatial_npy[s0 + start:s0 + end].copy()).float()
            else:
                patches = self.spatial[start:end].float()
            if valid_len < self.segment_len:
                pad = patches[-1:].expand(
                    self.segment_len - valid_len, -1, -1)
                patches = torch.cat([patches, pad])
            result["patches"] = patches

        # --- Optical flow (可选) ---
        if self.flow is not None:
            flow = self.flow[start:end].float()       # [valid_len, n_patches]
            if valid_len < self.segment_len:
                pad = flow[-1:].expand(
                    self.segment_len - valid_len, -1)
                flow = torch.cat([flow, pad])          # [segment_len, n_patches]
            result["flow_mag"] = flow

        return result


# ---------------------------------------------------------------------------
# Dataset 2: Clip Folder (export/ directory)
# ---------------------------------------------------------------------------

class ClipFolderDataset(Dataset):
    """扫描 export/ 目录的带标注 clip folder.

    目录结构:
        data_dir/
        └── {video_id}/
            └── {clip_id}/
                ├── annot.json
                └── features/{backbone}/{player}/*.pt
    """

    def __init__(self, data_dir, backbone="dinov2", radius=5, max_len=2700):
        self.backbone = backbone
        self.radius   = radius
        self.max_len  = max_len
        self.data_dir = Path(data_dir)
        self.feat_dim = None

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

                hits         = annot.get("hits", [])
                hitters      = annot.get("hitters", [])
                total_frames = annot.get("total_frames", 0)

                if total_frames == 0 or not hits:
                    continue

                player_hits = {}
                for h, pl in zip(hits, hitters):
                    player_hits.setdefault(f"p{pl}", []).append(h)

                for pkey, phits in player_hits.items():
                    feat_dir = clip_dir / "features" / backbone / pkey
                    if not feat_dir.exists():
                        continue

                    if self.feat_dim is None:
                        pts = sorted(feat_dir.glob("*.pt"))
                        if pts:
                            t = torch.load(pts[0], weights_only=True)
                            self.feat_dim = t.shape[-1] if t.dim() > 0 else t.numel()

                    self.samples.append(
                        (str(clip_dir), pkey, sorted(phits), total_frames))

        if self.feat_dim is None:
            self.feat_dim = 512

        print(f"ClipFolderDataset: {len(self.samples)} samples, "
              f"feat_dim={self.feat_dim}, backbone={backbone}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip_dir, player, hits, total = self.samples[idx]
        feat_dir = Path(clip_dir) / "features" / self.backbone / player

        feats = []
        for fi in range(total):
            pt = feat_dir / f"{fi}.pt"
            if pt.exists():
                t = torch.load(pt, weights_only=True)
                if t.dim() > 1:
                    t = t.squeeze()
                feats.append(t)
            elif feats:
                feats.append(feats[-1].clone())

        if not feats:
            feats = [torch.zeros(self.feat_dim)]

        feats = torch.stack(feats)       # [T_clip, D]
        T = feats.shape[0]
        valid_len = min(T, self.max_len)

        if T < self.max_len:
            pad = feats[-1:].expand(self.max_len - T, -1)
            feats = torch.cat([feats, pad])
        elif T > self.max_len:
            feats = feats[:self.max_len]
            hits = [h for h in hits if h < self.max_len]

        cls_target, disp_target, disp_mask = cls_displacement_target(
            hits, self.max_len, radius=self.radius)

        return {
            "features":      feats,
            "cls_target":    cls_target,
            "disp_target":   disp_target,
            "disp_mask":     disp_mask,
            "hit_indices":   hits,
            "valid_len":     valid_len,
            "segment_start": 0,
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _collate(batch):
    """patches / flow_mag 字段只在 spatial 模式下存在, 按需 stack."""
    result = {
        "features":      torch.stack([b["features"]      for b in batch]),
        "cls_target":    torch.stack([b["cls_target"]    for b in batch]),
        "disp_target":   torch.stack([b["disp_target"]   for b in batch]),
        "disp_mask":     torch.stack([b["disp_mask"]     for b in batch]),
        "hit_indices":   [b["hit_indices"]               for b in batch],
        "valid_len":     torch.tensor([b["valid_len"]    for b in batch]),
        "segment_start": torch.tensor([b["segment_start"] for b in batch]),
    }
    if "patches" in batch[0]:
        result["patches"]  = torch.stack([b["patches"]  for b in batch])
        # [B, segment_len, 196, 1024]
    if "flow_mag" in batch[0]:
        result["flow_mag"] = torch.stack([b["flow_mag"] for b in batch])
        # [B, segment_len, 196]
    return result


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_multihit_dataloaders(
    # Full video mode
    features_pt=None,
    hit_json=None,
    spatial_pt=None,      # 可选: [N, 196, 1024] spatial patch tokens
    flow_pt=None,         # 可选: [N, 196] Farneback flow magnitude
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
    """构建 train/val DataLoader.

    Returns:
        (train_loader, val_loader, feat_dim)
        eval_only=True 时 train_loader=None.
    """
    if features_pt and hit_json:
        if eval_only:
            ds = FullVideoDataset(
                features_pt, hit_json,
                spatial_pt=spatial_pt, flow_pt=flow_pt,
                segment_len=segment_len, stride=stride, radius=radius,
            )
            val_loader = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                collate_fn=_collate, pin_memory=True,
            )
            return None, val_loader, ds.feat_dim

        # Temporal split: front 80% train, back 20% val
        tmp = torch.load(features_pt, weights_only=True)
        N = tmp.shape[0]
        del tmp
        split = int(N * (1 - val_split))

        train_ds = FullVideoDataset(
            features_pt, hit_json,
            spatial_pt=spatial_pt, flow_pt=flow_pt,
            segment_len=segment_len,
            stride=stride or 900,
            radius=radius,
            frame_range=(0, split),
        )
        val_ds = FullVideoDataset(
            features_pt, hit_json,
            spatial_pt=spatial_pt, flow_pt=flow_pt,
            segment_len=segment_len,
            stride=segment_len,   # val: no overlap
            radius=radius,
            frame_range=(split, N),
        )
        feat_dim = train_ds.feat_dim

        print(f"Temporal split: train [0:{split}] {len(train_ds)} segs, "
              f"val [{split}:{N}] {len(val_ds)} segs, feat_dim={feat_dim}")
        spatial_mode = train_ds.spatial is not None
        print(f"Spatial mode: {spatial_mode}  "
              f"Flow mode: {train_ds.flow is not None}")

    elif clip_data_dir:
        ds = ClipFolderDataset(
            clip_data_dir, backbone=backbone,
            radius=radius, max_len=segment_len,
        )
        feat_dim = ds.feat_dim

        if eval_only:
            val_loader = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                collate_fn=_collate, pin_memory=True,
            )
            return None, val_loader, feat_dim

        n = len(ds)
        n_val = max(1, int(n * val_split))
        gen = torch.Generator().manual_seed(seed)
        train_ds, val_ds = torch.utils.data.random_split(
            ds, [n - n_val, n_val], generator=gen)
    else:
        raise ValueError("需要 (features_pt + hit_json) 或 clip_data_dir")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_collate, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate, pin_memory=True,
    )
    return train_loader, val_loader, feat_dim
