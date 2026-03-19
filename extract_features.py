"""Pre-extract per-frame features: features/{backbone}/{player}/*.pt alongside frames/."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

BBOX_PAD = 1.5
# Per-backbone image size: ViT-B-16 requires exactly 224; ResNets can use 448
BACKBONE_IMG_SIZE = {"resnet18": 448, "resnet34": 448, "vit_small": 224, "dinov2": 224}

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_backbone(name: str, device: str) -> tuple[nn.Module, int]:
    """Build frozen backbone. Returns (model, feature_dim)."""
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4, m.avgpool,
        )
        feat_dim = 512
    elif name == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        backbone = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4, m.avgpool,
        )
        feat_dim = 512
    elif name == "vit_small":
        # NOTE: despite the CLI alias "vit_small", this loads ViT-B-16 (~86M params, D=768),
        # NOT ViT-Small (~22M params, D=384). Presentations/papers must cite "ViT-B-16".
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        feat_dim = 768

        class _ViTFeats(nn.Module):
            def __init__(self, vit):
                super().__init__()
                self.vit = vit

            def forward(self, x):
                x = self.vit._process_input(x)
                B = x.shape[0]
                cls = self.vit.class_token.expand(B, -1, -1)
                x = torch.cat([cls, x], dim=1)
                x = self.vit.encoder(x)
                return x[:, 0]  # CLS → [B, 768]

        backbone = _ViTFeats(m)
    elif name == "dinov2":
        from transformers import AutoModel

        _dinov2_model = AutoModel.from_pretrained("facebook/dinov2-large")
        feat_dim = 1024  # DINOv2 ViT-L hidden_size

        class _DINOv2Feats(nn.Module):
            def __init__(self, dinov2):
                super().__init__()
                self.dinov2 = dinov2

            def forward(self, x):
                # x: [B, 3, 224, 224] already ImageNet-normalized
                outputs = self.dinov2(pixel_values=x)
                return outputs.last_hidden_state[:, 0, :]  # CLS token [B, 1024]

        backbone = _DINOv2Feats(_dinov2_model)
    else:
        raise ValueError(f"Unknown backbone: {name}")

    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone, feat_dim


def crop_square(img: np.ndarray, bbox: dict | None, img_size: int = 448) -> np.ndarray:
    """Crop square around bbox center, resize to img_size."""
    H, W = img.shape[:2]
    if bbox is None:
        side = min(H, W)
        cy, cx = H // 2, W // 2
    else:
        x1, y1 = int(bbox["x1"]), int(bbox["y1"])
        x2, y2 = int(bbox["x2"]), int(bbox["y2"])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        side = int(max(x2 - x1, y2 - y1) * BBOX_PAD)

    half = side // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1c = min(W, x0 + side)
    y1c = min(H, y0 + side)
    x0 = max(0, x1c - side)
    y0 = max(0, y1c - side)

    crop = img[y0:y1c, x0:x1c]
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        crop = img
    return cv2.resize(crop, (img_size, img_size))


def get_bbox(frames_dict: dict, fi: int, player: str) -> dict | None:
    """Get bbox for a frame with nearest-frame fallback."""
    import bisect
    fd = frames_dict.get(str(fi))
    if fd:
        b = fd.get(player)
        if b:
            return b
        other = "p2" if player == "p1" else "p1"
        return fd.get(other)

    annotated = sorted(int(k) for k in frames_dict if k.isdigit())
    if not annotated:
        return None
    pos = bisect.bisect_left(annotated, fi)
    cands = []
    if pos < len(annotated):
        cands.append(annotated[pos])
    if pos > 0:
        cands.append(annotated[pos - 1])
    nearest = min(cands, key=lambda x: abs(x - fi))
    fd = frames_dict.get(str(nearest), {})
    return fd.get(player) or fd.get("p2" if player == "p1" else "p1")


def extract_clip(clip_dir: Path, annot: dict, backbone: nn.Module,
                 backbone_name: str, device: str, batch_size: int,
                 img_size: int = 448) -> None:
    """Extract features for all frames × all hitters in a clip."""
    frames_dir = clip_dir / "frames"
    total = annot.get("total_frames", 0)
    frames_dict = annot.get("frames", {})
    hitters = annot.get("hitters", [0])

    players = set()
    for h in hitters:
        if h in (1, 2):
            players.add(f"p{h}")
    if not players:
        players = {"p1"}

    for player in sorted(players):
        # features/{backbone_name}/{player}/
        feat_dir = clip_dir / "features" / backbone_name / player
        feat_dir.mkdir(parents=True, exist_ok=True)

        # Skip if complete
        existing = len(list(feat_dir.glob("*.pt")))
        if existing >= total:
            continue

        for batch_start in range(0, total, batch_size):
            batch_idx = list(range(batch_start, min(batch_start + batch_size, total)))
            tensors = []
            valid_idx = []

            for fi in batch_idx:
                if (feat_dir / f"{fi}.pt").exists():
                    continue

                jpg = frames_dir / f"{fi}.jpg"
                img = None
                if jpg.exists():
                    img = cv2.imread(str(jpg))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img is None:
                    continue

                bbox = get_bbox(frames_dict, fi, player)
                crop = crop_square(img, bbox, img_size=img_size)
                tensors.append(IMAGENET_TRANSFORM(crop))
                valid_idx.append(fi)

            if not tensors:
                continue

            batch_t = torch.stack(tensors).to(device)
            with torch.no_grad():
                feats = backbone(batch_t)
                if feats.dim() > 2:
                    feats = feats.flatten(1)

            for fi, feat in zip(valid_idx, feats):
                torch.save(feat.cpu(), feat_dir / f"{fi}.pt")


def main():
    parser = argparse.ArgumentParser(description="Pre-extract backbone features")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "vit_small", "dinov2"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    backbone, feat_dim = build_backbone(args.backbone, device)
    img_size = BACKBONE_IMG_SIZE.get(args.backbone, 448)
    print(f"Backbone: {args.backbone} | D={feat_dim} | IMG={img_size} | Device: {device}")

    root = Path(args.data_dir)
    annots = sorted(root.rglob("annot.json"))
    print(f"Found {len(annots)} clips")

    for i, ap in enumerate(annots):
        clip_dir = ap.parent
        with open(ap) as f:
            annot = json.load(f)
        total = annot.get("total_frames", 0)
        print(f"[{i+1}/{len(annots)}] {clip_dir.name}: {total} frames ...", end=" ", flush=True)
        extract_clip(clip_dir, annot, backbone, args.backbone, device, args.batch_size,
                     img_size=img_size)
        print("done")

    print(f"\nAll done. Features at: features/{args.backbone}/p*/")


if __name__ == "__main__":
    main()
