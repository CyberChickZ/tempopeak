"""TempoPeak model: optional backbone + temporal head + linear projection."""

import torch
import torch.nn as nn
import torchvision.models as models

from temporal_heads import build_head, HEAD_REGISTRY


class TempoPeakModel(nn.Module):
    """Temporal head → Linear → logits [B, T].

    Two modes:
      - Image mode:   input [B, T, 3, H, W] → frozen backbone → [B, T, 512] → head
      - Feature mode:  input [B, T, D] → optional projection → head (no backbone)
    """

    def __init__(self, temporal_head: str = "identity",
                 feat_dim: int = 512, use_backbone: bool = True):
        super().__init__()
        self.use_backbone = use_backbone

        if use_backbone:
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(
                backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
                backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
                backbone.avgpool,
            )
            for p in self.backbone.parameters():
                p.requires_grad = False
            backbone_dim = 512
        else:
            self.backbone = None
            backbone_dim = feat_dim

        # Project to 512 if feature dim differs (e.g. ViT 768 → 512)
        if backbone_dim != 512:
            self.feat_proj = nn.Linear(backbone_dim, 512)
        else:
            self.feat_proj = None

        # Temporal head
        self.head = build_head(temporal_head)
        head_out_dim = HEAD_REGISTRY[temporal_head].out_dim

        # Projection to scalar per timestep
        self.proj = nn.Linear(head_out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, T, 3, H, W] (image mode) or [B, T, D] (feature mode).

        Returns:
            logits: [B, T] raw scores.
        """
        B, T = x.shape[:2]

        if self.use_backbone and x.dim() == 5:
            with torch.no_grad():
                feats = self.backbone(x.reshape(B * T, *x.shape[2:]))
            feats = feats.flatten(1).reshape(B, T, -1)  # [B, T, 512]
        else:
            feats = x  # already [B, T, D]

        if self.feat_proj is not None:
            feats = self.feat_proj(feats)

        h = self.head(feats)
        logits = self.proj(h).squeeze(-1)  # [B, T]
        return logits
