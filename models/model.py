"""TempoPeak model: frozen ResNet-18 backbone + temporal head + linear projection."""

import torch
import torch.nn as nn
import torchvision.models as models

from temporal_heads import build_head, HEAD_REGISTRY


class TempoPeakModel(nn.Module):
    """ResNet-18 (frozen) → Temporal Head → Linear → logits [B, T]."""

    def __init__(self, temporal_head: str = "identity"):
        super().__init__()

        # Frozen ResNet-18 backbone (avgpool output → 512-d)
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            backbone.avgpool,
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Temporal head
        self.head = build_head(temporal_head)
        head_out_dim = HEAD_REGISTRY[temporal_head].out_dim

        # Projection to scalar per timestep
        self.proj = nn.Linear(head_out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, T, 3, 224, 224] video frames.

        Returns:
            logits: [B, T] raw scores (apply softmax externally).
        """
        B, T = x.shape[:2]

        # Per-frame backbone features
        with torch.no_grad():
            feats = self.backbone(x.reshape(B * T, *x.shape[2:]))
        feats = feats.flatten(1).reshape(B, T, -1)  # [B, T, 512]

        # Temporal modelling
        h = self.head(feats)  # [B, T, D_out]

        # Project to logits
        logits = self.proj(h).squeeze(-1)  # [B, T]
        return logits
