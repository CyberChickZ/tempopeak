import torch
import torch.nn as nn
import torchvision.models as models
from mamba_ssm import Mamba


class EventModel(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()

        # --- Backbone ---
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )

        self.feature_dim = 512

        # --- Projection ---
        self.proj = nn.Linear(self.feature_dim, d_model)

        # --- Bi-Mamba ---
        self.mamba_fwd = Mamba(d_model=d_model)
        self.mamba_bwd = Mamba(d_model=d_model)

        # --- Score head ---
        self.score = nn.Linear(d_model * 2, 1)

    def forward(self, x):
        # x: [B, T, 3, H, W]

        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)

        feat = self.feature_extractor(x)
        feat = torch.flatten(feat, 1)

        feat = feat.view(B, T, -1)

        z = self.proj(feat)

        # Forward
        h_fwd = self.mamba_fwd(z)

        # Backward
        z_rev = torch.flip(z, dims=[1])
        h_bwd = self.mamba_bwd(z_rev)
        h_bwd = torch.flip(h_bwd, dims=[1])

        h = torch.cat([h_fwd, h_bwd], dim=-1)

        scores = self.score(h).squeeze(-1)

        return scores
