import torch
import torch.nn as nn
from mamba_ssm import Mamba

class BiMamba2(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.fwd = Mamba(d_model=d_model)
        self.bwd = Mamba(d_model=d_model)

    def forward(self, x):
        # x: [B, T, D]
        h_fwd = self.fwd(x)

        x_rev = torch.flip(x, dims=[1])
        h_bwd = self.bwd(x_rev)
        h_bwd = torch.flip(h_bwd, dims=[1])

        return torch.cat([h_fwd, h_bwd], dim=-1)
