"""All 6 temporal heads for TempoPeak hit frame detection."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Identity
# ---------------------------------------------------------------------------

class IdentityHead(nn.Module):
    """Passthrough — no temporal modelling. Output dim = input dim (512)."""
    out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ---------------------------------------------------------------------------
# 2. BiLSTM
# ---------------------------------------------------------------------------

class BiLSTMHead(nn.Module):
    """Bidirectional LSTM. Output dim = 256 (128*2)."""
    out_dim = 256

    def __init__(self, d_in: int = 512, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(d_in, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers,
                            batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        out, _ = self.lstm(x)
        return out


# ---------------------------------------------------------------------------
# 3. MS-TCN
# ---------------------------------------------------------------------------

class _DilatedResBlock(nn.Module):
    """Single dilated residual block for MS-TCN."""

    def __init__(self, d: int, dilation: int):
        super().__init__()
        self.conv_d = nn.Conv1d(d, d, 3, padding=dilation, dilation=dilation)
        self.conv_1 = nn.Conv1d(d, d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_d(x))
        out = self.conv_1(out)
        return x + out


class _TCNStage(nn.Module):
    """One MS-TCN stage: stack of dilated residual blocks."""

    def __init__(self, d_in: int, d: int, num_layers: int):
        super().__init__()
        self.conv_in = nn.Conv1d(d_in, d, 1)
        self.layers = nn.ModuleList(
            [_DilatedResBlock(d, 2 ** i) for i in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for layer in self.layers:
            x = layer(x)
        return x


class MSTCNHead(nn.Module):
    """Multi-Stage TCN. Output dim = 128."""
    out_dim = 128

    def __init__(self, d_in: int = 512, hidden: int = 128,
                 stages: int = 2, layers_per_stage: int = 4):
        super().__init__()
        self.stages = nn.ModuleList()
        for i in range(stages):
            din = d_in if i == 0 else hidden
            self.stages.append(_TCNStage(din, hidden, layers_per_stage))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] → conv expects [B, D, T]
        x = x.transpose(1, 2)
        for stage in self.stages:
            x = stage(x)
        return x.transpose(1, 2)


# ---------------------------------------------------------------------------
# 4. Transformer
# ---------------------------------------------------------------------------

class TransformerHead(nn.Module):
    """Transformer encoder with learnable positional encoding. Output dim = 512."""
    out_dim = 512

    def __init__(self, d_model: int = 512, nhead: int = 8, layers: int = 2,
                 max_len: int = 128):
        super().__init__()
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        x = x + self.pos_enc[:, :T, :]
        return self.encoder(x)


# ---------------------------------------------------------------------------
# 5/6. Pure-PyTorch SSM (Mamba2 / BiMamba2)
# ---------------------------------------------------------------------------

class _SSMLayer(nn.Module):
    """Simplified selective state-space layer (pure PyTorch, no CUDA kernels).

    Implements a gated SSM: input → (gate, ssm_input) → discretised recurrence → output.
    Compatible with both CPU and CUDA.
    """

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Input projections: expand to 2*d for gate + ssm paths
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)

        # SSM parameters (input-dependent)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)

        # Learnable log(A) — initialised as negative (stable)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0).expand(d_model, -1).clone()
        )

        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)  # [B, T, 2D]
        x_ssm, z = xz.chunk(2, dim=-1)  # each [B, T, D]

        # Compute selective parameters
        dt = F.softplus(self.dt_proj(x_ssm))  # [B, T, D]
        B_t = self.B_proj(x_ssm)  # [B, T, N]
        C_t = self.C_proj(x_ssm)  # [B, T, N]
        A = -torch.exp(self.A_log)  # [D, N]

        # Discretise and scan
        y = self._scan(x_ssm, dt, A, B_t, C_t)  # [B, T, D]

        # Combine with gate
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_ssm
        y = y * F.silu(z)
        y = self.out_proj(y)

        return y + residual

    def _scan(self, x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor,
              B_t: torch.Tensor, C_t: torch.Tensor) -> torch.Tensor:
        """Sequential scan — simple loop, works on any device."""
        B_batch, T, D = x.shape
        N = self.d_state

        # dA: [B, T, D, N],  dB_x: [B, T, D, N]
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = dt.unsqueeze(-1) * B_t.unsqueeze(2)  # [B, T, D, N]
        dB_x = dB * x.unsqueeze(-1)  # [B, T, D, N]

        h = torch.zeros(B_batch, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dB_x[:, t]  # [B, D, N]
            y_t = (h * C_t[:, t].unsqueeze(1)).sum(-1)  # [B, D]
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # [B, T, D]


class Mamba2Head(nn.Module):
    """Forward-only (causal) SSM. Output dim = 256."""
    out_dim = 256

    def __init__(self, d_in: int = 512, d_model: int = 256, layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList([_SSMLayer(d_model) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class BiMamba2Head(nn.Module):
    """Bidirectional SSM (fwd + bwd concat). Output dim = 512."""
    out_dim = 512

    def __init__(self, d_in: int = 512, d_model: int = 256, layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.fwd_layers = nn.ModuleList(
            [_SSMLayer(d_model) for _ in range(layers)])
        self.bwd_layers = nn.ModuleList(
            [_SSMLayer(d_model) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        fwd = x
        for layer in self.fwd_layers:
            fwd = layer(fwd)

        bwd = torch.flip(x, [1])
        for layer in self.bwd_layers:
            bwd = layer(bwd)
        bwd = torch.flip(bwd, [1])

        return torch.cat([fwd, bwd], dim=-1)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

HEAD_REGISTRY: dict[str, type[nn.Module]] = {
    "identity": IdentityHead,
    "bilstm": BiLSTMHead,
    "mstcn": MSTCNHead,
    "transformer": TransformerHead,
    "mamba2": Mamba2Head,
    "bimamba2": BiMamba2Head,
}


def build_head(name: str) -> nn.Module:
    """Instantiate a temporal head by name."""
    return HEAD_REGISTRY[name]()
