import torch
import torch.nn as nn
from torch.optim import AdamW

from models.model import EventModel

def main():
    torch.manual_seed(0)

    device = "cuda"
    B, T, H, W = 4, 64, 224, 224

    model = EventModel(d_model=128).to(device)
    model.train()

    # NOTE: use logits for CE; do NOT softmax inside the model for training.
    # We will temporarily re-compute logits here by calling model parts.
    # If your model currently returns p (softmax), we will adjust in the next step.

    # Synthetic batch: random images
    x = torch.randn(B, T, 3, H, W, device=device)

    # Random target frame index per sample
    t_gt = torch.randint(low=0, high=T, size=(B,), device=device)

    # Build a "logits-only" forward by reusing your model modules.
    # This avoids editing models/model.py in this step.
    with torch.no_grad():
        pass

    criterion = nn.CrossEntropyLoss()
    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)

    # Train a few steps and print loss; should go down a bit (not strictly monotonic).
    for step in range(20):
        optim.zero_grad(set_to_none=True)

        # forward (reconstruct logits from model internals)
        B2, T2, C, H2, W2 = x.shape
        x2 = x.view(B2 * T2, C, H2, W2)

        feat = model.feature_extractor(x2)
        feat = torch.flatten(feat, 1).view(B2, T2, -1)
        z = model.proj(feat)

        h_fwd = model.mamba_fwd(z)
        z_rev = torch.flip(z, dims=[1])
        h_bwd = torch.flip(model.mamba_bwd(z_rev), dims=[1])
        h = torch.cat([h_fwd, h_bwd], dim=-1)

        logits = model.score(h).squeeze(-1)  # [B, T]

        loss = criterion(logits, t_gt)
        loss.backward()
        optim.step()

        if step % 1 == 0:
            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                acc = (pred == t_gt).float().mean().item()
            print(f"step {step:02d}  loss {loss.item():.4f}  acc {acc:.3f}")

if __name__ == "__main__":
    main()
