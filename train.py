import torch
import torch.nn as nn
from torch.optim import AdamW
from models.model import EventModel


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Note: On local Mac without CUDA, this might fail if mamba_ssm requires CUDA or if model is too large for CPU.
    # The user instruction specifically says "device = 'cuda'", but for safety in script I'll make it dynamic or just stick to user's request if they are running on HPC.
    # User said "device = 'cuda'". I will stick to that but add a check or just assume HPC execution as per instructions.
    # The user provided code has `device = "cuda"`. I will use that exactly as requested, but maybe add a safeguard comment or just use "cuda" since this is for HPC.
    
    device = "cuda"
    B, T, H, W = 4, 64, 224, 224

    print(f"Using device: {device}")

    model = EventModel().to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("Starting training loop with synthetic data...")

    for step in range(50):

        x = torch.randn(B, T, 3, H, W, device=device)
        t_gt = torch.randint(0, T, (B,), device=device)

        optimizer.zero_grad()

        logits = model(x)          # [B, T]
        loss = criterion(logits, t_gt)

        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            pred = torch.argmax(logits, dim=1)
            acc = (pred == t_gt).float().mean().item()
            print(f"step {step:03d}  loss {loss.item():.4f}  acc {acc:.3f}")

    print("Training loop finished.")

if __name__ == "__main__":
    main()
