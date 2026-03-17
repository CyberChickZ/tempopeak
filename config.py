"""Argument parsing for TempoPeak training."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    p = argparse.ArgumentParser(description="TempoPeak temporal head training")

    # Model
    p.add_argument("--temporal_head", type=str, default="identity",
                   choices=["identity", "bilstm", "mstcn", "transformer",
                            "mamba2", "bimamba2"])
    p.add_argument("--t_max", type=int, default=32, choices=[16, 32, 64])

    # Loss
    p.add_argument("--target_type", type=str, default="gaussian",
                   choices=["onehot", "gaussian"])
    p.add_argument("--sigma", type=float, default=2.0)

    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    # Data
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--backbone", type=str, default="resnet18",
                   choices=["resnet18", "resnet34", "vit_small"],
                   help="backbone for feature extraction / feature dir lookup")
    p.add_argument("--no_features", action="store_true",
                   help="force image mode, skip pre-extracted features")

    # Device
    p.add_argument("--device", type=str, default="auto",
                   help="auto = cuda if available else cpu")

    return p.parse_args()


def resolve_device(device_str: str) -> str:
    """Resolve device string to actual device."""
    import torch
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str
