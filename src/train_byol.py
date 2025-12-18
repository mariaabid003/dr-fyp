"""
Training script for BYOL self-supervised pretraining
on unlabeled fundus images.

Responsibilities:
- Build BYOL model
- Create optimizer
- Train with EMA target update
- Save encoder checkpoints

This script is called from a notebook or CLI.
"""

import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.byol import BYOL
from src.data import UnlabeledDRDataset


# --------------------------------------------------
# Training function
# --------------------------------------------------
def train_byol(
    unlabeled_dir,
    save_dir,
    epochs=100,
    batch_size=16,
    lr=3e-4,
    image_size=512,
    backbone="resnet50",
    device="cuda"
):
    """
    Train BYOL on unlabeled fundus images
    """

    os.makedirs(save_dir, exist_ok=True)

    # --------------------------
    # Dataset & Loader
    # --------------------------
    dataset = UnlabeledDRDataset(
        root_dir=unlabeled_dir,
        image_size=image_size,
        byol=True
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # --------------------------
    # Model
    # --------------------------
    model = BYOL(
        backbone_name=backbone,
        image_size=image_size
    ).to(device)

    model.train()

    # --------------------------
    # Optimizer
    # --------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    # --------------------------
    # Training loop
    # --------------------------
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch [{epoch}/{epochs}]")

        for view1, view2 in pbar:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            loss = model(view1, view2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update
            model.update_target_network()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(loader)

        print(f"Epoch {epoch}: Avg BYOL Loss = {avg_loss:.4f}")

        # --------------------------
        # Save checkpoint
        # --------------------------
        if epoch % 10 == 0 or epoch == epochs:
            ckpt_path = os.path.join(
                save_dir, f"byol_epoch_{epoch}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                ckpt_path
            )
            print(f"Saved checkpoint: {ckpt_path}")

    print("BYOL pretraining completed.")


# --------------------------------------------------
# Optional CLI entry point
# --------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--unlabeled_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="models/byol")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--backbone", type=str, default="resnet50")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_byol(
        unlabeled_dir=args.unlabeled_dir,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        backbone=args.backbone,
        device=device
    )
