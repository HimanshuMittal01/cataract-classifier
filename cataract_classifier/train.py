"""Module for training pytorch CNN models"""

import json
from pathlib import Path

import timm
import torch
import numpy as np
import albumentations as A
from rich import print
from rich.progress import track
from torch.utils.data import DataLoader
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from cataract_classifier.data import CataractDataset


def train_one_epoch(
    model,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: str = "cuda",
):
    # Initialize the average loss for the current epoch
    train_step_losses = []
    valid_step_losses = []

    # Run steps on training dataloader
    for X, y in track(train_dataloader, description=f"Epoch {epoch+1}:"):
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)
        loss = criterion(y_pred, y.unsqueeze(1))

        # Compute gradeients and update model parameters
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update metrics
        train_step_losses.append(loss.detach().item())

    # Evaluate validation dataset
    with torch.no_grad():
        model.eval()
        for X, y in valid_dataloader:
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_pred = model(X)
            loss = criterion(y_pred, y.unsqueeze(1))

            # Update metrics
            valid_step_losses.append(loss.detach().item())

    return train_step_losses, valid_step_losses


def train(
    train_img_paths: list[Path],
    valid_img_paths: list[Path],
    save_path: str,
    batch_size: int = 32,
    num_epochs: int = 10,
    lr: float = 2e-4,
    weight_decay: float = 1e-2,
    random_seed: int = 0,
):
    # Initialize model
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)
    model.to(device)

    # Get model transform
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )

    # Define training and validation dataset
    # Augmentation should not be applied to the validation dataset
    train_dataset = CataractDataset(
        img_filepaths=train_img_paths,
        transform=transform,
        augmentation=A.Compose([A.Rotate(limit=45, p=0.5)], seed=random_seed),
    )
    valid_dataset = CataractDataset(
        img_filepaths=valid_img_paths,
        transform=transform,
        augmentation=None,
    )

    # Define dataloader
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    # Define loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True
    )

    # Create directory for save_path if it does not exist
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)

    # Define training loop
    train_epoch_losses = []
    valid_epoch_losses = []
    best_loss = np.inf
    for epoch in range(num_epochs):
        # Run epoch
        train_step_losses, valid_step_losses = train_one_epoch(
            model=model,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
        )

        train_loss = np.average(train_step_losses)
        valid_loss = np.average(valid_step_losses)
        train_epoch_losses.append(train_loss)
        valid_epoch_losses.append(valid_loss)

        # Checkpoint best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best Model saved at [green]{save_path}[/green]")

            metadata = {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            }

            # Save metadata in a JSON file
            with open(
                Path(save_path).parent / "training_metadata.json", "w"
            ) as f:
                json.dump(metadata, f)

        print(
            f"Train Loss: {train_loss:.5f}, Valid Loss: {valid_loss:.5f}, Best Loss: {best_loss:.5f}"
        )
