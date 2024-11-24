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
from cataract_classifier.evaluate import evaluate


def train_one_epoch(
    model,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: str = "cuda",
):
    """Iterate through train dataloader once for training set to update model weights.

    Then, iterate through validation dataloader once for evaluation.
    """
    # Initialize the average loss for the current epoch
    train_step_losses = []
    valid_step_losses = []

    # Run steps on training dataloader
    train_preds = []
    train_actuals = []
    for X, y in track(train_dataloader, description=f"Epoch {epoch+1}:"):
        X, y = X.to(device), y.to(device).unsqueeze(1)

        # Forward pass
        y_pred = model(X)
        loss = criterion(y_pred, y)

        # Compute gradeients and update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        train_step_losses.append(loss.detach().item())

        train_preds.append(y_pred)
        train_actuals.append(y)

    # Evaluate training set
    train_accuracy, train_aucroc, train_cm = evaluate(
        y_true=torch.vstack(train_actuals),
        y_pred=torch.vstack(train_preds),
        device=device,
    )

    # Predict on validation dataset
    valid_preds = []
    valid_actuals = []
    with torch.no_grad():
        model.eval()
        for X, y in valid_dataloader:
            X, y = X.to(device), y.to(device).unsqueeze(1)

            # Forward pass
            y_pred = model(X)
            loss = criterion(y_pred, y)

            # Update metrics
            valid_step_losses.append(loss.detach().item())

            valid_preds.append(y_pred)
            valid_actuals.append(y)

    # Evaluate validation set
    valid_accuracy, valid_aucroc, valid_cm = evaluate(
        y_true=torch.cat(valid_actuals),
        y_pred=torch.cat(valid_preds),
        device=device,
    )

    return {
        "train": {
            "epoch_loss": np.average(train_step_losses),
            "step_losses": train_step_losses,
            "accuracy": train_accuracy,
            "aucroc": train_aucroc,
            "cm": train_cm,
        },
        "valid": {
            "epoch_loss": np.average(valid_step_losses),
            "step_losses": valid_step_losses,
            "accuracy": valid_accuracy,
            "aucroc": valid_aucroc,
            "cm": valid_cm,
        },
    }


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
    best_valid_epoch = 0
    for epoch in range(num_epochs):
        # Run epoch
        epoch_eval = train_one_epoch(
            model=model,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
        )

        train_epoch_losses.append(epoch_eval["train"]["epoch_loss"])
        valid_epoch_losses.append(epoch_eval["valid"]["epoch_loss"])

        # Checkpoint best model
        if epoch_eval["valid"]["epoch_loss"] < best_loss:
            best_loss = epoch_eval["valid"]["epoch_loss"]
            best_valid_epoch = epoch + 1

            torch.save(model.state_dict(), save_path)
            print(f"Best Model saved at [green]{save_path}[/green]")

            metadata = {
                "epoch": epoch,
                "train_loss": epoch_eval["train"]["epoch_loss"],
                "valid_loss": epoch_eval["valid"]["epoch_loss"],
            }

            # Save metadata in a JSON file
            with open(
                Path(save_path).parent / "training_metadata.json", "w"
            ) as f:
                json.dump(metadata, f)

        # Log training and validation metrics
        train_metric_logs = ", ".join(
            [
                f"Train {metric}: {epoch_eval['train'][metric]:.6f}"
                for metric in ["epoch_loss", "accuracy", "aucroc"]
            ]
        )
        valid_metric_logs = ", ".join(
            [
                f"Valid {metric}: {epoch_eval['valid'][metric]:.6f}"
                for metric in ["epoch_loss", "accuracy", "aucroc"]
            ]
        )
        print(train_metric_logs)
        print(valid_metric_logs)
        print(f"Best Loss: {best_loss:.5f} at epoch {best_valid_epoch}!!")
