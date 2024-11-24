"""Module for training PyTorch CNN models.

This module includes functions for training a convolutional neural network (CNN) model on a cataract classification task.
It supports training, validation, model checkpointing, and saving training metadata.

Functions:
    - train_one_epoch: Runs one epoch of training and validation, updating model weights and evaluating performance.
    - train: Main function for training the model on the provided datasets, handling training loops, evaluation,
      model saving, and logging metrics.
"""

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
    """Perform one epoch of training and validation for the model.

    This function iterates over the training dataset to update the model weights,
    and then iterates over the validation dataset to evaluate the model's performance.

    Args:
        model (torch.nn.Module): The model being trained.
        train_dataloader : Dataloader for the training dataset.
        valid_dataloader : Dataloader for the validation dataset.
        criterion : The loss function used to train the model.
        optimizer : The optimizer used for model parameter updates.
        epoch : The current epoch number.
        device : The device on which to perform training ("cpu", "cuda", or "mps"). Default is "cuda".

    Returns:
        dict: A dictionary containing the training and validation metrics for the epoch:
            - "train": Training metrics (loss, accuracy, AUC-ROC, confusion matrix).
            - "valid": Validation metrics (loss, accuracy, AUC-ROC, confusion matrix).
    """
    # Initialize the average loss for the current epoch
    train_step_losses = []
    valid_step_losses = []

    # Train the model on the training dataset
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

    # Evaluate training set metrics
    train_accuracy, train_aucroc, train_cm = evaluate(
        y_true=torch.cat(train_actuals),
        y_pred=torch.sigmoid(torch.cat(train_preds)),
        device=device,
    )

    # Predict and evaluate on the validation dataset
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

    # Evaluate validation set metrics
    valid_accuracy, valid_aucroc, valid_cm = evaluate(
        y_true=torch.cat(valid_actuals),
        y_pred=torch.sigmoid(torch.cat(valid_preds)),
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
    """
    Train a CNN model for cataract classification.

    This function trains the model for a specified number of epochs, performs validation at the end of each epoch,
    saves the best model based on validation loss, and logs training and validation metrics.

    Args:
        train_img_paths : List of paths to the training images.
        valid_img_paths : List of paths to the validation images.
        save_path : The path where the best model will be saved.
        batch_size : The batch size for training and validation. Default is 32.
        num_epochs : The number of epochs to train the model. Default is 10.
        lr : The learning rate for the optimizer. Default is 2e-4.
        weight_decay : The weight decay (L2 regularization) for the optimizer. Default is 1e-2.
        random_seed : The random seed to use for data augmentation. Default is 0.

    Returns:
        None: The function saves the best model and training metadata.
    """

    # Initialize model
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)
    model.to(device)

    # Set up image transformations for training and validation
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )

    # Define datasets for training and validation (with augmentation on training data)
    train_dataset = CataractDataset(
        img_filepaths=train_img_paths,
        transform=transform,
        augmentation=A.Compose(
            [
                A.Rotate(limit=30, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Blur(p=0.25),
                A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), p=0.5),
            ],
            seed=random_seed,
        ),
    )
    valid_dataset = CataractDataset(
        img_filepaths=valid_img_paths,
        transform=transform,
        augmentation=None,  # No augmentation for validation
    )

    # Create dataloaders for training and validation
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    # Define the loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True
    )

    # Create the save path if it doesn't exist
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)

    # Training loop
    train_epoch_losses = []
    valid_epoch_losses = []
    best_loss = np.inf
    best_valid_epoch = 0
    for epoch in range(num_epochs):
        # Perform training and validation for one epoch
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

        # Checkpoint best model (based on validation loss)
        if epoch_eval["valid"]["epoch_loss"] < best_loss:
            best_loss = epoch_eval["valid"]["epoch_loss"]
            best_valid_epoch = epoch + 1

            # Save the best model
            torch.save(model.state_dict(), save_path)
            print(f"Best Model saved at [green]{save_path}[/green]")

            # Save metadata in json about the best epoch
            metadata = {
                "epoch": epoch,
                "train_loss": epoch_eval["train"]["epoch_loss"],
                "valid_loss": epoch_eval["valid"]["epoch_loss"],
            }
            with open(
                Path(save_path).parent / "training_metadata.json", "w"
            ) as f:
                json.dump(metadata, f)

        # Log the metrics for this epoch
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
