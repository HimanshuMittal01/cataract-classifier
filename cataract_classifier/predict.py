"""Module for predicting cataract on images.

This module is intended for use by the backend for loading models, performing image predictions,
and evaluating the model on a test set. It supports both single image prediction and batch predictions
on a test set.
"""

import json
from pathlib import Path

import timm
import torch
import matplotlib.pyplot as plt
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader

from cataract_classifier.data import CataractDataset
from cataract_classifier.evaluate import evaluate


def load_model(model_name: str, weights_path: str):
    """
    Load a pre-trained EfficientNet model for cataract classification.

    Args:
        weights_path (str): Path to the model weights file.

    Returns:
        The pre-trained model with loaded weights.
    """
    if not Path(weights_path).exists():
        raise ValueError(f"Cannot load weights. {weights_path} does not exist.")

    model = timm.create_model(model_name, pretrained=True, num_classes=1)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    return model


def predict_single_image(model, image, device: str = "cpu"):
    """Predict whether a single image shows cataracts or not.

    Args:
        model (torch.nn.Module): The trained model.
        image (PIL.Image): The input image to classify.
        device (str): The device ("cpu", "cuda", or "mps") for running the model. Default is "cpu".

    Returns:
        float: The predicted probability of cataract presence.
    """
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    img = transform(image)

    model.to(device)
    model.eval()
    prediction = model(img.to(device).unsqueeze(0))

    pred_prob = torch.sigmoid(prediction).item()
    return pred_prob


def predict_on_testset(
    test_img_paths: list[Path],
    model_name: str,
    results_path: str,
    batch_size: int = 32,
):
    """Predict and evaluate the model on a test set of images.

    Args:
        test_img_paths: List of paths to images in the test set.
        model_name: Name of the backbone network.
        results_path: Path to the model results directory.
            Weights will be searched in 'weights_dir / {model_name} / "finetuned_{model_name}.pt"'
        batch_size: The batch size for inference. Default is 32.

    Returns:
        None: Prints and save the evaluation results (accuracy, AUC-ROC, confusion matrix).
    """
    # Initialize model
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load the fine-tuned model
    results_path = Path(results_path) / model_name
    model = load_model(model_name, results_path / f"finetuned_{model_name}.pt")
    model.to(device)

    # Define the transformation to apply to the images (same as used during training)
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )

    # Initialize the test dataset (no augmentation applied during testing)
    test_dataset = CataractDataset(
        img_filepaths=test_img_paths,
        transform=transform,
        augmentation=None,
    )

    # Define dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Predict on test dataset
    test_preds = []
    test_actuals = []
    with torch.no_grad():
        model.eval()
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device).unsqueeze(1)

            # Forward pass
            y_pred = model(X)

            # Store predictions and actual labels for evaluation
            test_preds.append(y_pred)
            test_actuals.append(y)

    # Evaluate validation set
    test_eval = evaluate(
        y_true=torch.cat(test_actuals),
        y_pred=torch.sigmoid(torch.cat(test_preds)),
        device=device,
    )

    # Print accuracy and AUCROC
    print(f"Test Accuracy: {test_eval['accuracy']:.6f}")
    print(f"Test AUCROC: {test_eval['aucroc']:.6f}")

    # Save metrics in json
    scaler_metrics = ["accuracy", "aucroc", "precision", "recall", "f1"]
    metadata = {m: test_eval[m] for m in scaler_metrics}
    with open(results_path / "testing_eval.json", "w") as f:
        json.dump(metadata, f)

    # Save figure metrics like confusion matrix
    plot_metrics = ["cm", "roc"]
    for m in plot_metrics:
        test_eval[m].savefig(results_path / f"test_{m}.png")

        # cleanup memory
        test_eval[m].clf()
        plt.close()
