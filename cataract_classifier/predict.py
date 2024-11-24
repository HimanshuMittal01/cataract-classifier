"""Module for predicting cataract on images.

It is intended that it will be used by backend.
"""

from pathlib import Path

import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader

from cataract_classifier.data import CataractDataset
from cataract_classifier.evaluate import evaluate


def load_model(weights_path: str):
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    return model


def predict_single_image(model, image):
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    img = transform(image)
    prediction = model(img.unsqueeze(0))

    pred_prob = torch.sigmoid(prediction).item()
    return pred_prob


def predict_on_testset(
    test_img_paths: list[Path],
    weights_path: str,
    batch_size: int = 32,
):
    # Initialize model
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = load_model(weights_path)
    model.to(device)

    # Get model transform
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )

    # Define training and validation dataset
    # Augmentation should not be applied to the validation dataset
    train_dataset = CataractDataset(
        img_filepaths=test_img_paths,
        transform=transform,
        augmentation=None,
    )

    # Define dataloader
    test_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    # Predict on test dataset
    test_preds = []
    test_actuals = []
    with torch.no_grad():
        model.eval()
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device).unsqueeze(1)

            # Forward pass
            y_pred = model(X)

            # Update metrics
            test_preds.append(y_pred)
            test_actuals.append(y)

    # Evaluate validation set
    test_accuracy, test_aucroc, test_cm = evaluate(
        y_true=torch.cat(test_actuals),
        y_pred=torch.cat(test_preds),
        device=device,
    )

    print(f"Test Accuracy: {test_accuracy:.6f}")
    print(f"Test AUCROC: {test_aucroc:.6f}")
