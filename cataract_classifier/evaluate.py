"""Contains evaluation metrics and functions for model performance."""

import torch
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinaryROC,
)


def evaluate(y_pred, y_true, device="cpu"):
    """Evaluate predictions based on various metrics: accuracy, ROC-AUC, and confusion matrix.

    Args:
        y_pred (Tensor): The predicted values (e.g., output from the model).
        y_true (Tensor): The true labels (ground truth).
        device (str): The device to run the evaluation on (default is "cpu").

    Returns:
        A dict containing:
            - accuracy (float): The accuracy of the model's predictions.
            - aucroc (float): The Area Under the ROC Curve (AUC-ROC) score.
            - precision (float): The precision of the model's predictions.
            - recall (float): The recall of the model's predictions.
            - f1 (float): The F1-score of the model's predictions.
            - cm_ (matplotlib.pyplot.Figure): The confusion matrix plot figure.
            - roc (matplotlib.pyplot.Figure): The ROC curve plot figure.
    """
    # Binary Accuracy metric (with a default threshold of 0.5)
    accuracy_metric = BinaryAccuracy(threshold=0.5).to(device)
    accuracy = accuracy_metric(y_pred, y_true).item()

    # Binary AUC-ROC metric (setting thresholds=None is most accurate but also memory consuming)
    aucroc_metric = BinaryAUROC(thresholds=None).to(device)
    aucroc = aucroc_metric(y_pred, y_true).item()

    # Confusion Matrix (with a threshold of 0.5)
    cm_metric = BinaryConfusionMatrix(threshold=0.5).to(device)
    cm_ = cm_metric(y_pred, y_true)

    # ROC Curve
    roc_metric = BinaryROC(thresholds=None).to(device)
    roc_metric.update(y_pred, y_true.type(torch.int8))

    # Calculate precision = TP / (TP + FP)
    precision = (cm_[1, 1] / (cm_[1, 1] + cm_[0, 1])).item()

    # Calculate recall = TP / (TP + FN)
    recall = (cm_[1, 1] / (cm_[1, 1] + cm_[1, 0])).item()

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall)

    # Plot confusion matrix and ROC curve
    cm_fig = cm_metric.plot()[0]
    roc_fig = roc_metric.plot()[0]

    return {
        "accuracy": accuracy,
        "aucroc": aucroc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm": cm_fig,
        "roc": roc_fig,
    }
