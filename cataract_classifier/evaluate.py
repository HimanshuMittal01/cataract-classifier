"""Contains evaluation metrics and functions for model performance."""

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryConfusionMatrix,
)


def evaluate(y_pred, y_true, device="cpu"):
    """Evaluate predictions based on various metrics: accuracy, ROC-AUC, and confusion matrix.

    Args:
        y_pred (Tensor): The predicted values (e.g., output from the model).
        y_true (Tensor): The true labels (ground truth).
        device (str): The device to run the evaluation on (default is "cpu").

    Returns:
        tuple: A tuple containing:
            - accuracy (float): The accuracy of the model's predictions.
            - aucroc (float): The Area Under the ROC Curve (AUC-ROC) score.
            - cm_ (Tensor): The confusion matrix as a 2x2 tensor.
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

    return accuracy, aucroc, cm_
