"""It contains evaluation metrics and functions."""

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryConfusionMatrix,
)


def evaluate(y_pred, y_true, device="cpu"):
    """Evaluate predictions on accuracy, ROC-AUC, etc"""
    accuracy_metric = BinaryAccuracy(threshold=0.5).to(device)
    accuracy = accuracy_metric(y_pred, y_true).item()

    # setting thresholds=None is most accurate but also memory consuming
    aucroc_metric = BinaryAUROC(thresholds=None).to(device)
    aucroc = aucroc_metric(y_pred, y_true).item()

    # output a figure
    cm_metric = BinaryConfusionMatrix(threshold=0.5).to(device)
    cm_ = cm_metric(y_pred, y_true)

    return accuracy, aucroc, cm_
