"""
Evaluation metrics: Macro-F1, Precision, Recall, AUROC, accuracy, confusion matrix.
"""
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
from typing import Dict, Any, Optional


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    num_classes: int = 4,
) -> Dict[str, Any]:
    """
    y_true, y_pred: (N,) integer labels.
    y_score: (N, num_classes) optional probability scores for AUROC.
    """
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist(),
    }
    if y_score is not None and y_score.ndim == 2 and y_score.shape[1] == num_classes:
        try:
            out["auroc"] = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
        except ValueError:
            out["auroc"] = float("nan")
    else:
        out["auroc"] = float("nan")
    return out
