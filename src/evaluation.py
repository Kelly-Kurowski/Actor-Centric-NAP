
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


def evaluate_predictions(y_true, y_pred, y_train):
    """
    Evaluate next-activity predictions on the held-out test set.

    Reports accuracy, macro precision, recall, and F1,
    along with test-set class coverage diagnostics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_train = np.asarray(y_train)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "Prediction count does not match test samples."
        )

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty test set.")

    if pd.isna(y_true).any() or pd.isna(y_pred).any():
        raise ValueError(
            "True labels and predictions cannot contain missing values."
        )

    # Include classes present in either the true labels or predictions.
    labels = np.unique(np.concatenate([y_true, y_pred]))

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


    # Measure how many test targets belong to classes seen in training.
    known_classes = set(y_train)
    known_mask = np.array(
        [target in known_classes for target in y_true]
    )
    unseen_classes = set(y_true) - known_classes

    return {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "class_coverage": float(known_mask.mean()),
        "samples_known_classes": int(known_mask.sum()),
        "classes_unseen_in_train": int(len(unseen_classes)),
    }
