def sklearn_metrics(y_true, y_pred):
    """Compute accuracy, precision, recall, and F1 using scikit-learn."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "Accuracy": f"{accuracy_score(y_true, y_pred):.4f}",
        "Precision": f"{precision_score(y_true, y_pred, zero_division=0):.4f}",
        "Recall": f"{recall_score(y_true, y_pred, zero_division=0):.4f}",
        "F1": f"{f1_score(y_true, y_pred, zero_division=0):.4f}",
    }


def calculate_confusion_matrix(predicted_positive, actual_positive, total_samples):
    """
    Calculate the confusion matrix components (TP, FN, FP, TN) and compute
    Precision, Recall, and F1 Score based on predicted and actual values.

    Parameters
    ----------
    predicted_positive : int
        The total number of predicted positives (PP).
    actual_positive : int
        The total number of actual positives (AP).
    total_samples : int
        The total number of samples in the dataset.

    Returns
    -------
    tuple
        A tuple containing:
        - precision : float
            Precision (TP / (TP + FP)).
        - recall : float
            Recall (TP / (TP + FN)).
        - f1_score : float
            F1 score (2 * Precision * Recall / (Precision + Recall)).
        - TP : float
            True Positives (Predicted Positive ∩ Actual Positive).
        - FN : float
            False Negatives (Actual Positive - TP).
        - FP : float
            False Positives (Predicted Positive - TP).
        - TN : float
            True Negatives (Actual Negative - FP).

    Notes
    -----
    - Precision, Recall, and F1 Score are key metrics for evaluating classification models.
    - This function assumes the data is binary (positive/negative classification).
    """

    # Calculate Actual Negatives (AN)
    actual_negative = total_samples - actual_positive

    # Calculate True Positives (TP)
    TP = min(predicted_positive, actual_positive)

    # Calculate False Positives (FP)
    FP = predicted_positive - TP

    # Calculate False Negatives (FN)
    FN = actual_positive - TP

    # Calculate True Negatives (TN)
    TN = actual_negative - FP

    # Calculate Precision
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0

    # Calculate Recall
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    # Calculate F1 Score
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) != 0
        else 0
    )

    return precision, recall, f1_score, TP, FN, FP, TN
