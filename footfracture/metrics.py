import numpy as np


def calculate_metrics(matrix):
    """
    Calculate sensitivity, specificity, and accuracy from a 2x2 matrix.

    Parameters:
    matrix (dict): A dictionary containing the values of the 2x2 matrix with keys:
                   - TP: True Positives
                   - FN: False Negatives
                   - FP: False Positives
                   - TN: True Negatives

    Returns:
    dict: A dictionary containing the sensitivity, specificity, and accuracy.
    """
    TP = matrix.get('TP', 0)
    FN = matrix.get('FN', 0)
    FP = matrix.get('FP', 0)
    TN = matrix.get('TN', 0)

    # Calculate sensitivity
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0

    # Calculate specificity
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

    # Calculate accuracy
    accuracy = (TP + TN) / (TP + FN + FP + TN) if (TP + FN + FP + TN) != 0 else 0

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy
    }


def calculate_multiclass_metrics(confusion_matrix):
    """Per-class (one-vs-rest) sensitivity, specificity and accuracy for an NxN confusion matrix."""
    # True Positives, False Positives, True Negatives, False Negatives
    TP = np.diag(confusion_matrix)
    FP = np.sum(confusion_matrix, axis=0) - TP
    FN = np.sum(confusion_matrix, axis=1) - TP
    TN = np.sum(confusion_matrix) - (TP + FP + FN)
    
    # Sensitivity, Specificity, Accuracy
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    total_accuracy = (TP + TN) / np.sum(confusion_matrix)
    total_accuracy = TP / np.sum(confusion_matrix)
    
    metrics = {
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Total Accuracy': total_accuracy,
    }
    
    # Create the metrics string in a table format
    metrics_str = "Metric          Class 1         Class 2         Class 3\n"
    for key, value in metrics.items():
        metrics_str += f"{key:<15}"
        for v in value:
            metrics_str += f"{v:<15.6f}"
        metrics_str += "\n"
    
    return metrics_str


if __name__ == '__main__':
    # Example usage: step-2 (fracture vs. edema) validation confusion matrix,
    # with fracture as the positive class
    matrix = {
        'TP': 34,  # True Positives
        'FN': 3,   # False Negatives
        'FP': 6,   # False Positives
        'TN': 41   # True Negatives
    }

    results = calculate_metrics(matrix)
    print(f"Sensitivity: {results['sensitivity']:.4f}")
    print(f"Specificity: {results['specificity']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")

    # 3-way example: rows = true class, columns = predicted class
    conf_matrix = np.array([[390, 56, 2],
                            [33, 425, 14],
                            [9, 21, 278]])
    print(calculate_multiclass_metrics(conf_matrix))
