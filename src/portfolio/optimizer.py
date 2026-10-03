import numpy as np


def calculate_weights(predictions: np.ndarray) -> np.ndarray:
    """Convert non-negative return predictions into long-only weights."""
    predictions = np.maximum(np.asarray(predictions), 0)

    if predictions.sum() == 0:
        return np.ones(len(predictions)) / len(predictions)

    return predictions / predictions.sum()
