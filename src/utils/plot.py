import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


def plot(
    data: pd.DataFrame,
    ground_truth: pd.DataFrame,
    target_col: str,
    prediction: np.ndarray,
    boundaries: Optional[np.ndarray] = None,
    level: Optional[float] = None,
):
    prediction = prediction[0].tolist()
    length = len(prediction)
    x = data[target_col].tail(length * 2).tolist()
    y = ground_truth[target_col].tolist()

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(x)), x, label="Input Sequence (x)", color="blue", linestyle="--")

    plt.plot(range(len(x), len(x) + len(prediction)), prediction, label='Prediction', color='red', linestyle='--')
    plt.plot(range(len(x), len(x) + len(y)), y, label='True Sequence (y)', color='green')

    if boundaries is not None:
        lower = boundaries[0][0]
        upper = boundaries[0][1]
        plt.fill_between(range(len(x), len(x) + len(prediction)), lower, upper, color="gray", alpha=0.3, label="prediction interval")
    
    plt.legend()
    
    plt.tight_layout()
    plt.show()
