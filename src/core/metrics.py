import numpy as np


def cv_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
    y_mean = np.mean(y_true)
    rmse = np.sqrt(np.sum((y_true - y_pred) ** 2) / n)
    return float(rmse / y_mean * 100)


def nmbe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
    y_mean = np.mean(y_true)
    return float(np.sum(y_true - y_pred) / (n * y_mean) * 100)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "cv_rmse": cv_rmse(y_true, y_pred),
        "nmbe": nmbe(y_true, y_pred),
        "r_squared": r_squared(y_true, y_pred),
    }
