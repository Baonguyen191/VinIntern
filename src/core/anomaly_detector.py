import numpy as np
import pandas as pd


def detect_point_anomalies(
    residuals: np.ndarray, k: float = 3.0
) -> tuple[np.ndarray, float]:
    sigma = np.std(residuals)
    threshold = k * sigma
    mask = np.abs(residuals) > threshold
    return mask, float(threshold)


def detect_drift_cusum(
    residuals: np.ndarray, k_slack: float = 0.5, h: float = 5.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = np.mean(residuals)
    sigma = np.std(residuals)
    n = len(residuals)

    s_plus = np.zeros(n)
    s_minus = np.zeros(n)

    h_threshold = h * sigma
    drift_up = np.zeros(n, dtype=bool)
    drift_down = np.zeros(n, dtype=bool)

    for i in range(1, n):
        s_plus[i] = max(0, s_plus[i - 1] + (residuals[i] - mu) - k_slack * sigma)
        s_minus[i] = max(0, s_minus[i - 1] - (residuals[i] - mu) - k_slack * sigma)
        if s_plus[i] > h_threshold:
            drift_up[i] = True
            s_plus[i] = 0
        if s_minus[i] > h_threshold:
            drift_down[i] = True
            s_minus[i] = 0

    return s_plus, s_minus, drift_up, drift_down


def detect_context_anomalies(
    daily: pd.DataFrame,
    residuals: np.ndarray,
    y_pred: np.ndarray,
    pct_threshold: float = 0.3,
) -> np.ndarray:
    temp = daily["temp_f"].values
    q25, q75 = np.percentile(temp, 25), np.percentile(temp, 75)
    temp_normal = (temp >= q25) & (temp <= q75)
    high_residual = residuals > pct_threshold * y_pred
    return temp_normal & high_residual


def classify_anomalies(
    dates: pd.DatetimeIndex,
    residuals: np.ndarray,
    point_mask: np.ndarray,
    drift_up: np.ndarray,
    drift_down: np.ndarray,
    context_mask: np.ndarray,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": dates,
            "residual": residuals,
            "point_anomaly": point_mask,
            "drift_up": drift_up,
            "drift_down": drift_down,
            "context_anomaly": context_mask,
        }
    )
    df["is_anomaly"] = point_mask | drift_up | drift_down | context_mask

    def _label(row):
        parts = []
        if row["point_anomaly"]:
            parts.append("point")
        if row["drift_up"]:
            parts.append("drift_up")
        if row["drift_down"]:
            parts.append("drift_down")
        if row["context_anomaly"]:
            parts.append("context")
        return ",".join(parts) if parts else "normal"

    df["anomaly_type"] = df.apply(_label, axis=1)
    return df
