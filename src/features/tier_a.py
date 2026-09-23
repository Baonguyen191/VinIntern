import numpy as np
import pandas as pd

TOW_SLOTS = 7 * 96  # 672


def time_of_week_index(timestamps: pd.DatetimeIndex) -> np.ndarray:
    return timestamps.dayofweek * 96 + timestamps.hour * 4 + timestamps.minute // 15


def compute_temperature_breakpoints(
    temperatures: np.ndarray, n_breakpoints: int = 6
) -> np.ndarray:
    quantiles = np.linspace(0, 100, n_breakpoints + 2)[1:-1]
    return np.percentile(temperatures, quantiles)


def piecewise_linear_temperature(
    temperatures: np.ndarray, breakpoints: np.ndarray
) -> np.ndarray:
    return np.maximum(0, temperatures[:, None] - breakpoints[None, :])


def build_tier_a_features(
    df: pd.DataFrame,
    breakpoints: np.ndarray | None = None,
    n_breakpoints: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    tow_idx = time_of_week_index(df.index)
    tow_onehot = np.zeros((len(df), TOW_SLOTS), dtype=np.float32)
    tow_onehot[np.arange(len(df)), tow_idx] = 1.0

    temp = df["Temperature(F)"].values.astype(np.float64)
    if breakpoints is None:
        breakpoints = compute_temperature_breakpoints(temp, n_breakpoints)
    pw_temp = piecewise_linear_temperature(temp, breakpoints).astype(np.float32)

    X = np.hstack([tow_onehot, pw_temp])
    return X, breakpoints
