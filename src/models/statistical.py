import numpy as np
import pandas as pd


def _compute_seasonal_index(train_daily: pd.DataFrame) -> dict[int, float]:
    overall_median = train_daily["kWh"].median()
    if overall_median == 0:
        return {m: 1.0 for m in range(1, 13)}
    index = {}
    for m in range(1, 13):
        month_data = train_daily[train_daily["month"] == m]["kWh"]
        index[m] = month_data.median() / overall_median if len(month_data) > 0 else 1.0
    return index


def _compute_trend_slope(kwh_series: pd.Series, window: int = 28) -> float:
    recent = kwh_series.iloc[-window:] if len(kwh_series) >= window else kwh_series
    if len(recent) < 7:
        return 0.0
    x = np.arange(len(recent), dtype=float)
    y = recent.values.astype(float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def compute_dow_baseline(
    train_daily: pd.DataFrame,
    test_daily: pd.DataFrame,
    window: int = 28,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Adaptive per-DOW rolling median with seasonal decomposition and trend."""
    seasonal_idx = _compute_seasonal_index(train_daily)
    overall_median = float(train_daily["kWh"].median())

    cols = ["kWh", "day_of_week", "month"]
    all_data = pd.concat([train_daily[cols], test_daily[cols]])
    all_data = all_data.copy()
    all_data["kWh_deseasoned"] = all_data["kWh"] / all_data["month"].map(seasonal_idx)

    test_start = len(train_daily)
    n_weeks = max(window // 7, 2)

    dow_medians = {}
    for dow in range(7):
        same_dow = train_daily[train_daily["day_of_week"] == dow]["kWh"]
        dow_medians[dow] = float(same_dow.median()) if len(same_dow) > 0 else overall_median

    trend_slope = _compute_trend_slope(train_daily["kWh"], window)

    y_test = test_daily["kWh"].values
    y_pred = np.empty(len(test_daily))

    for i in range(len(test_daily)):
        idx = test_start + i
        dow = int(all_data.iloc[idx]["day_of_week"])
        test_month = int(all_data.iloc[idx]["month"])

        history = all_data.iloc[:idx]
        same_dow_hist = history[history["day_of_week"] == dow]["kWh_deseasoned"]
        if len(same_dow_hist) >= n_weeks:
            base = same_dow_hist.iloc[-n_weeks:].median()
        elif len(same_dow_hist) > 0:
            base = same_dow_hist.median()
        else:
            base = history["kWh_deseasoned"].median()

        slope = _compute_trend_slope(all_data["kWh"].iloc[:idx], window)
        trend_adj = slope * (i + 1) / max(len(test_daily), 1)

        y_pred[i] = base * seasonal_idx.get(test_month, 1.0) + trend_adj

    model_info = {
        "model": "dow_median",
        "overall_median": overall_median,
        "seasonal_index": {str(k): round(v, 6) for k, v in seasonal_idx.items()},
        "dow_medians": {str(k): round(v, 2) for k, v in dow_medians.items()},
        "trend_slope": round(trend_slope, 6),
        "window": window,
        "n_weeks": n_weeks,
        "train_days": len(train_daily),
    }

    return y_test, y_pred, model_info


def compute_overall_baseline(
    train_daily: pd.DataFrame,
    test_daily: pd.DataFrame,
    window: int = 28,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Adaptive rolling median with seasonal decomposition and trend."""
    seasonal_idx = _compute_seasonal_index(train_daily)
    overall_median = float(train_daily["kWh"].median())

    cols = ["kWh", "month"]
    all_data = pd.concat([train_daily[cols], test_daily[cols]])
    all_data = all_data.copy()
    all_data["kWh_deseasoned"] = all_data["kWh"] / all_data["month"].map(seasonal_idx)

    test_start = len(train_daily)
    trend_slope = _compute_trend_slope(train_daily["kWh"], window)

    y_test = test_daily["kWh"].values
    y_pred = np.empty(len(test_daily))

    for i in range(len(test_daily)):
        idx = test_start + i
        test_month = int(all_data.iloc[idx]["month"])

        start = max(0, idx - window)
        base = all_data["kWh_deseasoned"].iloc[start:idx].median()

        slope = _compute_trend_slope(all_data["kWh"].iloc[:idx], window)
        trend_adj = slope * (i + 1) / max(len(test_daily), 1)

        y_pred[i] = base * seasonal_idx.get(test_month, 1.0) + trend_adj

    model_info = {
        "model": "overall_median",
        "overall_median": overall_median,
        "seasonal_index": {str(k): round(v, 6) for k, v in seasonal_idx.items()},
        "trend_slope": round(trend_slope, 6),
        "window": window,
        "train_days": len(train_daily),
    }

    return y_test, y_pred, model_info
