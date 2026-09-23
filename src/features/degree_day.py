import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def find_balance_point(
    daily: pd.DataFrame,
    temp_min: float | None = None,
    temp_max: float | None = None,
    step: float = 0.5,
) -> float:
    temps = daily["temp_f"].values
    y = daily["kWh"].values
    humidity = daily["humidity"].values
    dow = pd.get_dummies(daily["day_of_week"], prefix="dow", dtype=float)
    dow = dow.iloc[:, :-1].values

    if temp_min is None:
        temp_min = float(np.percentile(temps, 5))
    if temp_max is None:
        temp_max = float(np.percentile(temps, 95))

    best_bp, best_rmse = temp_min, np.inf

    for bp in np.arange(temp_min, temp_max + step, step):
        hdd = np.maximum(0, bp - temps)
        cdd = np.maximum(0, temps - bp)
        X = np.column_stack([hdd, cdd, humidity, dow])
        model = LinearRegression()
        model.fit(X, y)
        rmse = np.sqrt(np.mean((y - model.predict(X)) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_bp = float(bp)

    return best_bp


def build_dd_features(
    daily: pd.DataFrame, balance_point: float
) -> tuple[np.ndarray, list[str]]:
    temps = daily["temp_f"].values
    hdd = np.maximum(0, balance_point - temps)
    cdd = np.maximum(0, temps - balance_point)
    humidity = daily["humidity"].values
    is_holiday = daily["is_holiday"].values

    dow = pd.get_dummies(daily["day_of_week"], prefix="dow", dtype=float)
    dow = dow.iloc[:, :-1]

    month = daily.index.month.values.astype(float)
    week = daily.index.isocalendar().week.values.astype(float)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    week_sin = np.sin(2 * np.pi * week / 52)
    week_cos = np.cos(2 * np.pi * week / 52)
    day_index = (daily.index - daily.index[0]).days.values.astype(float)
    trend = day_index / max(day_index.max(), 1)

    names = (["hdd", "cdd", "humidity"] + list(dow.columns)
             + ["is_holiday", "month_sin", "month_cos",
                "week_sin", "week_cos", "trend"])
    X = np.column_stack([hdd, cdd, humidity, dow.values, is_holiday,
                         month_sin, month_cos, week_sin, week_cos, trend])
    return X, names


def build_raw_features(
    daily: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[int]]:
    X = daily[["temp_f", "humidity", "day_of_week", "is_holiday", "month"]].values.copy()
    names = ["temp_f", "humidity", "day_of_week", "is_holiday", "month"]
    categorical_indices = [2, 3, 4]
    return X, names, categorical_indices
