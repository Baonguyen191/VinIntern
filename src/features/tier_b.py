import numpy as np
import pandas as pd

SHAPE_STATS = [
    "near_base_load",
    "near_peak_load",
    "rise_time",
    "fall_time",
    "high_load_duration",
    "average_load",
]


def compute_daily_shape_stats(df: pd.DataFrame) -> pd.DataFrame:
    daily_groups = df.groupby(df.index.date)

    records = []
    for date, group in daily_groups:
        values = group["Value"].values
        if len(values) < 48:
            continue

        near_base = np.percentile(values, 2.5)
        near_peak = np.percentile(values, 95.0)
        avg_load = np.mean(values)

        min_idx = int(np.argmin(values))
        max_idx = int(np.argmax(values))
        rise_time = abs(max_idx - min_idx)

        midpoint = (near_base + near_peak) / 2
        after_peak = values[max_idx:]
        below_mid = np.where(after_peak < midpoint)[0]
        fall_time = int(below_mid[0]) if len(below_mid) > 0 else len(after_peak)

        high_load_duration = int(np.sum(values > midpoint))

        records.append(
            {
                "date": pd.Timestamp(date),
                "near_base_load": near_base,
                "near_peak_load": near_peak,
                "rise_time": float(rise_time),
                "fall_time": float(fall_time),
                "high_load_duration": float(high_load_duration),
                "average_load": avg_load,
            }
        )

    return pd.DataFrame(records).set_index("date")


def compute_daily_weather(merged_df: pd.DataFrame) -> pd.DataFrame:
    daily = merged_df.groupby(merged_df.index.date)["Temperature(F)"].mean()
    daily.index = pd.DatetimeIndex(daily.index)
    return daily.to_frame()


def build_tier_b_features(
    daily_stats: pd.DataFrame,
    weather_daily: pd.DataFrame,
    breakpoints: np.ndarray | None = None,
    n_breakpoints: int = 6,
) -> tuple[np.ndarray, dict, np.ndarray, pd.DatetimeIndex]:
    common_dates = daily_stats.index.intersection(weather_daily.index)
    daily_stats = daily_stats.loc[common_dates]
    weather_daily = weather_daily.loc[common_dates]

    dow = daily_stats.index.dayofweek
    dow_onehot = np.zeros((len(daily_stats), 7), dtype=np.float32)
    dow_onehot[np.arange(len(daily_stats)), dow] = 1.0

    temp = weather_daily["Temperature(F)"].values.astype(np.float64)
    if breakpoints is None:
        quantiles = np.linspace(0, 100, n_breakpoints + 2)[1:-1]
        breakpoints = np.percentile(temp, quantiles)
    pw_temp = np.maximum(0, temp[:, None] - breakpoints[None, :]).astype(np.float32)

    X = np.hstack([dow_onehot, pw_temp])
    y_dict = {stat: daily_stats[stat].values for stat in SHAPE_STATS}

    return X, y_dict, breakpoints, common_dates
