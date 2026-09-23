import numpy as np
import pandas as pd

try:
    import holidays as _holidays_lib
    _HAS_HOLIDAYS = True
except ImportError:
    _HAS_HOLIDAYS = False

from ..io.eweld_merger import TAIWAN_HOLIDAYS

# Extended Taiwan holiday blocks (Lunar New Year multi-day closures)
_TAIWAN_EXTENDED_HOLIDAYS: dict[int, list[tuple[str, str]]] = {
    2016: [("2016-02-06", "2016-02-14")],
    2017: [("2017-01-27", "2017-02-01")],
    2018: [("2018-02-14", "2018-02-20")],
    2019: [("2019-02-02", "2019-02-10")],
    2020: [("2020-01-23", "2020-01-29")],
    2021: [("2021-02-10", "2021-02-16")],
    2022: [("2022-01-29", "2022-02-06")],
}


def _get_extended_holiday_dates(country: str, years: list[int]) -> set[pd.Timestamp]:
    dates: set[pd.Timestamp] = set()
    if country == "TW":
        for y in years:
            for start, end in _TAIWAN_EXTENDED_HOLIDAYS.get(y, []):
                for d in pd.date_range(start, end, freq="D"):
                    dates.add(d)
        dates |= {pd.Timestamp(d) for d in TAIWAN_HOLIDAYS}
    elif _HAS_HOLIDAYS:
        if country == "CA":
            cal = _holidays_lib.Canada(years=years)
        elif country == "UK":
            cal = _holidays_lib.UnitedKingdom(years=years)
        else:
            cal = _holidays_lib.US(years=years)
        for d in cal.keys():
            dates.add(pd.Timestamp(d))
    return dates


def clean_daily_consumption(
    daily: pd.DataFrame,
    zero_pct: float = 5.0,
    mad_window: int = 31,
    mad_k: float = 3.0,
    country: str = "TW",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean daily electricity consumption data.

    Returns (cleaned_daily, outlier_report).
    cleaned_daily has outlier kWh replaced with NaN.
    outlier_report has columns: date, kWh, reason.
    """
    df = daily.copy()
    outliers: list[dict] = []

    # Step 1: Adaptive zero-drop filter
    threshold = np.percentile(df["kWh"].dropna().values, zero_pct)
    threshold = max(threshold, 1.0)  # floor at 1 kWh
    zero_mask = df["kWh"] < threshold
    for idx in df.index[zero_mask]:
        outliers.append({"date": idx, "kWh": df.loc[idx, "kWh"], "reason": "zero_drop"})
    df.loc[zero_mask, "kWh"] = np.nan

    # Step 2: Rolling MAD outlier removal
    valid = df["kWh"].copy()
    rolling_med = valid.rolling(window=mad_window, center=True, min_periods=mad_window // 2).median()
    rolling_mad = valid.rolling(window=mad_window, center=True, min_periods=mad_window // 2).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )
    scaled_mad = 1.4826 * rolling_mad
    lower = rolling_med - mad_k * scaled_mad
    upper = rolling_med + mad_k * scaled_mad
    mad_mask = (valid < lower) | (valid > upper)
    mad_mask = mad_mask & ~zero_mask  # don't double-count
    for idx in df.index[mad_mask]:
        if not pd.isna(df.loc[idx, "kWh"]):
            outliers.append({"date": idx, "kWh": df.loc[idx, "kWh"], "reason": "rolling_mad"})
    df.loc[mad_mask, "kWh"] = np.nan

    # Step 3: Extended holiday masking
    years = sorted(df.index.year.unique().tolist())
    ext_holidays = _get_extended_holiday_dates(country, years)
    hol_mask = df.index.normalize().isin(ext_holidays) & df["kWh"].notna()
    for idx in df.index[hol_mask]:
        outliers.append({"date": idx, "kWh": df.loc[idx, "kWh"], "reason": "holiday"})
    df.loc[hol_mask, "kWh"] = np.nan

    report = pd.DataFrame(outliers)
    return df, report
