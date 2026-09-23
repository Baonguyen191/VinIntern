import pandas as pd

TAIWAN_HOLIDAYS = {
    # 2015
    "2015-01-01", "2015-02-18", "2015-02-19", "2015-02-20", "2015-02-21",
    "2015-02-28", "2015-04-04", "2015-04-05",
    "2015-06-20", "2015-09-27", "2015-10-10",
    # 2016
    "2016-01-01", "2016-02-07", "2016-02-08", "2016-02-09", "2016-02-10",
    "2016-02-28", "2016-02-29", "2016-04-04",
    "2016-06-09", "2016-09-15", "2016-10-10",
    # 2017
    "2017-01-01", "2017-01-02", "2017-01-27", "2017-01-28", "2017-01-29",
    "2017-01-30", "2017-02-28", "2017-04-04",
    "2017-05-30", "2017-10-04", "2017-10-10",
    # 2018
    "2018-01-01", "2018-02-15", "2018-02-16", "2018-02-17", "2018-02-18",
    "2018-02-28", "2018-04-04", "2018-04-05",
    "2018-06-18", "2018-09-24", "2018-10-10",
    # 2019
    "2019-01-01", "2019-02-04", "2019-02-05", "2019-02-06", "2019-02-07",
    "2019-02-28", "2019-04-04", "2019-04-05",
    "2019-06-07", "2019-09-13", "2019-10-10",
    # 2020
    "2020-01-01", "2020-01-24", "2020-01-25", "2020-01-26", "2020-01-27",
    "2020-02-28", "2020-04-04",
    "2020-06-25", "2020-10-01", "2020-10-10",
    # 2021
    "2021-01-01", "2021-02-11", "2021-02-12", "2021-02-13", "2021-02-14",
    "2021-02-28", "2021-03-01", "2021-04-04",
    "2021-06-14", "2021-09-21", "2021-10-10", "2021-10-11",
    # 2022
    "2022-01-01", "2022-01-31", "2022-02-01", "2022-02-02", "2022-02-03",
    "2022-02-28", "2022-04-04", "2022-04-05",
    "2022-06-03", "2022-09-10", "2022-10-10",
}
_HOLIDAY_SET = {pd.Timestamp(d) for d in TAIWAN_HOLIDAYS}


def merge_electricity_weather(
    elec_df: pd.DataFrame, weather_df: pd.DataFrame
) -> pd.DataFrame:
    # Electricity timestamps are end-of-interval (00:15 = period 00:00-00:15).
    # Weather timestamps are start-of-interval (00:00 = same period).
    # Shift electricity back 15 min to align.
    elec = elec_df.copy()
    elec.index = elec.index - pd.Timedelta(minutes=15)
    merged = elec.join(weather_df, how="left")
    merged = merged.dropna(subset=["Temperature(F)"])
    return merged


STORM_PREFIXES = ("08", "09", "10", "11")


def build_ew_daily_flags(
    ew_df: pd.DataFrame, date_index: pd.DatetimeIndex
) -> pd.Series:
    storm_days: set[pd.Timestamp] = set()
    for _, row in ew_df.iterrows():
        if str(row["Weather"])[:2] in STORM_PREFIXES:
            start = pd.Timestamp(row["Start Time"]).normalize()
            end = pd.Timestamp(row["End Time"]).normalize()
            for d in pd.date_range(start, end, freq="D"):
                storm_days.add(d)
    return pd.Series(
        date_index.normalize().isin(storm_days), index=date_index, name="is_storm"
    )


def aggregate_daily(merged_15min: pd.DataFrame) -> pd.DataFrame:
    daily = merged_15min.resample("D").agg(
        {"Value": "sum", "Temperature(F)": "mean", "Humidity(%)": "mean"}
    )
    daily.columns = ["kWh", "temp_f", "humidity"]
    daily["day_of_week"] = daily.index.dayofweek
    daily["month"] = daily.index.month
    daily["is_holiday"] = daily.index.normalize().isin(_HOLIDAY_SET).astype(int)
    return daily.dropna()
