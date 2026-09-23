import pandas as pd

try:
    import holidays as _holidays_lib

    def _make_holiday_set(country: str, years: list[int]) -> set[pd.Timestamp]:
        if country == "CA":
            cal = _holidays_lib.Canada(years=years)
        elif country == "UK":
            cal = _holidays_lib.UnitedKingdom(years=years)
        else:
            cal = _holidays_lib.US(years=years)
        return {pd.Timestamp(d) for d in cal.keys()}

except ImportError:
    def _make_holiday_set(country: str, years: list[int]) -> set[pd.Timestamp]:
        return set()


def merge_bdg2_electricity_weather(
    elec_df: pd.DataFrame, weather_df: pd.DataFrame
) -> pd.DataFrame:
    # BDG2 is hourly start-of-interval — no shift needed (unlike EWELD 15-min)
    merged = elec_df.join(weather_df, how="left")
    merged = merged.dropna(subset=["Temperature(F)"])
    return merged


def aggregate_bdg2_daily(
    merged_hourly: pd.DataFrame, country: str = "US"
) -> pd.DataFrame:
    daily = merged_hourly.resample("D").agg(
        Value=("Value", lambda x: x.sum(min_count=20)),
        temp_f=("Temperature(F)", "mean"),
        humidity=("Humidity(%)", "mean"),
    )
    daily = daily.rename(columns={"Value": "kWh"})
    daily["day_of_week"] = daily.index.dayofweek
    daily["month"] = daily.index.month

    years = sorted(daily.index.year.unique().tolist())
    hol_set = _make_holiday_set(country, years)
    daily["is_holiday"] = daily.index.normalize().isin(hol_set).astype(int)

    return daily.dropna()
