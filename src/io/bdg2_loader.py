import numpy as np
import pandas as pd
from pathlib import Path

BDG2_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "bdg2" / "data"

_metadata_cache: pd.DataFrame | None = None


def _load_metadata() -> pd.DataFrame:
    global _metadata_cache
    if _metadata_cache is None:
        _metadata_cache = pd.read_csv(BDG2_DIR / "metadata" / "metadata.csv")
    return _metadata_cache


def get_bdg2_site(building_id: str) -> str:
    return building_id.split("_")[0]


def get_bdg2_country(building_id: str) -> str:
    meta = _load_metadata()
    row = meta.loc[meta["building_id"] == building_id]
    if row.empty:
        raise ValueError(f"Building {building_id} not found in metadata")
    tz = row.iloc[0]["timezone"]
    if pd.isna(tz):
        return "US"
    if "Canada" in tz or tz == "US/Eastern" and get_bdg2_site(building_id) == "Moose":
        return "CA"
    if "Europe" in tz:
        return "UK"
    return "US"


def load_bdg2_metadata(building_id: str) -> dict:
    meta = _load_metadata()
    row = meta.loc[meta["building_id"] == building_id]
    if row.empty:
        raise ValueError(f"Building {building_id} not found in metadata")
    return row.iloc[0].to_dict()


def load_bdg2_electricity(building_id: str, variant: str = "raw") -> pd.DataFrame:
    if variant == "raw":
        path = BDG2_DIR / "meters" / "raw" / "electricity.csv"
    elif variant == "cleaned":
        path = BDG2_DIR / "meters" / "cleaned" / "electricity_cleaned.csv"
    else:
        raise ValueError(f"variant must be 'raw' or 'cleaned', got '{variant}'")

    df = pd.read_csv(path, usecols=["timestamp", building_id], parse_dates=["timestamp"])
    df = df.rename(columns={"timestamp": "Time", building_id: "Value"})
    df = df.set_index("Time").sort_index()
    return df


def load_bdg2_weather(site_id: str) -> pd.DataFrame:
    df = pd.read_csv(
        BDG2_DIR / "weather" / "weather.csv",
        parse_dates=["timestamp"],
    )
    df = df[df["site_id"] == site_id].copy()
    df = df.set_index("timestamp").sort_index()
    df = df.drop(columns=["site_id"])

    # Interpolate missing dew point before humidity calc
    df["dewTemperature"] = df["dewTemperature"].interpolate(method="linear", limit=6)
    df["airTemperature"] = df["airTemperature"].interpolate(method="linear", limit=6)

    # °C → °F
    df["Temperature(F)"] = df["airTemperature"] * 9.0 / 5.0 + 32.0
    df["Dew Point(F)"] = df["dewTemperature"] * 9.0 / 5.0 + 32.0

    # Relative humidity via Magnus formula
    t = df["airTemperature"].values
    td = df["dewTemperature"].values
    rh = 100.0 * np.exp((17.625 * td) / (243.04 + td)) / np.exp((17.625 * t) / (243.04 + t))
    df["Humidity(%)"] = np.clip(rh, 0, 100)

    # Wind: m/s → mph
    df["Wind Speed(mph)"] = df["windSpeed"] * 2.23694
    df["Wind Gust(mph)"] = np.nan

    # Pressure: hPa → inHg
    df["Pressure(in)"] = df["seaLvlPressure"] * 0.02953

    df["Wind"] = "CALM"
    df["Condition"] = ""

    keep = [
        "Temperature(F)", "Dew Point(F)", "Humidity(%)",
        "Wind", "Wind Speed(mph)", "Wind Gust(mph)",
        "Pressure(in)", "Condition",
    ]
    return df[keep]
