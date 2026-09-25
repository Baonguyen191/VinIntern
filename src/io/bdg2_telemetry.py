"""Long-format BDG2 telemetry for the anomaly pipelines (replaces data_normalized/).

BDG2 meter files are wide (one column per building, hourly, 2016–2017). This
module turns them into long frames with the columns the anomaly code uses:
entity_id, ts, value, plus temperature, humidity and day_type for electricity.
Results are cached as parquet under data/bdg2/cache/.
"""
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from src.io.bdg2_loader import BDG2_DIR, get_bdg2_site

CACHE_DIR = BDG2_DIR.parent / "cache"
METER_FILES = {
    ("electricity", "raw"): "meters/raw/electricity.csv",
    ("electricity", "cleaned"): "meters/cleaned/electricity_cleaned.csv",
    ("chilledwater", "raw"): "meters/raw/chilledwater.csv",
    ("chilledwater", "cleaned"): "meters/cleaned/chilledwater_cleaned.csv",
}
WORKDAY, WEEKEND, HOLIDAY = "ngay_lam_viec", "cuoi_tuan", "le"


def _metadata() -> pd.DataFrame:
    return pd.read_csv(BDG2_DIR / "metadata" / "metadata.csv")


def _country(timezone: str | float, site: str) -> str:
    """Same rule as bdg2_loader.get_bdg2_country, without re-reading metadata."""
    if pd.isna(timezone):
        return "US"
    if "Canada" in timezone or (timezone == "US/Eastern" and site == "Moose"):
        return "CA"
    if "Europe" in timezone:
        return "UK"
    return "US"


def _read_wide(meter: str, variant: str, buildings: list[str] | None) -> pd.DataFrame:
    path = BDG2_DIR / METER_FILES[(meter, variant)]
    header = pd.read_csv(path, nrows=0).columns
    cols = [c for c in header if c != "timestamp" and (buildings is None or c in buildings)]
    df = pd.read_csv(path, usecols=["timestamp", *cols], parse_dates=["timestamp"])
    return df.set_index("timestamp").astype("float32")


def _weather() -> pd.DataFrame:
    w = pd.read_csv(
        BDG2_DIR / "weather" / "weather.csv",
        usecols=["timestamp", "site_id", "airTemperature", "dewTemperature"],
        parse_dates=["timestamp"],
    )
    w = w.sort_values(["site_id", "timestamp"])
    for col in ("airTemperature", "dewTemperature"):
        w[col] = w.groupby("site_id")[col].transform(lambda s: s.interpolate(limit=6))
    t, td = w["airTemperature"].to_numpy(), w["dewTemperature"].to_numpy()
    rh = 100.0 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t))
    return pd.DataFrame({
        "site": w["site_id"].to_numpy(),
        "ts": w["timestamp"].to_numpy(),
        "temperature": w["airTemperature"].astype("float32").to_numpy(),
        "humidity": np.clip(rh, 0, 100).astype("float32"),
    })


def _day_types(ts: pd.DatetimeIndex, country: str) -> np.ndarray:
    years = sorted(set(ts.year))
    hol = holidays.country_holidays(country, years=years)
    dates = ts.normalize()
    is_hol = np.array([d in hol for d in dates.date])
    return np.where(is_hol, HOLIDAY, np.where(ts.dayofweek >= 5, WEEKEND, WORKDAY))


def load_telemetry(
    meter: str = "electricity",
    variant: str = "cleaned",
    usage: str | None = "Office",
    buildings: list[str] | None = None,
    start: str | None = "2016-01-01",
    end: str | None = "2017-12-31 23:00",
    with_context: bool = True,
) -> pd.DataFrame:
    """Long BDG2 telemetry: entity_id, ts, value [, temperature, humidity, day_type, primaryspaceusage].

    `usage` filters by metadata primaryspaceusage; `buildings` restricts further.
    Rows with a missing reading are dropped (missing = absent row, as in real feeds).
    """
    key = f"{meter}_{variant}_{usage or 'all'}_{start or ''}_{end or ''}_{int(with_context)}"
    key = key.replace(":", "").replace(" ", "T")
    cache = CACHE_DIR / f"{key}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        meta = _metadata()
        if usage is not None:
            meta = meta[meta["primaryspaceusage"] == usage]
        wide = _read_wide(meter, variant, meta["building_id"].tolist())
        wide = wide.loc[start:end].copy()
        df = (
            wide.rename_axis("ts").reset_index()
            .melt(id_vars="ts", var_name="entity_id", value_name="value")
            .dropna(subset=["value"])
        )
        if with_context:
            df = _add_context(df, meta)
        df = df.sort_values(["entity_id", "ts"], ignore_index=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)

    if buildings is not None:
        df = df[df["entity_id"].isin(buildings)].reset_index(drop=True)
    return df


def _add_context(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    df["site"] = df["entity_id"].map(get_bdg2_site)
    df = df.merge(_weather(), on=["site", "ts"], how="left")

    info = meta.set_index("building_id")
    df["primaryspaceusage"] = df["entity_id"].map(info["primaryspaceusage"])
    site_country = {
        site: _country(tz, site)
        for site, tz in info.groupby(info.index.map(get_bdg2_site))["timezone"].first().items()
    }
    df["day_type"] = ""
    for site, idx in df.groupby("site").groups.items():
        df.loc[idx, "day_type"] = _day_types(pd.DatetimeIndex(df.loc[idx, "ts"]), site_country[site])
    df["day_type"] = df["day_type"].astype("category")
    return df.drop(columns=["site"])


def load_electricity(**kwargs) -> pd.DataFrame:
    """Hourly electricity (kWh per hour = average kW) with weather and day type."""
    return load_telemetry("electricity", **kwargs)


def load_cooling(buildings: list[str] | None = None, variant: str = "cleaned", **kwargs) -> pd.DataFrame:
    """Hourly chilled-water energy for the given buildings: entity_id, ts, value."""
    return load_telemetry(
        "chilledwater", variant=variant, usage=None, buildings=buildings, with_context=False, **kwargs
    )
