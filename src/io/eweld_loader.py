import pandas as pd
from pathlib import Path

EWELD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "eweld"

CLUSTER_TO_STATION = {"CT1": "W1", "CT2": "W2", "CT3": "W3"}


def load_electricity(user_id: str) -> pd.DataFrame:
    ec_dir = EWELD_DIR / "Electricity Consumption"
    for sector_dir in ec_dir.iterdir():
        if not sector_dir.is_dir():
            continue
        for sub_dir in sector_dir.iterdir():
            if not sub_dir.is_dir():
                continue
            csv_path = sub_dir / f"{user_id}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, parse_dates=["Time"])
                df = df.set_index("Time").sort_index()
                return df
    raise FileNotFoundError(f"No electricity data for {user_id}")


def load_weather(station_id: str) -> pd.DataFrame:
    path = EWELD_DIR / "Weather Data" / f"{station_id}.csv"
    df = pd.read_csv(path, parse_dates=["Time"])
    df = df.set_index("Time").sort_index()
    return df


def load_extreme_weather(cluster_id: str) -> pd.DataFrame:
    ew_dir = EWELD_DIR / "Extreme Weather" / f"EW_{cluster_id}"
    frames = []
    for f in sorted(ew_dir.glob("*_interval.csv")):
        sub = pd.read_csv(f, parse_dates=["Start Time", "End Time"])
        if not sub.empty:
            frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["Time", "Start Time", "End Time", "Weather"])
    return pd.concat(frames, ignore_index=True)


def load_user_cluster(cluster_id: str) -> list[str]:
    path = EWELD_DIR / "User Location" / f"U_{cluster_id}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df["User No."].tolist()


def get_user_cluster(user_id: str) -> str:
    for cid in ("CT1", "CT2", "CT3"):
        if user_id in load_user_cluster(cid):
            return cid
    raise ValueError(f"{user_id} not found in any cluster")


def get_weather_station(cluster_id: str) -> str:
    return CLUSTER_TO_STATION[cluster_id]
